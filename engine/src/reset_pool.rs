//! Replay-state reset pool: episode resets drawn from real human play.
//!
//! Kickoff and `curriculum::random_reset` between them cover only two slices of
//! the state space a Rocket League policy actually has to solve: the opening
//! formation, and unstructured scrambles that no human would ever produce. Both
//! under-sample the states that decide real games — a 50/50 on the wall, a
//! backboard read, a slow roll into the corner with two cars converging. This
//! module loads a corpus of such states, snapshotted tick-by-tick out of parsed
//! GC2 duel replays (`replay/src/reset_pool.rs` writes it), so the curriculum can
//! start episodes from positions humans genuinely reached.
//!
//! The types below are wire-compatible mirrors of `replay/src/reset_pool.rs`.
//! They are **redefined here rather than imported**: `replay/Cargo.toml` already
//! declares `construct-engine = { path = "../engine" }`, so an engine -> replay
//! import would be a dependency cycle.
//!
//! Loading is deliberately infallible (`load_or_empty`). The pool is a 600 MB
//! artifact that lives in `data/`, which is synced separately from the code; a
//! training box whose `data/` is stale or absent must degrade to today's
//! kickoff/random behavior, not take a 192-arena run down with it.

use std::fs::File;
use std::io::{BufRead, BufReader, Read};

use rocketsim_rs::math::{RotMat, Vec3};
use serde::Deserialize;

use crate::sampler::Pcg32;

#[derive(Debug, Clone, Deserialize)]
pub struct BallSpawn {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct CarSpawn {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
    /// `[x, y, z, w]` — converted to a `RotMat` at apply time.
    pub quat: [f32; 4],
    /// `0..1` on the wire, NOT `0..100` like `CarState::boost`.
    pub boost: f32,
    /// `0` = blue, `1` = orange.
    pub team: u8,
    pub on_ground: bool,
}

/// Duels-only by construction: the loader rejects any line whose car count is
/// not 2 (100% of the current corpus is 2). The fixed array saves 44 B/state vs
/// `Vec<CarSpawn>` (~169 MB vs ~216 MB at full scale) and makes the team-size
/// gate in `reset_episode` a compile-time-shaped check. Extending to 2v2/3v3
/// means bucketing the pool by car count, not widening this.
#[derive(Debug, Clone, Deserialize)]
pub struct ResetState {
    pub ball: BallSpawn,
    pub cars: [CarSpawn; 2],
}

/// Why a parsed state was dropped. See `accept` for the rationale per rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Reject {
    OriginBall,
    BallPastGoal,
    FrozenCar,
    CarsOverlap,
    GroundedFrozenHigh,
    NonFinite,
    TeamOrder,
    Implausible,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RejectCounts {
    pub origin_ball: usize,
    pub ball_past_goal: usize,
    pub frozen_car: usize,
    pub cars_overlap: usize,
    pub grounded_frozen_high: usize,
    pub nonfinite: usize,
    pub team_order: usize,
    pub implausible: usize,
    pub malformed: usize,
    pub read: usize,
}

impl RejectCounts {
    fn bump(&mut self, r: Reject) {
        match r {
            Reject::OriginBall => self.origin_ball += 1,
            Reject::BallPastGoal => self.ball_past_goal += 1,
            Reject::FrozenCar => self.frozen_car += 1,
            Reject::CarsOverlap => self.cars_overlap += 1,
            Reject::GroundedFrozenHigh => self.grounded_frozen_high += 1,
            Reject::NonFinite => self.nonfinite += 1,
            Reject::TeamOrder => self.team_order += 1,
            Reject::Implausible => self.implausible += 1,
        }
    }
}

/// `Arena::IsBallScored` fires on `|ball.y| > goalBaseThresholdY + ballRadius`
/// (`RocketSim/src/Sim/Arena/Arena.cpp:900`), i.e. `5124.25 + 91.25`.
const BALL_SCORE_Y: f32 = 5124.25 + 91.25;
/// Ball `|y|` beyond this is rejected. NOT the goal line itself: `step` advances
/// `tick_skip` ticks before the policy acts, so a ball *inside* the goal line
/// travelling goalward crosses `BALL_SCORE_Y` during the very first step and
/// pays a full goal reward for a state the policy had no agency over. A ball is
/// speed-capped at `BALL_MAX_SPEED` (6000 uu/s) every tick, so 8 ticks move it
/// at most `6000 * 8/120 = 400` uu; keeping the band `400 + margin` inside
/// `BALL_SCORE_Y` makes a first-step goal unreachable rather than merely rare.
/// Measured cost: 5.0% of the accepted corpus (46,973 of 938,801 states).
///
/// This bound is only sound because F8 caps ball speed at `MAX_BALL_SPEED`; an
/// uncapped state could out-travel the band on tick 1 before RocketSim clamps.
const GOAL_LINE_Y: f32 = 4800.0;
/// Cars are clamped to `CAR_MAX_ANG_SPEED` (5.5 rad/s) by RocketSim on the first
/// tick, but the *reset obs* the policy acts on is written before any tick, so a
/// junk value reaches the net unclamped and un-normalized (`schema/v0.toml`
/// divides by 5.5, so 96 rad/s is an obs of 17.5 against a design range of ±1).
/// Set above the physical cap, not at it: ~10% of corpus cars sit just over 5.5
/// from parse jitter and are legitimate, while the junk tail is orders of
/// magnitude out.
const MAX_CAR_ANG_SPEED: f32 = 6.0;
/// `CAR_MAX_SPEED` is 2300 uu/s; same reasoning as `MAX_CAR_ANG_SPEED`.
const MAX_CAR_SPEED: f32 = 2400.0;
/// `BALL_MAX_SPEED` is 6000 uu/s. Zero observed rejects today — this is the
/// guard that keeps `GOAL_LINE_Y`'s travel budget honest.
const MAX_BALL_SPEED: f32 = 6100.0;

/// `schema/v0.toml`'s tick_skip. Duplicated as a literal because the filter runs
/// at load time, before any `Schema` exists; the const assert below is what
/// keeps the duplicate honest.
const TICK_SKIP_FOR_BAND: f32 = 8.0;
/// Compile-time proof of the F2 travel budget: a ball sitting exactly at the
/// band boundary, moving goalward at the fastest speed F8 admits, cannot reach
/// the score threshold within one `step`. If anyone widens `GOAL_LINE_Y` back
/// toward the real goal line, raises `MAX_BALL_SPEED`, or bumps tick_skip, the
/// build fails here rather than the corpus quietly resuming first-step goals.
const _: () = assert!(GOAL_LINE_Y + MAX_BALL_SPEED * (TICK_SKIP_FOR_BAND / 120.0) < BALL_SCORE_Y);
/// Octane's hitbox is 118x84x36 uu; centers closer than this genuinely overlap.
const MIN_CAR_SEPARATION: f32 = 100.0;
/// Above this altitude, an `on_ground` car at a dead stop is garbage, not driving.
const GROUNDED_HIGH_Z: f32 = 50.0;

fn is_zero3(v: &[f32; 3]) -> bool {
    v[0] == 0.0 && v[1] == 0.0 && v[2] == 0.0
}

fn all_finite(vs: &[&[f32]]) -> bool {
    vs.iter().all(|s| s.iter().all(|x| x.is_finite()))
}

/// Euclidean magnitude — RocketSim clamps on magnitude (`Car.cpp:198` compares
/// `angVel.length2()`), so the plausibility bounds do too.
fn mag(v: &[f32; 3]) -> f32 {
    (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
}

/// The pool as written by `replay/src/reset_pool.rs` is NOT pre-filtered — it
/// maps every sampled tick unconditionally and never consults v5's
/// `episode_marker` or `replay_demoed`. None of the bad states trip the engine's
/// own `episode::state_is_sane` gate (its bounds are 2-4x looser than anything
/// here), so they would load silently and start episodes from unrecoverable or
/// physically absurd positions. Filtering happens here or not at all.
///
/// Rules are checked in order and the first failure wins, so the counters read
/// as a disjoint partition rather than overlapping tallies.
pub(crate) fn accept(st: &ResetState) -> Result<(), Reject> {
    // F1: dead/despawned-ball frames (post-goal freeze, actor teardown). z=0 is
    // 93 uu BELOW rest height, i.e. the ball spawns inside the floor. Highest-
    // yield single filter (~10% of the corpus). The sub-floor check is not
    // limited to the exact origin: 34 corpus states put the ball as deep as
    // z=-165 (fully buried given the 91.25 uu radius) and 32 put a car below
    // z=0, which is the same spawn-inside-the-geometry hazard the origin case
    // describes.
    if is_zero3(&st.ball.pos)
        || st.ball.pos[2] < 0.0
        || st.cars.iter().any(|c| c.pos[2] < 0.0)
    {
        return Err(Reject::OriginBall);
    }
    // F2: a ball near the goal fires `is_ball_scored()` during the first step,
    // before the policy has acted. See `GOAL_LINE_Y` for why the band sits well
    // inside the actual goal line.
    if st.ball.pos[1].abs() > GOAL_LINE_Y {
        return Err(Reject::BallPastGoal);
    }
    // F3: demo proxy. `CarSpawn` carries no `demoed` flag, but
    // `reconstruct.rs` zeroes a demoed car's vel AND ang_vel, so both being
    // exactly zero identifies one. Loaded naively these become live cars
    // frozen at a demo position.
    if st.cars.iter().any(|c| is_zero3(&c.vel) && is_zero3(&c.ang_vel)) {
        return Err(Reject::FrozenCar);
    }
    // F4: overlapping hitboxes — Bullet fires a separation impulse on step 1.
    let (a, b) = (&st.cars[0].pos, &st.cars[1].pos);
    let d2 = (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2);
    if d2 < MIN_CAR_SEPARATION * MIN_CAR_SEPARATION {
        return Err(Reject::CarsOverlap);
    }
    // F5: frozen garbage grounded at altitude. Deliberately NOT rejecting
    // `on_ground` at altitude WITH velocity — that is legitimate wall and
    // corner-ramp driving, which is exactly the kind of state this lever exists
    // to expose the policy to.
    if st
        .cars
        .iter()
        .any(|c| c.on_ground && c.pos[2] > GROUNDED_HIGH_Z && is_zero3(&c.vel))
    {
        return Err(Reject::GroundedFrozenHigh);
    }
    // F6: free insurance against a future re-parse. Zero observed today.
    if !all_finite(&[&st.ball.pos, &st.ball.vel, &st.ball.ang_vel]) {
        return Err(Reject::NonFinite);
    }
    for c in &st.cars {
        if !all_finite(&[&c.pos, &c.vel, &c.ang_vel, &c.quat, &[c.boost]]) {
            return Err(Reject::NonFinite);
        }
    }
    // F7: blue-then-orange ordering is what lets `apply_replay_state` map
    // `cars[i]` onto `car_ids[i]` without a search. Enforce it, don't assume it.
    if !(st.cars[0].team == 0 && st.cars[1].team == 1) {
        return Err(Reject::TeamOrder);
    }
    // F8: physical plausibility. F6 only proves a value is *finite*, so a bad
    // future re-parse could hand us 1e30 and every rule above would pass it.
    // Today's corpus already carries a junk tail — the reconstructor's own
    // comment (`replay/src/reconstruct.rs:310`) calls out ~500 rad/s spins —
    // whose worst accepted car spins at 96.5 rad/s, 17.5x RocketSim's cap.
    // Those states reach the policy as wildly out-of-range obs and sit only
    // 3.6% under `episode::state_is_sane`'s ang_vel bound, so a re-parse that
    // shifts the tail slightly would start tripping the debug assert and the
    // release-mode blowup containment. Rejects 47 of 938,801 states (0.005%).
    if st.cars.iter().any(|c| mag(&c.ang_vel) > MAX_CAR_ANG_SPEED || mag(&c.vel) > MAX_CAR_SPEED)
        || mag(&st.ball.vel) > MAX_BALL_SPEED
    {
        return Err(Reject::Implausible);
    }
    Ok(())
}

/// Public predicate over the `accept` filter, for callers outside this crate
/// (integration tests) that need to assert a hand-built state's disposition
/// without depending on the crate-private `Reject` taxonomy.
pub fn is_acceptable(st: &ResetState) -> bool {
    accept(st).is_ok()
}

/// Fixed internal reservoir seed. Fixed (not derived from the arena rng) so the
/// same file always yields the same subset — the pool must be reproducible
/// across processes and across training boxes.
const RESERVOIR_SEED: u64 = 0x5245_5345_5400_0001;

/// Longest line the loader will buffer. A real state serializes to ~600 B, so
/// 64 KiB is ~100x headroom; anything longer is a mis-shaped file, not a state.
const MAX_LINE_BYTES: u64 = 1 << 16;

/// Streaming load + filter, returning the kept states and a rejection tally.
/// `None` iff the file could not be opened at all. Never returns an `Err`:
/// malformed lines are counted and skipped, never fatal.
pub(crate) fn load_counted(path: &str, max_states: usize) -> Option<(Vec<ResetState>, RejectCounts)> {
    let f = File::open(path).ok()?;
    let mut counts = RejectCounts::default();
    let mut kept: Vec<ResetState> = Vec::new();
    let mut rng = Pcg32::new(RESERVOIR_SEED);
    // Index of the current ACCEPTED state, which is what drives the reservoir.
    let mut accepted = 0usize;

    // Streamed line by line: the production file is 611 MB, and reading it into
    // a single String would triple peak RSS for no benefit.
    //
    // Bytes, not `lines()`, for two reasons. (1) `lines()` yields a single
    // `Err(InvalidData)` for the whole iterator on one non-UTF-8 byte; the old
    // handler broke out of the loop and returned down the SUCCESS path, so one
    // flipped byte 10k lines into a 1.08M-line file silently truncated the pool
    // to a biased prefix of a handful of matches while the summary still
    // reported a healthy keep rate. Decoding per line downgrades that to one
    // malformed line. (2) `lines()` buffers each line unboundedly, so a
    // mis-shaped file (a single-line JSON array instead of JSONL) is read whole
    // into RAM before being discarded — a 611 MB transient spike on a box
    // already holding 192 arenas.
    let mut rdr = BufReader::new(f);
    let mut raw: Vec<u8> = Vec::with_capacity(1024);
    'lines: loop {
        raw.clear();
        match (&mut rdr).take(MAX_LINE_BYTES).read_until(b'\n', &mut raw) {
            Ok(0) => break,
            Ok(_) => {}
            // A genuine IO error (e.g. `path` is a directory) would otherwise
            // repeat forever. Keep what we have, but say so — a silently short
            // pool is the same seed and config producing different trajectories
            // on different boxes, with nothing downstream able to notice.
            Err(e) => {
                eprintln!(
                    "[curriculum] WARNING: replay pool '{path}' read error after {} lines \
                     ({e}); pool TRUNCATED at {} states",
                    counts.read,
                    kept.len()
                );
                break;
            }
        }
        counts.read += 1;
        // Hit the cap without reaching a newline: an over-long line. Count it
        // malformed and discard the remainder through the next newline, reusing
        // `raw` so the skip itself stays bounded.
        if raw.len() as u64 == MAX_LINE_BYTES && !raw.ends_with(b"\n") {
            counts.malformed += 1;
            loop {
                raw.clear();
                match (&mut rdr).take(MAX_LINE_BYTES).read_until(b'\n', &mut raw) {
                    Ok(0) => break 'lines,
                    Ok(_) if raw.ends_with(b"\n") => break,
                    Ok(_) => continue,
                    Err(_) => break 'lines,
                }
            }
            continue;
        }
        let line = match std::str::from_utf8(&raw) {
            Ok(s) => s,
            Err(_) => {
                counts.malformed += 1;
                continue;
            }
        };
        if line.trim().is_empty() {
            counts.malformed += 1;
            continue;
        }
        // Typed derive, never an intermediate `serde_json::Value`: measured at
        // 1.63 s / 172 MB peak for the full file, vs. several times that via
        // `Value`. A wrong car count fails here (serde rejects a 3-element
        // array against `[CarSpawn; 2]`) and lands in `malformed`.
        let st: ResetState = match serde_json::from_str(&line) {
            Ok(s) => s,
            Err(_) => {
                counts.malformed += 1;
                continue;
            }
        };
        if let Err(r) = accept(&st) {
            counts.bump(r);
            continue;
        }
        if max_states == 0 || kept.len() < max_states {
            kept.push(st);
        } else {
            // Single-pass reservoir: unbiased over the whole file and
            // deterministic. Taking the first N instead would be a biased
            // sample of a handful of matches, since the file is in replay order.
            let j = (rng.next_u32() as usize) % (accepted + 1);
            if j < max_states {
                kept[j] = st;
            }
        }
        accepted += 1;
    }
    Some((kept, counts))
}

/// Loads the replay reset pool. **Infallible by contract**: every failure mode
/// returns a (possibly empty) `Vec` and warns on stderr. A missing or corrupt
/// pool must degrade a training box to kickoff/random resets, never panic — a
/// panic here kills a 192-arena run.
pub fn load_or_empty(path: &str, max_states: usize) -> Vec<ResetState> {
    let t0 = std::time::Instant::now();
    let Some((kept, c)) = load_counted(path, max_states) else {
        eprintln!(
            "[curriculum] WARNING: replay pool '{path}' unreadable; \
             replay resets DISABLED, falling back to kickoff/random"
        );
        return Vec::new();
    };
    let pct = if c.read > 0 { 100.0 * kept.len() as f64 / c.read as f64 } else { 0.0 };
    let mb = (kept.len() * std::mem::size_of::<ResetState>()) as f64 / (1024.0 * 1024.0);
    eprintln!(
        "[curriculum] replay pool {path}: {} read, {} kept ({pct:.1}%), rejected: \
         origin_ball {}, frozen_car {}, ball_past_goal {}, cars_overlap {}, \
         grounded_frozen_high {}, nonfinite {}, team_order {}, implausible {}, malformed {}  \
         ({mb:.1} MB, {:.2} s)",
        c.read,
        kept.len(),
        c.origin_ball,
        c.frozen_car,
        c.ball_past_goal,
        c.cars_overlap,
        c.grounded_frozen_high,
        c.nonfinite,
        c.team_order,
        c.implausible,
        c.malformed,
        t0.elapsed().as_secs_f64(),
    );
    kept
}

/// Rotate `v` by unit quaternion `q = [x, y, z, w]` (active rotation,
/// `v' = q * v * q^-1`), via the standard cross-product expansion.
fn rotate_vector(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let qv = [q[0], q[1], q[2]];
    let qw = q[3];
    let t = [
        2.0 * (qv[1] * v[2] - qv[2] * v[1]),
        2.0 * (qv[2] * v[0] - qv[0] * v[2]),
        2.0 * (qv[0] * v[1] - qv[1] * v[0]),
    ];
    let cross_qv_t = [
        qv[1] * t[2] - qv[2] * t[1],
        qv[2] * t[0] - qv[0] * t[2],
        qv[0] * t[1] - qv[1] * t[0],
    ];
    [
        v[0] + qw * t[0] + cross_qv_t[0],
        v[1] + qw * t[1] + cross_qv_t[1],
        v[2] + qw * t[2] + cross_qv_t[2],
    ]
}

/// Build a RocketSim `RotMat` (forward/right/up in world space) from an
/// `[x, y, z, w]` quaternion: the columns are the images of local x/y/z.
///
/// Ported verbatim from `replay/src/reconstruct.rs`. Two hazards:
///
/// 1. **`RotMat::from_quat` is not available as built.** It exists at
///    `rocketsim_rs-0.37.0/src/glam_ext.rs`, but `glam_ext` is behind
///    `#[cfg(feature = "glam")]`, `glam` is not in that crate's default
///    features, and `engine/Cargo.toml` pins `rocketsim_rs = "=0.37.0"` with no
///    features. Flipping a feature on a pinned dep has a far wider blast radius
///    than these 20 tested lines.
/// 2. **This is a second copy of a construction `replay/src/bc_obs.rs` depends
///    on being exact.** `bc_obs` inverts shard-stored quaternions with
///    precisely this construction; drift between the copies would silently skew
///    every exported obs rotation against what the engine feeds the policy.
///    `quat_to_rotmat_matches_replay_construction` is the anti-drift pin.
pub(crate) fn quat_to_rotmat(q: [f32; 4]) -> RotMat {
    let f = rotate_vector(q, [1.0, 0.0, 0.0]);
    let r = rotate_vector(q, [0.0, 1.0, 0.0]);
    let u = rotate_vector(q, [0.0, 0.0, 1.0]);
    RotMat {
        forward: Vec3::new(f[0], f[1], f[2]),
        right: Vec3::new(r[0], r[1], r[2]),
        up: Vec3::new(u[0], u[1], u[2]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "tests/fixtures/reset_pool_mini.jsonl";
    const GOLDEN: &str = r#"{"ball":{"pos":[1377.0392,-1571.8032,93.15],"vel":[355.37018,-410.93005,0.0],"ang_vel":[4.5153623,3.9048707,0.6028991]},"cars":[{"pos":[1136.5748,-1442.4575,17.011116],"vel":[1027.664,-812.5248,0.13396516],"ang_vel":[-0.038042724,0.037549272,-4.791631],"quat":[0.0014666612,0.004585884,-0.32203302,0.9467162],"boost":0.121568635,"team":0,"on_ground":true},{"pos":[1723.522,-569.9986,17.00996],"vel":[1001.0869,-945.01855,-0.0047683716],"ang_vel":[0.0059034913,-0.0074107987,-5.499992],"quat":[0.0026252908,0.004022506,-0.49989405,0.86607325],"boost":0.03512711,"team":1,"on_ground":true}]}"#;

    fn golden() -> ResetState {
        serde_json::from_str(GOLDEN).expect("golden line parses")
    }

    /// Writes `lines` to a uniquely-named temp file and returns its path.
    fn tmp_jsonl(name: &str, lines: &[String]) -> String {
        let p = std::env::temp_dir().join(format!("construct_reset_pool_{name}.jsonl"));
        std::fs::write(&p, lines.join("\n") + "\n").unwrap();
        p.to_str().unwrap().to_string()
    }

    // ---- T1 ----------------------------------------------------------------

    #[test]
    fn parses_golden_line() {
        let s = golden();
        assert_eq!(s.ball.pos, [1377.0392, -1571.8032, 93.15]);
        assert_eq!(s.ball.vel, [355.37018, -410.93005, 0.0]);
        assert_eq!(s.ball.ang_vel, [4.5153623, 3.9048707, 0.6028991]);

        let b = &s.cars[0];
        assert_eq!(b.pos, [1136.5748, -1442.4575, 17.011116]);
        assert_eq!(b.vel, [1027.664, -812.5248, 0.13396516]);
        assert_eq!(b.ang_vel, [-0.038042724, 0.037549272, -4.791631]);
        assert_eq!(b.quat, [0.0014666612, 0.004585884, -0.32203302, 0.9467162]);
        assert_eq!(b.boost, 0.121568635);
        assert_eq!(b.team, 0);
        assert!(b.on_ground);

        let o = &s.cars[1];
        assert_eq!(o.pos, [1723.522, -569.9986, 17.00996]);
        assert_eq!(o.vel, [1001.0869, -945.01855, -0.0047683716]);
        assert_eq!(o.ang_vel, [0.0059034913, -0.0074107987, -5.499992]);
        assert_eq!(o.quat, [0.0026252908, 0.004022506, -0.49989405, 0.86607325]);
        assert_eq!(o.boost, 0.03512711);
        assert_eq!(o.team, 1);
        assert!(o.on_ground);

        assert!(accept(&s).is_ok(), "the golden line must survive the filter");
    }

    // ---- T2..T7: accept filter ---------------------------------------------

    #[test]
    fn rejects_origin_ball() {
        let mut s = golden();
        s.ball.pos = [0.0, 0.0, 0.0];
        assert_eq!(accept(&s), Err(Reject::OriginBall));

        // z=0 alone is not the rule — all three components must be exactly zero.
        let mut t = golden();
        t.ball.pos = [10.0, 0.0, 0.0];
        assert!(accept(&t).is_ok());

        let path = tmp_jsonl("origin_ball", &[serde_json::to_string(&serde_json::json!({
            "ball": {"pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
            "cars": [car_json(0, [100.0, 0.0, 17.0]), car_json(1, [500.0, 0.0, 17.0])],
        })).unwrap()]);
        let (kept, counts) = load_counted(&path, 0).unwrap();
        assert!(kept.is_empty());
        assert_eq!(counts.origin_ball, 1);
    }

    fn car_json(team: u8, pos: [f32; 3]) -> serde_json::Value {
        serde_json::json!({
            "pos": pos, "vel": [10.0, 0.0, 0.0], "ang_vel": [0.1, 0.0, 0.0],
            "quat": [0.0, 0.0, 0.0, 1.0], "boost": 0.5, "team": team, "on_ground": true,
        })
    }

    #[test]
    fn rejects_ball_past_goal_line() {
        let mut s = golden();
        s.ball.pos[1] = 5300.0;
        assert_eq!(accept(&s), Err(Reject::BallPastGoal));

        s.ball.pos[1] = -5300.0;
        assert_eq!(accept(&s), Err(Reject::BallPastGoal), "both signs");

        // Inside the real goal line but inside the first-step travel band too:
        // these used to be accepted, and were measured firing `is_ball_scored`
        // on step 1 at ~2.7e-4 per reset.
        s.ball.pos[1] = 5119.0;
        assert_eq!(accept(&s), Err(Reject::BallPastGoal), "inside the goal line is NOT enough");
        s.ball.pos[1] = -5024.9;
        assert_eq!(accept(&s), Err(Reject::BallPastGoal), "an observed first-step-goal case");

        s.ball.pos[1] = 4799.0;
        assert!(accept(&s).is_ok(), "inside the band is legal");

        s.ball.pos[1] = GOAL_LINE_Y;
        assert!(accept(&s).is_ok(), "boundary is `>`, not `>=`");
    }

    /// The arithmetic that makes "no first-step goal" a proof rather than a
    /// measured rate. If someone widens `GOAL_LINE_Y` back toward the real goal
    /// line, or `tick_skip` grows, this fails before the corpus does.
    #[test]
    fn f2_band_covers_one_tick_skip_of_travel() {
        const TICK_SKIP: f32 = 8.0;
        const TICKS_PER_SEC: f32 = 120.0;
        // F8 caps ball speed at MAX_BALL_SPEED; RocketSim re-clamps to
        // BALL_MAX_SPEED (6000) every tick, so this is the loosest possible
        // per-step displacement.
        let max_travel = MAX_BALL_SPEED * (TICK_SKIP / TICKS_PER_SEC);
        assert!(
            GOAL_LINE_Y + max_travel < BALL_SCORE_Y,
            "a ball at the F2 boundary must not reach the score threshold in one step: \
             {GOAL_LINE_Y} + {max_travel} >= {BALL_SCORE_Y}"
        );
    }

    #[test]
    fn rejects_implausible_magnitudes() {
        // The real corpus worst case: 96.5 rad/s, 17.5x RocketSim's 5.5 cap.
        let mut s = golden();
        s.cars[0].ang_vel = [0.0, 0.0, 96.53176];
        assert_eq!(accept(&s), Err(Reject::Implausible));

        // Just over the physical cap is NOT rejected — ~10% of corpus cars sit
        // there from parse jitter and RocketSim clamps them on tick 1.
        let mut ok = golden();
        ok.cars[0].ang_vel = [0.0, 0.0, 5.9];
        assert!(accept(&ok).is_ok(), "near-cap spin is legitimate");

        // Bound is on MAGNITUDE, matching RocketSim's own clamp.
        let mut m = golden();
        m.cars[1].ang_vel = [3.5, 3.5, 3.5]; // mag 6.06
        assert_eq!(accept(&m), Err(Reject::Implausible), "magnitude, not per-component");

        let mut v = golden();
        v.cars[0].vel = [2500.0, 0.0, 0.0];
        assert_eq!(accept(&v), Err(Reject::Implausible), "car above CAR_MAX_SPEED");

        let mut b = golden();
        b.ball.vel = [0.0, 7000.0, 0.0];
        assert_eq!(accept(&b), Err(Reject::Implausible), "ball above BALL_MAX_SPEED");

        // F6 only proves finiteness — this is the case F8 exists for.
        let mut huge = golden();
        huge.cars[0].vel = [1e30, 0.0, 0.0];
        assert!(huge.cars[0].vel[0].is_finite());
        assert_eq!(accept(&huge), Err(Reject::Implausible), "finite-but-absurd must not pass");
    }

    #[test]
    fn rejects_sub_floor_ball_and_cars() {
        // 34 corpus states bury the ball as deep as z=-165 (radius is 91.25),
        // and 32 put a car below the floor — the same spawn-inside-the-geometry
        // hazard the origin-ball rule describes, which the exact-zero test missed.
        let mut b = golden();
        b.ball.pos[2] = -165.2858;
        assert_eq!(accept(&b), Err(Reject::OriginBall));

        let mut c = golden();
        c.cars[1].pos[2] = -39.721916;
        assert_eq!(accept(&c), Err(Reject::OriginBall));

        let mut ok = golden();
        ok.ball.pos[2] = 0.0;
        assert!(accept(&ok).is_ok(), "resting ON the floor plane is not sub-floor");
    }

    #[test]
    fn rejects_frozen_car_demo_proxy() {
        let mut s = golden();
        s.cars[1].vel = [0.0, 0.0, 0.0];
        s.cars[1].ang_vel = [0.0, 0.0, 0.0];
        assert_eq!(accept(&s), Err(Reject::FrozenCar));

        // The proxy needs BOTH zero: a car can legitimately be momentarily at
        // rest translationally while still rotating.
        let mut t = golden();
        t.cars[1].vel = [0.0, 0.0, 0.0];
        t.cars[1].ang_vel = [0.0, 0.0, 1.5];
        assert!(accept(&t).is_ok(), "zero vel with nonzero ang_vel is a live car");
    }

    #[test]
    fn rejects_overlapping_cars() {
        let mut s = golden();
        s.cars[1].pos = [s.cars[0].pos[0] + 60.0, s.cars[0].pos[1], s.cars[0].pos[2]];
        assert_eq!(accept(&s), Err(Reject::CarsOverlap));

        s.cars[1].pos = [s.cars[0].pos[0] + 101.0, s.cars[0].pos[1], s.cars[0].pos[2]];
        assert!(accept(&s).is_ok());

        s.cars[1].pos = [s.cars[0].pos[0] + 100.0, s.cars[0].pos[1], s.cars[0].pos[2]];
        assert!(accept(&s).is_ok(), "boundary is `<`, not `<=`");
    }

    #[test]
    fn rejects_grounded_frozen_high() {
        // on_ground at altitude WITH velocity is legitimate wall/corner driving.
        let mut ok = golden();
        ok.cars[0].pos[2] = 900.0;
        ok.cars[0].on_ground = true;
        assert!(accept(&ok).is_ok(), "wall driving must survive");

        let mut bad = golden();
        bad.cars[0].pos[2] = 900.0;
        bad.cars[0].on_ground = true;
        bad.cars[0].vel = [0.0, 0.0, 0.0];
        assert_eq!(accept(&bad), Err(Reject::GroundedFrozenHigh));
    }

    #[test]
    fn rejects_nonfinite() {
        for mutate in [
            (|s: &mut ResetState| s.ball.pos[0] = f32::NAN) as fn(&mut ResetState),
            |s| s.ball.vel[1] = f32::NAN,
            |s| s.ball.ang_vel[2] = f32::INFINITY,
            |s| s.cars[0].pos[2] = f32::INFINITY,
            |s| s.cars[0].vel[0] = f32::NAN,
            |s| s.cars[1].ang_vel[1] = f32::NAN,
            |s| s.cars[1].quat[3] = f32::NAN,
            |s| s.cars[1].boost = f32::NEG_INFINITY,
        ] {
            let mut s = golden();
            mutate(&mut s);
            assert_eq!(accept(&s), Err(Reject::NonFinite), "non-finite must be rejected");
        }
    }

    #[test]
    fn rejects_wrong_car_count_and_team_order() {
        // Wrong car counts fail to PARSE (serde rejects them against [CarSpawn; 2])
        // and must be counted malformed rather than panicking.
        let three = serde_json::json!({
            "ball": {"pos": [0.0, 100.0, 93.15], "vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
            "cars": [car_json(0, [0.0, 0.0, 17.0]), car_json(1, [500.0, 0.0, 17.0]), car_json(1, [900.0, 0.0, 17.0])],
        });
        let one = serde_json::json!({
            "ball": {"pos": [0.0, 100.0, 93.15], "vel": [0.0, 0.0, 0.0], "ang_vel": [0.0, 0.0, 0.0]},
            "cars": [car_json(0, [0.0, 0.0, 17.0])],
        });
        let path = tmp_jsonl("car_count", &[three.to_string(), one.to_string()]);
        let (kept, counts) = load_counted(&path, 0).unwrap();
        assert!(kept.is_empty());
        assert_eq!(counts.malformed, 2);

        // Right count, wrong team order -> F7.
        let mut s = golden();
        s.cars[0].team = 1;
        s.cars[1].team = 0;
        assert_eq!(accept(&s), Err(Reject::TeamOrder));

        let mut same = golden();
        same.cars[1].team = 0;
        assert_eq!(accept(&same), Err(Reject::TeamOrder), "both blue is not blue-then-orange");
    }

    // ---- T8..T10: loader ---------------------------------------------------

    #[test]
    fn skips_malformed_lines_without_failing() {
        let good = GOLDEN.to_string();
        let lines = vec![
            good.clone(),
            "{\"ball\":{\"pos\":[1.0,2.0".to_string(), // truncated
            good.clone(),
            String::new(), // empty
            "this is not json at all".to_string(),
            good.clone(),
        ];
        let path = tmp_jsonl("malformed", &lines);
        let (kept, counts) = load_counted(&path, 0).unwrap();
        assert_eq!(kept.len(), 3, "exactly the 3 good lines survive");
        assert_eq!(counts.malformed, 3);
    }

    /// REGRESSION: one non-UTF-8 byte used to abort `lines()` for the whole
    /// iterator, and the handler broke out down the SUCCESS path — so a single
    /// flipped byte halfway through the 1.08M-line corpus silently truncated the
    /// pool to a biased prefix of a handful of matches, with an unchanged
    /// keep-rate in the summary and nothing downstream able to notice.
    #[test]
    fn invalid_utf8_line_does_not_truncate_the_pool() {
        let p = std::env::temp_dir().join("construct_reset_pool_badutf8.jsonl");
        let mut bytes: Vec<u8> = Vec::new();
        for i in 0..20 {
            if i == 10 {
                // a whole line of invalid UTF-8
                bytes.extend_from_slice(&[0xFF, 0xFE, 0xFD]);
                bytes.push(b'\n');
            }
            bytes.extend_from_slice(GOLDEN.as_bytes());
            bytes.push(b'\n');
        }
        std::fs::write(&p, &bytes).unwrap();
        let (kept, counts) = load_counted(p.to_str().unwrap(), 0).unwrap();
        assert_eq!(kept.len(), 20, "every good line after the bad byte must survive");
        assert_eq!(counts.read, 21);
        assert_eq!(counts.malformed, 1, "the bad line is malformed, not fatal");
        let _ = std::fs::remove_file(&p);
    }

    /// A mis-shaped file (a single-line JSON array instead of JSONL) must not be
    /// buffered whole: `lines()` grew one unbounded `String`, so a 611 MB pool
    /// written that way was a 611 MB transient spike on a box already holding
    /// 192 arenas.
    #[test]
    fn over_long_line_is_malformed_not_buffered() {
        let p = std::env::temp_dir().join("construct_reset_pool_giantline.jsonl");
        let mut bytes: Vec<u8> = Vec::new();
        bytes.push(b'[');
        for i in 0..600 {
            if i > 0 {
                bytes.push(b',');
            }
            bytes.extend_from_slice(GOLDEN.as_bytes());
        }
        bytes.extend_from_slice(b"]\n");
        assert!(bytes.len() as u64 > MAX_LINE_BYTES * 3, "fixture must exceed the cap severalfold");
        // A real line after the giant one, to prove the skip resynchronizes on
        // the next newline instead of eating the rest of the file.
        bytes.extend_from_slice(GOLDEN.as_bytes());
        bytes.push(b'\n');
        std::fs::write(&p, &bytes).unwrap();

        let (kept, counts) = load_counted(p.to_str().unwrap(), 0).unwrap();
        assert_eq!(kept.len(), 1, "only the well-formed trailing line is kept");
        assert!(counts.malformed >= 1);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn missing_file_yields_empty_pool() {
        assert!(load_or_empty("/nonexistent/definitely/absent.jsonl", 0).is_empty());
        // A directory is openable but unreadable as lines — must not panic.
        assert!(load_or_empty("/tmp", 0).is_empty());
        assert!(load_or_empty("", 0).is_empty());
    }

    #[test]
    fn max_states_cap_respected_and_deterministic() {
        let full = load_or_empty(FIXTURE, 0);
        // 78 lines: 67 clean, 11 filtered. The count moved from 64 when F2's
        // band tightened to 4800 (dropping 3 previously-accepted near-goal
        // lines) and 6 real corpus states at the new boundary were appended so
        // the integration test for "no first-step goal" stops being vacuous.
        assert_eq!(full.len(), 67, "fixture has 67 clean lines (11 bad are filtered)");

        let a = load_or_empty(FIXTURE, 10);
        let b = load_or_empty(FIXTURE, 10);
        assert_eq!(a.len(), 10);
        // Fixed internal seed -> the same file always yields the same subset.
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.ball.pos, y.ball.pos);
            assert_eq!(x.cars[0].pos, y.cars[0].pos);
            assert_eq!(x.cars[1].quat, y.cars[1].quat);
        }
        // Every sampled state really came from the file.
        for x in &a {
            assert!(full.iter().any(|f| f.ball.pos == x.ball.pos && f.cars[0].pos == x.cars[0].pos));
        }
        // A cap larger than the file is a no-op.
        assert_eq!(load_or_empty(FIXTURE, 1000).len(), 67);
        // The reservoir must not just be the first N lines of a replay-ordered file.
        let prefix: Vec<[f32; 3]> = full.iter().take(10).map(|s| s.ball.pos).collect();
        let sampled: Vec<[f32; 3]> = a.iter().map(|s| s.ball.pos).collect();
        assert_ne!(prefix, sampled, "reservoir must not degenerate to the file prefix");
    }

    // ---- T11..T12: quaternion port -----------------------------------------

    #[test]
    fn quat_to_rotmat_matches_replay_construction() {
        // ANTI-DRIFT PIN. `replay/src/reconstruct.rs::quat_to_rotmat` is the
        // original of this construction, and `replay/src/bc_obs.rs` inverts
        // shard-stored quaternions with EXACTLY it. If this engine-side copy
        // ever drifts, every exported BC obs rotation is silently skewed
        // against what the engine feeds the policy at train time. These
        // literals were produced by the replay-side implementation.
        let q = [0.0014666612, 0.004585884, -0.32203302, 0.9467162];
        let m = quat_to_rotmat(q);
        let close = |a: f32, b: f32, what: &str| {
            assert!((a - b).abs() < 1e-6, "{what}: {a} vs {b}");
        };
        close(m.forward.x, 0.7925474047660828, "forward.x");
        close(m.forward.y, -0.609734296798706, "forward.y");
        close(m.forward.z, -0.009627687744796276, "forward.z");
        close(m.right.x, 0.6097612380981445, "right.x");
        close(m.right.y, 0.7925851345062256, "right.y");
        close(m.right.z, -0.00017658830620348454, "right.z");
        close(m.up.x, 0.007738434709608555, "up.x");
        close(m.up.y, -0.005730636417865753, "up.y");
        close(m.up.z, 0.9999536275863647, "up.z");
    }

    #[test]
    fn quat_to_rotmat_orthonormal() {
        let id = quat_to_rotmat([0.0, 0.0, 0.0, 1.0]);
        assert_eq!((id.forward.x, id.forward.y, id.forward.z), (1.0, 0.0, 0.0));
        assert_eq!((id.right.x, id.right.y, id.right.z), (0.0, 1.0, 0.0));
        assert_eq!((id.up.x, id.up.y, id.up.z), (0.0, 0.0, 1.0));

        let mut rng = Pcg32::new(0xBEEF);
        let dot = |a: &Vec3, b: &Vec3| a.x * b.x + a.y * b.y + a.z * b.z;
        for _ in 0..200 {
            // uniform-ish quats via 4 gaussian-free draws, then normalize
            let raw = [
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
                rng.next_f32() * 2.0 - 1.0,
            ];
            let n = (raw.iter().map(|x| x * x).sum::<f32>()).sqrt();
            if n < 1e-3 {
                continue;
            }
            let q = [raw[0] / n, raw[1] / n, raw[2] / n, raw[3] / n];
            let m = quat_to_rotmat(q);
            for (v, name) in [(&m.forward, "forward"), (&m.right, "right"), (&m.up, "up")] {
                assert!((dot(v, v).sqrt() - 1.0).abs() < 1e-5, "{name} not unit");
            }
            assert!(dot(&m.forward, &m.right).abs() < 1e-5);
            assert!(dot(&m.forward, &m.up).abs() < 1e-5);
            assert!(dot(&m.right, &m.up).abs() < 1e-5);
            // right-handed: forward x right == up
            let cx = m.forward.y * m.right.z - m.forward.z * m.right.y;
            let cy = m.forward.z * m.right.x - m.forward.x * m.right.z;
            let cz = m.forward.x * m.right.y - m.forward.y * m.right.x;
            let det = cx * m.up.x + cy * m.up.y + cz * m.up.z;
            assert!((det - 1.0).abs() < 1e-5, "det must be +1, got {det}");
        }
    }
}
