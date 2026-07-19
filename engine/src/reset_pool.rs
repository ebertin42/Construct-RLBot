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
use std::io::{BufRead, BufReader};

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
        }
    }
}

/// Ball `|y|` beyond this is past the goal line.
const GOAL_LINE_Y: f32 = 5120.0;
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
    // yield single filter (~10% of the corpus).
    if is_zero3(&st.ball.pos) {
        return Err(Reject::OriginBall);
    }
    // F2: ball already past the goal line would fire `is_ball_scored()` on the
    // very first step, terminating the episode before the policy acts.
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
    Ok(())
}

/// Fixed internal reservoir seed. Fixed (not derived from the arena rng) so the
/// same file always yields the same subset — the pool must be reproducible
/// across processes and across training boxes.
const RESERVOIR_SEED: u64 = 0x5245_5345_5400_0001;

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
    for line in BufReader::new(f).lines() {
        let line = match line {
            Ok(l) => l,
            // A read error (e.g. `path` is a directory) would otherwise repeat
            // forever — stop and keep whatever we already have.
            Err(_) => break,
        };
        if line.trim().is_empty() {
            counts.malformed += 1;
            counts.read += 1;
            continue;
        }
        counts.read += 1;
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
         grounded_frozen_high {}, nonfinite {}, team_order {}, malformed {}  ({mb:.1} MB, {:.2} s)",
        c.read,
        kept.len(),
        c.origin_ball,
        c.frozen_car,
        c.ball_past_goal,
        c.cars_overlap,
        c.grounded_frozen_high,
        c.nonfinite,
        c.team_order,
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

        s.ball.pos[1] = 5119.0;
        assert!(accept(&s).is_ok(), "inside the goal line is legal");

        s.ball.pos[1] = 5120.0;
        assert!(accept(&s).is_ok(), "boundary is `>`, not `>=`");
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
        assert_eq!(full.len(), 64, "fixture has 64 clean lines (8 bad are filtered)");

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
        assert_eq!(load_or_empty(FIXTURE, 1000).len(), 64);
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
