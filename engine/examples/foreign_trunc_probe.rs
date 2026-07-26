// INVESTIGATION PROBE (do not ship). Answers: can Element and Immortal --
// the two EASIEST ported bots (at a shared p4 vs ck_001171502080: immortal
// 0.896, element 0.677 vs nexto 0.146) -- drive a car in a 2v2/3v3 arena if we
// hand them a TRUNCATED AdvancedObs (self + exactly ONE other car), instead of
// being locked out of the team blocks by the per-kind guard in
// engine/src/engine.rs?
//
// Their net has a HARD 107-wide input (immortal net.0.weight (512,107), element
// fc1.weight (256,107)) and obs_advanced.rs builds 107 = ball 9 + prev_action 8
// + pads 34 + self 25 + 31 * (other cars) with exactly ONE other car, so the
// width is only satisfiable by showing one car. Upstream RLMarlbot loops over
// all other cars with no padding and no truncation, so upstream crashes at 2v2
// too -- the limitation is in THIS ARTIFACT, not in the idea of the bot. Nothing
// stops US from choosing which single car to show.
//
// The probe drives the bots through the SAME primitives collect uses
// (set_foreign_overrides -> step_v1), with the obs built by cloning the live
// GameState and RETAINING TWO CARS (self + one opponent) before calling the
// unmodified `build_advanced_obs`. That is byte-identical to the real builder at
// 1v1 (asserted below), so the 1v1 rows in the table ARE the reference behaviour
// and every other row is measured against them on the same harness.
//
// Three candidate selectors for "which other car":
//   nearest-self  the opponent closest to me       (the immediate threat)
//   nearest-ball  the opponent closest to the ball (the ball carrier)
//   lowest-id     a fixed opponent                 (no switching, no relevance)
// A 4th row, `mate-ball`, is the DELIBERATELY WRONG control: it puts a TEAMMATE
// in the slot the net was trained to read as the enemy.
//
// Run: cargo run --release --example foreign_trunc_probe   (from engine/)
//      PROBE_STEPS=4000 by default (~266s of sim per row).
use std::collections::HashMap;
use std::fs::File;

use construct_engine::actions;
use construct_engine::curriculum::CurriculumConfig;
use construct_engine::element::ElementNet;
use construct_engine::episode::{EpisodeArena, ObsMode, StepFlags};
use construct_engine::foreign::ForeignMlp;
use construct_engine::obs_advanced::{build_advanced_obs, ADV_OBS_SIZE};
use construct_engine::obs_v1::{ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT};
use construct_engine::reward::RewardConfig;
use construct_engine::schema::Schema;
use construct_engine::sim_init;
use ndarray::IxDyn;
use ndarray_npy::NpzReader;
use rocketsim_rs::sim::Team;
use rocketsim_rs::GameState;

fn weights(name: &str) -> Option<HashMap<String, (Vec<f32>, Vec<usize>)>> {
    let home = std::env::var("HOME").unwrap_or_default();
    let p = std::path::Path::new(&home).join(format!(".cache/construct/{name}_weights.npz"));
    if !p.exists() {
        eprintln!("SKIP: {} absent", p.display());
        return None;
    }
    let mut npz = NpzReader::new(File::open(&p).unwrap()).unwrap();
    let mut map = HashMap::new();
    for n in npz.names().unwrap() {
        let a: ndarray::Array<f32, IxDyn> = npz.by_name(&n).unwrap();
        map.insert(n.trim_end_matches(".npy").to_string(),
                   (a.iter().copied().collect(), a.shape().to_vec()));
    }
    Some(map)
}

struct Rng(u64);
impl Rng {
    fn next_u32(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        (x >> 32) as u32
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sel { NearestSelf, NearestBall, LowestId, MateBall }
impl Sel {
    fn name(self) -> &'static str {
        match self {
            Sel::NearestSelf => "near-self",
            Sel::NearestBall => "near-ball",
            Sel::LowestId => "lowest-id",
            Sel::MateBall => "MATE(ctl)",
        }
    }
}

/// The two nets we can drive. Both consume AdvancedObs-107 and both zero YAW
/// while jump is held (`ForeignKind::zero_yaw_on_jump`).
enum Net { Immortal(ForeignMlp, Vec<[f32; 8]>), Element(ElementNet) }
impl Net {
    fn act(&self, obs: &[f32]) -> [f32; 8] {
        match self {
            Net::Immortal(mlp, table) => match mlp.forward(obs, 1, ADV_OBS_SIZE) {
                Ok(l) => {
                    let i = l.iter().enumerate()
                        .fold((0usize, f32::NEG_INFINITY),
                              |(bi, bv), (i, &x)| if x > bv { (i, x) } else { (bi, bv) }).0;
                    table[i]
                }
                Err(_) => [0.0; 8],
            },
            Net::Element(n) => n.decide(obs).unwrap_or([0.0; 8]),
        }
    }
}

#[inline]
fn d2(a: rocketsim_rs::math::Vec3, b: rocketsim_rs::math::Vec3) -> f32 {
    (a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)
}

/// Index into `gs.cars` of the single car to show `me_j`. Returns None only if
/// there is nobody of the requested kind (never happens in a real arena).
fn pick(gs: &GameState, me_j: usize, sel: Sel) -> Option<usize> {
    let me = &gs.cars[me_j];
    let want_mate = sel == Sel::MateBall;
    let cand: Vec<usize> = (0..gs.cars.len())
        .filter(|&j| j != me_j && ((gs.cars[j].team == me.team) == want_mate))
        .collect();
    let key = |j: usize| -> f32 {
        match sel {
            Sel::NearestSelf => d2(gs.cars[j].state.pos, me.state.pos),
            Sel::NearestBall | Sel::MateBall => d2(gs.cars[j].state.pos, gs.ball.pos),
            Sel::LowestId => gs.cars[j].id as f32,
        }
    };
    cand.into_iter().min_by(|&a, &b| key(a).partial_cmp(&key(b)).unwrap())
}

/// The truncated obs: clone the state, keep [me, other], build the UNMODIFIED
/// 107-float AdvancedObs over that 2-car world. `me` lands at index 0 and
/// `build_advanced_obs`'s ally-then-enemy sort has a single element, so the
/// layout is bit-for-bit the 1v1 layout the net was trained on.
fn trunc_obs(gs: &GameState, me_j: usize, other_j: usize, prev: &[f32; 8], out: &mut [f32]) {
    let mut sub = gs.clone();
    sub.cars = vec![gs.cars[me_j], gs.cars[other_j]];
    build_advanced_obs(&sub, 0, prev, out);
}

#[derive(Default)]
struct Stat {
    goals_for: usize,      // the ported bot's team scored
    goals_against: usize,
    touches: usize,
    car_steps: usize,
    facing: f64,           // mean forward . unit(car->ball)
    approach: f64,         // mean unit(vel) . unit(car->ball), only when moving
    approach_n: usize,
    speed: f64,
    idle: usize,           // steps with |vel| < 200 uu/s
    saturated_steer: usize,
    ground_spin: f64,      // mean |ang_vel.z| while on ground
    ground_n: usize,
    dist: f64,
    jump: usize,
    boost: usize,
    switches: usize,       // shown-car identity changed between consecutive decisions
    distinct: std::collections::HashSet<[u32; 8]>,
    insane: usize,
    ms: u128,
}

/// `net = None` means "orange acts uniformly at random from our own 92-row
/// table" -- the FLOOR row, so every behavioural number has a scale.
#[allow(clippy::too_many_arguments)]
fn run(net: Option<&Net>, m: usize, fcars: usize, sel: Sel, steps: usize, seed: u32) -> Stat {
    let sch = Schema::load("../schema/v1.toml").expect("schema v1");
    // reward_v0 is the neutral scoring tape: goal = raw +/-10, so |rew| >= 9.4 is
    // exactly one goal and the probe counts them without a curriculum term.
    let cfg = RewardConfig::load("../configs/reward_v0.toml").expect("reward_v0");
    let cur = CurriculumConfig::load("../configs/curriculum_v3_match.toml").ok();
    let mut ar = EpisodeArena::new_full(m, m, sch.tick_skip, cfg, sch.normalization.clone(),
                                        seed, cur, ObsMode::V1);
    let n = ar.num_agents();
    let (ek, mk, qk, pk) = (MAX_ENT * ENT_FEAT, MAX_ENT, Q_FEAT, PREV_ACTIONS);
    let mut rew = vec![0f32; n];
    let mut flags = vec![StepFlags::default(); n];
    let (mut fe, mut fm, mut fq, mut fp) =
        (vec![0f32; n * ek], vec![false; n * mk], vec![0f32; n * qk], vec![0i64; n * pk]);
    let mut acts = vec![0i64; n];
    let table_v1 = actions::TABLE_SIZE_V1 as u32;

    let mut rng = Rng(0x243f_6a88_85a3_08d3 ^ seed as u64);
    let mut st = Stat::default();
    let mut prev: HashMap<u32, [f32; 8]> = HashMap::new();
    let mut shown: HashMap<u32, u32> = HashMap::new();
    let mut last_hit: HashMap<u32, u64> = HashMap::new();
    let mut obs = vec![0f32; ADV_OBS_SIZE];

    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        for a in 0..n { acts[a] = (rng.next_u32() % table_v1) as i64; }
        let gs = ar.game_state();
        // the bot drives the FIRST `fcars` orange cars by ascending id -- the same
        // set collect's partial-foreign-team path drives (engine.rs `fcars`).
        let mut orange: Vec<usize> = (0..gs.cars.len())
            .filter(|&j| gs.cars[j].team == Team::Orange).collect();
        orange.sort_by_key(|&j| gs.cars[j].id);
        let driven: Vec<usize> = orange.iter().copied().take(fcars).collect();

        let mut ov: Vec<(u32, [f32; 8])> = Vec::with_capacity(driven.len());
        for &j in &driven {
            let cid = gs.cars[j].id;
            let Some(other) = pick(&gs, j, sel) else { continue };
            let oid = gs.cars[other].id;
            if let Some(&p) = shown.get(&cid) { if p != oid { st.switches += 1; } }
            shown.insert(cid, oid);

            let p = *prev.get(&cid).unwrap_or(&[0.0; 8]);
            let raw = match net {
                Some(nt) => { trunc_obs(&gs, j, other, &p, &mut obs); nt.act(&obs) }
                None => actions::make_lookup_table_v1()[(rng.next_u32() % table_v1) as usize],
            };
            prev.insert(cid, raw);
            let mut out = raw;
            // Immortal and Element both zero YAW while jump is held (their bot.py
            // `yaw = 0 if action[5] > 0 else action[3]`); the RAW action is what
            // goes back into the obs. Same split as ForeignPolicy::decide.
            if out[5] > 0.0 { out[3] = 0.0; }
            ov.push((cid, out));

            // behaviour metrics, on the DRIVEN cars only
            let c = &gs.cars[j].state;
            let to_ball = [gs.ball.pos.x - c.pos.x, gs.ball.pos.y - c.pos.y, gs.ball.pos.z - c.pos.z];
            let db = (to_ball[0].powi(2) + to_ball[1].powi(2) + to_ball[2].powi(2)).sqrt().max(1e-3);
            let u = [to_ball[0] / db, to_ball[1] / db, to_ball[2] / db];
            let f = c.rot_mat.forward;
            st.facing += (f.x * u[0] + f.y * u[1] + f.z * u[2]) as f64;
            let sp = (c.vel.x.powi(2) + c.vel.y.powi(2) + c.vel.z.powi(2)).sqrt();
            if sp > 100.0 {
                st.approach += ((c.vel.x * u[0] + c.vel.y * u[1] + c.vel.z * u[2]) / sp) as f64;
                st.approach_n += 1;
            }
            st.speed += sp as f64;
            if sp < 200.0 { st.idle += 1; }
            if c.is_on_ground { st.ground_spin += c.ang_vel.z.abs() as f64; st.ground_n += 1; }
            st.dist += db as f64;
            if out[5] > 0.0 { st.jump += 1; }
            if out[6] > 0.0 { st.boost += 1; }
            if out[1].abs() >= 1.0 { st.saturated_steer += 1; }
            st.distinct.insert(out.map(|x| x.to_bits()));
            st.car_steps += 1;
        }
        ar.set_foreign_overrides(ov);
        ar.step_v1(&acts, &mut rew, &mut flags, &mut fe, &mut fm, &mut fq, &mut fp);

        // reward_v0: blue agent 0's tape. +10 = blue scored, -10 = orange scored.
        if rew[0] >= 9.4 { st.goals_against += 1; }
        if rew[0] <= -9.4 { st.goals_for += 1; }
        if flags.iter().any(|f| f.terminated || f.truncated) {
            prev.clear();
            shown.clear();
            last_hit.clear();
        }
        let now = ar.game_state();
        if !construct_engine::episode::state_is_sane(&now) { st.insane += 1; }
        for &j in &driven {
            let Some(c) = now.cars.iter().find(|c| c.id == gs.cars[j].id) else { continue };
            let h = &c.state.ball_hit_info;
            let e = last_hit.entry(c.id).or_insert(u64::MAX);
            if h.is_valid && h.tick_count_when_hit != *e {
                st.touches += 1;
                *e = h.tick_count_when_hit;
            }
        }
    }
    st.ms = t0.elapsed().as_millis();
    st
}

fn row(label: &str, mode: &str, fc: usize, sel: &str, steps: usize, st: &Stat) {
    let cs = st.car_steps.max(1) as f64;
    let mins = steps as f64 * 8.0 / 120.0 / 60.0;
    // goals per 600s, so the rows are comparable to the nexto/necto numbers in
    // engine/examples/foreign_team_probe.rs (67/71/67 and 62/61/65 at 1/2/3v3).
    let per600 = 600.0 / (steps as f64 * 8.0 / 120.0);
    println!("{:<9} {:>4} {:>3} {:>10} {:>7.0} {:>7.0} {:>8.1} {:>7.3} {:>7.3} {:>7.0} {:>6.1}% {:>7.2} {:>6.2} {:>6.1}% {:>6.1}% {:>6} {:>7.3} {:>5} {:>6}",
             label, mode, fc, sel,
             st.goals_for as f64 * per600, st.goals_against as f64 * per600,
             st.touches as f64 / mins / fc.max(1) as f64,
             st.facing / cs,
             if st.approach_n > 0 { st.approach / st.approach_n as f64 } else { f64::NAN },
             st.speed / cs,
             100.0 * st.idle as f64 / cs,
             st.dist / cs / 1000.0,
             if st.ground_n > 0 { st.ground_spin / st.ground_n as f64 } else { f64::NAN },
             100.0 * st.jump as f64 / cs,
             100.0 * st.boost as f64 / cs,
             st.distinct.len(),
             st.switches as f64 / cs,
             st.insane, st.ms);
}

/// EXPLOITABILITY. Every row above scores the bot against uniform-random blue,
/// which cannot punish "I am blind to two of the four cars". This runs the
/// truncated bot against NEXTO -- a real 1s-3s bot whose EARL obs sizes itself
/// from `state.cars.len()`, so it sees the whole arena -- and reports the goal
/// share. If truncation is a fatal handicap, the share must fall off a cliff
/// from 1v1 (where truncation is a no-op) to 3v3.
/// `net = None` drives orange with NECTO instead -- the CONTROL. Necto sees the
/// whole arena (EARL, sized from `state.cars.len()`), so whatever share it loses
/// from 1v1 to 3v3 is the cost of TEAM SIZE against nexto, not the cost of
/// truncation. Without this row an element collapse is unreadable: measurement
/// already shows even OUR net goes 41:21 at 1v1 p6 and 4:115 at 3v3 p6 vs nexto.
fn vs_nexto(net: Option<&Net>, label: &str, m: usize, fcars: usize, sel: Sel, steps: usize,
            seed: u32) {
    use construct_engine::foreign::{ForeignKind, ForeignPolicy};
    let Some(nw) = weights("nexto") else { return };
    let mut nx = ForeignPolicy::new(&nw, ForeignKind::Nexto).expect("nexto");
    let mut nc = weights("necto")
        .and_then(|w| ForeignPolicy::new(&w, ForeignKind::Necto).ok());
    if net.is_none() && nc.is_none() { return }
    let sch = Schema::load("../schema/v1.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let cur = CurriculumConfig::load("../configs/curriculum_v3_match.toml").ok();
    let mut ar = EpisodeArena::new_full(m, m, sch.tick_skip, cfg, sch.normalization.clone(),
                                        seed, cur, ObsMode::V1);
    let n = ar.num_agents();
    let (ek, mk, qk, pk) = (MAX_ENT * ENT_FEAT, MAX_ENT, Q_FEAT, PREV_ACTIONS);
    let mut rew = vec![0f32; n];
    let mut flags = vec![StepFlags::default(); n];
    let (mut fe, mut fm, mut fq, mut fp) =
        (vec![0f32; n * ek], vec![false; n * mk], vec![0f32; n * qk], vec![0i64; n * pk]);
    let mut acts = vec![0i64; n];
    // ORANGE cars the bot does NOT drive stand in for v9's "mirror" cars (our own
    // policy). Uniform-random is the honest stand-in for a from-scratch net; a
    // frozen action index would be strictly worse than random and would flatter
    // nexto. (First cut of this probe had them frozen on index 0 -- it moved
    // element's 2v2/fc1 share from 0.100 to 0.000.)
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ seed as u64);
    let mut prev: HashMap<u32, [f32; 8]> = HashMap::new();
    let mut obs = vec![0f32; ADV_OBS_SIZE];
    let (mut gf, mut ga) = (0usize, 0usize);
    let (mut fac, mut cs) = (0f64, 0usize);
    for _ in 0..steps {
        for a in 0..n { acts[a] = (rng.next_u32() % actions::TABLE_SIZE_V1 as u32) as i64; }
        let gs = ar.game_state();
        let mut ov: Vec<(u32, [f32; 8])> = Vec::with_capacity(n);
        // blue = nexto, through the REAL ForeignPolicy path (its own obs/net/table)
        for j in 0..gs.cars.len() {
            if gs.cars[j].team != Team::Blue { continue; }
            let cid = gs.cars[j].id;
            ov.push((cid, nx.decide(&gs, j, cid as u64)));
        }
        let mut orange: Vec<usize> = (0..gs.cars.len())
            .filter(|&j| gs.cars[j].team == Team::Orange).collect();
        orange.sort_by_key(|&j| gs.cars[j].id);
        for &j in orange.iter().take(fcars) {
            let cid = gs.cars[j].id;
            let out = match net {
                Some(nt) => {
                    let Some(other) = pick(&gs, j, sel) else { continue };
                    let p = *prev.get(&cid).unwrap_or(&[0.0; 8]);
                    trunc_obs(&gs, j, other, &p, &mut obs);
                    let raw = nt.act(&obs);
                    prev.insert(cid, raw);
                    let mut o = raw;
                    if o[5] > 0.0 { o[3] = 0.0; }
                    o
                }
                // CONTROL: necto's own full-arena obs, through the real path.
                None => nc.as_mut().unwrap().decide(&gs, j, 1u64 << 40 | cid as u64),
            };
            ov.push((cid, out));
            let c = &gs.cars[j].state;
            let tb = [gs.ball.pos.x - c.pos.x, gs.ball.pos.y - c.pos.y, gs.ball.pos.z - c.pos.z];
            let db = (tb[0].powi(2) + tb[1].powi(2) + tb[2].powi(2)).sqrt().max(1e-3);
            let f = c.rot_mat.forward;
            fac += ((f.x * tb[0] + f.y * tb[1] + f.z * tb[2]) / db) as f64;
            cs += 1;
        }
        ar.set_foreign_overrides(ov);
        ar.step_v1(&acts, &mut rew, &mut flags, &mut fe, &mut fm, &mut fq, &mut fp);
        if rew[0] >= 9.4 { ga += 1; }
        if rew[0] <= -9.4 { gf += 1; }
        if flags.iter().any(|f| f.terminated || f.truncated) {
            prev.clear();
            nx.reset();
            if let Some(p) = nc.as_mut() { p.reset(); }
        }
    }
    let tot = (gf + ga).max(1) as f64;
    println!("{:<9} {:>4} {:>3} {:>10} {:>6} {:>6} {:>10.3} {:>9.3}",
             label, format!("{m}v{m}"), fcars, sel.name(), gf, ga, gf as f64 / tot, fac / cs.max(1) as f64);
}

fn header() {
    println!("{:<9} {:>4} {:>3} {:>10} {:>7} {:>7} {:>8} {:>7} {:>7} {:>7} {:>7} {:>7} {:>6} {:>7} {:>7} {:>6} {:>7} {:>5} {:>6}",
             "bot", "mode", "fc", "shown", "gF/600s", "gA/600s", "tch/min", "facing", "apprch",
             "speed", "idle%", "dist/k", "spinZ", "jump%", "boost%", "nCtrl", "switch", "insn", "ms");
}

/// The harness must be the reference at 1v1 or none of the other rows mean
/// anything: cloning the state and retaining 2 cars has to reproduce
/// `build_advanced_obs` on the untouched state, float for float.
fn harness_identity_check() {
    let sch = Schema::load("../schema/v1.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let mut ar = EpisodeArena::new_full(1, 1, sch.tick_skip, cfg, sch.normalization.clone(),
                                        11, None, ObsMode::V1);
    let n = ar.num_agents();
    let (ek, mk, qk, pk) = (MAX_ENT * ENT_FEAT, MAX_ENT, Q_FEAT, PREV_ACTIONS);
    let (mut r, mut f) = (vec![0f32; n], vec![StepFlags::default(); n]);
    let (mut fe, mut fm, mut fq, mut fp) =
        (vec![0f32; n * ek], vec![false; n * mk], vec![0f32; n * qk], vec![0i64; n * pk]);
    let prev = [0.3f32, -1.0, 0.5, 0.0, 1.0, 1.0, 0.0, 1.0];
    let mut bad = 0usize;
    for k in 0..60 {
        ar.step_v1(&vec![(k % 92) as i64; n], &mut r, &mut f, &mut fe, &mut fm, &mut fq, &mut fp);
        let gs = ar.game_state();
        for j in 0..gs.cars.len() {
            let mut a = vec![0f32; ADV_OBS_SIZE];
            let mut b = vec![0f32; ADV_OBS_SIZE];
            build_advanced_obs(&gs, j, &prev, &mut a);
            let other = pick(&gs, j, Sel::NearestSelf).unwrap();
            trunc_obs(&gs, j, other, &prev, &mut b);
            if a.iter().zip(&b).any(|(x, y)| x.to_bits() != y.to_bits()) { bad += 1; }
        }
    }
    println!("harness identity @1v1: {} of 120 obs differ from build_advanced_obs \
              (0 required)\n", bad);
    assert_eq!(bad, 0, "the truncation harness is not the reference at 1v1");
}

/// The width claim, checked instead of asserted: the UNTRUNCATED builder on a
/// 4-car state must run off the end of the 107-float buffer.
fn width_check() {
    let sch = Schema::load("../schema/v1.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let mut ar = EpisodeArena::new_full(2, 2, sch.tick_skip, cfg, sch.normalization.clone(),
                                        5, None, ObsMode::V1);
    let gs = ar.game_state();
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut o = vec![0f32; ADV_OBS_SIZE];
        build_advanced_obs(&gs, 0, &[0.0; 8], &mut o);
    }));
    println!("untruncated build_advanced_obs @2v2: {}",
             if r.is_err() { "PANICS (107 is exactly one other car)" } else { "ran?!" });
    let need = 9 + 8 + 34 + 25 + 31 * (gs.cars.len() - 1);
    println!("  width needed for {} cars = {need}, net input = {ADV_OBS_SIZE}\n", gs.cars.len());
}

fn main() {
    sim_init::ensure_init(None);
    let steps: usize = std::env::var("PROBE_STEPS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4000);
    println!("steps/arena = {steps}  (tick_skip 8 -> {}s of sim per row)", steps * 8 / 120);
    println!("blue = uniform-random from our 92-action table; orange = the ported bot.");
    println!("fc = foreign cars driven (v9's team blocks start at 1).");
    println!("facing  = mean forward . unit(car->ball)   [1.0 = pointed straight at the ball]");
    println!("apprch  = mean unit(vel) . unit(car->ball) [1.0 = driving straight at the ball]");
    println!("spinZ   = mean |ang_vel.z| on ground       [a spinner sits near the 5.5 rad/s cap]");
    println!("switch  = shown-opponent identity flips per car-step\n");
    std::panic::set_hook(Box::new(|_| {}));
    width_check();
    let _ = std::panic::take_hook();
    harness_identity_check();

    // PROBE_FOCUS=1: the selector question only, over several seeds. The wide
    // grid below is one seed per cell, and the between-selector gaps there
    // (e.g. immortal 3v3 fc=3 lowest-id 20 vs near-ball 47 goals/600s) are the
    // same size as single-run noise -- so re-measure the cells that matter.
    if std::env::var("PROBE_FOCUS").is_ok() {
        header();
        for name in ["element", "immortal"] {
            let Some(w) = weights(name) else { continue };
            let net = if name == "element" {
                Net::Element(ElementNet::new(&w, ADV_OBS_SIZE).expect("element net"))
            } else {
                Net::Immortal(ForeignMlp::from_named(&w, ADV_OBS_SIZE, 512, 126, 6).unwrap(),
                              actions::make_immortal_table())
            };
            for (m, fc) in [(1usize, 1usize), (2, 1), (3, 1), (3, 3)] {
                for sel in [Sel::NearestSelf, Sel::NearestBall, Sel::LowestId, Sel::MateBall] {
                    if m == 1 && sel != Sel::NearestSelf { continue; }
                    for seed in [7u32, 23, 101] {
                        let st = run(Some(&net), m, fc, sel, steps, seed);
                        row(name, &format!("{m}v{m}/s{seed}"), fc,
                            if m == 1 { "ref" } else { sel.name() }, steps, &st);
                    }
                }
            }
            println!();
        }
        return;
    }

    // PROBE_VS=1: the truncated bot against NEXTO (sees the whole arena).
    if std::env::var("PROBE_VS").is_ok() {
        println!("{:<9} {:>4} {:>3} {:>10} {:>6} {:>6} {:>10} {:>9}",
                 "bot", "mode", "fc", "shown", "gFor", "gAg", "goalShare", "facing");
        // CONTROL FIRST: necto (full-arena obs) vs nexto, same harness.
        for (m, fc) in [(1usize, 1usize), (2, 1), (2, 2), (3, 1), (3, 3)] {
            vs_nexto(None, "necto(ctl)", m, fc, Sel::NearestBall, steps, 7);
        }
        println!();
        for name in ["element", "immortal"] {
            let Some(w) = weights(name) else { continue };
            let net = if name == "element" {
                Net::Element(ElementNet::new(&w, ADV_OBS_SIZE).expect("element net"))
            } else {
                Net::Immortal(ForeignMlp::from_named(&w, ADV_OBS_SIZE, 512, 126, 6).unwrap(),
                              actions::make_immortal_table())
            };
            for (m, fc) in [(1usize, 1usize), (2, 1), (2, 2), (3, 1), (3, 3)] {
                vs_nexto(Some(&net), name, m, fc, Sel::NearestBall, steps, 7);
            }
            println!();
        }
        return;
    }

    header();
    // FLOOR: orange acting uniformly at random. Any bot row must beat this or
    // "it behaves sanely" is unfalsifiable.
    for m in [1usize, 2, 3] {
        let st = run(None, m, m, Sel::NearestBall, steps, 7);
        row("random", &format!("{m}v{m}"), m, "-", steps, &st);
    }
    println!();

    for name in ["element", "immortal"] {
        let Some(w) = weights(name) else { continue };
        let net = if name == "element" {
            Net::Element(ElementNet::new(&w, ADV_OBS_SIZE).expect("element net"))
        } else {
            Net::Immortal(ForeignMlp::from_named(&w, ADV_OBS_SIZE, 512, 126, 6).expect("immortal net"),
                          actions::make_immortal_table())
        };
        // 1v1 = the REFERENCE row (truncation is a no-op there).
        let st = run(Some(&net), 1, 1, Sel::NearestSelf, steps, 7);
        row(name, "1v1", 1, "ref", steps, &st);
        for m in [2usize, 3] {
            for fc in [1usize, m] {
                for sel in [Sel::NearestSelf, Sel::NearestBall, Sel::LowestId, Sel::MateBall] {
                    let st = run(Some(&net), m, fc, sel, steps, 7);
                    row(name, &format!("{m}v{m}"), fc, sel.name(), steps, &st);
                }
            }
        }
        println!();
    }
}
