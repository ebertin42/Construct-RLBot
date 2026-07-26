// INVESTIGATION PROBE (do not ship). Answers: can the ported Nexto/Necto bots
// actually drive 2v2 and 3v3 arenas, or is the blanket 1v1 guard in
// engine/src/engine.rs:319-325 protecting something real for them too?
//
// The guard's stated reason ("their obs width is fixed") is TRUE for Immortal
// and Element (AdvancedObs-107 == exactly one other car; build_advanced_obs
// would run off the end of the buffer) but Nexto/Necto size their obs from
// state.cars.len(), so it should be vacuous for them. This probe checks that
// claim against the real weights + real physics instead of by reading.
//
// It runs the SAME code path collect uses (EpisodeArena::foreign_controls ->
// ForeignPolicy::decide -> set_foreign_overrides -> step_v1) at 1v1/2v2/3v3,
// with orange driven by the ported bot and blue driven by uniform-random
// actions from our own table, and reports: does it run, does it play (goals
// against a random opponent), do the several cars of one team get INDEPENDENT
// controls (per-car state keying), and is the reward tape sane.
//
// Run: cargo run --release --example foreign_team_probe   (from engine/)
use std::collections::HashMap;
use std::fs::File;

use construct_engine::actions;
use construct_engine::curriculum::CurriculumConfig;
use construct_engine::episode::{EpisodeArena, ObsMode, StepFlags};
use construct_engine::foreign::{ForeignKind, ForeignPolicy};
use construct_engine::obs_v1::{ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT};
use construct_engine::reward::RewardConfig;
use construct_engine::schema::Schema;
use construct_engine::sim_init;
use ndarray::IxDyn;
use ndarray_npy::NpzReader;

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

/// xorshift so blue's "random" opponent is reproducible without pulling the
/// engine's own rng discipline into the probe.
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

struct Stat {
    steps: usize,
    goals_for: usize,   // orange (the ported bot) scored
    goals_against: usize,
    touches: usize,
    jump_frac: f64,
    boost_frac: f64,
    distinct_controls: usize,
    /// steps where at least two cars of the ported team issued DIFFERENT controls
    steps_cars_differ: usize,
    max_abs_rew: f32,
    mean_abs_rew: f64,
    nonfinite: usize,
    insane_states: usize,
    ms: u128,
}

fn run(kind: ForeignKind, w: &HashMap<String, (Vec<f32>, Vec<usize>)>, m: usize,
       steps: usize, period: u32, seed: u32) -> Stat {
    let sch = Schema::load("../schema/v1.toml").expect("schema v1");
    // reward_v0 is the neutral scoring tape: goal = raw +/-10, so |rew| >= 9.4
    // is exactly one goal and the probe can count them without a curriculum.
    let cfg = RewardConfig::load("../configs/reward_v0.toml").expect("reward_v0");
    let cur = CurriculumConfig::load("../configs/curriculum_v3_match.toml").ok();
    let mut ar = EpisodeArena::new_full(m, m, sch.tick_skip, cfg, sch.normalization.clone(),
                                        seed, cur, ObsMode::V1);
    let mut pol = ForeignPolicy::new(w, kind).expect("build foreign policy");
    pol.set_decision_period(period);

    let n = ar.num_agents();
    let bc = ar.blue_count();
    let (ek, mk, qk, pk) = (MAX_ENT * ENT_FEAT, MAX_ENT, Q_FEAT, PREV_ACTIONS);
    let mut rew = vec![0f32; n];
    let mut flags = vec![StepFlags::default(); n];
    let (mut fe, mut fm, mut fq, mut fp) =
        (vec![0f32; n * ek], vec![false; n * mk], vec![0f32; n * qk], vec![0i64; n * pk]);
    let mut acts = vec![0i64; n];
    let table_v1 = actions::TABLE_SIZE_V1 as u32;

    let mut rng = Rng(0x243f_6a88_85a3_08d3 ^ seed as u64);
    let mut st = Stat {
        steps, goals_for: 0, goals_against: 0, touches: 0, jump_frac: 0.0, boost_frac: 0.0,
        distinct_controls: 0, steps_cars_differ: 0, max_abs_rew: 0.0, mean_abs_rew: 0.0,
        nonfinite: 0, insane_states: 0, ms: 0,
    };
    let mut seen: std::collections::HashSet<[u32; 8]> = std::collections::HashSet::new();
    let mut jumps = 0usize;
    let mut boosts = 0usize;
    let mut decisions = 0usize;
    let mut sum_abs = 0f64;
    let mut last_hit: HashMap<u32, u64> = HashMap::new();

    let t0 = std::time::Instant::now();
    for _ in 0..steps {
        for a in 0..n {
            acts[a] = (rng.next_u32() % table_v1) as i64;
        }
        let mut ov = Vec::with_capacity(n - bc);
        let mut ctrls: Vec<[f32; 8]> = Vec::with_capacity(n - bc);
        for a in bc..n {
            let (cid, _key, c) = ar.foreign_controls(a, &mut pol, 0);
            ov.push((cid, c));
            ctrls.push(c);
        }
        ar.set_foreign_overrides(ov);
        for c in &ctrls {
            decisions += 1;
            if c[5] > 0.0 { jumps += 1; }
            if c[6] > 0.0 { boosts += 1; }
            seen.insert(c.map(|x| x.to_bits()));
        }
        if ctrls.len() > 1 && ctrls.iter().any(|c| c != &ctrls[0]) {
            st.steps_cars_differ += 1;
        }
        ar.step_v1(&acts, &mut rew, &mut flags, &mut fe, &mut fm, &mut fq, &mut fp);
        for &r in rew.iter() {
            if !r.is_finite() { st.nonfinite += 1; continue; }
            sum_abs += r.abs() as f64;
            if r.abs() > st.max_abs_rew { st.max_abs_rew = r.abs(); }
        }
        // reward_v0: blue agent 0's tape. +10 = blue scored, -10 = orange scored.
        if rew[0] >= 9.4 { st.goals_against += 1; }
        if rew[0] <= -9.4 { st.goals_for += 1; }
        let gs = ar.game_state();
        if !construct_engine::episode::state_is_sane(&gs) { st.insane_states += 1; }
        // touches BY THE PORTED TEAM only: per-car ball_hit_info tick changed
        // (the same detector reward.rs uses), so "is it engaging the ball at
        // all" is measured on the bot's cars, not on blue's random flailing.
        for c in gs.cars.iter().filter(|c| c.team == rocketsim_rs::sim::Team::Orange) {
            let h = &c.state.ball_hit_info;
            let e = last_hit.entry(c.id).or_insert(u64::MAX);
            if h.is_valid && h.tick_count_when_hit != *e {
                st.touches += 1;
                *e = h.tick_count_when_hit;
            }
        }
    }
    st.ms = t0.elapsed().as_millis();
    st.jump_frac = jumps as f64 / decisions.max(1) as f64;
    st.boost_frac = boosts as f64 / decisions.max(1) as f64;
    st.distinct_controls = seen.len();
    st.mean_abs_rew = sum_abs / (steps * n) as f64;
    st
}

/// Two things that only bite above 1v1:
///  (a) obs sanity -- kv must stay finite and O(1) with 6 cars, and the mask must
///      still be all-false (nothing is padding in a full complement).
///  (b) Necto's STATEFUL timers are per ARENA but are advanced inside
///      build_necto_obs, i.e. once per CAR. With one foreign car per arena that
///      is once per step (correct); with 3 it is 3x per step, so the pad-respawn
///      and demo timers decay 2-3x too fast AND the 3 cars of one team disagree
///      about the same pad within a single frame.
fn obs_and_timer_checks() {
    use construct_engine::obs_necto::{build_necto_obs, n_entities as necto_ent, NectoTimers,
                                      NECTO_KV, NECTO_Q};
    use construct_engine::obs_nexto::{build_nexto_obs, n_entities as nexto_ent, NEXTO_KV, NEXTO_Q};
    let sch = Schema::load("../schema/v1.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let mut ar = EpisodeArena::new_full(3, 3, sch.tick_skip, cfg, sch.normalization.clone(),
                                        3, None, ObsMode::V1);
    let acts = vec![7i64; ar.num_agents()];
    let n = ar.num_agents();
    let (ek, mk, qk, pk) = (MAX_ENT * ENT_FEAT, MAX_ENT, Q_FEAT, PREV_ACTIONS);
    let (mut r, mut f) = (vec![0f32; n], vec![StepFlags::default(); n]);
    let (mut fe, mut fm, mut fq, mut fp) =
        (vec![0f32; n * ek], vec![false; n * mk], vec![0f32; n * qk], vec![0i64; n * pk]);
    for _ in 0..40 {
        ar.step_v1(&acts, &mut r, &mut f, &mut fe, &mut fm, &mut fq, &mut fp);
    }
    let gs = ar.game_state();
    println!("\n-- obs sanity at 3v3 (6 cars) --");
    let ne = nexto_ent(gs.cars.len());
    let mut q = vec![0f32; NEXTO_Q];
    let mut kv = vec![0f32; ne * NEXTO_KV];
    let mut mask = vec![false; ne];
    build_nexto_obs(&gs, 0, &[0.0; 8], &mut q, &mut kv, &mut mask);
    println!("nexto: n_ent={ne} kv_len={} all_finite={} max|kv|={:.3} masked={}",
             kv.len(), kv.iter().all(|x| x.is_finite()),
             kv.iter().fold(0f32, |m, x| m.max(x.abs())), mask.iter().filter(|m| **m).count());
    // exactly one is_self, and the mate/opp flags must split 2/3 (self excluded)
    let selfs = (0..gs.cars.len()).filter(|r| kv[r * NEXTO_KV] == 1.0).count();
    let mates = (0..gs.cars.len()).filter(|r| kv[r * NEXTO_KV + 1] == 1.0).count();
    let opps = (0..gs.cars.len()).filter(|r| kv[r * NEXTO_KV + 2] == 1.0).count();
    println!("nexto: is_self rows={selfs} is_mate rows={mates} is_opp rows={opps} (expect 1 / 3 / 3)");

    let ne = necto_ent(gs.cars.len());
    let mut q = vec![0f32; NECTO_Q];
    let mut kv = vec![0f32; ne * NECTO_KV];
    let mut mask = vec![false; ne];
    let mut timers = NectoTimers::new(gs.cars.len());
    // three calls in ONE frame = what a 3-car foreign team does per step
    let pad_row = 1 + gs.cars.len() + 3; // a big pad
    let mut pad_seen = Vec::new();
    let mut demo_seen = Vec::new();
    for car in [3usize, 4, 5] {
        build_necto_obs(&gs, car, &[0.0; 8], &mut timers, &mut q, &mut kv, &mut mask);
        pad_seen.push(kv[pad_row * NECTO_KV + 21]);
        demo_seen.push(kv[(1 + car) * NECTO_KV + 21]);
    }
    println!("necto: n_ent={ne} all_finite={} max|kv|={:.3} masked={}",
             kv.iter().all(|x| x.is_finite()),
             kv.iter().fold(0f32, |m, x| m.max(x.abs())), mask.iter().filter(|m| **m).count());
    println!("necto TIMER ALIASING (same frame, 3 cars): pad col21 = {:?}", pad_seen);
    println!("                                            demo col21 = {:?}", demo_seen);
    println!("  -> one frame of decay is {:.5}; the 3 calls consumed {:.5} of pad timer",
             8.0 / 1200.0, pad_seen[0] - pad_seen[2]);
}

fn main() {
    sim_init::ensure_init(None);
    let steps: usize = std::env::var("PROBE_STEPS").ok()
        .and_then(|s| s.parse().ok()).unwrap_or(4000);
    println!("steps/arena = {steps}  (tick_skip 8 -> ~{}s of sim)", steps * 8 / 120);
    println!("blue = uniform-random from our 92-action table; orange = the ported bot\n");
    println!("{:<8} {:>4} {:>5} {:>7} {:>8} {:>8} {:>6} {:>6} {:>7} {:>8} {:>9} {:>7} {:>6} {:>7}",
             "bot", "mode", "per", "n_ent", "goalsFor", "goalsAg", "touch", "jump%", "boost%",
             "nCtrl", "carsDiff", "meanRew", "nonfin", "ms");
    for (name, kind) in [("nexto", ForeignKind::Nexto), ("necto", ForeignKind::Necto)] {
        let Some(w) = weights(name) else { continue };
        for m in [1usize, 2, 3] {
            for period in [1u32, 4] {
                let st = run(kind, &w, m, steps, period, 7);
                let n_ent = match kind {
                    ForeignKind::Nexto => construct_engine::obs_nexto::n_entities(2 * m),
                    _ => construct_engine::obs_necto::n_entities(2 * m),
                };
                println!("{:<8} {:>4} {:>5} {:>7} {:>8} {:>8} {:>6} {:>5.1}% {:>6.1}% {:>8} {:>9} {:>7.4} {:>6} {:>7}",
                         name, format!("{m}v{m}"), period, n_ent, st.goals_for, st.goals_against,
                         st.touches, 100.0 * st.jump_frac, 100.0 * st.boost_frac,
                         st.distinct_controls, st.steps_cars_differ, st.mean_abs_rew,
                         st.nonfinite, st.ms);
                if st.insane_states > 0 {
                    println!("   !! {} insane physics states", st.insane_states);
                }
            }
        }
    }
    // Physics-only baseline: the same arenas with NO foreign forward, so the
    // rows above can be read as "cost of N ported-bot forwards per step" -- a
    // foreign team arena runs one SEQUENTIAL forward per opposing car.
    println!("\n-- physics-only baseline (no foreign forward) --");
    for m in [1usize, 2, 3] {
        let sch = Schema::load("../schema/v1.toml").unwrap();
        let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
        let cur = CurriculumConfig::load("../configs/curriculum_v3_match.toml").ok();
        let mut ar = EpisodeArena::new_full(m, m, sch.tick_skip, cfg,
                                            sch.normalization.clone(), 7, cur, ObsMode::V1);
        let n = ar.num_agents();
        let (ek, mk, qk, pk) = (MAX_ENT * ENT_FEAT, MAX_ENT, Q_FEAT, PREV_ACTIONS);
        let (mut r, mut f) = (vec![0f32; n], vec![StepFlags::default(); n]);
        let (mut fe, mut fm, mut fq, mut fp) =
            (vec![0f32; n * ek], vec![false; n * mk], vec![0f32; n * qk], vec![0i64; n * pk]);
        let mut rng = Rng(0x243f_6a88_85a3_08d3 ^ 7);
        let mut acts = vec![0i64; n];
        let t0 = std::time::Instant::now();
        for _ in 0..steps {
            for a in 0..n { acts[a] = (rng.next_u32() % actions::TABLE_SIZE_V1 as u32) as i64; }
            ar.step_v1(&acts, &mut r, &mut f, &mut fe, &mut fm, &mut fq, &mut fp);
        }
        println!("{m}v{m}: {} ms for {steps} steps", t0.elapsed().as_millis());
    }

    obs_and_timer_checks();

    // Immortal is the CONTROL: its obs really is fixed-width, so 2v2 must fail
    // loudly here. Caught so the probe reports it instead of aborting.
    if let Some(w) = weights("immortal") {
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run(ForeignKind::Immortal, &w, 2, 20, 1, 7)
        }));
        println!("\nimmortal @2v2 (control): {}",
                 if r.is_err() { "PANICKED as expected (fixed 107-wide obs)" } else { "ran?!" });
    }
}
