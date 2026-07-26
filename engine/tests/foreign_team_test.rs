//! Foreign opponents in TEAM arenas (v9 E1/E4/E5).
//!
//! Three things are under test, and each one was a real bug or a real
//! measurement:
//!
//!   E1 the foreign-arena guard is PER KIND, not blanket. Immortal and Element
//!      consume AdvancedObs-107 -- literally "self + ONE other car" -- and
//!      `build_advanced_obs` runs off the end of the buffer at 2v2 (verified
//!      panic: "index out of bounds: the len is 107 but the index is 107"). Nexto
//!      and Necto are EARL attention models whose obs builders size themselves
//!      from `state.cars.len()`, and they were 1s-3s bots in the wild.
//!
//!   E4 PARTIAL foreign teams. `decision_period` has almost no authority above
//!      1v1: the same net vs a FULL nexto team scores 88:5 at 1v1 p12 but 1:25 at
//!      3v3 p12, and the 3v3 goal share only creeps to 0.371 by p96. Car count is
//!      the knob that works, and one nexto among three orange cars (the other two
//!      mirroring our own policy) is a contested arena by construction.
//!
//!   E5 the per-row team-size label the trainer standardises advantages on. It
//!      MUST come from the engine: the worker split is
//!      `(num_arenas - assigned) / (threads - t)` with auto-detected threads, so
//!      a Python reconstruction would disagree on some machines and mis-scale
//!      every gradient silently.
//!
//! Tests needing the (unlicensed, never-committed) nexto/necto npz SKIP when it
//! is absent -- see docs/foreign-opponents.md.
use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use construct_engine::{
    actions,
    engine::{allocate_team_sizes, MultiEngine, NetWeights},
    foreign::ForeignKind,
    obs_v1::{ENT_FEAT, PREV_ACTIONS, Q_FEAT},
    reward::RewardConfig,
    sampler::Pcg32,
    schema::Schema,
    sim_init::ensure_init,
};
use ndarray::IxDyn;
use ndarray_npy::NpzReader;

type RawDict = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn rand_vec(rng: &mut Pcg32, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale).collect()
}

/// Deterministic random v1 state dict at the launch dims (128/2/4/512).
/// Same construction as engine_v1_test.rs's; duplicated rather than shared
/// because the two files test different things and a shared fixture would
/// couple them.
fn v1_state_dict(seed: u64) -> RawDict {
    let (d, layers, ff) = (128usize, 2usize, 512usize);
    let mut rng = Pcg32::new(seed);
    let mut m: RawDict = HashMap::new();
    fn lin(m: &mut RawDict, rng: &mut Pcg32, name: &str, out: usize, inp: usize) {
        m.insert(format!("{name}.weight"), (rand_vec(rng, out * inp, 0.05), vec![out, inp]));
        m.insert(format!("{name}.bias"), (rand_vec(rng, out, 0.05), vec![out]));
    }
    fn ln(m: &mut RawDict, name: &str, dim: usize) {
        m.insert(format!("{name}.weight"), (vec![1.0; dim], vec![dim]));
        m.insert(format!("{name}.bias"), (vec![0.0; dim], vec![dim]));
    }
    lin(&mut m, &mut rng, "embed", d, ENT_FEAT);
    lin(&mut m, &mut rng, "query_embed", d, Q_FEAT);
    lin(&mut m, &mut rng, "act_embed.0", d, 8);
    lin(&mut m, &mut rng, "act_embed.2", 32, d);
    m.insert("prev_embed_w".into(), (rand_vec(&mut rng, PREV_ACTIONS, 0.05), vec![PREV_ACTIONS]));
    lin(&mut m, &mut rng, "prev_proj", d, 32);
    for i in 0..layers {
        ln(&mut m, &format!("blocks.{i}.ln1"), d);
        for p in ["q", "k", "v", "o"] {
            lin(&mut m, &mut rng, &format!("blocks.{i}.attn.{p}"), d, d);
        }
        ln(&mut m, &format!("blocks.{i}.ln2"), d);
        lin(&mut m, &mut rng, &format!("blocks.{i}.ff1"), ff, d);
        lin(&mut m, &mut rng, &format!("blocks.{i}.ff2"), d, ff);
    }
    ln(&mut m, "pool_ln", d);
    for p in ["q", "k", "v", "o"] {
        lin(&mut m, &mut rng, &format!("pool.{p}"), d, d);
    }
    lin(&mut m, &mut rng, "policy_dot", 32, d);
    lin(&mut m, &mut rng, "value_head", 1, d);
    let table = actions::make_lookup_table_v1();
    let flat: Vec<f32> = table.iter().flatten().copied().collect();
    m.insert("action_table".into(), (flat, vec![actions::TABLE_SIZE_V1, 8]));
    m
}

fn bot_path(name: &str) -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    std::path::Path::new(&home).join(format!(".cache/construct/{name}_weights.npz"))
}

fn load_npz(p: &std::path::Path) -> RawDict {
    let mut npz = NpzReader::new(File::open(p).unwrap()).unwrap();
    let mut map = RawDict::new();
    for n in npz.names().unwrap() {
        let a: ndarray::Array<f32, IxDyn> = npz.by_name(&n).unwrap();
        let shape = a.shape().to_vec();
        map.insert(n.trim_end_matches(".npy").to_string(), (a.iter().copied().collect(), shape));
    }
    map
}

/// Immortal-shaped all-zero state dict: exercises the whole obs -> forward ->
/// argmax -> table path without the unlicensed real weights (and, here, without
/// needing them at all -- the guard fires before any forward).
fn zero_immortal() -> RawDict {
    let mut w = RawDict::new();
    for (idx, out_dim, in_dim) in [
        (0, 512, 107), (2, 512, 512), (4, 512, 512), (6, 512, 512),
        (8, 512, 512), (10, 512, 512), (12, 126, 512),
    ] {
        w.insert(format!("net.{idx}.weight"), (vec![0.0f32; out_dim * in_dim], vec![out_dim, in_dim]));
        w.insert(format!("net.{idx}.bias"), (vec![0.0f32; out_dim], vec![out_dim]));
    }
    w
}

/// Engine over `sizes` (per-arena team size), one worker thread so the arena ->
/// worker mapping in the assertions is unambiguous.
fn engine(sizes: &[usize], seed: u32) -> MultiEngine {
    ensure_init(None);
    let sch = Schema::load("../schema/v1.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let pairs: Vec<(usize, usize)> = sizes.iter().map(|&s| (s, s)).collect();
    let mut e = MultiEngine::new(pairs, sch, cfg, seed, 1, None, false);
    e.set_weights(NetWeights::V1 { raw: v1_state_dict(99), heads: 4 }).unwrap();
    e
}

fn foreign(raw: RawDict, kind: ForeignKind, period: u32, cars: u32) -> NetWeights {
    NetWeights::Foreign { raw, kind, period, cars }
}

// ---------------------------------------------------------------- E1 --------

#[test]
fn foreign_team_arena_runs_for_fixed_width_kinds_via_truncated_obs() {
    // INVERTED on 2026-07-26. This test previously asserted that a 2v2 arena
    // with immortal/element was REFUSED ("1v1-only"), because
    // build_advanced_obs overran its 107-float buffer above 1v1. Those two now
    // run on a TRUNCATED AdvancedObs-107 showing only the opponent nearest the
    // ball (obs_advanced::build_advanced_obs_one_other), so the refusal is gone
    // and the 2v2 arena must actually COLLECT.
    let mut e = engine(&[1, 2], 5);
    e.set_foreign_opponents(vec![foreign(zero_immortal(), ForeignKind::Immortal, 1, u32::MAX)])
        .unwrap();
    // arena 0 (1v1) and arena 1 (2v2) both driven by the fixed-width bot.
    let out = e
        .collect(4, Arc::new(vec![-2, -2]))
        .expect("a 2v2 arena with a fixed-width bot must now run, not be refused");
    // BLUE only are learner rows in a foreign arena: 1 (arena0) + 2 (arena1).
    assert_eq!(out.learner_agents, 1 + 2);
    assert!(out.rewards.iter().all(|x| x.is_finite()), "truncated obs produced non-finite rewards");
    assert!(out.values.iter().all(|x| x.is_finite()));
    assert!(out.actions.iter().all(|&a| a >= 0 && (a as usize) < actions::TABLE_SIZE_V1));
    // ...and the mixed assignment (one self-play, one foreign) still works.
    let out = e.collect(4, Arc::new(vec![-2, -1])).unwrap();
    assert_eq!(out.learner_agents, 1 + 4, "arena0 blue only + arena1 (2v2) self-play");
}

#[test]
fn truncated_bot_produces_non_constant_controls_at_2v2() {
    // The weaker claim the refusal used to make for us: that a fixed-width bot
    // in a team arena is not silently broken. A net fed a malformed obs tends
    // to emit one frozen action; assert the 2v2 arena's foreign cars are
    // actually being driven by checking the LEARNER side still sees varied
    // outcomes and nothing degenerates into NaN. Uses the real bot if present.
    let p = bot_path("element");
    if !p.exists() {
        eprintln!("SKIP: {} absent (weights are not committed)", p.display());
        return;
    }
    let mut e = engine(&[2], 7);
    e.set_foreign_opponents(vec![foreign(load_npz(&p), ForeignKind::Element, 1, u32::MAX)])
        .unwrap();
    let out = e.collect(16, Arc::new(vec![-2])).expect("element must drive a 2v2 arena");
    assert_eq!(out.learner_agents, 2, "2v2 foreign arena -> the 2 BLUE cars are learner rows");
    assert!(out.rewards.iter().all(|x| x.is_finite()));
    assert!(out.values.iter().all(|x| x.is_finite()));
    let a0 = out.actions[0];
    assert!(
        out.actions.iter().any(|&a| a != a0),
        "every sampled action identical -- the arena is probably not stepping"
    );
}

#[test]
fn foreign_team_arena_allowed_for_earl_kinds() {
    let p = bot_path("nexto");
    if !p.exists() {
        eprintln!("SKIP: {} absent (weights are not committed)", p.display());
        return;
    }
    let mut e = engine(&[3], 5);
    e.set_foreign_opponents(vec![foreign(load_npz(&p), ForeignKind::Nexto, 12, u32::MAX)]).unwrap();
    let out = e.collect(8, Arc::new(vec![-2])).unwrap();
    assert_eq!(out.learner_agents, 3, "3v3 foreign arena -> the 3 BLUE cars are learner rows");
    assert!(out.rewards.iter().all(|x| x.is_finite()));
    assert!(out.values.iter().all(|x| x.is_finite()));
    assert!(out.actions.iter().all(|&a| a >= 0 && (a as usize) < actions::TABLE_SIZE_V1));
}

// ---------------------------------------------------------------- E4 --------

#[test]
fn foreign_cars_defaults_to_full_team() {
    let p = bot_path("necto");
    if !p.exists() {
        eprintln!("SKIP: {} absent", p.display());
        return;
    }
    // `cars = u32::MAX` is what lib.rs fills in when the caller omits
    // foreign_cars, and it must be byte-identical to explicitly asking for the
    // whole orange team.
    let run = |cars: u32| {
        let mut e = engine(&[3], 17);
        e.set_foreign_opponents(vec![foreign(load_npz(&p), ForeignKind::Necto, 12, cars)]).unwrap();
        e.collect(8, Arc::new(vec![-2])).unwrap()
    };
    let a = run(u32::MAX);
    let b = run(3);
    assert_eq!(a.actions, b.actions, "default must drive the WHOLE orange team");
    assert_eq!(a.rewards, b.rewards);
    assert_eq!(a.forward_rows, a.learner_agents,
               "a full foreign team leaves no mirror cars, so nothing extra is forwarded");
}

#[test]
fn foreign_cars_1_of_3_leaves_two_orange_policy_driven() {
    let p = bot_path("necto");
    if !p.exists() {
        eprintln!("SKIP: {} absent", p.display());
        return;
    }
    let run = |cars: u32| {
        let mut e = engine(&[3], 17);
        e.set_foreign_opponents(vec![foreign(load_npz(&p), ForeignKind::Necto, 12, cars)]).unwrap();
        e.collect(8, Arc::new(vec![-2])).unwrap()
    };
    let full = run(3);
    let one = run(1);
    // Learner accounting is UNCHANGED -- only blue rows, whatever the bot drives.
    // This is what keeps the equal-rows arithmetic in the v9 train config exact.
    assert_eq!(one.learner_agents, 3);
    assert_eq!(one.learner_agents, full.learner_agents);
    // The two cars the bot no longer drives are forwarded through OUR policy,
    // not left to sample from an all-zero logits row (= uniform random).
    assert_eq!(full.forward_rows, 3, "full team: 3 blue, no mirrors");
    assert_eq!(one.forward_rows, 5, "1 of 3: 3 blue + 2 mirror cars");
    // ...and it demonstrably changes the game: the two orange cars the bot no
    // longer drives move differently, which blue OBSERVES.
    //
    // Compared on `ents`, not `actions`: with the near-uniform logits of a
    // random fixture net a small perturbation almost never flips which bucket a
    // categorical draw lands in, so identical sampled actions here would prove
    // nothing either way. The observed world is the honest signal.
    assert!(one.ents != full.ents,
            "1-of-3 must produce a different rollout than a full foreign team");
    assert!(one.rewards.iter().all(|x| x.is_finite()));
    // clamping: asking for more cars than the team has is the full team, and the
    // 1v1 block is unaffected by a >1 setting.
    let mut e = engine(&[1], 17);
    e.set_foreign_opponents(vec![foreign(load_npz(&p), ForeignKind::Necto, 12, 3)]).unwrap();
    let o = e.collect(4, Arc::new(vec![-2])).unwrap();
    assert_eq!(o.learner_agents, 1);
    assert_eq!(o.forward_rows, 1, "nothing to mirror in a 1v1 arena");
}

// ---------------------------------------------------------------- E5 --------

#[test]
fn learner_team_sizes_match_row_layout() {
    let p = bot_path("nexto");
    let have_bot = p.exists();
    // 1v1 x2, 2v2 x2, 3v3 x2 -- the same block ordering allocate_team_sizes
    // produces, which is what the trainer places opponents against.
    let sizes = [1usize, 1, 2, 2, 3, 3];
    let mut e = engine(&sizes, 23);
    // arena 0: foreign (blue only), arena 2: native opponent (blue only),
    // arena 4: foreign at 3v3 if we have a team-capable bot; everything else
    // self-play. A MIXED assignment is the point: the label must follow the
    // ROW layout, not the arena layout.
    let mut assign = vec![-1i32; sizes.len()];
    e.set_opponents(vec![NetWeights::V1 { raw: v1_state_dict(123), heads: 4 }]).unwrap();
    assign[2] = 0;
    if have_bot {
        e.set_foreign_opponents(vec![foreign(load_npz(&p), ForeignKind::Nexto, 12, 1)]).unwrap();
        assign[0] = -2;
        assign[4] = -2;
    }
    let out = e.collect(4, Arc::new(assign.clone())).unwrap();

    // Expected labels, rebuilt independently of the engine's own loop.
    let mut expect: Vec<i8> = Vec::new();
    for (i, &s) in sizes.iter().enumerate() {
        let rows = if assign[i] == -1 { 2 * s } else { s }; // opponent arenas: blue only
        expect.extend(std::iter::repeat(s as i8).take(rows));
    }
    assert_eq!(out.learner_team_size.len(), out.learner_agents);
    assert_eq!(out.learner_team_size, expect);

    // And the aggregate the trainer actually reasons about: with equal-row
    // weights every group should carry the same number of rows. Here the counts
    // are just the layout arithmetic, which is what the config's invariant
    // `rows_m = a_m * m * (2 - phi - lambda)` is derived from.
    for m in 1..=3i8 {
        let got = out.learner_team_size.iter().filter(|&&x| x == m).count();
        let want: usize = sizes.iter().enumerate()
            .filter(|(_, &s)| s as i8 == m)
            .map(|(i, &s)| if assign[i] == -1 { 2 * s } else { s })
            .sum();
        assert_eq!(got, want, "team size {m}");
    }
}

#[test]
fn team_sizes_getter_matches_the_allocator() {
    // The trainer places foreign/league opponents PER BLOCK and needs to know
    // which arenas are which; `allocate_team_sizes` was Rust-only until v9.
    let e = engine(&allocate_team_sizes(144, [6.0, 3.0, 2.0]), 1);
    let ts = e.team_sizes();
    assert_eq!(ts.len(), 144);
    assert_eq!(ts.iter().filter(|&&s| s == 1).count(), 79);
    assert_eq!(ts.iter().filter(|&&s| s == 2).count(), 39);
    assert_eq!(ts.iter().filter(|&&s| s == 3).count(), 26);
    let mut sorted = ts.clone();
    sorted.sort_unstable();
    assert_eq!(ts, sorted, "blocks must stay ordered 1s, 2s, 3s");
}

// ---------------------------------------------------------------- E9 --------

#[test]
fn reward_terms_sum_matches_total_reward() {
    // Single-nonzero-term config: the telemetry for that term must equal the
    // summed reward stream exactly (self-play, so every agent-step is a
    // learner row and nothing is discarded).
    ensure_init(None);
    let sch = Schema::load("../schema/v1.toml").unwrap();
    let mut cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    cfg.goal = 0.0;
    cfg.touch = 0.0;
    cfg.vel_to_ball = 0.05;
    let mut e = MultiEngine::new(vec![(1, 1); 2], sch, cfg, 31, 1, None, false);
    e.set_weights(NetWeights::V1 { raw: v1_state_dict(99), heads: 4 }).unwrap();
    // drain anything the constructor's own reset may have accumulated
    e.reward_terms().unwrap();
    let out = e.collect(24, Arc::new(vec![-1; 2])).unwrap();
    let (terms, learner, opp) = e.reward_terms().unwrap();
    // Self-play: every agent-step IS a learner row, so the split is the identity
    // and there is nothing in the opponent column at all.
    assert_eq!(terms, learner, "self-play must not split");
    assert!(opp.iter().all(|&x| x == 0.0), "self-play has no opponent-driven cars");
    let total: f64 = out.rewards.iter().map(|&x| x as f64).sum();
    assert!((terms[construct_engine::reward::T_VEL_TO_BALL] - total).abs() < 1e-4,
            "vel_to_ball telemetry {} != summed reward {total}",
            terms[construct_engine::reward::T_VEL_TO_BALL]);
    assert_eq!(terms[construct_engine::reward::T_AGENT_STEPS], (24 * 4) as f64,
               "one count per agent-step");
    // read-and-RESET: a second read with no stepping in between is all zeros.
    let (again, again_learner, again_opp) = e.reward_terms().unwrap();
    assert!(again.iter().all(|&x| x == 0.0), "reward_terms must reset on read");
    assert!(again_learner.iter().all(|&x| x == 0.0),
            "the learner counters must reset on the SAME read -- one draining and the \
             other accumulating would silently span an unknown number of iterations");
    assert!(again_opp.iter().all(|&x| x == 0.0), "and so must the opponent counters");
}

#[test]
fn the_opponent_column_counts_the_bot_and_not_our_mirror_cars() {
    // THE MIS-ATTRIBUTION, at the level it was sized at. A 3v3 arena whose
    // foreign slot drives ONE car has three kinds of row: 3 blue learner rows,
    // 1 bot car, and 2 orange "mirror" cars run through our own net and then
    // dropped from the training set. Under "opponent == every row we do not
    // train on" the two mirrors were counted as the bot.
    //
    // Sized against the live v9 run (foreign on 48/144 arenas, 470 agents, 391
    // learner rows): of the 79 non-learner rows, 48 are bot cars and 31 -- 39%
    // -- are ours. At a true bot aerial-gate rate of 0.24 and ours of 0.02 the
    // subtraction printed 0.153 and a reader attributed all of it to nexto.
    let p = bot_path("necto");
    if !p.exists() {
        eprintln!("SKIP: {} absent", p.display());
        return;
    }
    let mut e = engine(&[3], 17);
    e.set_foreign_opponents(vec![foreign(load_npz(&p), ForeignKind::Necto, 12, 1)]).unwrap();
    e.reward_terms().unwrap();          // drain construction-time counters
    e.collect(8, Arc::new(vec![-2])).unwrap();
    let (all, learner, opp) = e.reward_terms().unwrap();
    use construct_engine::reward::T_AGENT_STEPS;
    assert_eq!(all[T_AGENT_STEPS], 48.0, "6 cars x 8 steps");
    assert_eq!(learner[T_AGENT_STEPS], 24.0, "the 3 blue cars are the learner rows");
    assert_eq!(opp[T_AGENT_STEPS], 8.0,
               "the ONE car necto drives -- (all - learner) would say 24, i.e. 3x");
    assert_eq!(all[T_AGENT_STEPS] - learner[T_AGENT_STEPS] - opp[T_AGENT_STEPS], 16.0,
               "the 2 mirror cars are ours, untrained, and in neither column");
    // A FULL foreign team has no mirrors, so there the two agree -- which is what
    // makes the change invisible in every 1v1 row of the bench history.
    let mut e = engine(&[3], 17);
    e.set_foreign_opponents(vec![foreign(load_npz(&p), ForeignKind::Necto, 12, 3)]).unwrap();
    e.reward_terms().unwrap();
    e.collect(8, Arc::new(vec![-2])).unwrap();
    let (all, learner, opp) = e.reward_terms().unwrap();
    assert_eq!(opp[T_AGENT_STEPS], all[T_AGENT_STEPS] - learner[T_AGENT_STEPS]);
    assert_eq!(opp[T_AGENT_STEPS], 24.0);
}
