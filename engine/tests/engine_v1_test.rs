//! Task T6 (entity-transformer plan): engine v1 collect path. Drives
//! `MultiEngine` directly (no Python) with `schema/v1.toml` and a random
//! fixed-seed v1 state dict, asserting collect shapes/mask pattern/finiteness,
//! fixed-config determinism, the `emit_v0_obs` kickstart buffer, and league
//! (opponent-slot) compatibility. The v0 path's byte-identity gate is the
//! EXISTING test suite staying green untouched.

use construct_engine::{
    actions,
    engine::{MultiEngine, NetWeights},
    obs::OBS_SIZE,
    obs_v1::{ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT},
    reward::RewardConfig,
    sampler::Pcg32,
    schema::Schema,
    sim_init::ensure_init,
};
use std::collections::HashMap;
use std::sync::Arc;

type RawDict = HashMap<String, (Vec<f32>, Vec<usize>)>;

fn rand_vec(rng: &mut Pcg32, n: usize, scale: f32) -> Vec<f32> {
    (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale).collect()
}

/// Deterministic random v1 state dict with the launch dims (d_model=128,
/// layers=2, heads=4, ff=512) and the real 92-row action table. Key names are
/// the policy_v1.rs mapping-table contract.
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

/// 2 arenas of 1v1 on schema v1, single worker thread (fixed config).
fn v1_engine(seed: u32, emit_v0_obs: bool) -> MultiEngine {
    ensure_init(None);
    let sch = Schema::load("../schema/v1.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let mut e = MultiEngine::new(vec![(1, 1); 2], sch, cfg, seed, 1, None, emit_v0_obs);
    e.set_weights(NetWeights::V1 { raw: v1_state_dict(99), heads: 4 }).unwrap();
    e
}

/// Expected 1v1 mask row: self + 1 opp + ball + 6 pads + 4 pred present,
/// both mate slots and opp slots 1/2 masked.
const MASK_1V1: [bool; MAX_ENT] = [
    false, // self
    true, true, // mates absent
    false, true, true, // one opp
    false, // ball
    false, false, false, false, false, false, // 6 big pads
    false, false, false, false, // 4 ball-pred
];

#[test]
fn v1_collect_smoke_shapes_mask_finite() {
    let mut e = v1_engine(7, false);
    let steps = 64usize;
    let out = e.collect(steps, Arc::new(vec![-1; 2])).unwrap();
    let n = out.learner_agents;
    assert_eq!(n, 4, "2 arenas x 1v1 self-play -> 4 learner agents");

    // shapes
    assert_eq!(out.ents.len(), steps * n * MAX_ENT * ENT_FEAT);
    assert_eq!(out.mask.len(), steps * n * MAX_ENT);
    assert_eq!(out.query.len(), steps * n * Q_FEAT);
    assert_eq!(out.prev.len(), steps * n * PREV_ACTIONS);
    assert_eq!(out.actions.len(), steps * n);
    assert_eq!(out.logprobs.len(), steps * n);
    assert_eq!(out.values.len(), steps * n);
    assert_eq!(out.rewards.len(), steps * n);
    assert_eq!(out.final_values.len(), steps * n);
    assert_eq!(out.last_values.len(), n);
    assert!(out.obs.is_empty(), "v0 obs buffer must be unused in v1 mode");
    assert!(out.obs_v0.is_empty(), "obs_v0 must be absent when emit_v0_obs=false");

    // finiteness
    assert!(out.ents.iter().all(|x| x.is_finite()));
    assert!(out.query.iter().all(|x| x.is_finite()));
    assert!(out.logprobs.iter().all(|x| x.is_finite()));
    assert!(out.values.iter().all(|x| x.is_finite()));
    assert!(out.rewards.iter().all(|x| x.is_finite()));
    assert!(out.final_values.iter().all(|x| x.is_finite()));
    assert!(out.last_values.iter().all(|x| x.is_finite()));

    // actions in range [0, 92)
    assert!(out.actions.iter().all(|&a| a >= 0 && (a as usize) < actions::TABLE_SIZE_V1));

    // mask pattern: every (t, agent) row must be the 1v1 pattern
    for t in 0..steps {
        for a in 0..n {
            let row = &out.mask[(t * n + a) * MAX_ENT..(t * n + a + 1) * MAX_ENT];
            assert_eq!(row, &MASK_1V1, "mask mismatch at t={t} agent={a}");
        }
    }

    // prev-action ring: t=0 is all zeros (fresh episodes); at t=1, slot 0 of
    // each agent's ring is the action executed at t=0 (no resets happen on
    // the first kickoff step).
    for a in 0..n {
        assert!(!out.terminated[a] && !out.truncated[a], "unexpected episode end at t=0");
        let ring0 = &out.prev[a * PREV_ACTIONS..(a + 1) * PREV_ACTIONS];
        assert!(ring0.iter().all(|&x| x == 0), "t=0 ring must be zeros, got {ring0:?}");
        let ring1 = &out.prev[(n + a) * PREV_ACTIONS..(n + a + 1) * PREV_ACTIONS];
        assert_eq!(ring1[0], out.actions[a], "t=1 ring slot 0 must hold t=0's action");
        assert!(ring1[1..].iter().all(|&x| x == 0), "t=1 ring tail must still be zeros");
    }
}

#[test]
fn v1_collect_is_deterministic_for_fixed_config() {
    let mut a = v1_engine(7, false);
    let mut b = v1_engine(7, false);
    let steps = 32usize;
    let oa = a.collect(steps, Arc::new(vec![-1; 2])).unwrap();
    let ob = b.collect(steps, Arc::new(vec![-1; 2])).unwrap();
    assert_eq!(oa.ents, ob.ents, "ents must be bit-identical");
    assert_eq!(oa.mask, ob.mask);
    assert_eq!(oa.query, ob.query);
    assert_eq!(oa.prev, ob.prev);
    assert_eq!(oa.actions, ob.actions, "actions must be identical");
    assert_eq!(oa.logprobs, ob.logprobs);
    assert_eq!(oa.values, ob.values);
    assert_eq!(oa.rewards, ob.rewards);
    assert_eq!(oa.last_values, ob.last_values);
}

#[test]
fn emit_v0_obs_populates_kickstart_buffer() {
    let mut e = v1_engine(3, true);
    let steps = 16usize;
    let out = e.collect(steps, Arc::new(vec![-1; 2])).unwrap();
    let n = out.learner_agents;
    assert_eq!(out.obs_v0.len(), steps * n * OBS_SIZE, "obs_v0 must be [T,N,94]");
    assert!(out.obs_v0.iter().all(|x| x.is_finite()));
    // v0 obs is not all-zero (real kickoff state)
    assert!(out.obs_v0.iter().any(|&x| x != 0.0));
}

#[test]
fn v1_opponent_slots_shrink_learner_rows() {
    let mut e = v1_engine(11, false);
    e.set_opponents(vec![NetWeights::V1 { raw: v1_state_dict(123), heads: 4 }]).unwrap();
    let steps = 16usize;
    // arena 0 plays opponent slot 0 (orange filtered out), arena 1 self-play
    let out = e.collect(steps, Arc::new(vec![0, -1])).unwrap();
    assert_eq!(out.learner_agents, 3, "arena0 blue + arena1 both = 3 learner rows");
    assert_eq!(out.ents.len(), steps * 3 * MAX_ENT * ENT_FEAT);
    assert_eq!(out.prev.len(), steps * 3 * PREV_ACTIONS);
    assert!(out.ents.iter().all(|x| x.is_finite()));
    assert!(out.values.iter().all(|x| x.is_finite()));
    assert!(out.actions.iter().all(|&a| a >= 0 && (a as usize) < actions::TABLE_SIZE_V1));
}
