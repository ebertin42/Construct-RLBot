//! Integration gates for the replay-state reset lever.
//!
//! The unit tests in `reset_pool.rs`/`episode.rs` pin the loader and the branch
//! algebra; these run the real engine end to end, and exist mainly to prove the
//! two operational contracts: a replay reset is a *steppable* state (not just a
//! well-formed one), and a training box whose `data/` is stale degrades to
//! today's behavior instead of taking a 192-arena run down.

use std::sync::Arc;

use std::collections::HashMap;

use construct_engine::actions;
use construct_engine::curriculum::CurriculumConfig;
use construct_engine::engine::{MultiEngine, NetWeights};
use construct_engine::obs_v1::{ENT_FEAT, PREV_ACTIONS, Q_FEAT};
use construct_engine::sampler::Pcg32;
use construct_engine::episode::{EpisodeArena, StepFlags};
use construct_engine::obs::OBS_SIZE;
use construct_engine::reset_pool;
use construct_engine::reward::RewardConfig;
use construct_engine::schema::Schema;
use construct_engine::sim_init::ensure_init;

const FIXTURE: &str = "tests/fixtures/reset_pool_mini.jsonl";

fn base_curriculum() -> CurriculumConfig {
    CurriculumConfig::load("../configs/curriculum_v1.toml").unwrap()
}

/// v2-shaped curriculum backed by the mini fixture rather than the 611 MB
/// corpus (which may not exist on a fresh checkout).
fn fixture_curriculum(replay: f32, kickoff: f32, random: f32) -> CurriculumConfig {
    let mut c = base_curriculum();
    c.replay_weight = replay;
    c.kickoff_weight = kickoff;
    c.random_weight = random;
    c.pool = Arc::new(reset_pool::load_or_empty(FIXTURE, 0));
    assert!(!c.pool.is_empty(), "fixture pool must load");
    c
}

fn mk(blue: usize, orange: usize, seed: u32, c: CurriculumConfig) -> EpisodeArena {
    ensure_init(None);
    let s = Schema::load("../schema/v0.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    EpisodeArena::new_with_curriculum(blue, orange, s.tick_skip, cfg, s.normalization, seed, Some(c))
}

// ---- T27 ----
#[test]
fn curriculum_v2_config_loads_from_repo() {
    let c = CurriculumConfig::load("../configs/curriculum_v2.toml").unwrap();
    assert_eq!(c.replay_weight, 0.7);
    assert_eq!(c.kickoff_weight, 0.1);
    assert_eq!(c.random_weight, 0.2);
    let rp = c.replay_pool.as_ref().expect("v2 declares a [replay_pool]");
    assert_eq!(rp.path, "data/reset_pool_v5.jsonl");
    assert_eq!(rp.max_states, 0);
    // Deliberately NOT asserting the pool is non-empty: `data/` is synced
    // separately from the code and is absent on a fresh checkout. That case
    // must degrade gracefully, not fail the suite.
}

// ---- T28 ----
#[test]
fn stepping_after_replay_reset_is_stable() {
    let mut a = mk(1, 1, 21, fixture_curriculum(1.0, 0.0, 0.0));
    let mut r = vec![0.0f32; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0f32; 2 * OBS_SIZE];
    let mut obs = vec![0.0f32; 2 * OBS_SIZE];
    for i in 0..600 {
        a.step(&[0, 45], &mut r, &mut f, &mut fo);
        a.write_obs(&mut obs);
        assert!(r.iter().all(|x| x.is_finite()), "reward went nonfinite at step {i}");
        assert!(obs.iter().all(|x| x.is_finite()), "obs went nonfinite at step {i}");
        assert!(fo.iter().all(|x| x.is_finite()), "final obs went nonfinite at step {i}");
    }
    // NOT `== 0`. Blowups here are contact-solver chaos and depend on heap
    // layout: repeating an identical 2000-reset run 8 times gave replay
    // `3 2 0 0 0 0 0 0` and the random branch this displaces `2 2 0 1 0 2 0 0`,
    // so a hard zero is a coincidence of allocation, not an invariant. What is
    // worth gating is that replay resets are not a NEW source of instability
    // relative to the branch they displace, and that containment keeps the run
    // alive — `episode::step_impl` zeroes rewards and terminates on containment,
    // so the finite asserts above are what protect the learner.
    let blowups = a.blowup_count();
    assert!(
        blowups <= 8,
        "replay resets drove {blowups} solver blowups over 600 steps — far above the \
         handful-per-thousand baseline the random branch also shows"
    );
}

// ---- T29 ----
#[test]
fn replay_reset_never_terminates_on_first_step() {
    // Direct proof that filter F2 keeps a replay reset from firing
    // `is_ball_scored()` on the very first step of the episode.
    //
    // This test used to be vacuous. F2 rejected only |ball.y| > 5120 (the real
    // goal line), but `step` advances tick_skip=8 ticks before the policy acts,
    // so a ball INSIDE the line travelling goalward crossed the 5215.5 score
    // threshold during that first step. Measured against the full 938,801-state
    // corpus: 15 first-step terminations in 50,000 resets (~3e-4), each paying a
    // full ±goal reward — worst observed reward 24.65 — for a state the policy
    // had zero agency over. It passed anyway because the 64-line fixture topped
    // out at |ball.y| = 5067.7 and never produced one.
    //
    // Two changes make the assertion real. F2's band is now 4800, which is
    // `BALL_MAX_SPEED * tick_skip / 120 = 400` uu plus margin inside the score
    // threshold, so a first-step goal is arithmetically unreachable rather than
    // merely rare (pinned by `f2_band_covers_one_tick_skip_of_travel`). And the
    // fixture now carries 6 real corpus states sitting AT that boundary with
    // hard goalward velocity (|y| up to 4800.0, |vy| up to 2750), so the loop
    // below actually exercises the case it claims to.
    let pool = reset_pool::load_or_empty(FIXTURE, 0);
    let hardest = pool
        .iter()
        .filter(|s| s.ball.pos[1] * s.ball.vel[1] > 0.0)
        .map(|s| s.ball.pos[1].abs())
        .fold(0.0f32, f32::max);
    assert!(
        hardest > 4700.0,
        "fixture must contain goalward near-boundary states or this test proves nothing; \
         hardest |ball.y| with goalward vel = {hardest}"
    );

    let mut a = mk(1, 1, 909, fixture_curriculum(1.0, 0.0, 0.0));
    let mut r = vec![0.0f32; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0f32; 2 * OBS_SIZE];
    let mut contained = 0u64;
    for i in 0..2_000 {
        let before = a.blowup_count();
        a.debug_force_reset();
        a.step(&[0, 0], &mut r, &mut f, &mut fo);
        // A contained physics blowup also terminates the episode, and that is
        // BY DESIGN — `step_impl` zeroes the rewards, so nothing reaches the
        // learner. Only a goal termination is the hazard F2 exists to stop, so
        // separate the two rather than asserting a blanket "never terminates"
        // that a rebuild would trip.
        if a.blowup_count() > before {
            contained += 1;
            assert!(
                r.iter().all(|x| *x == 0.0),
                "containment must zero rewards, got {r:?} at reset {i}"
            );
            continue;
        }
        assert!(!f[0].terminated && !f[1].terminated, "instant goal termination at reset {i}");
        assert!(!f[0].truncated && !f[1].truncated, "instant truncation at reset {i}");
    }
    // Guard against the loop passing because every reset blew up and got
    // `continue`d past.
    assert!(contained < 20, "{contained}/2000 resets blew up — that is a regression, not noise");
}

/// Companion to T29: the states F2 now rejects are exactly the ones that used to
/// score on step 1. Drives them through the engine WITHOUT the filter to show
/// the hazard is real and that the band is what removes it — otherwise a future
/// widening of `GOAL_LINE_Y` looks free.
#[test]
fn near_goal_states_would_terminate_on_first_step_without_f2() {
    use construct_engine::reset_pool::{BallSpawn, CarSpawn, ResetState};

    // Hand-built rather than corpus-loaded: `accept` rejects this now, which is
    // the whole point. The ball figures are a real case observed firing on step
    // 1 against the pre-fix filter (rewards [24.65, -22.17]): y=5063.5 with
    // vy=2843.5 reaches 5253 in 8 ticks, past the 5215.5 score threshold. Cars
    // are parked at midfield and well apart so the episode ends on the GOAL and
    // not on a contact-solver blowup, which would zero the rewards instead.
    let car = |team: u8, x: f32| CarSpawn {
        pos: [x, 0.0, 17.0],
        vel: [0.0, 100.0, 0.0],
        ang_vel: [0.0, 0.0, 0.1],
        quat: [0.0, 0.0, 0.0, 1.0],
        boost: 0.5,
        team,
        on_ground: true,
    };
    let st = ResetState {
        ball: BallSpawn {
            pos: [-44.2, 5063.5, 98.2],
            vel: [-1021.2, 2843.5, 56.7],
            ang_vel: [0.0; 3],
        },
        cars: [car(0, -1500.0), car(1, 1500.0)],
    };
    assert!(
        !reset_pool::is_acceptable(&st),
        "F2 must reject this state — if it doesn't, the band regressed"
    );

    let mut c = base_curriculum();
    c.replay_weight = 1.0;
    c.kickoff_weight = 0.0;
    c.random_weight = 0.0;
    c.pool = Arc::new(vec![st]);
    let mut a = mk(1, 1, 4242, c);
    let mut r = vec![0.0f32; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0f32; 2 * OBS_SIZE];
    a.debug_force_reset();
    a.step(&[0, 0], &mut r, &mut f, &mut fo);
    assert!(
        f[0].terminated || f[1].terminated,
        "the hazard F2 exists to prevent must actually be a hazard"
    );
    assert!(
        r.iter().any(|x| x.abs() > 1.0),
        "and it must pay a real goal reward: {r:?}"
    );
}

// ---- T30: end-to-end gate for the team-size fallback ----
#[test]
fn mixed_team_sizes_with_pool_run_clean() {
    ensure_init(None);
    let s = Schema::load("../schema/v1.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let sizes = vec![(1usize, 1usize), (1, 1), (2, 2), (3, 3)];
    let mut e = MultiEngine::new(
        sizes,
        s,
        cfg,
        4242,
        2,
        Some(fixture_curriculum(0.7, 0.1, 0.2)),
        false,
    );
    e.set_weights(NetWeights::V1 { raw: v1_state_dict(99), heads: 4 }).unwrap();
    let out = e.collect(200, Arc::new(vec![-1; 4])).unwrap();
    assert!(out.ents.iter().all(|x| x.is_finite()), "entity obs must stay finite");
    assert!(out.query.iter().all(|x| x.is_finite()), "query obs must stay finite");
    assert!(out.rewards.iter().all(|x| x.is_finite()), "rewards must stay finite");
}

// ---- T31: the single most important test here ----
#[test]
fn missing_pool_file_does_not_fail_engine_construction() {
    // A de-synced `data/` directory must degrade a training box to current
    // behavior, never kill the run.
    let p = std::env::temp_dir().join("construct_curriculum_absent_pool.toml");
    std::fs::write(
        &p,
        "kickoff_weight = 0.1\nrandom_weight = 0.2\nreplay_weight = 0.7\n\n\
         [replay_pool]\npath = \"data/definitely_absent.jsonl\"\n",
    )
    .unwrap();
    let c = CurriculumConfig::load(p.to_str().unwrap()).expect("absent pool must not fail load()");
    assert!(c.pool.is_empty());
    let _ = std::fs::remove_file(&p);

    ensure_init(None);
    let s = Schema::load("../schema/v1.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let mut e = MultiEngine::new(vec![(1, 1); 4], s, cfg, 7, 2, Some(c), false);
    e.set_weights(NetWeights::V1 { raw: v1_state_dict(99), heads: 4 }).unwrap();
    let out = e.collect(100, Arc::new(vec![-1; 4])).unwrap();
    assert!(out.rewards.iter().all(|x| x.is_finite()));
}

/// Deterministic v1 state dict at the launch dims (d_model=128, layers=2,
/// heads=4, ff=512) — same construction as `engine_v1_test.rs`. These tests
/// only care that the reset/physics path stays finite, not what the net says.
fn v1_state_dict(seed: u64) -> HashMap<String, (Vec<f32>, Vec<usize>)> {
    type RawDict = HashMap<String, (Vec<f32>, Vec<usize>)>;
    let (d, layers, ff) = (128usize, 2usize, 512usize);
    let mut rng = Pcg32::new(seed);
    let mut m: RawDict = HashMap::new();
    fn rand_vec(rng: &mut Pcg32, n: usize, scale: f32) -> Vec<f32> {
        (0..n).map(|_| (rng.next_f32() * 2.0 - 1.0) * scale).collect()
    }
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
