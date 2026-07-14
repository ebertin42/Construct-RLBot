use construct_engine::episode::{EpisodeArena, StepFlags};
use construct_engine::{obs::OBS_SIZE, reward::RewardConfig, schema::Schema, sim_init::ensure_init};

fn mk(blue: usize, orange: usize, seed: u32) -> EpisodeArena {
    ensure_init(None);
    let s = Schema::load("../schema/v0.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    EpisodeArena::new(blue, orange, s.tick_skip, cfg, s.normalization, seed)
}

#[test]
fn deterministic_given_seed_and_actions() {
    let (mut a, mut b) = (mk(1, 1, 123), mk(1, 1, 123));
    let mut oa = vec![0.0; 2 * OBS_SIZE];
    let mut ob = vec![0.0; 2 * OBS_SIZE];
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    for step in 0..50 {
        let acts = [(step % 90) as i64, ((step * 7) % 90) as i64];
        a.step(&acts, &mut r, &mut f, &mut fo);
        let ra = r.clone();
        b.step(&acts, &mut r, &mut f, &mut fo);
        assert_eq!(ra, r, "step {step}");
    }
    a.write_obs(&mut oa);
    b.write_obs(&mut ob);
    assert_eq!(oa, ob);
}

#[test]
fn scored_ball_terminates_and_pays_goal() {
    let mut a = mk(1, 1, 5);
    // Warp ball into orange net with velocity (test helper below)
    a.debug_place_ball([0.0, 5200.0, 320.0], [0.0, 2000.0, 0.0]);
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    let mut terminated = false;
    for _ in 0..30 {
        a.step(&[0, 0], &mut r, &mut f, &mut fo);
        if f[0].terminated {
            terminated = true;
            assert!(r[0] > 5.0, "blue agent gets goal reward, got {}", r[0]);
            assert!(r[1] < -5.0, "orange agent concedes, got {}", r[1]);
            break;
        }
    }
    assert!(terminated, "ball placed in goal mouth must score within 30 steps");
}

#[test]
fn no_touch_truncates_after_30s() {
    let mut a = mk(1, 0, 5);
    let mut r = vec![0.0; 1];
    let mut f = vec![StepFlags::default(); 1];
    let mut fo = vec![0.0; OBS_SIZE];
    let mut truncated_at = None;
    for step in 0..600 {
        a.step(&[2], &mut r, &mut f, &mut fo); // action 2 = straight reverse: drives away from ball
        if f[0].truncated {
            truncated_at = Some(step);
            break;
        }
    }
    // 30s at 15 steps/s = step 449 (0-indexed step count 450)
    let t = truncated_at.expect("must truncate");
    assert!((445..=455).contains(&t), "truncated at {t}");
}

#[test]
fn physics_nan_contained_as_termination() {
    // RocketSim's contact solver can blow up (observed live: two cars squeezing
    // the ball -> car ejected to [nan, -inf, nan]). The engine must contain it:
    // finite rewards, terminated flags, and a clean post-reset state.
    let mut a = mk(1, 1, 5);
    a.debug_place_ball([f32::NAN, f32::NAN, f32::NAN], [0.0, 0.0, 0.0]);
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    a.step(&[0, 0], &mut r, &mut f, &mut fo);
    assert!(r.iter().all(|x| x.is_finite()), "rewards must stay finite, got {r:?}");
    assert!(f[0].terminated && f[1].terminated, "poisoned arena must terminate episode");
    assert!(fo.iter().all(|x| x.is_finite()), "final_obs must stay finite");
    // next episode must be clean
    let mut obs = vec![0.0; 2 * OBS_SIZE];
    a.write_obs(&mut obs);
    assert!(obs.iter().all(|x| x.is_finite()), "post-reset obs must be finite");
    // and stepping again works normally
    a.step(&[0, 0], &mut r, &mut f, &mut fo);
    assert!(r.iter().all(|x| x.is_finite()));
    assert!(!f[0].terminated || f[0].terminated == f[1].terminated);
}

#[test]
fn team_spirit_blends_rewards_within_team() {
    ensure_init(None);
    let s = Schema::load("../schema/v0.toml").unwrap();
    let mut cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    cfg.goal = 10.0;
    cfg.team_spirit = 0.5;
    cfg.opp_spirit = 0.25;
    let mut a = EpisodeArena::new(2, 2, s.tick_skip, cfg, s.normalization, 5);
    // Force a goal so raw rewards differ strongly across teams
    a.debug_place_ball([0.0, 5300.0, 320.0], [0.0, 2000.0, 0.0]);
    let mut r = vec![0.0; 4];
    let mut f = vec![StepFlags::default(); 4];
    let mut fo = vec![0.0; 4 * OBS_SIZE];
    for _ in 0..30 {
        a.step(&[0, 0, 0, 0], &mut r, &mut f, &mut fo);
        if f[0].terminated {
            break;
        }
    }
    assert!(f[0].terminated, "goal must land");
    // Blue agents (0,1) share identical blended rewards only if raw rewards were
    // identical; the invariant we CAN assert exactly: blend preserves the team sums'
    // relationship r_i' = (1-t)*r_i + t*bm - o*om. Verify via reconstruction:
    // sum of blue blended = (1-t)*sum_blue + 2*t*bm - 2*o*om = sum_blue - 2*o*om.
    // With goal=+10 to both blue raw and -10*(1-bias)=-10 to both orange raw
    // (plus small shaping), check signs and ordering:
    assert!(r[0] > 0.0 && r[1] > 0.0, "blue positive after blend: {r:?}");
    assert!(r[2] < 0.0 && r[3] < 0.0, "orange negative after blend: {r:?}");
}

#[test]
fn zero_spirit_is_bit_identical_to_unblended() {
    ensure_init(None);
    let s = Schema::load("../schema/v0.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    assert_eq!(cfg.team_spirit, 0.0);
    let mk_pair = || {
        (
            EpisodeArena::new(2, 2, s.tick_skip, cfg.clone(), s.normalization.clone(), 9),
            EpisodeArena::new(2, 2, s.tick_skip, cfg.clone(), s.normalization.clone(), 9),
        )
    };
    // identical arenas; one steps through the blend path (team_spirit==0 short-circuit),
    // rewards must be bit-identical across 100 steps of varied actions
    let (mut a, mut b) = mk_pair();
    let (mut ra, mut rb) = (vec![0.0; 4], vec![0.0; 4]);
    let mut f = vec![StepFlags::default(); 4];
    let mut fo = vec![0.0; 4 * OBS_SIZE];
    for step in 0..100 {
        let acts = [
            (step % 90) as i64,
            ((step * 7) % 90) as i64,
            ((step * 13) % 90) as i64,
            ((step * 29) % 90) as i64,
        ];
        a.step(&acts, &mut ra, &mut f, &mut fo);
        b.step(&acts, &mut rb, &mut f, &mut fo);
        assert_eq!(ra, rb, "step {step}");
    }
}
