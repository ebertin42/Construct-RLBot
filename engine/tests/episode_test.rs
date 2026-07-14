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
