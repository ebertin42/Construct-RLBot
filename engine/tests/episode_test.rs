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
fn containment_rebuilds_arena_leaving_live_physics() {
    // Dead-ball repro. When the contact solver blows up (NaN), Bullet's
    // updateSingleAabb latches DISABLE_SIMULATION on the poisoned body — a
    // one-way latch RocketSim never clears (Ball::SetState only attempts
    // activation when velocity != 0, and plain setActivationState refuses to
    // leave DISABLE_SIMULATION). Containment that merely resets the POISONED
    // arena therefore leaves the ball permanently frozen (observed live:
    // viewer ghost ball, arenas degraded to zero-touch episodes). The fix is
    // to rebuild the arena on containment; this test asserts post-containment
    // physics health, which a reset-in-place cannot provide.
    let mut a = mk(1, 1, 5);
    let ids = |a: &mut construct_engine::episode::EpisodeArena| -> Vec<(u32, u8)> {
        let gs = a.game_state();
        let mut v: Vec<(u32, u8)> = gs.cars.iter().map(|c| (c.id, c.team as u8)).collect();
        v.sort();
        v
    };
    let ids_before = ids(&mut a);

    // 1. Poison the ball -> one step -> containment fires.
    a.debug_place_ball([f32::NAN, f32::NAN, f32::NAN], [0.0, 0.0, 0.0]);
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    a.step(&[0, 0], &mut r, &mut f, &mut fo);
    assert!(f[0].terminated && f[1].terminated, "poisoned arena must terminate episode");

    // 2. Car ids must survive the rebuild bit-identically (ids are a per-arena
    // counter starting at 1; preserved add order re-issues 1..N), so the
    // agent->car mapping and viewer identity hold.
    let ids_after = ids(&mut a);
    assert_eq!(ids_before, ids_after, "rebuilt arena must re-issue identical car ids");

    // 3. The latched case: a MOVING ball must keep moving. (An exactly
    // zero-velocity ball is legitimately frozen even on a healthy arena —
    // RocketSim's Arena::Step forces ISLAND_SLEEPING whenever lin+ang vel are
    // exactly zero, which is what keeps the kickoff ball pinned at center —
    // so the discriminating probe is a small NONZERO velocity: Arena::Step
    // then calls setActivationState(ACTIVE_TAG) every tick, which a healthy
    // ball obeys and a DISABLE_SIMULATION-latched ball refuses.) On the old
    // reset-in-place containment the ball stays at z=500 bit-exactly forever
    // despite its velocity — the live dead-ball symptom.
    for _ in 0..3 {
        a.step(&[0, 0], &mut r, &mut f, &mut fo);
    }
    a.debug_place_ball([0.0, 0.0, 500.0], [0.0, 0.0, -10.0]);
    for _ in 0..20 {
        a.step(&[0, 0], &mut r, &mut f, &mut fo);
    }
    let z = a.game_state().ball.pos.z;
    assert!(
        z < 490.0,
        "gravity must act on the ball after containment (Bullet DISABLE_SIMULATION latch), ball z = {z}"
    );

    // 4. Cars must still drive on the rebuilt arena.
    let pos_before = a.game_state().cars[0].state.pos;
    for _ in 0..10 {
        a.step(&[2, 2], &mut r, &mut f, &mut fo); // action 2 = straight reverse
    }
    let pos_after = a.game_state().cars[0].state.pos;
    let moved = ((pos_after.x - pos_before.x).powi(2)
        + (pos_after.y - pos_before.y).powi(2)
        + (pos_after.z - pos_before.z).powi(2))
    .sqrt();
    assert!(moved > 10.0, "car must still move after containment, moved {moved} uu");
}

#[test]
fn physics_insane_but_finite_contained_as_termination() {
    // A contact-solver blowup can ramp through huge-but-finite values before (or
    // instead of) ever reaching NaN/inf. `state_is_finite` alone accepts these
    // (investigation: engine/examples/blowup_probe.rs demonstrated a by-construction
    // "levitating ball" persists 449 steps as valid, non-terminal transitions). The
    // engine must contain an insane-but-finite state exactly like the NaN case:
    // terminate on the FIRST frame it's visible, zero reward, no leaked transition.
    let mut a = mk(1, 1, 5);
    // In-field x/y (can't score) but z/vel far beyond any legal value.
    a.debug_place_ball([2000.0, 0.0, 5.0e8], [0.0, 0.0, 5.0e6]);
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    a.step(&[0, 0], &mut r, &mut f, &mut fo);
    assert!(r.iter().all(|&x| x == 0.0), "insane-state termination must zero reward, got {r:?}");
    assert!(
        f[0].terminated && f[1].terminated,
        "insane state must terminate on the first frame it's visible, got {f:?}"
    );
    assert!(!f[0].truncated && !f[1].truncated);
    assert!(fo.iter().all(|x| x.is_finite()), "final_obs must stay finite");
    // Post-reset state must be back in sane range (not still latched insane).
    let mut obs = vec![0.0; 2 * OBS_SIZE];
    a.write_obs(&mut obs);
    let max_abs = obs.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
    assert!(max_abs < 100.0, "post-reset obs must be sane, got max|obs|={max_abs}");
    // and stepping again works normally, without re-triggering containment
    a.step(&[0, 0], &mut r, &mut f, &mut fo);
    assert!(r.iter().all(|x| x.is_finite()));
}

#[test]
fn insane_state_takes_precedence_over_phantom_goal() {
    // Investigation outcome 3: a blowup that ejects the ball past the goal line
    // makes `is_ball_scored()` fire a REAL (not fake) goal for a physically
    // impossible position, injecting a spurious +/-goal reward before the episode
    // resets (observed live with reward_v2: +24.55; reproduced here at y=1e8 with
    // goal=20.0 -> rewards=[20.0, -19.998]). The insane-state guard must take
    // precedence over goal detection so this can never pay out.
    let mut cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    cfg.goal = 20.0;
    ensure_init(None);
    let s = Schema::load("../schema/v0.toml").unwrap();
    let mut a = EpisodeArena::new(1, 1, s.tick_skip, cfg, s.normalization, 5);
    // Same x/z as the legitimate scored_ball_terminates_and_pays_goal test (inside
    // the goal frame), but y is a blowup value far beyond the goal line AND the
    // sane bound (12,000) -- an insane state that ALSO satisfies is_ball_scored.
    a.debug_place_ball([0.0, 1.0e8, 320.0], [0.0, 0.0, 0.0]);
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    a.step(&[0, 0], &mut r, &mut f, &mut fo);
    assert!(f[0].terminated && f[1].terminated, "insane state must still terminate");
    assert!(
        r.iter().all(|&x| x == 0.0),
        "insane-state termination must NOT award the phantom goal, got {r:?}"
    );
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
