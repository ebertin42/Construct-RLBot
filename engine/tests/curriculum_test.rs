use construct_engine::curriculum::CurriculumConfig;
use construct_engine::episode::{EpisodeArena, StepFlags};
use construct_engine::obs::OBS_SIZE;
use construct_engine::reward::RewardConfig;
use construct_engine::schema::Schema;
use construct_engine::sim_init::ensure_init;

fn mk(curriculum: Option<CurriculumConfig>, seed: u32) -> EpisodeArena {
    ensure_init(None);
    let s = Schema::load("../schema/v0.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    EpisodeArena::new_with_curriculum(1, 1, s.tick_skip, cfg, s.normalization, seed, curriculum)
}

fn all_random() -> CurriculumConfig {
    let mut c = CurriculumConfig::load("../configs/curriculum_v1.toml").unwrap();
    c.kickoff_weight = 0.0;
    c.random_weight = 1.0;
    c
}

#[test]
fn config_loads() {
    let c = CurriculumConfig::load("../configs/curriculum_v1.toml").unwrap();
    assert!(c.kickoff_weight > 0.0 && c.random_weight > 0.0);
    assert_eq!(c.random.min_separation, 300.0);
}

#[test]
fn random_resets_vary_and_respect_bounds() {
    let mut a = mk(Some(all_random()), 42);
    let mut ball_positions = Vec::new();
    for _ in 0..30 {
        let gs = a.game_state();
        let b = gs.ball.pos;
        assert!(b.x.abs() <= 3500.0 && b.y.abs() <= 4500.0, "ball xy in bounds: {b:?}");
        assert!(b.z >= 93.0 && b.z <= 1700.0, "ball z in bounds: {}", b.z);
        let bv = gs.ball.vel;
        assert!((bv.x * bv.x + bv.y * bv.y + bv.z * bv.z).sqrt() <= 2500.0 * 1.001);
        for c in &gs.cars {
            let p = c.state.pos;
            assert!(p.x.abs() <= 3500.0 && p.y.abs() <= 4500.0 && p.z >= 16.0 && p.z <= 1700.0);
            let d = ((p.x - b.x).powi(2) + (p.y - b.y).powi(2) + (p.z - b.z).powi(2)).sqrt();
            // separation is best-effort (10 attempts) — assert it holds in the vast majority
            ball_positions.push((b.x, b.y, d));
        }
        a.debug_force_reset(); // test helper: trigger reset_episode directly
    }
    // variety: ball must not always sit at the kickoff spot
    let at_origin = ball_positions.iter().filter(|(x, y, _)| x.abs() < 1.0 && y.abs() < 1.0).count();
    assert!(at_origin < ball_positions.len() / 2, "random resets look like kickoffs");
    let sep_ok = ball_positions.iter().filter(|(_, _, d)| *d >= 300.0).count();
    assert!(sep_ok * 10 >= ball_positions.len() * 9, "separation holds >=90%: {sep_ok}/{}", ball_positions.len());
}

#[test]
fn deterministic_given_seed() {
    let (mut a, mut b) = (mk(Some(all_random()), 7), mk(Some(all_random()), 7));
    for _ in 0..10 {
        let (ga, gb) = (a.game_state(), b.game_state());
        assert_eq!(ga.ball.pos.x, gb.ball.pos.x);
        assert_eq!(ga.cars[0].state.pos.y, gb.cars[0].state.pos.y);
        a.debug_force_reset();
        b.debug_force_reset();
    }
}

#[test]
fn no_curriculum_means_kickoff_only() {
    let mut a = mk(None, 3);
    for _ in 0..5 {
        let gs = a.game_state();
        assert!(gs.ball.pos.x.abs() < 1.0 && gs.ball.pos.y.abs() < 1.0, "kickoff ball at center");
        a.debug_force_reset();
    }
}

#[test]
fn stepping_after_random_reset_is_stable() {
    let mut a = mk(Some(all_random()), 21);
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    for _ in 0..200 {
        a.step(&[0, 45], &mut r, &mut f, &mut fo);
        assert!(r.iter().all(|x| x.is_finite()));
    }
}
