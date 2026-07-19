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
        assert!(gs.ball.pos.x.abs() < 12.0 && gs.ball.pos.y.abs() < 12.0, "kickoff ball near center (±10uu jitter)");
        a.debug_force_reset();
    }
}

// Kickoff formation + jitter bound: a fresh no-curriculum EpisodeArena's
// constructor kickoff starts from the SAME RocketSim kickoff formation as a
// raw arena kicked off with the RAW engine seed (the LCG seed advance
// happens only between episodes, never before the first) — this was
// previously bit-identical, but kickoff spawn jitter (episode.rs's
// `jitter_kickoff_spawns`, added to break the symmetric car-car-ball pinch
// that was manufacturing solver blowups nearly every kickoff) now adds small
// independent per-car x/y + yaw noise on top of it. The ball is never
// jittered (stays bit-identical); car positions must land within the
// documented jitter bound. Pins the post-jitter behavior against the actual
// RocketSim reference rather than magic numbers.
#[test]
fn constructor_kickoff_matches_raw_arena_within_jitter_bounds() {
    use rocketsim_rs::sim::{Arena, CarConfig, Team};

    // Mirrors episode.rs's KICKOFF_JITTER_POS (private to that module — kept
    // in sync here rather than exposed as a public constant just for this).
    const KICKOFF_JITTER_POS: f32 = 150.0;

    ensure_init(None);
    let s = Schema::load("../schema/v0.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    let mut a = EpisodeArena::new(1, 1, s.tick_skip, cfg, s.normalization, 42);
    let gs = a.game_state();

    let mut raw = Arena::default_standard();
    let _ = raw.pin_mut().add_car(Team::Blue, CarConfig::octane());
    let _ = raw.pin_mut().add_car(Team::Orange, CarConfig::octane());
    raw.pin_mut().reset_to_random_kickoff(Some(42));
    let gr = raw.pin_mut().get_game_state();

    // Ball spawn now carries ±10uu horizontal jitter (the decisive symmetry
    // breaker); z stays exact.
    assert!((gs.ball.pos.x - gr.ball.pos.x).abs() <= 10.0 + 1e-3, "ball x beyond jitter bound");
    assert!((gs.ball.pos.y - gr.ball.pos.y).abs() <= 10.0 + 1e-3, "ball y beyond jitter bound");
    assert_eq!(gs.ball.pos.z, gr.ball.pos.z, "ball z must be exact");
    assert_eq!(gs.cars.len(), gr.cars.len());
    for (c, r) in gs.cars.iter().zip(gr.cars.iter()) {
        let dx = c.state.pos.x - r.state.pos.x;
        let dy = c.state.pos.y - r.state.pos.y;
        assert!(
            dx.abs() <= KICKOFF_JITTER_POS + 1e-3 && dy.abs() <= KICKOFF_JITTER_POS + 1e-3,
            "car {} jittered pos too far from raw seed-42 kickoff: dx={dx} dy={dy}",
            c.id
        );
        // z is never jittered (grounded kickoff rest height is untouched).
        assert_eq!(c.state.pos.z, r.state.pos.z, "car {} z must be untouched by jitter", c.id);
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
