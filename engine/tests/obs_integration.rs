use construct_engine::{obs, schema::Schema, sim_init::ensure_init};
use rocketsim_rs::sim::{Arena, CarConfig, Team};

fn norm() -> construct_engine::schema::Normalization {
    Schema::load("../schema/v0.toml").unwrap().normalization
}

#[test]
fn kickoff_obs_known_values() {
    ensure_init(None);
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(7));
    let gs = arena.pin_mut().get_game_state();
    let mut o = [0.0f32; obs::OBS_SIZE];
    obs::build_obs(&gs, 0, &norm(), &mut o);
    assert!((o[15] - (100.0 / 3.0) / 100.0).abs() < 1e-4, "kickoff boost 33.33");
    assert_eq!(o[16], 1.0, "on ground at kickoff");
    assert!((o[19].abs() + o[20].abs()) < 1e-6, "ball at x=y=0");
    assert!((o[21] - 93.15 * norm().pos_norm as f32).abs() < 1e-4);
    let slot = &o[34..46];
    assert_eq!(slot[10], 0.0, "first other is opponent (1v1): same_team=0");
    assert_eq!(slot[11], 1.0, "opponent alive");
    assert!(o[46..94].iter().all(|&x| x == 0.0), "remaining slots padded");
}

#[test]
fn orange_obs_mirrors_blue_at_kickoff() {
    ensure_init(None);
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(7));
    let gs = arena.pin_mut().get_game_state();
    let (mut b, mut o) = ([0.0f32; obs::OBS_SIZE], [0.0f32; obs::OBS_SIZE]);
    obs::build_obs(&gs, 0, &norm(), &mut b);
    obs::build_obs(&gs, 1, &norm(), &mut o);
    // Kickoff spawns are 180deg-rotation symmetric -> mirrored obs must match
    for i in 0..obs::OBS_SIZE {
        assert!((b[i] - o[i]).abs() < 1e-4, "idx {i}: {} vs {}", b[i], o[i]);
    }
}
