use construct_engine::sim_init::ensure_init;
use rocketsim_rs::sim::{Arena, CarConfig, Team};

#[test]
fn arena_steps_and_ball_rests_at_spawn_height() {
    ensure_init(None);
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(42));
    arena.pin_mut().step(8);
    let ball = arena.pin_mut().get_ball();
    assert!((ball.pos.z - 93.15).abs() < 1.0, "ball z = {}", ball.pos.z);
    assert_eq!(arena.get_tick_count(), 8);
}
