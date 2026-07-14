use rocketsim_rs::{sim::Team, GameState};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RewardConfig {
    pub goal: f32,
    pub touch: f32,
    pub vel_to_ball: f32,
}

impl RewardConfig {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        toml::from_str(&text).map_err(|e| format!("{path}: {e}"))
    }
}

/// Reward for one agent for the transition prev -> cur.
/// `scored`: Some(team) if a goal was scored during this step.
///
/// Touch detection (deviates from the task brief's draft — verified against the
/// rocketsim_rs 0.37.0 source at `src/sim/ball_hit_info.rs`):
///
/// `BallHitInfo` has the fields the brief assumed (`is_valid: bool`,
/// `tick_count_when_hit: u64`, ...), but the brief's comparison
/// `hit_now > hit_before` is unsound. A car that has never touched the ball has
/// `tick_count_when_hit == u64::MAX` (a sentinel written by the underlying sim reset,
/// not Rust's `#[derive(Default)]` value of 0 — confirmed empirically: a fresh car has
/// `is_valid == false` and `tick_count_when_hit == u64::MAX`). That means a car's
/// *first-ever* touch transitions `tick_count_when_hit` from `u64::MAX` down to a small
/// real tick number, so `hit_now > hit_before` is false on exactly the touch that
/// matters most, and the touch bonus would never fire on a car's first-ever contact.
///
/// Fix: treat a touch as "the hit tick changed AND is now valid AND falls inside this
/// step's tick window (prev.tick_count, cur.tick_count]". This is correct for the
/// bootstrap MAX -> real-tick case above, for ordinary repeat touches (old real tick ->
/// new real tick), and for arena reuse across episode resets (Task 7's EpisodeArena): if
/// `tick_count` is reset to a small value at the start of a new episode while a stale
/// `tick_count_when_hit` from the previous episode is still numerically larger, a plain
/// `!=` check without the tick-window bound could misfire; bounding by
/// `(prev.tick_count, cur.tick_count]` prevents that.
pub fn compute(
    prev: &GameState,
    cur: &GameState,
    car_idx: usize,
    scored: Option<Team>,
    cfg: &RewardConfig,
) -> f32 {
    let me = &cur.cars[car_idx];
    let mut r = 0.0f32;

    if let Some(team) = scored {
        r += if team == me.team { cfg.goal } else { -cfg.goal };
    }

    // touch: ball_hit_info recorded a new, valid hit during this step's tick window
    let hit_now = &me.state.ball_hit_info;
    let hit_before = &prev.cars[car_idx].state.ball_hit_info;
    let touched = hit_now.is_valid
        && hit_now.tick_count_when_hit != hit_before.tick_count_when_hit
        && hit_now.tick_count_when_hit > prev.tick_count
        && hit_now.tick_count_when_hit <= cur.tick_count;
    if touched {
        r += cfg.touch;
    }

    // vel_to_ball: projection of car velocity onto unit vector toward ball
    let (bp, mp, mv) = (cur.ball.pos, me.state.pos, me.state.vel);
    let d = [bp.x - mp.x, bp.y - mp.y, bp.z - mp.z];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let proj = (mv.x * d[0] + mv.y * d[1] + mv.z * d[2]) / dist;
    r += cfg.vel_to_ball * (proj / 2300.0).clamp(0.0, 1.0);

    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim_init::ensure_init;
    use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

    fn cfg() -> RewardConfig {
        RewardConfig { goal: 10.0, touch: 0.5, vel_to_ball: 0.05 }
    }

    #[test]
    fn goal_reward_signed_by_team() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let gs = arena.pin_mut().get_game_state();

        // Deviation from the brief's literal test: `gs.cars` is NOT in add_car order
        // (verified empirically — with Blue added then Orange, `get_game_state()`
        // returns cars[0] = Orange, cars[1] = Blue for this arena/binding version), so
        // we look up each team's index rather than assume car_idx 0 == Blue. The
        // reward-signing behavior under test (compute() sides with the car's own
        // `me.team`, independent of array position) is unaffected and still verified.
        let blue_idx = gs.cars.iter().position(|c| c.team == Team::Blue).unwrap();
        let orange_idx = gs.cars.iter().position(|c| c.team == Team::Orange).unwrap();

        let blue = compute(&gs, &gs, blue_idx, Some(Team::Blue), &cfg());
        let orange = compute(&gs, &gs, orange_idx, Some(Team::Blue), &cfg());
        assert_eq!(blue, 10.0 + expected_shaping(&gs, blue_idx, &cfg()));
        assert_eq!(orange, -10.0 + expected_shaping(&gs, orange_idx, &cfg()));
    }

    #[test]
    fn driving_toward_ball_pays_positive() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let prev = arena.pin_mut().get_game_state();
        // full throttle toward ball for 1 second
        for _ in 0..15 {
            let id = prev.cars[0].id;
            arena.pin_mut()
                .set_car_controls(id, CarControls { throttle: 1.0, ..Default::default() })
                .unwrap();
            arena.pin_mut().step(8);
        }
        let cur = arena.pin_mut().get_game_state();
        let r = compute(&prev, &cur, 0, None, &cfg());
        assert!(r > 0.0, "moving at ball should reward, got {r}");
        assert!(r <= 0.05 + 0.5, "bounded by vel_to_ball + touch weights, got {r}");
    }

    // helper used by test 1: shaping-only value (no goal term)
    fn expected_shaping(gs: &rocketsim_rs::GameState, idx: usize, c: &RewardConfig) -> f32 {
        compute(gs, gs, idx, None, c)
    }

    // Not in the brief's test list, but added to de-risk configs/reward_v0.toml itself
    // (the two brief tests build RewardConfig by hand and never exercise `load`).
    #[test]
    fn loads_v0() {
        let c = RewardConfig::load("../configs/reward_v0.toml").unwrap();
        assert_eq!(c.goal, 10.0);
        assert_eq!(c.touch, 0.5);
        assert_eq!(c.vel_to_ball, 0.05);
    }

    // Not in the brief's test list. Guards the touch-detection contract documented
    // above `compute`: a car that has never touched the ball has `is_valid == false`
    // and `tick_count_when_hit == u64::MAX` (a sentinel, NOT 0 — see module doc comment
    // for how this was verified), so it must never register a false touch, even on the
    // very first step of a fresh arena.
    #[test]
    fn no_false_touch_before_any_contact() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let prev = arena.pin_mut().get_game_state();
        assert!(!prev.cars[0].state.ball_hit_info.is_valid);
        assert_eq!(prev.cars[0].state.ball_hit_info.tick_count_when_hit, u64::MAX);
        arena.pin_mut().step(1);
        let cur = arena.pin_mut().get_game_state();
        let r = compute(&prev, &cur, 0, None, &cfg());
        // No touch yet, and negligible motion in one tick, so reward should be ~0
        // and in particular must not include the touch bonus.
        assert!(r < cfg().touch, "spurious touch bonus on first step, got {r}");
    }

    // Not in the brief's test list. Regression test for the sentinel bug described in
    // the module doc comment: places the car directly on the ball (deterministic
    // overlap, no reliance on kickoff-drive physics) and confirms a car's *first-ever*
    // touch pays the touch bonus. With the brief's original `hit_now > hit_before`
    // comparison this test fails, because MAX -> small-real-tick is a decrease, not an
    // increase.
    #[test]
    fn first_touch_ever_pays_touch_bonus() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let prev = arena.pin_mut().get_game_state();
        assert!(!prev.cars[0].state.ball_hit_info.is_valid, "precondition: never touched");

        let id = prev.cars[0].id;
        let mut cs = prev.cars[0].state;
        cs.pos = prev.ball.pos; // full overlap: deterministic contact regardless of hitbox geometry
        cs.vel = rocketsim_rs::math::Vec3::new(500.0, 0.0, 0.0);
        arena.pin_mut().set_car(id, cs).unwrap();
        arena.pin_mut().step(2);

        let cur = arena.pin_mut().get_game_state();
        assert!(cur.cars[0].state.ball_hit_info.is_valid, "expected a touch to register");

        let r = compute(&prev, &cur, 0, None, &cfg());
        assert!(r >= cfg().touch, "expected touch bonus in reward, got {r}");
    }
}
