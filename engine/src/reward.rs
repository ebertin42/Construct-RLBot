use rocketsim_rs::{sim::Team, GameState};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct RewardConfig {
    pub goal: f32,
    pub touch: f32,
    pub vel_to_ball: f32,
    #[serde(default)]
    pub aggression_bias: f32, // concede = -goal*(1-bias)
    #[serde(default)]
    pub touch_accel: f32, // impact-scaled touch
    #[serde(default)]
    pub vel_ball_to_goal: f32, // ball velocity toward opp net
    #[serde(default)]
    pub offensive_potential: f32, // KRC(vel_to_ball, ball-goal alignment)
    /// Team-spirit blending (spec §4): r_i' = (1-t)*r_i + t*mean(team) - opp_spirit*mean(opponents).
    /// Applied in EpisodeArena::step, not here (needs all agents' raw rewards).
    #[serde(default)]
    pub team_spirit: f32,
    #[serde(default)]
    pub opp_spirit: f32,
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

    // goal / concede with aggression bias (bias 0.0 == old symmetric behavior)
    if let Some(team) = scored {
        r += if team == me.team {
            cfg.goal
        } else {
            -cfg.goal * (1.0 - cfg.aggression_bias)
        };
    }

    // touch: ball_hit_info recorded a new, valid hit during this step's tick window
    let hit_now = &me.state.ball_hit_info;
    let hit_before = &prev.cars[car_idx].state.ball_hit_info;
    let touched = hit_now.is_valid
        && hit_now.tick_count_when_hit != hit_before.tick_count_when_hit
        && hit_now.tick_count_when_hit > prev.tick_count
        && hit_now.tick_count_when_hit <= cur.tick_count;

    // NOTE: `touch` (flat) and `touch_accel` (impact-scaled) both fire on the same
    // contact if both weights are nonzero — they stack. v1 configs set touch = 0.0;
    // combining them is legal but double-pays contact, so do it deliberately.
    if touched {
        r += cfg.touch;
    }

    // vel_to_ball: projection of car velocity onto unit vector toward ball
    let (bp, mp, mv) = (cur.ball.pos, me.state.pos, me.state.vel);
    let d = [bp.x - mp.x, bp.y - mp.y, bp.z - mp.z];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let proj = (mv.x * d[0] + mv.y * d[1] + mv.z * d[2]) / dist;
    r += cfg.vel_to_ball * (proj / 2300.0).clamp(0.0, 1.0);

    // --- v1 components (all zero-cost when weights are 0.0) ---
    let opp_goal_y: f32 = if me.team == Team::Blue { 5120.0 } else { -5120.0 };

    if cfg.touch_accel != 0.0 && touched {
        // impact = ball velocity change during the step, normalized by 2300 uu/s
        let dv = [
            cur.ball.vel.x - prev.ball.vel.x,
            cur.ball.vel.y - prev.ball.vel.y,
            cur.ball.vel.z - prev.ball.vel.z,
        ];
        let impact = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt() / 2300.0;
        r += cfg.touch_accel * impact.clamp(0.0, 1.0);
    }

    if cfg.vel_ball_to_goal != 0.0 {
        // Ball velocity toward the opponent's net, per each agent's own goal vector.
        // NOT antisymmetric between teams in general (each team normalizes by its own
        // goal distance; exact antisymmetry holds only for a ball on the y-axis) —
        // this is intentional: shaping components are per-agent, not zero-sum (spec §4
        // reserves zero-sum wrapping for contested quantities like goals/boost/demos).
        let g = [0.0 - cur.ball.pos.x, opp_goal_y - cur.ball.pos.y, 0.0];
        let gn = (g[0] * g[0] + g[1] * g[1]).sqrt().max(1e-6);
        let toward = (cur.ball.vel.x * g[0] + cur.ball.vel.y * g[1]) / gn;
        r += cfg.vel_ball_to_goal * (toward / 6000.0).clamp(-1.0, 1.0);
    }

    if cfg.offensive_potential != 0.0 {
        // KRC-2 (Lucy-SKG style geometric mean): sqrt(vel-to-ball+ * ball-goal-alignment+)
        let vtb = (proj / 2300.0).clamp(0.0, 1.0); // reuse the projection computed above
        let bg = [0.0 - cur.ball.pos.x, opp_goal_y - cur.ball.pos.y];
        let bgn = (bg[0] * bg[0] + bg[1] * bg[1]).sqrt().max(1e-6);
        let cb = [cur.ball.pos.x - me.state.pos.x, cur.ball.pos.y - me.state.pos.y];
        let cbn = (cb[0] * cb[0] + cb[1] * cb[1]).sqrt().max(1e-6);
        let align = ((cb[0] * bg[0] + cb[1] * bg[1]) / (cbn * bgn)).clamp(0.0, 1.0);
        r += cfg.offensive_potential * (vtb * align).sqrt();
    }

    r
}

/// Win probability in [0,1] from the perspective of a team leading by
/// `score_diff`, with `t_frac` of the match clock remaining (1.0 at kickoff,
/// 0.0 at the final whistle).
///
/// The slope sharpens as the clock runs down: a one-goal lead is nearly
/// meaningless at kickoff and nearly decisive with seconds left. `t_floor`
/// caps that sharpening so `k` stays finite at t_frac = 0.
///
/// This is a POTENTIAL, not a reward. Its accuracy affects only how fast the
/// policy learns, never what it converges to (Ng, Harada & Russell 1999), so a
/// crude analytic form is a legitimate starting point.
pub fn win_prob(score_diff: i32, t_frac: f32, k_base: f32, t_floor: f32) -> f32 {
    let k = k_base / t_frac.max(t_floor);
    1.0 / (1.0 + (-k * score_diff as f32).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim_init::ensure_init;
    use rocketsim_rs::sim::{Arena, CarConfig, CarControls, Team};

    fn cfg() -> RewardConfig {
        RewardConfig {
            goal: 10.0,
            touch: 0.5,
            vel_to_ball: 0.05,
            aggression_bias: 0.0,
            touch_accel: 0.0,
            vel_ball_to_goal: 0.0,
            offensive_potential: 0.0,
            team_spirit: 0.0,
            opp_spirit: 0.0,
        }
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

    fn v1_cfg() -> RewardConfig {
        RewardConfig {
            goal: 20.0,
            touch: 0.0,
            vel_to_ball: 0.02,
            aggression_bias: 0.2,
            touch_accel: 2.0,
            vel_ball_to_goal: 0.5,
            offensive_potential: 0.3,
            team_spirit: 0.0,
            opp_spirit: 0.0,
        }
    }

    fn v0_cfg() -> RewardConfig {
        // exactly the fields reward_v0.toml sets; new fields must default to 0.0
        let parsed: RewardConfig =
            toml::from_str("goal = 10.0\ntouch = 0.5\nvel_to_ball = 0.05").unwrap();
        parsed
    }

    #[test]
    fn v0_toml_parses_with_zero_defaults() {
        let c = v0_cfg();
        assert_eq!(c.aggression_bias, 0.0);
        assert_eq!(c.touch_accel, 0.0);
        assert_eq!(c.vel_ball_to_goal, 0.0);
        assert_eq!(c.offensive_potential, 0.0);
    }

    #[test]
    fn v0_behavior_unchanged_by_new_fields() {
        // regression gate: with all new weights zero, compute() must equal the
        // v0 formula exactly on a stepped arena state (goal + touch + vel_to_ball only)
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(11));
        let prev = arena.pin_mut().get_game_state();
        for _ in 0..30 {
            let ids: Vec<u32> = prev.cars.iter().map(|c| c.id).collect();
            for id in &ids {
                arena.pin_mut()
                    .set_car_controls(*id, CarControls { throttle: 1.0, boost: true, ..Default::default() })
                    .unwrap();
            }
            arena.pin_mut().step(8);
        }
        let cur = arena.pin_mut().get_game_state();
        let c = v0_cfg();
        for idx in 0..2 {
            let r = compute(&prev, &cur, idx, None, &c);
            let expected = v0_reference(&prev, &cur, idx, &c);
            assert_eq!(r, expected, "car {idx}: v0 behavior drifted");
        }
    }

    // Reference copy of the v0 formula, frozen for the regression test. The touch-window
    // condition below is copied verbatim from the CURRENT compute() (not the brief's
    // illustrative draft) so this test actually pins today's behavior.
    fn v0_reference(prev: &GameState, cur: &GameState, car_idx: usize, cfg: &RewardConfig) -> f32 {
        let me = &cur.cars[car_idx];
        let mut r = 0.0f32;

        // (goal/scored is not part of this reference: the caller always passes
        // scored = None for this test, so the goal block never fires either way.)

        let hit_now = &me.state.ball_hit_info;
        let hit_before = &prev.cars[car_idx].state.ball_hit_info;
        let touched = hit_now.is_valid
            && hit_now.tick_count_when_hit != hit_before.tick_count_when_hit
            && hit_now.tick_count_when_hit > prev.tick_count
            && hit_now.tick_count_when_hit <= cur.tick_count;
        if touched {
            r += cfg.touch;
        }

        let (bp, mp, mv) = (cur.ball.pos, me.state.pos, me.state.vel);
        let d = [bp.x - mp.x, bp.y - mp.y, bp.z - mp.z];
        let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
        let proj = (mv.x * d[0] + mv.y * d[1] + mv.z * d[2]) / dist;
        r += cfg.vel_to_ball * (proj / 2300.0).clamp(0.0, 1.0);

        r
    }

    #[test]
    fn aggression_bias_softens_concede_only() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let gs = arena.pin_mut().get_game_state();
        let cfg = v1_cfg();
        let blue_idx = gs.cars.iter().position(|c| c.team == Team::Blue).unwrap();
        let orange_idx = gs.cars.iter().position(|c| c.team == Team::Orange).unwrap();
        let shaping_b = compute(&gs, &gs, blue_idx, None, &cfg);
        let shaping_o = compute(&gs, &gs, orange_idx, None, &cfg);
        let rb = compute(&gs, &gs, blue_idx, Some(Team::Blue), &cfg) - shaping_b;
        let ro = compute(&gs, &gs, orange_idx, Some(Team::Blue), &cfg) - shaping_o;
        assert_eq!(rb, 20.0);
        assert!((ro - (-16.0)).abs() < 1e-5, "concede should be -goal*(1-0.2), got {ro}");
    }

    #[test]
    fn vel_ball_to_goal_signed_by_team() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(2));
        // ball at center flying straight at the ORANGE net (+y): good for Blue
        let mut ball = arena.pin_mut().get_ball();
        ball.pos = rocketsim_rs::math::Vec3::new(0.0, 0.0, 500.0);
        ball.vel = rocketsim_rs::math::Vec3::new(0.0, 3000.0, 0.0);
        arena.pin_mut().set_ball(ball);
        let gs = arena.pin_mut().get_game_state();
        let mut cfg = v1_cfg();
        // isolate the component
        cfg.vel_to_ball = 0.0;
        cfg.offensive_potential = 0.0;
        cfg.touch_accel = 0.0;
        let blue_idx = gs.cars.iter().position(|c| c.team == Team::Blue).unwrap();
        let orange_idx = gs.cars.iter().position(|c| c.team == Team::Orange).unwrap();
        let rb = compute(&gs, &gs, blue_idx, None, &cfg);
        let ro = compute(&gs, &gs, orange_idx, None, &cfg);
        assert!(rb > 0.2, "ball flying at orange net pays blue, got {rb}");
        assert!(ro < -0.2, "and costs orange, got {ro}");
        // antisymmetry holds only in this x=0 configuration (see component comment)
        assert!((rb + ro).abs() < 1e-5, "antisymmetric at x=0 by construction");
    }

    #[test]
    fn touch_accel_pays_for_impact_not_contact() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(3));
        // drive car into a resting ball at speed
        let gs0 = arena.pin_mut().get_game_state();
        let car_id = gs0.cars[0].id;
        let mut cs = arena.pin_mut().get_car(car_id);
        let ball = arena.pin_mut().get_ball();
        cs.pos = rocketsim_rs::math::Vec3::new(ball.pos.x - 200.0, ball.pos.y, 17.0);
        cs.vel = rocketsim_rs::math::Vec3::new(1800.0, 0.0, 0.0);
        arena.pin_mut().set_car(car_id, cs).unwrap();
        let prev = arena.pin_mut().get_game_state();
        for _ in 0..8 {
            arena.pin_mut()
                .set_car_controls(car_id, CarControls { throttle: 1.0, ..Default::default() })
                .unwrap();
            arena.pin_mut().step(8);
        }
        let cur = arena.pin_mut().get_game_state();
        let mut cfg = v1_cfg();
        cfg.vel_to_ball = 0.0;
        cfg.vel_ball_to_goal = 0.0;
        cfg.offensive_potential = 0.0;
        let idx = cur.cars.iter().position(|c| c.id == car_id).unwrap();
        let r = compute(&prev, &cur, idx, None, &cfg);
        assert!(r > 0.3, "hard hit should pay meaningfully, got {r}");
        assert!(r <= cfg.touch_accel, "bounded by weight, got {r}");
    }

    #[test]
    fn offensive_potential_in_unit_range() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(4));
        let gs = arena.pin_mut().get_game_state();
        let mut cfg = v1_cfg();
        cfg.vel_to_ball = 0.0;
        cfg.vel_ball_to_goal = 0.0;
        cfg.touch_accel = 0.0;
        for idx in 0..2 {
            let r = compute(&gs, &gs, idx, None, &cfg);
            assert!(r >= 0.0 && r <= cfg.offensive_potential, "car {idx}: {r}");
        }
    }

    #[test]
    fn v1_toml_and_v0_toml_parse_with_zero_spirit_defaults() {
        let v0 = RewardConfig::load("../configs/reward_v0.toml").unwrap();
        assert_eq!(v0.team_spirit, 0.0);
        assert_eq!(v0.opp_spirit, 0.0);
    }

    // --- reward v4: symmetric zero-sum regime ---
    // (levers-roadmap-2026-07-19.md lever #3; configs/reward_v4.toml has the full
    // WHY. v3's +10/-8 asymmetry bred a positive-sum goal-trading exploit; v3.1's
    // +10/-12 fix bred an avoidance equilibrium instead. v4's fix is symmetric
    // goal/concede (aggression_bias = 0.0) PLUS full zero-sum opponent-subtraction
    // (opp_spirit = 1.0) via episode::blend_team_spirit, applied to the whole
    // per-step reward (goal + shaping), not just the goal term.)
    use crate::episode::blend_team_spirit;

    #[test]
    fn v4_toml_matches_documented_design() {
        // Guards against silent typos in configs/reward_v4.toml: every field must
        // match the design documented in that file's header comments.
        let c = RewardConfig::load("../configs/reward_v4.toml").unwrap();
        assert_eq!(c.goal, 10.0);
        assert_eq!(c.touch, 0.0);
        assert_eq!(c.vel_to_ball, 0.05);
        assert_eq!(c.aggression_bias, 0.0, "symmetric goal/concede is the whole point");
        assert_eq!(c.touch_accel, 0.0);
        assert_eq!(c.vel_ball_to_goal, 0.0);
        assert_eq!(c.offensive_potential, 0.0);
        assert_eq!(c.team_spirit, 0.3);
        assert_eq!(c.opp_spirit, 1.0, "full zero-sum opponent subtraction is the fix");
    }

    // Test 1 (brief): with the real v4 config loaded, a 1v1 step where blue scores
    // must produce a post-blend reward vector that sums to ~0, and specifically
    // +20 / -20 (goal scale doubles under opp_spirit=1.0 — see config header).
    // Composes reward::compute (raw, per-agent) with episode::blend_team_spirit
    // (team/opponent blending), exactly as EpisodeArena::step does internally.
    #[test]
    fn v4_zero_sum_on_goal_event() {
        ensure_init(None);
        let cfg = RewardConfig::load("../configs/reward_v4.toml").unwrap();
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let gs = arena.pin_mut().get_game_state();
        // Kickoff spawn velocity is 0, so vel_to_ball's shaping term is exactly 0
        // here (proj = dot(vel, dir)/dist = 0) — the raw per-agent reward below is
        // driven purely by the goal term, matching the config header's ±10 claim
        // exactly rather than approximately.
        let blue_idx = gs.cars.iter().position(|c| c.team == Team::Blue).unwrap();
        let orange_idx = gs.cars.iter().position(|c| c.team == Team::Orange).unwrap();

        let raw_blue = compute(&gs, &gs, blue_idx, Some(Team::Blue), &cfg);
        let raw_orange = compute(&gs, &gs, orange_idx, Some(Team::Blue), &cfg);
        assert_eq!(raw_blue, 10.0, "symmetric score, no shaping at kickoff velocity");
        assert_eq!(raw_orange, -10.0, "symmetric concede, no shaping at kickoff velocity");

        // blue_count = 1 (agent order: blue then orange, matching EpisodeArena's
        // car_ids convention documented on that struct).
        let mut rewards = [raw_blue, raw_orange];
        blend_team_spirit(&mut rewards, 1, cfg.team_spirit, cfg.opp_spirit);

        assert!((rewards[0] - 20.0).abs() < 1e-4, "scorer should net +20, got {}", rewards[0]);
        assert!((rewards[1] - (-20.0)).abs() < 1e-4, "conceder should net -20, got {}", rewards[1]);
        assert!(
            (rewards[0] + rewards[1]).abs() < 1e-4,
            "post-blend reward vector must sum to ~0 (zero-sum), got {} + {} = {}",
            rewards[0],
            rewards[1],
            rewards[0] + rewards[1]
        );
    }

    // Test 2 (brief): the regression test that pins WHY v4 exists. Simulates the
    // goal-trading exploit arithmetically for two alternating goal events (blue
    // scores, then orange scores) under both the real v3 config and the real v4
    // config.
    //
    // v3 side deliberately uses v3's RAW (unblended) goal/concede reward — i.e.
    // opp_spirit treated as 0 here, NOT v3.toml's actual opp_spirit=0.3 — because
    // the historical trading-exploit diagnosis (journal 2026-07-18 09:40, commit
    // aa560ab; "partners alternate goals, each netting +2 per exchange") was the
    // pure asymmetric goal/concede arithmetic, independent of team-spirit
    // blending (a separate mechanism: per-teammate reward pooling, orthogonal to
    // the goal/concede skew that caused the exploit). This isolates the exact
    // mechanism v4 fixes.
    //
    // v4 side uses the REAL config end-to-end, including its opp_spirit=1.0
    // blend, because the blend IS the fix being tested.
    #[test]
    fn trading_is_unprofitable_under_v4_but_not_v3() {
        let v3 = RewardConfig::load("../configs/reward_v3.toml").unwrap();
        let v4 = RewardConfig::load("../configs/reward_v4.toml").unwrap();

        // Sanity-pin the raw numbers the brief's design is stated in terms of.
        assert_eq!(v3.goal, 10.0);
        assert_eq!(v3.aggression_bias, 0.2);
        let v3_concede = -v3.goal * (1.0 - v3.aggression_bias);
        assert!((v3_concede - (-8.0)).abs() < 1e-5, "v3 concede should be -8.0, got {v3_concede}");

        // --- v3: raw (unblended) arithmetic, opp_spirit treated as 0 ---
        // Event A: blue scores (+goal for blue, concede for orange).
        // Event B: orange scores (+goal for orange, concede for blue).
        let v3_blue_net = v3.goal + v3_concede; // score then concede
        let v3_orange_net = v3_concede + v3.goal; // concede then score
        assert!((v3_blue_net - 2.0).abs() < 1e-5, "v3 blue should net +2/exchange, got {v3_blue_net}");
        assert!(
            (v3_orange_net - 2.0).abs() < 1e-5,
            "v3 orange should net +2/exchange, got {v3_orange_net}"
        );
        // Positive-sum: both sides profit from trading regardless of cooperation.
        assert!(v3_blue_net > 0.0 && v3_orange_net > 0.0);

        // --- v4: real config end-to-end, including its zero-sum blend ---
        assert_eq!(v4.goal, 10.0);
        assert_eq!(v4.aggression_bias, 0.0);
        let v4_concede = -v4.goal * (1.0 - v4.aggression_bias);
        assert!((v4_concede - (-10.0)).abs() < 1e-5);

        // Event A: blue scores.
        let mut event_a = [v4.goal, v4_concede]; // [blue, orange]
        blend_team_spirit(&mut event_a, 1, v4.team_spirit, v4.opp_spirit);
        // Event B: orange scores.
        let mut event_b = [v4_concede, v4.goal]; // [blue, orange]
        blend_team_spirit(&mut event_b, 1, v4.team_spirit, v4.opp_spirit);

        let v4_blue_net = event_a[0] + event_b[0];
        let v4_orange_net = event_a[1] + event_b[1];
        assert!(
            v4_blue_net.abs() < 1e-4,
            "v4 blue should net exactly 0.0/exchange, got {v4_blue_net}"
        );
        assert!(
            v4_orange_net.abs() < 1e-4,
            "v4 orange should net exactly 0.0/exchange, got {v4_orange_net}"
        );
    }

    // --- win_prob: pure potential for match-win shaping (Task 1) ---
    const K: f32 = 0.6;
    const TF: f32 = 0.05;

    #[test]
    fn win_prob_is_half_at_level_score_for_every_clock() {
        // A tied game is a coin flip whatever the clock says. If this drifts,
        // kickoff carries a bias and the shaped game is no longer zero-sum.
        for t in [1.0, 0.75, 0.5, 0.25, 0.0] {
            assert!((super::win_prob(0, t, K, TF) - 0.5).abs() < 1e-6, "t_frac={t}");
        }
    }

    #[test]
    fn win_prob_is_complementary_between_teams() {
        for d in [-3, -1, 0, 1, 3] {
            for t in [1.0, 0.5, 0.1] {
                let a = super::win_prob(d, t, K, TF);
                let b = super::win_prob(-d, t, K, TF);
                assert!((a + b - 1.0).abs() < 1e-6, "d={d} t={t}: {a} + {b}");
            }
        }
    }

    #[test]
    fn a_lead_is_worth_more_as_the_clock_runs_down() {
        let early = super::win_prob(1, 0.9, K, TF);
        let late = super::win_prob(1, 0.1, K, TF);
        assert!(late > early, "late {late} should exceed early {early}");
    }

    #[test]
    fn win_prob_stays_bounded_at_zero_time() {
        // t_floor is what stops k blowing up in the final tick.
        let v = super::win_prob(5, 0.0, K, TF);
        assert!(v.is_finite() && (0.0..=1.0).contains(&v), "got {v}");
    }
}
