use rocketsim_rs::{sim::Team, GameState};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
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
    /// Potential-based match-win shaping (task #56 Phase 1). Zero in every
    /// pre-v5 config, which is what keeps those configs bit-identical.
    #[serde(default)]
    pub win_prob_weight: f32,
    /// Logistic slope at full clock; see `win_prob`.
    #[serde(default)]
    pub win_prob_k_base: f32,
    /// Lower bound on `t_frac` in the slope denominator, so `k` stays finite.
    #[serde(default)]
    pub win_prob_t_floor: f32,
    /// MUST equal the trainer's `[ppo] gamma`. Any other value breaks the
    /// potential-based guarantee that the optimum is unchanged. Checked at
    /// trainer startup — see `python/construct/learn/train.py`.
    #[serde(default)]
    pub win_prob_gamma: f32,

    // --- v9 AIR PLAY (2026-07-26). All `serde(default)` -> 0.0/0, which is
    // what keeps reward_v0.toml and every historical config bit-identical
    // (`v0_behavior_unchanged_by_new_fields` is the regression gate). ---
    /// Minimum ticks between two PAID flat touches. 0 = historical (uncapped).
    ///
    /// MEASURED (engine/examples/aerial_reward_probe.rs, 2026-07-26): RocketSim
    /// writes `ballHitInfo.tickCountWhenHit` inside Bullet's per-contact-point
    /// callback with NO cooldown (RocketSim/src/Sim/Ball/Ball.cpp:256) -- while
    /// the extra-impulse path six lines below (:262) has an explicit 1-tick
    /// cooldown. Under held contact the `touched` predicate below therefore
    /// fires 12.0-14.4 times per SECOND, so `touch = 0.5` pays 6.00-7.19/s =
    /// 1800-2160 per 300s match, ~200 goals' worth, to a bot that just holds the
    /// ball. That hazard is live in reward_v8_team.toml today; v9 is the run
    /// that would find it, because carries and air dribbles HOLD contact and we
    /// are about to pay the bot to go airborne.
    ///
    /// The guard is a gap test between SUCCESSIVE RECORDED HITS, which bounds
    /// paid touches at one per `touch_cooldown_ticks` exactly: a payment at
    /// t_j needs t_j - t_{j-1} >= c, and t_{j-1} >= t_i (the previous payment),
    /// so t_j - t_i >= c. At c = 30 that is 4 payments/s = 2.0 reward/s.
    /// Applies to `touch` ONLY -- `touch_accel` is already impulse-bounded and
    /// gating it would break its proportionality to delivered impulse.
    #[serde(default)]
    pub touch_cooldown_ticks: u32,
    /// THE headline v9 term: paid only for a touch made with NO surface contact,
    /// on a ball too high for a double jump, and only for the impulse delivered
    /// TOWARD the opponent's net. See `configs/reward_v9_aerial.toml` for the
    /// measurement behind each of the four gates.
    #[serde(default)]
    pub aerial_touch: f32,
    /// Height ramp for `aerial_touch` / `air_setup`: h = 0 at or below `lo`,
    /// 1 at or above `hi`. A RAMP, not a step, so the exact threshold is not
    /// load-bearing. MEASURED apexes from a standing start with 100 boost:
    /// hop 231.9, double jump 393.6, hop+boost 612.6, full aerial 1238.1; a car
    /// contacts a ball ~130uu above its own centre, so a hop reaches ball
    /// z ~= 360 (h = 0) and a double jump z ~= 525 (h = 0.125 at 400/1400).
    #[serde(default)]
    pub aerial_z_lo: f32,
    #[serde(default)]
    pub aerial_z_hi: f32,
    /// Potential-based aerial-SETUP shaping, staged OFF at v9 launch (see
    /// `air_potential` / `air_shaping`). Shares `win_prob_gamma` as its gamma,
    /// which is why `validate` demands that gamma once this is nonzero.
    #[serde(default)]
    pub air_setup: f32,
}

/// Per-term reward telemetry (E9). Index space for the `[f64; N_TERMS]`
/// counter array `EpisodeArena` accumulates and `Engine.reward_terms()`
/// exposes. Existing in this form because "is the new term actually firing,
/// and is it being farmed?" must be answerable from the log without a probe
/// run -- the 2026-07-20 confound was an arm that ran fully INERT while every
/// log line looked normal.
///
/// The last four entries are COUNTS, not reward: they are what the §8.3
/// farming tripwires (touches/min/car, airborne-touch fraction, reward per
/// goal) are computed from.
pub const N_TERMS: usize = 13;
pub const T_GOAL: usize = 0;
pub const T_TOUCH: usize = 1;
pub const T_VEL_TO_BALL: usize = 2;
pub const T_TOUCH_ACCEL: usize = 3;
pub const T_VEL_BALL_TO_GOAL: usize = 4;
pub const T_OFFENSIVE_POTENTIAL: usize = 5;
pub const T_AERIAL_TOUCH: usize = 6;
pub const T_AIR_SETUP: usize = 7;
pub const T_WIN_PROB: usize = 8;
pub const T_TOUCH_EVENTS: usize = 9;
pub const T_AIRBORNE_TOUCH_EVENTS: usize = 10;
pub const T_GOAL_EVENTS: usize = 11;
pub const T_AGENT_STEPS: usize = 12;
pub const TERM_NAMES: [&str; N_TERMS] = [
    "goal",
    "touch",
    "vel_to_ball",
    "touch_accel",
    "vel_ball_to_goal",
    "offensive_potential",
    "aerial_touch",
    "air_setup",
    "win_prob",
    "touch_events",
    "airborne_touch_events",
    "goal_events",
    "agent_steps",
];

/// Ball radius. The `aerial_z_lo` floor in `validate`: a ramp whose bottom is
/// below the resting ball centre would pay for a ball rolling on the floor.
const BALL_RADIUS: f32 = 93.15;

/// Height ramp shared by `aerial_touch` and `air_potential`.
///
/// `validate` guarantees `hi > lo` whenever either weight is nonzero, so the
/// division is never 0/0 here. That guarantee is load-bearing, not decorative:
/// with `hi == lo == 0` (the "someone set aerial_touch and nothing else"
/// config) this is inf/NaN, which poisons the run while looking normal --
/// exactly the failure the `win_prob` validator was written for.
#[inline]
fn height_ramp(ball_z: f32, cfg: &RewardConfig) -> f32 {
    ((ball_z - cfg.aerial_z_lo) / (cfg.aerial_z_hi - cfg.aerial_z_lo)).clamp(0.0, 1.0)
}

impl RewardConfig {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let cfg: RewardConfig = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        cfg.validate().map_err(|e| format!("{path}: {e}"))?;
        Ok(cfg)
    }

    /// Reject half-configured shaping. With `win_prob_weight` set but
    /// `k_base`/`t_floor` left at their zero defaults, PHI collapses to a
    /// constant 0.5, the shaping term is identically zero, and the run trains
    /// on nothing while looking entirely normal.
    pub fn validate(&self) -> Result<(), String> {
        self.validate_aerial()?;
        if self.win_prob_weight == 0.0 {
            return Ok(());
        }
        if self.win_prob_k_base <= 0.0 {
            return Err(format!(
                "win_prob_weight={} but win_prob_k_base={}: PHI would be a \
                 constant 0.5 and the shaping term identically zero",
                self.win_prob_weight, self.win_prob_k_base));
        }
        if !(0.0..1.0).contains(&self.win_prob_t_floor) || self.win_prob_t_floor <= 0.0 {
            return Err(format!(
                "win_prob_t_floor={} must be in (0,1)", self.win_prob_t_floor));
        }
        Ok(())
    }

    /// Reject a half-configured AIR block (E8). Same failure mode as the
    /// win_prob validator above, one step worse: `(z - lo) / (hi - lo)` with
    /// `hi == lo == 0` is inf or NaN, and a NaN reward does not look like a
    /// misconfiguration -- it looks like a run that trains and then quietly
    /// produces a dead policy. Both air terms share `height_ramp`, so BOTH
    /// demand the bounds.
    // `!(a > b)` rather than `a <= b` is DELIBERATE and load-bearing: it also
    // rejects NaN (every comparison with NaN is false, so `!(NaN > x)` is true).
    // A NaN weight from a hand-edited toml would otherwise sail through and turn
    // every reward in the run into NaN, which is precisely the class of silent
    // failure this validator exists for. Do not "simplify" these.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn validate_aerial(&self) -> Result<(), String> {
        if self.aerial_touch == 0.0 && self.air_setup == 0.0 {
            return Ok(());
        }
        if !(self.aerial_z_hi > self.aerial_z_lo) {
            return Err(format!(
                "aerial_touch={} / air_setup={} need aerial_z_hi > aerial_z_lo, got \
                 lo={} hi={}: the height ramp (z-lo)/(hi-lo) would be inf/NaN and \
                 would poison every reward in the run while looking normal",
                self.aerial_touch, self.air_setup, self.aerial_z_lo, self.aerial_z_hi));
        }
        if self.aerial_z_lo < BALL_RADIUS {
            return Err(format!(
                "aerial_z_lo={} is below the ball radius ({BALL_RADIUS}): a ramp that \
                 starts under a RESTING ball pays for ground play",
                self.aerial_z_lo));
        }
        // air_setup is potential-based and its gamma MUST be the trainer's
        // gamma (Ng/Harada/Russell). Sharing win_prob_gamma is deliberate --
        // there is one potential gamma per run -- but it means air_setup with
        // the default 0.0 gamma silently degenerates to `-PHI(prev)`, a pure
        // penalty for having been airborne. train.py's check_win_prob_gamma is
        // extended to cover this case; this is the engine-side half.
        if self.air_setup != 0.0 && !(self.win_prob_gamma > 0.0) {
            return Err(format!(
                "air_setup={} needs win_prob_gamma > 0 (the SHARED potential gamma, \
                 which must equal the trainer's [ppo] gamma); got {}. At gamma=0 the \
                 shaping degenerates to -PHI(prev), i.e. a pure penalty for takeoff",
                self.air_setup, self.win_prob_gamma));
        }
        Ok(())
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
    compute_terms(prev, cur, car_idx, scored, cfg, &mut [0.0; N_TERMS])
}

/// `compute` plus per-term telemetry (E9). `compute` is a thin wrapper over
/// this with a throwaway accumulator, so there is exactly ONE copy of the
/// reward arithmetic: the sum is accumulated into `r` in the same order and by
/// the same expressions whether or not anyone is looking at `terms`, which is
/// what makes the split safe for the gate's byte-identity (a second, parallel
/// implementation would drift the moment either was edited).
pub fn compute_terms(
    prev: &GameState,
    cur: &GameState,
    car_idx: usize,
    scored: Option<Team>,
    cfg: &RewardConfig,
    terms: &mut [f64; N_TERMS],
) -> f32 {
    let me = &cur.cars[car_idx];
    let mut r = 0.0f32;
    terms[T_AGENT_STEPS] += 1.0;

    // goal / concede with aggression bias (bias 0.0 == old symmetric behavior)
    if let Some(team) = scored {
        let g = if team == me.team {
            cfg.goal
        } else {
            -cfg.goal * (1.0 - cfg.aggression_bias)
        };
        r += g;
        terms[T_GOAL] += g as f64;
        terms[T_GOAL_EVENTS] += 1.0;
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
    //
    // COOLDOWN (v9, default 0 == historical): pay the flat touch only when this
    // hit is at least `touch_cooldown_ticks` after the PREVIOUS recorded hit.
    // See the field's doc comment for the measured 12-14 events/s carry farm this
    // closes and the proof that the gap test bounds paid touches at one per
    // cooldown. Three cases are deliberately paid unconditionally:
    //   * cooldown == 0            -> the historical config space, untouched;
    //   * !hit_before.is_valid     -> a car's FIRST-EVER touch (and the first
    //                                 after any reset, since reset_to_random_kickoff
    //                                 re-invalidates ball_hit_info) -- the sentinel
    //                                 is u64::MAX, so a naive subtraction would
    //                                 underflow and starve the touch that matters most;
    //   * hit ticks going BACKWARDS -> the arena's tick_count restarted (blowup
    //                                 containment rebuilds the arena, see
    //                                 rebuild_arena), so the gap is meaningless.
    //
    // Hoisted out of the `if touched` block below because `aerial_touch` takes
    // the SAME gate (2026-07-26 review): the original design exempted it on the
    // argument that its payout is bounded by delivered impulse, and that
    // argument does not survive a ball the WALL hands back to you -- see the
    // aerial block. One shared gap test means one rate bound for both terms.
    let cooldown_ok = cfg.touch_cooldown_ticks == 0
        || !hit_before.is_valid
        || hit_now.tick_count_when_hit < hit_before.tick_count_when_hit
        || hit_now.tick_count_when_hit - hit_before.tick_count_when_hit
            >= cfg.touch_cooldown_ticks as u64;
    if touched {
        terms[T_TOUCH_EVENTS] += 1.0;
        if !me.state.is_on_ground {
            terms[T_AIRBORNE_TOUCH_EVENTS] += 1.0;
        }
        if cooldown_ok {
            r += cfg.touch;
            terms[T_TOUCH] += cfg.touch as f64;
        }
    }

    // vel_to_ball: projection of car velocity onto unit vector toward ball
    let (bp, mp, mv) = (cur.ball.pos, me.state.pos, me.state.vel);
    let d = [bp.x - mp.x, bp.y - mp.y, bp.z - mp.z];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().max(1e-6);
    let proj = (mv.x * d[0] + mv.y * d[1] + mv.z * d[2]) / dist;
    let v2b = cfg.vel_to_ball * (proj / 2300.0).clamp(0.0, 1.0);
    r += v2b;
    terms[T_VEL_TO_BALL] += v2b as f64;

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
        let ta = cfg.touch_accel * impact.clamp(0.0, 1.0);
        r += ta;
        terms[T_TOUCH_ACCEL] += ta as f64;
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
        let vbg = cfg.vel_ball_to_goal * (toward / 6000.0).clamp(-1.0, 1.0);
        r += vbg;
        terms[T_VEL_BALL_TO_GOAL] += vbg as f64;
    }

    if cfg.offensive_potential != 0.0 {
        // KRC-2 (Lucy-SKG style geometric mean): sqrt(vel-to-ball+ * ball-goal-alignment+)
        let vtb = (proj / 2300.0).clamp(0.0, 1.0); // reuse the projection computed above
        let bg = [0.0 - cur.ball.pos.x, opp_goal_y - cur.ball.pos.y];
        let bgn = (bg[0] * bg[0] + bg[1] * bg[1]).sqrt().max(1e-6);
        let cb = [cur.ball.pos.x - me.state.pos.x, cur.ball.pos.y - me.state.pos.y];
        let cbn = (cb[0] * cb[0] + cb[1] * cb[1]).sqrt().max(1e-6);
        let align = ((cb[0] * bg[0] + cb[1] * bg[1]) / (cbn * bgn)).clamp(0.0, 1.0);
        let op = cfg.offensive_potential * (vtb * align).sqrt();
        r += op;
        terms[T_OFFENSIVE_POTENTIAL] += op as f64;
    }

    // --- v9: aerial_touch. THE headline term. -------------------------------
    //     h    = clamp((ball_z - aerial_z_lo)/(aerial_z_hi - aerial_z_lo), 0, 1)
    //     dv   = ball.vel(cur) - ball.vel(prev)
    //     fwd  = clamp(sign(opp_goal_y) * dv.y / 2300, -1, 1)      <- SYMMETRIC
    //     mine = clamp(cos(dv, ball_pos - car_pos), 0, 1)          <- attribution
    //     r   += aerial_touch * h * fwd * mine
    //            iff touched && !is_on_ground && cooldown_ok
    //
    // Each factor closes a MEASURED exploit (full numbers in
    // configs/reward_v9_aerial.toml; probe: engine/examples/aerial_reward_probe.rs):
    //
    //  * `!is_on_ground` kills WALL-RIDING. RocketSim defines isOnGround as
    //    `numWheelsInContact >= 3` (Car.cpp:118), SURFACE-AGNOSTIC: a car driving
    //    the side wall at z = 1065 reads is_on_ground = TRUE (measured). Without
    //    this gate a "height-scaled touch" is a wall-play farm with zero air
    //    skill. Do NOT swap in `world_contact.has_contact` -- it reads FALSE up
    //    there (body contact, not wheels), which would re-open the farm.
    //  * `h` kills HOP-TAPS. Measured apexes from a standing start: hop 231.9,
    //    double jump 393.6, hop+boost 612.6, full aerial 1238.1; a car contacts
    //    the ball ~130uu above its own centre, so a double jump reaches ball
    //    z ~= 525 -> h = 0.125 at 400/1400. A low aerial (800) pays 0.40.
    //  * `fwd` kills JUGGLING: a vertical juggle has dv.y ~ 0 and pays ~0.
    //
    // TWO CORRECTIONS FROM THE 2026-07-26 REVIEW. Both were farms, both measured:
    //
    //  (1) `fwd` used to be `dot(dv_xy, unit(goal - ball_xy))` CLAMPED TO [0,1],
    //      i.e. the ball-relative goal direction with undoing-your-own-work free.
    //      Two closed cycles paid on BOTH legs and returned the ball to its
    //      starting state -- 2.391 aerial_touch per round trip, 2.1 laps per
    //      goal's worth:
    //        * a y-axis juggle: the forward leg paid 1.198 and the return leg
    //          paid exactly 0.000 instead of -1.198. A pure ratchet.
    //        * a LATERAL cycle, which the [0,1] clamp alone does NOT fix: the
    //          goal vector is BALL-RELATIVE, so at ball (x=+3000, y=4500) it is
    //          98% lateral, and after the ball crosses to x=-3000 it has FLIPPED.
    //          Batting the ball back and forth across the corner paid 1.198 then
    //          1.193 -- both legs positive, ball never advances, and at z=1200 no
    //          goal is even possible (crossbar 642). Invisible to the
    //          touches/min tripwire: a 6000uu leg at 2400uu/s is 24 touches/min.
    //      FIX: project onto the FIELD-FORWARD axis (a constant per team) instead
    //      of the ball-relative goal vector, and clamp SYMMETRICALLY. A constant
    //      axis is what makes `fwd` a difference of a state function that the
    //      unpaid flight phase does not silently restore: the ball's y-velocity
    //      is unchanged by lateral travel, so a lateral cycle now pays 0 on BOTH
    //      legs, and any closed ball-velocity cycle the CAR creates sums to ~0
    //      because the return leg is charged exactly what the forward leg paid
    //      (pinned by `aerial_touch_telescopes_to_zero_over_a_closed_ball_
    //      velocity_cycle`, run at ball x != 0 so it covers the lateral case).
    //      The price is that a cross-field pass from the corner now pays ~0. That
    //      was ALREADY the documented intent ("a sideways setup pass pays zero");
    //      it just was not true.
    //      The residual, stated so nobody is surprised: a ratchet still exists
    //      whenever something OTHER than the car undoes the impulse for free (the
    //      back wall handing the ball straight back). That is what
    //      `touch_cooldown_ticks` now bounds -- the exemption this term used to
    //      claim rested on the impulse bound, and the bound only holds for a
    //      monotone one-way strike.
    //  (2) `dv` is the ball's TOTAL velocity change over the step, from every
    //      source, and the only per-car gate was `touched && !is_on_ground`. A
    //      motionless airborne car parked beside the ball collected 0.896 --
    //      byte-identical to the striker's 0.896 -- including when the impulse
    //      came from an OPPONENT. In a run whose whole point is equal 1s/2s/3s
    //      share, 2/3 of experience is in arenas where a teammate supplies the
    //      impulse: one strike would pay N grazing teammates N x 0.896, 11 free
    //      grazes to a goal, against a design budget of 4-10 for the entire term
    //      per match. It also teaches the double-commit habit `team_spirit`
    //      exists to suppress.
    //      FIX: `mine`, the cosine between the impulse and the direction from the
    //      paying car TO the ball. A real strike pushes the ball AWAY from the
    //      striker (cos ~ 1, unchanged); a side-graze or a passenger under the
    //      ball is perpendicular (cos ~ 0).
    //      3D, NOT the 2D form, and that choice has a price on both sides. In 2D
    //      a car hovering DIRECTLY UNDER the ball has a near-zero xy offset, the
    //      normalisation floors out, and it collects arbitrary credit -- the exact
    //      passenger this factor exists to stop. In 3D that case is a clean zero.
    //      The price: a car nosing the ball forward from underneath (the psyonix
    //      extra-impulse case, where dv is NOT along the contact normal) gets
    //      partial credit -- ~0.58 for dv (0,2000,500) at an offset of (0,50,130).
    //      Under-paying a real strike is the safe direction for a shaping term
    //      (L8); over-paying a hover is not.
    //      One-sided clamp: a car that genuinely drives the ball backwards is
    //      still charged, and a car that contributed nothing is neither paid nor
    //      charged. It does NOT separate two cars stacked on the same side of the
    //      ball in one 8-tick window -- that is a real 50-50, not a free graze.
    if cfg.aerial_touch != 0.0 && touched && !me.state.is_on_ground && cooldown_ok {
        let h = height_ramp(cur.ball.pos.z, cfg);
        let dv = [
            cur.ball.vel.x - prev.ball.vel.x,
            cur.ball.vel.y - prev.ball.vel.y,
            cur.ball.vel.z - prev.ball.vel.z,
        ];
        // Field-forward axis: +y for blue, -y for orange. Same team convention as
        // `opp_goal_y` above, deliberately WITHOUT its ball-relative direction.
        let fwd = (opp_goal_y.signum() * dv[1] / 2300.0).clamp(-1.0, 1.0);
        let dvn = (dv[0] * dv[0] + dv[1] * dv[1] + dv[2] * dv[2]).sqrt().max(1e-6);
        // `d`/`dist` are the car->ball vector and its length, already computed
        // for vel_to_ball above (and already floored away from 0/0).
        let mine =
            ((dv[0] * d[0] + dv[1] * d[1] + dv[2] * d[2]) / dvn / dist).clamp(0.0, 1.0);
        let at = cfg.aerial_touch * h * fwd * mine;
        r += at;
        terms[T_AERIAL_TOUCH] += at as f64;
    }

    r
}

/// PHI for the aerial-setup shaping (`air_setup`), in [0, 1]:
///
/// ```text
/// PHI = 0                                        if the car is on the ground
///     = h(ball_z) * clamp(1 - dist3d/2500, 0, 1)  otherwise
/// ```
///
/// Zero on the ground is what makes the term teach the TAKEOFF DECISION ("the
/// ball is going up, jump NOW") -- which `aerial_touch` structurally cannot,
/// because the curriculum's random resets already start ~50% of cars airborne
/// and therefore only ever teach "you are already up there, hit it".
///
/// Kept as a free function OUTSIDE `compute` (with `air_shaping`) so
/// `compute`'s signature -- and with it every gate's byte-identity -- survives:
/// this needs the (prev, cur) pair at the episode level, exactly like
/// `win_prob_shaping`.
pub fn air_potential(
    on_ground: bool,
    car_pos: [f32; 3],
    ball_pos: [f32; 3],
    cfg: &RewardConfig,
) -> f32 {
    if on_ground {
        return 0.0;
    }
    let d = [ball_pos[0] - car_pos[0], ball_pos[1] - car_pos[1], ball_pos[2] - car_pos[2]];
    let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    height_ramp(ball_pos[2], cfg) * (1.0 - dist / 2500.0).clamp(0.0, 1.0)
}

/// Potential-based shaping on `air_potential`: `w * (gamma * PHI(s') - PHI(s))`.
///
/// FARM-PROOF BY CONSTRUCTION (Ng/Harada/Russell 1999): the sum telescopes, so
/// ANY closed loop nets exactly zero. Fly in circles -> 0. Jump and land without
/// converting -> the takeoff payment is refunded exactly on landing. Loitering at
/// high PHI actually COSTS `w*(gamma-1)*PHI = -0.0046w` per step at the live
/// gamma. The only way to bank it is to convert, which is `aerial_touch`'s job.
///
/// TWO BOUNDED LEAKS, stated so nobody is surprised by them:
///   1. a goal/terminal firing while airborne never delivers the refund, worth up
///      to `+w` per goal (3% of a goal at w = 0.3). Mildly aligned with the intent.
///   2. random resets can start an episode at PHI > 0 (the curriculum spawns the
///      ball as high as z = 1700); GAE's bootstrap absorbs it.
///
/// The RESET BOUNDARY ITSELF IS SAFE and that is not an accident:
/// `episode.rs`'s `step_impl` sets `prev_state` to the fresh kickoff AFTER the
/// reward loop has run, so no (pre-reset, post-reset) state pair is ever fed to
/// this function. A later refactor that moves the reset above the reward loop
/// would break this silently -- it would look like a small, plausible cleanup and
/// would start paying a large spurious potential jump at every kickoff.
pub fn air_shaping(
    prev: &GameState,
    cur: &GameState,
    car_idx: usize,
    gamma: f32,
    cfg: &RewardConfig,
) -> f32 {
    if cfg.air_setup == 0.0 {
        return 0.0;
    }
    let phi = |gs: &GameState| {
        let c = &gs.cars[car_idx];
        air_potential(
            c.state.is_on_ground,
            [c.state.pos.x, c.state.pos.y, c.state.pos.z],
            [gs.ball.pos.x, gs.ball.pos.y, gs.ball.pos.z],
            cfg,
        )
    };
    cfg.air_setup * (gamma * phi(cur) - phi(prev))
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

/// The POTENTIAL used for shaping: `win_prob` recentred to [-0.5, +0.5].
///
/// The recentring is load-bearing. A raw sigmoid gives
/// `PHI_orange = 1 - PHI_blue`, not `-PHI_blue`, so the two teams' shaping sums
/// to `w*(gamma-1)` rather than zero -- a constant -0.046/step at the live
/// gamma, -207 across a match, which rewards ending episodes early. Recentring
/// makes the negation exact and the shaped game zero-sum at ANY gamma, and
/// costs nothing: a constant shift leaves potential-based shaping
/// potential-based, so the per-agent Ng guarantee is untouched.
pub fn win_potential(score_diff: i32, t_frac: f32, k_base: f32, t_floor: f32) -> f32 {
    win_prob(score_diff, t_frac, k_base, t_floor) - 0.5
}

/// Score and clock for one arena's current match. Phase 1 keeps this out of
/// the observation entirely (see the spec): it feeds the reward only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MatchState {
    pub score_blue: u32,
    pub score_orange: u32,
    /// Fraction of the match clock remaining: 1.0 at kickoff, 0.0 at the whistle.
    pub t_frac: f32,
}

impl MatchState {
    /// Goals ahead from `team`'s point of view. Signed, so the two teams'
    /// values are exact negations of each other -- necessary but not
    /// sufficient for the shaped game to be zero-sum. That additionally
    /// requires the potential to satisfy PHI_orange = -PHI_blue, which is
    /// what `win_potential`'s recentring provides (see its doc comment).
    pub fn score_diff(&self, team: Team) -> i32 {
        let (mine, theirs) = match team {
            Team::Blue => (self.score_blue, self.score_orange),
            Team::Orange => (self.score_orange, self.score_blue),
        };
        mine as i32 - theirs as i32
    }
}

/// Potential-based shaping: `w * (gamma * PHI(s') - PHI(s))`.
///
/// Summed over a match this telescopes to `w * (PHI(end) - PHI(start))` when
/// gamma == 1, and to a discounted equivalent otherwise — i.e. it delivers the
/// match outcome as dense per-step signal, each piece landing at the moment the
/// game state actually changes rather than 4500 steps later where the discount
/// would erase it.
pub fn win_prob_shaping(
    prev: &MatchState,
    cur: &MatchState,
    team: Team,
    gamma: f32,
    cfg: &RewardConfig,
) -> f32 {
    if cfg.win_prob_weight == 0.0 {
        return 0.0;
    }
    let phi = |m: &MatchState| {
        win_potential(m.score_diff(team), m.t_frac, cfg.win_prob_k_base, cfg.win_prob_t_floor)
    };
    cfg.win_prob_weight * (gamma * phi(cur) - phi(prev))
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
            ..Default::default()
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
            ..Default::default()
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
        // v9 air block: every new field must default to inert, or reward_v0 --
        // the neutral scoring tape every gate and bench number was measured on --
        // silently changes meaning (L9).
        assert_eq!(c.touch_cooldown_ticks, 0);
        assert_eq!(c.aerial_touch, 0.0);
        assert_eq!(c.aerial_z_lo, 0.0);
        assert_eq!(c.aerial_z_hi, 0.0);
        assert_eq!(c.air_setup, 0.0);
    }

    #[test]
    fn v0_behavior_unchanged_by_new_fields() {
        // THE L9 regression gate: with all new weights zero, compute() must equal
        // the v0 formula EXACTLY on a stepped arena state (goal + touch +
        // vel_to_ball only). Every gate number, every bench number and
        // GOAL_THRESHOLD 9.4 are properties of this function at these weights.
        // Extended 2026-07-26 with the v9 air block (touch_cooldown_ticks,
        // aerial_touch, aerial_z_lo/hi, air_setup) -- all `serde(default)` and
        // all absent from reward_v0.toml, so `v0_cfg()` below already carries
        // them at zero.
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

    #[test]
    fn win_potential_is_zero_at_level_score_for_every_clock() {
        for t in [1.0, 0.75, 0.5, 0.25, 0.0] {
            let v = super::win_potential(0, t, K, TF);
            assert!(v.abs() < 1e-6, "t_frac={t}: got {v}");
        }
    }

    #[test]
    fn win_potential_negates_exactly_between_teams() {
        // This is the property the recentring buys: PHI(-d) == -PHI(d), so
        // Team::Orange's potential is the exact negation of Team::Blue's and
        // the shaped game is zero-sum at any gamma.
        for d in [-3, -1, 0, 1, 3] {
            for t in [1.0, 0.5, 0.1, 0.0] {
                let a = super::win_potential(d, t, K, TF);
                let b = super::win_potential(-d, t, K, TF);
                assert!((a + b).abs() < 1e-6, "d={d} t={t}: {a} + {b} != 0");
            }
        }
    }

    // --- MatchState + win_prob_shaping: Task 2 (load-bearing) ---

    fn v5_cfg() -> super::RewardConfig {
        let mut c = super::RewardConfig {
            goal: 0.0, touch: 0.0, vel_to_ball: 0.0,
            ..Default::default()
        };
        c.win_prob_weight = 1.0;
        c.win_prob_k_base = 0.6;
        c.win_prob_t_floor = 0.05;
        c.win_prob_gamma = 1.0; // gamma=1 makes the telescoping exact and checkable
        c
    }

    fn ms(b: u32, o: u32, t: f32) -> super::MatchState {
        super::MatchState { score_blue: b, score_orange: o, t_frac: t }
    }

    #[test]
    fn shaping_telescopes_to_the_match_outcome() {
        // THE load-bearing test. Walk a whole match as a sequence of states and
        // sum the per-step shaped rewards; the total must equal the change in
        // potential from first state to last. If this fails, the shaping is not
        // potential-based and the optimum is no longer match-winning.
        let cfg = v5_cfg();
        let path = [
            ms(0, 0, 1.00), ms(0, 0, 0.80), ms(1, 0, 0.80), ms(1, 0, 0.50),
            ms(1, 1, 0.50), ms(1, 1, 0.20), ms(2, 1, 0.20), ms(2, 1, 0.00),
        ];
        let mut total = 0.0f32;
        for w in path.windows(2) {
            total += super::win_prob_shaping(&w[0], &w[1], Team::Blue, cfg.win_prob_gamma, &cfg);
        }
        let first = super::win_prob(path[0].score_diff(Team::Blue), path[0].t_frac,
                                   cfg.win_prob_k_base, cfg.win_prob_t_floor);
        let last = super::win_prob(path[path.len() - 1].score_diff(Team::Blue),
                                   path[path.len() - 1].t_frac,
                                   cfg.win_prob_k_base, cfg.win_prob_t_floor);
        assert!((total - (last - first)).abs() < 1e-5,
                "sum {total} != PHI(end)-PHI(start) {}", last - first);
    }

    #[test]
    fn shaping_is_zero_sum_between_teams() {
        // Must hold at every gamma, not just gamma=1.0 -- the raw sigmoid
        // (PHI_orange = 1 - PHI_blue, not -PHI_blue) only telescoped to zero
        // by coincidence at gamma=1.0, which is not the production value.
        // 0.9954 is the live gamma; the rest are sanity bracketing.
        for gamma in [0.9954, 1.0, 0.99, 0.9, 0.5] {
            let mut cfg = v5_cfg();
            cfg.win_prob_gamma = gamma;
            let (a, b) = (ms(0, 0, 0.5), ms(1, 0, 0.5));
            let blue = super::win_prob_shaping(&a, &b, Team::Blue, gamma, &cfg);
            let orange = super::win_prob_shaping(&a, &b, Team::Orange, gamma, &cfg);
            assert!((blue + orange).abs() < 1e-6,
                    "gamma={gamma}: blue {blue} + orange {orange} != 0");
        }
    }

    #[test]
    fn conceding_pays_negative_and_scoring_pays_positive() {
        let cfg = v5_cfg();
        let scored = super::win_prob_shaping(&ms(0, 0, 0.5), &ms(1, 0, 0.5),
                                             Team::Blue, cfg.win_prob_gamma, &cfg);
        let conceded = super::win_prob_shaping(&ms(0, 0, 0.5), &ms(0, 1, 0.5),
                                               Team::Blue, cfg.win_prob_gamma, &cfg);
        assert!(scored > 0.0, "scoring should pay positive, got {scored}");
        assert!(conceded < 0.0, "conceding should pay negative, got {conceded}");
    }

    #[test]
    fn a_late_goal_pays_more_than_an_early_one() {
        // The whole point of the clock term: the same 0-0 -> 1-0 transition is
        // worth more with 10% of the match left than with 90% left.
        let cfg = v5_cfg();
        let early = super::win_prob_shaping(&ms(0, 0, 0.9), &ms(1, 0, 0.9),
                                            Team::Blue, cfg.win_prob_gamma, &cfg);
        let late = super::win_prob_shaping(&ms(0, 0, 0.1), &ms(1, 0, 0.1),
                                           Team::Blue, cfg.win_prob_gamma, &cfg);
        assert!(late > early, "late {late} should exceed early {early}");
    }

    #[test]
    fn zero_weight_makes_shaping_inert() {
        // reward_v0/v3/v4_1 have no win_prob keys -> weight defaults to 0 -> the
        // term must contribute exactly nothing, so historical configs behave
        // bit-identically.
        let mut cfg = v5_cfg();
        cfg.win_prob_weight = 0.0;
        let r = super::win_prob_shaping(&ms(0, 0, 0.5), &ms(1, 0, 0.5),
                                        Team::Blue, cfg.win_prob_gamma, &cfg);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn historical_reward_configs_still_parse_with_defaulted_win_prob_keys() {
        for p in ["../configs/reward_v0.toml", "../configs/reward_v3.toml",
                  "../configs/reward_v4_1.toml"] {
            let c = super::RewardConfig::load(p).unwrap_or_else(|e| panic!("{p}: {e}"));
            assert_eq!(c.win_prob_weight, 0.0, "{p} must default win_prob_weight to 0");
        }
    }

    #[test]
    fn half_configured_shaping_is_rejected_not_silently_inert() {
        let mut c = v5_cfg();
        c.win_prob_k_base = 0.0;
        assert!(c.validate().is_err(), "must reject weight-without-slope");
    }

    #[test]
    fn zero_t_floor_is_rejected() {
        let mut c = v5_cfg();
        c.win_prob_t_floor = 0.0;
        assert!(c.validate().is_err());
    }

    #[test]
    fn shaping_off_needs_no_other_keys() {
        let c = super::RewardConfig { goal: 10.0, touch: 0.0, vel_to_ball: 0.0,
                                      ..Default::default() };
        assert!(c.validate().is_ok(), "historical configs must validate");
    }

    // ======================================================================
    // v9 AIR PLAY (2026-07-26). E7/E8/E9.
    // ======================================================================

    fn v9_cfg() -> RewardConfig {
        RewardConfig::load("../configs/reward_v9_aerial.toml").unwrap()
    }

    /// A 1v1 arena with the car placed at `car` moving at `car_v`, the ball at
    /// `ball` moving at `ball_v`, stepped once so a contact registers. Returns
    /// (prev, cur, car_idx).
    fn contact_state(
        car: [f32; 3], car_v: [f32; 3], ball: [f32; 3], ball_v: [f32; 3], steps: u32,
    ) -> (GameState, GameState, usize) {
        use rocketsim_rs::math::Vec3;
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(5));
        let gs0 = arena.pin_mut().get_game_state();
        let id = gs0.cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(car[0], car[1], car[2]);
        cs.vel = Vec3::new(car_v[0], car_v[1], car_v[2]);
        cs.is_on_ground = false;
        arena.pin_mut().set_car(id, cs).unwrap();
        let mut b = arena.pin_mut().get_ball();
        b.pos = Vec3::new(ball[0], ball[1], ball[2]);
        b.vel = Vec3::new(ball_v[0], ball_v[1], ball_v[2]);
        arena.pin_mut().set_ball(b);
        let prev = arena.pin_mut().get_game_state();
        arena.pin_mut().step(steps);
        let cur = arena.pin_mut().get_game_state();
        (prev, cur, 0)
    }

    /// A HAND-BUILT (prev, cur) pair: exact ball velocity before/after, exact
    /// car placement, and a forged `ball_hit_info` for the cars marked as
    /// touching. Physics-free on purpose -- these tests are about the reward
    /// ARITHMETIC (which cycles cancel, who gets credited), and a stepped arena
    /// cannot place a MOTIONLESS passenger in contact with the ball on a chosen
    /// frame, which is exactly the case the attribution factor exists for.
    ///
    /// `cars` is (team, position, did-this-car-register-a-hit). Every car is
    /// airborne. The previous hit is at tick 0 and the new one at 1004, so the
    /// gap clears any cooldown -- the cooldown itself is tested by rewriting
    /// `prev`'s hit tick.
    fn synth(
        ball_pos: [f32; 3],
        ball_v0: [f32; 3],
        ball_v1: [f32; 3],
        cars: &[(Team, [f32; 3], bool)],
    ) -> (GameState, GameState) {
        use rocketsim_rs::math::Vec3;
        ensure_init(None);
        let mut arena = Arena::default_standard();
        for _ in cars {
            arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        }
        arena.pin_mut().reset_to_random_kickoff(Some(7));
        let mut prev = arena.pin_mut().get_game_state();
        prev.tick_count = 1000;
        prev.ball.pos = Vec3::new(ball_pos[0], ball_pos[1], ball_pos[2]);
        prev.ball.vel = Vec3::new(ball_v0[0], ball_v0[1], ball_v0[2]);
        for (i, (team, pos, _)) in cars.iter().enumerate() {
            let c = &mut prev.cars[i];
            c.team = *team;
            c.state.pos = Vec3::new(pos[0], pos[1], pos[2]);
            c.state.vel = Vec3::new(0.0, 0.0, 0.0);
            c.state.is_on_ground = false;
            c.state.ball_hit_info.is_valid = true;
            c.state.ball_hit_info.tick_count_when_hit = 0;
        }
        let mut cur = prev.clone();
        cur.tick_count = 1008;
        cur.ball.vel = Vec3::new(ball_v1[0], ball_v1[1], ball_v1[2]);
        for (i, (_, _, touched)) in cars.iter().enumerate() {
            if *touched {
                cur.cars[i].state.ball_hit_info.tick_count_when_hit = 1004;
            }
        }
        (prev, cur)
    }

    /// `aerial_touch` alone for car `i` of a `synth` pair.
    fn aerial_of(prev: &GameState, cur: &GameState, i: usize, cfg: &RewardConfig) -> f64 {
        let mut terms = [0.0f64; N_TERMS];
        compute_terms(prev, cur, i, None, cfg, &mut terms);
        assert!(terms[T_TOUCH_EVENTS] > 0.0,
                "precondition: car {i}'s forged contact must register as a touch");
        assert!(terms[T_AIRBORNE_TOUCH_EVENTS] > 0.0,
                "precondition: car {i} must be airborne, or the term is trivially 0");
        terms[T_AERIAL_TOUCH]
    }

    #[test]
    fn v9_toml_matches_documented_design() {
        // Guards against a silent typo in the launch config. Every number here
        // is quoted in the spec's magnitude reading and in the config header.
        let c = v9_cfg();
        assert_eq!(c.goal, 10.0);
        assert_eq!(c.touch, 0.5);
        assert_eq!(c.touch_cooldown_ticks, 30, "0.25s -> flat touch capped at 2.0/s");
        assert_eq!(c.vel_to_ball, 0.05);
        assert_eq!(c.touch_accel, 0.5, "the un-farmable form of 'reward flips'");
        assert_eq!(c.aerial_touch, 1.5);
        assert_eq!(c.aerial_z_lo, 400.0);
        assert_eq!(c.aerial_z_hi, 1400.0);
        assert_eq!(c.air_setup, 0.0, "STAGED OFF at launch -- two new terms at once is unattributable");
        assert_eq!(c.team_spirit, 0.3);
        assert_eq!(c.opp_spirit, 0.0);
    }

    #[test]
    fn wall_riding_is_not_airborne() {
        // THE gate behind `!is_on_ground`. RocketSim's isOnGround is
        // `numWheelsInContact >= 3` and is SURFACE-AGNOSTIC (Car.cpp:118), so a
        // car driving the side wall high above the floor still reads TRUE. If a
        // RocketSim bump ever changes this, `aerial_touch` silently becomes a
        // wall-play farm -- this test is the tripwire, not decoration.
        use rocketsim_rs::math::{Angle, Vec3};
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(9));
        let id = arena.pin_mut().get_game_state().cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        // pinned against the +x side wall (x = 4096), driving along it at height
        cs.pos = Vec3::new(4096.0 - 17.0, 0.0, 1065.0);
        cs.rot_mat = Angle { yaw: 0.0, pitch: 0.0, roll: -std::f32::consts::FRAC_PI_2 }.to_rotmat();
        cs.vel = Vec3::new(0.0, 1200.0, 0.0);
        arena.pin_mut().set_car(id, cs).unwrap();
        for _ in 0..6 {
            arena.pin_mut()
                .set_car_controls(id, CarControls { throttle: 1.0, boost: true, ..Default::default() })
                .unwrap();
            arena.pin_mut().step(8);
        }
        let gs = arena.pin_mut().get_game_state();
        let c = &gs.cars[0];
        assert!(c.state.pos.z > 800.0, "precondition: still high on the wall, z={}", c.state.pos.z);
        assert!(
            c.state.is_on_ground,
            "a car on the WALL at z={} must read is_on_ground=true; if this flipped, \
             aerial_touch's airborne gate no longer excludes wall play",
            c.state.pos.z
        );
    }

    #[test]
    fn aerial_touch_is_zero_on_the_ground() {
        // Ground contact, high-speed strike toward the net: touch/touch_accel pay,
        // aerial_touch must not.
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(3));
        let gs0 = arena.pin_mut().get_game_state();
        let id = gs0.cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        let ball = arena.pin_mut().get_ball();
        cs.pos = rocketsim_rs::math::Vec3::new(ball.pos.x, ball.pos.y - 200.0, 17.0);
        cs.vel = rocketsim_rs::math::Vec3::new(0.0, 1800.0, 0.0);
        arena.pin_mut().set_car(id, cs).unwrap();
        let prev = arena.pin_mut().get_game_state();
        for _ in 0..8 {
            arena.pin_mut()
                .set_car_controls(id, CarControls { throttle: 1.0, ..Default::default() })
                .unwrap();
            arena.pin_mut().step(8);
        }
        let cur = arena.pin_mut().get_game_state();
        assert!(cur.cars[0].state.is_on_ground, "precondition: the car stayed grounded");
        let mut terms = [0.0f64; N_TERMS];
        let r = compute_terms(&prev, &cur, 0, None, &v9_cfg(), &mut terms);
        assert!(terms[T_TOUCH_EVENTS] > 0.0, "precondition: the strike registered a touch");
        assert_eq!(terms[T_AERIAL_TOUCH], 0.0, "a grounded strike must pay ZERO aerial_touch");
        assert!(r > 0.0, "...while the ordinary touch terms still pay: {r}");
    }

    #[test]
    fn aerial_touch_is_zero_below_z_lo() {
        // Airborne, hitting the ball toward the net, but at hop height. h = 0.
        let cfg = v9_cfg();
        // ball centre 300 < aerial_z_lo 400
        let (prev, cur, i) = contact_state(
            [0.0, -200.0, 300.0], [0.0, 1400.0, 0.0], [0.0, 0.0, 300.0], [0.0, 0.0, 0.0], 16);
        let mut terms = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, i, None, &cfg, &mut terms);
        assert!(terms[T_TOUCH_EVENTS] > 0.0, "precondition: contact happened");
        assert!(terms[T_AIRBORNE_TOUCH_EVENTS] > 0.0, "precondition: the car was airborne");
        assert_eq!(terms[T_AERIAL_TOUCH], 0.0,
                   "a hop-height touch is below the ramp and must pay nothing");
    }

    #[test]
    fn aerial_touch_pays_a_real_strike_and_is_bounded_by_its_weight() {
        let cfg = v9_cfg();
        // Real aerial height (well above the measured double-jump apex of ~525
        // ball z), driving the ball toward the ORANGE net (+y) for a blue car.
        let (prev, cur, i) = contact_state(
            [0.0, -220.0, 1000.0], [0.0, 1900.0, 0.0], [0.0, 0.0, 1000.0], [0.0, 0.0, 0.0], 16);
        let mut terms = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, i, None, &cfg, &mut terms);
        assert!(terms[T_AERIAL_TOUCH] > 0.05,
                "a clean high strike toward the net must pay: {}", terms[T_AERIAL_TOUCH]);
        assert!(terms[T_AERIAL_TOUCH] <= cfg.aerial_touch as f64,
                "both factors are in [0,1] so the WEIGHT is the per-step bound: {}",
                terms[T_AERIAL_TOUCH]);
    }

    #[test]
    fn aerial_touch_charges_for_hitting_the_ball_backwards() {
        // The SYMMETRIC clamp (2026-07-26 review). This used to assert exactly
        // 0.0 -- "pays zero, not negative" -- and that asymmetry was the whole
        // ratchet: a y-axis juggle paid 1.198 going forward and 0.000 coming
        // back, so undoing your own work was free and a closed cycle banked
        // 2.391 per lap. Same height, same impulse magnitude, aimed at our OWN
        // net now costs what the forward leg paid.
        let cfg = v9_cfg();
        let (prev, cur, i) = contact_state(
            [0.0, 220.0, 1000.0], [0.0, -1900.0, 0.0], [0.0, 0.0, 1000.0], [0.0, 0.0, 0.0], 16);
        let mut terms = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, i, None, &cfg, &mut terms);
        assert!(terms[T_TOUCH_EVENTS] > 0.0, "precondition: contact happened");
        assert!(terms[T_AERIAL_TOUCH] < -0.05,
                "impulse toward our OWN net must be CHARGED, not free: {}",
                terms[T_AERIAL_TOUCH]);
        assert!(terms[T_AERIAL_TOUCH] >= -(cfg.aerial_touch as f64),
                "the weight is still the bound in both directions: {}",
                terms[T_AERIAL_TOUCH]);
    }

    #[test]
    fn aerial_touch_telescopes_to_zero_over_a_closed_ball_velocity_cycle() {
        // THE farming regression test for `aerial_touch`, and the direct mirror
        // of `air_potential_telescopes_to_zero_over_a_jump_cycle`. Both legs are
        // run at ball x = +/-3000, y = 4500 -- the measured LATERAL farm
        // geometry, where the OLD ball-relative goal vector was 98% sideways and
        // flipped as the ball crossed the field, so batting the ball back and
        // forth across the corner paid 1.198 then 1.193, both positive, with the
        // ball ending exactly where it started and no goal possible (z = 1200 is
        // well above the 642 crossbar).
        let cfg = v9_cfg();
        const Z: f32 = 1200.0;

        // (a) LATERAL round trip: nothing about it advances the ball, so both
        // legs must pay EXACTLY zero -- the forward axis has no x component.
        let (p1, c1) = synth([3000.0, 4500.0, Z], [0.0; 3], [-2400.0, 0.0, 0.0],
                             &[(Team::Blue, [3300.0, 4500.0, Z], true)]);
        let (p2, c2) = synth([-3000.0, 4500.0, Z], [-2400.0, 0.0, 0.0], [2400.0, 0.0, 0.0],
                             &[(Team::Blue, [-3300.0, 4500.0, Z], true)]);
        let (out, back) = (aerial_of(&p1, &c1, 0, &cfg), aerial_of(&p2, &c2, 0, &cfg));
        assert_eq!((out, back), (0.0, 0.0),
                   "a lateral corner cycle must pay nothing on EITHER leg: {out} / {back}");

        // (b) FORWARD round trip at the same x: the return leg must refund the
        // forward leg exactly, so the closed cycle nets ~0.
        let (p3, c3) = synth([3000.0, 4500.0, Z], [0.0; 3], [0.0, 1800.0, 0.0],
                             &[(Team::Blue, [3000.0, 4300.0, Z], true)]);
        let (p4, c4) = synth([3000.0, 4500.0, Z], [0.0, 1800.0, 0.0], [0.0; 3],
                             &[(Team::Blue, [3000.0, 4700.0, Z], true)]);
        let (fwd, rev) = (aerial_of(&p3, &c3, 0, &cfg), aerial_of(&p4, &c4, 0, &cfg));
        assert!(fwd > 0.5, "precondition: the forward leg is a real payment: {fwd}");
        assert!((fwd + rev).abs() < 1e-6,
                "a closed ball-velocity cycle must net ~0: +{fwd} then {rev}");
    }

    #[test]
    fn aerial_touch_credits_the_striker_not_the_passenger() {
        // The attribution factor (2026-07-26 review). `dv` is the ball's TOTAL
        // velocity change from every source, so before this the motionless car
        // hanging beside the ball collected the striker's payout byte-identically
        // -- 0.896 each. With equal 1s/2s/3s share, 2/3 of experience is in
        // arenas where a teammate supplies the impulse, so one strike would have
        // paid N grazing teammates N x 0.896.
        let cfg = v9_cfg();
        let (prev, cur) = synth(
            [0.0, 0.0, 1000.0], [0.0; 3], [0.0, 2700.0, 0.0],
            &[
                (Team::Blue, [0.0, -220.0, 1000.0], true), // striker, behind the ball
                (Team::Blue, [300.0, 0.0, 1000.0], true),  // passenger, beside it
            ],
        );
        let (striker, passenger) = (aerial_of(&prev, &cur, 0, &cfg),
                                    aerial_of(&prev, &cur, 1, &cfg));
        assert!(striker > 0.5, "the real strike must be untouched by the fix: {striker}");
        assert!(passenger.abs() < 1e-6,
                "a car the impulse did not come from must be paid nothing: {passenger}");
    }

    #[test]
    fn aerial_touch_pays_no_passenger_when_the_impulse_is_an_opponents() {
        // Same shape, worse: the ORANGE car supplies the whole impulse and drives
        // the ball toward its OWN net. The blue passenger used to collect the
        // full 0.896 for hanging in the air while an opponent did the work.
        let cfg = v9_cfg();
        let (prev, cur) = synth(
            [0.0, 0.0, 1000.0], [0.0; 3], [0.0, 2700.0, 0.0],
            &[
                (Team::Orange, [0.0, -220.0, 1000.0], true), // orange striker
                (Team::Blue, [300.0, 0.0, 1000.0], true),    // blue passenger
            ],
        );
        let (orange, blue) = (aerial_of(&prev, &cur, 0, &cfg), aerial_of(&prev, &cur, 1, &cfg));
        assert!(blue.abs() < 1e-6, "the blue passenger contributed nothing: {blue}");
        assert!(orange < -0.5,
                "and the orange car that drove the ball at its own net is CHARGED: {orange}");
    }

    #[test]
    fn aerial_touch_takes_the_touch_cooldown() {
        // The exemption `aerial_touch` used to claim rested on "payout is
        // proportional to delivered impulse, which physics caps". That holds for
        // a monotone one-way strike and NOT for a ball something else hands back
        // (the back wall, an opponent), so the term now takes the same gap test
        // as the flat touch. Identical strike, only the gap to the PREVIOUS
        // recorded hit changes.
        let cfg = v9_cfg();
        let car = [(Team::Blue, [0.0, -220.0, 1000.0], true)];
        let (prev, cur) = synth([0.0, 0.0, 1000.0], [0.0; 3], [0.0, 2300.0, 0.0], &car);
        let paid = aerial_of(&prev, &cur, 0, &cfg);
        assert!(paid > 0.5, "precondition: a clean strike after a long gap pays: {paid}");

        let mut soon = prev.clone();
        // previous hit 4 ticks ago, well inside the 30-tick window
        soon.cars[0].state.ball_hit_info.tick_count_when_hit = 1000;
        let mut cur2 = cur.clone();
        cur2.cars[0].state.ball_hit_info.tick_count_when_hit = 1004;
        let mut terms = [0.0f64; N_TERMS];
        compute_terms(&soon, &cur2, 0, None, &cfg, &mut terms);
        assert!(terms[T_TOUCH_EVENTS] > 0.0, "precondition: the contact still REGISTERS");
        assert_eq!(terms[T_AERIAL_TOUCH], 0.0,
                   "a re-hit inside the cooldown must pay no aerial_touch either");
        assert_eq!(terms[T_TOUCH], 0.0, "...and no flat touch, as before");
    }

    #[test]
    fn touch_cooldown_starves_the_carry_farm() {
        // THE farming regression test. Hold the ball in permanent overlap so
        // RocketSim's uncooled `tickCountWhenHit` re-fires every step, and check
        // that (a) the raw touch EVENT rate really is pathological, and (b) the
        // v9 config pays at most the cooldown bound while reward_v8_team pays the
        // full farm. Without the cooldown this configuration is worth ~200 goals
        // per match to a bot that does nothing but sit on the ball.
        use rocketsim_rs::math::Vec3;
        ensure_init(None);
        const STEPS: u32 = 60; // 60 * 8 ticks = 480 ticks = 4.0 s
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(21));
        let id = arena.pin_mut().get_game_state().cars[0].id;

        let v9 = v9_cfg();
        let v8 = RewardConfig::load("../configs/reward_v8_team.toml").unwrap();
        let (mut t9, mut t8) = ([0.0f64; N_TERMS], [0.0f64; N_TERMS]);
        let mut prev = arena.pin_mut().get_game_state();
        for _ in 0..STEPS {
            // re-pin car and ball into overlap every step: a forced-contact
            // UPPER bound on the farm, deliberately more extreme than any real
            // carry (a scripted free-play dribble measured only ~0.1 events/s).
            let mut cs = arena.pin_mut().get_car(id);
            cs.pos = Vec3::new(0.0, 0.0, 100.0);
            cs.vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_car(id, cs).unwrap();
            let mut b = arena.pin_mut().get_ball();
            b.pos = Vec3::new(0.0, 0.0, 100.0);
            b.vel = Vec3::new(0.0, 0.0, 0.0);
            arena.pin_mut().set_ball(b);
            arena.pin_mut().step(8);
            let cur = arena.pin_mut().get_game_state();
            compute_terms(&prev, &cur, 0, None, &v9, &mut t9);
            compute_terms(&prev, &cur, 0, None, &v8, &mut t8);
            prev = cur;
        }
        let secs = (STEPS * 8) as f64 / 120.0;
        let events_per_s = t9[T_TOUCH_EVENTS] / secs;
        assert!(events_per_s > 5.0,
                "precondition: sustained overlap must re-fire `touched` many times a \
                 second (the hazard being closed); measured {events_per_s:.2}/s");
        // v8 (no cooldown) pays the full farm; v9 pays at most one touch per
        // cooldown window. `+ touch` covers the first, always-paid touch.
        let cap = (v9.touch * (secs as f32 * 120.0 / v9.touch_cooldown_ticks as f32 + 1.0)) as f64;
        assert!(t9[T_TOUCH] <= cap,
                "cooldown must bound flat touch at one payment per {} ticks: paid {} > cap {cap}",
                v9.touch_cooldown_ticks, t9[T_TOUCH]);
        assert!(t8[T_TOUCH] > 4.0 * t9[T_TOUCH],
                "the uncapped config must be dramatically more farmable, else this \
                 test is not measuring what it claims: v8 {} vs v9 {}",
                t8[T_TOUCH], t9[T_TOUCH]);
        // And the impulse-scaled term is starved in the same regime -- the
        // reason `touch_accel` is the sanctioned "reward flips" form.
        assert!(t9[T_TOUCH_ACCEL] < t8[T_TOUCH],
                "touch_accel must not replace the farm it was chosen to avoid: {}",
                t9[T_TOUCH_ACCEL]);
    }

    #[test]
    fn air_potential_telescopes_to_zero_over_a_jump_cycle() {
        // The farm-proof property of `air_setup`, checked as arithmetic rather
        // than physics: any path that STARTS and ENDS on the ground sums to
        // gamma-discounted zero. At gamma = 1 that is exactly zero, so a bot that
        // flies in circles and lands banks nothing.
        let mut cfg = v9_cfg();
        cfg.air_setup = 0.3;
        cfg.win_prob_gamma = 1.0;
        assert!(cfg.validate().is_ok());
        // (on_ground, car_pos, ball_pos) waypoints: grounded -> airborne climb ->
        // hover near a high ball -> descend -> grounded again.
        let path: [(bool, [f32; 3], [f32; 3]); 6] = [
            (true, [0.0, 0.0, 17.0], [0.0, 0.0, 1200.0]),
            (false, [0.0, 0.0, 300.0], [0.0, 0.0, 1200.0]),
            (false, [0.0, 0.0, 900.0], [0.0, 0.0, 1200.0]),
            (false, [0.0, 300.0, 1100.0], [0.0, 0.0, 1200.0]),
            (false, [0.0, 0.0, 400.0], [0.0, 0.0, 1200.0]),
            (true, [0.0, 0.0, 17.0], [0.0, 0.0, 1200.0]),
        ];
        let phi = |s: &(bool, [f32; 3], [f32; 3])| super::air_potential(s.0, s.1, s.2, &cfg);
        let mut total = 0.0f32;
        for w in path.windows(2) {
            total += cfg.air_setup * (cfg.win_prob_gamma * phi(&w[1]) - phi(&w[0]));
        }
        assert!(phi(&path[2]) > 0.0, "precondition: the mid-flight potential is nonzero");
        assert!(total.abs() < 1e-5,
                "a jump-and-land cycle must refund exactly: got {total}");
    }

    #[test]
    fn air_potential_is_zero_on_the_ground_at_any_ball_height() {
        let mut cfg = v9_cfg();
        cfg.air_setup = 0.3;
        for bz in [100.0, 800.0, 1400.0, 1900.0] {
            let p = super::air_potential(true, [0.0, 0.0, 17.0], [0.0, 0.0, bz], &cfg);
            assert_eq!(p, 0.0, "grounded PHI must be 0 (ball z {bz})");
        }
    }

    #[test]
    fn loitering_at_high_potential_costs_a_little_every_step() {
        // The anti-hover property: at the live gamma, holding PHI constant pays
        // w*(gamma-1)*PHI < 0. If this ever came out >= 0 the term would pay for
        // sitting in the air doing nothing.
        let mut cfg = v9_cfg();
        cfg.air_setup = 0.3;
        cfg.win_prob_gamma = 0.9954;
        let s = (false, [0.0, 0.0, 1100.0], [0.0, 0.0, 1200.0]);
        let phi = super::air_potential(s.0, s.1, s.2, &cfg);
        let step = cfg.air_setup * (cfg.win_prob_gamma * phi - phi);
        assert!(phi > 0.0 && step < 0.0, "phi {phi} step {step}");
    }

    #[test]
    fn air_shaping_is_inert_at_zero_weight() {
        // v9 ships air_setup = 0.0, and the episode-level call site is
        // UNCONDITIONAL, so this exact-zero return is what keeps every existing
        // config's per-step reward untouched.
        let cfg = v9_cfg();
        assert_eq!(cfg.air_setup, 0.0);
        let (prev, cur, i) = contact_state(
            [0.0, -220.0, 1000.0], [0.0, 1900.0, 0.0], [0.0, 0.0, 1000.0], [0.0, 0.0, 0.0], 16);
        assert_eq!(super::air_shaping(&prev, &cur, i, 0.9954, &cfg), 0.0);
    }

    #[test]
    fn half_configured_aerial_is_rejected_not_nan() {
        // aerial_touch with no ramp bounds: (z-0)/(0-0) is inf/NaN.
        let mut c = RewardConfig { goal: 10.0, ..Default::default() };
        c.aerial_touch = 1.5;
        assert!(c.validate().is_err(), "must reject weight-without-ramp");
        // a ramp that starts under a resting ball is a ground-play payout
        c.aerial_z_lo = 10.0;
        c.aerial_z_hi = 1400.0;
        assert!(c.validate().is_err(), "aerial_z_lo below the ball radius must be refused");
        c.aerial_z_lo = 400.0;
        assert!(c.validate().is_ok());
        // inverted bounds
        c.aerial_z_hi = 400.0;
        assert!(c.validate().is_err(), "hi == lo is the exact NaN case");
        c.aerial_z_hi = 1400.0;
        // air_setup shares the potential gamma and needs it > 0
        c.air_setup = 0.3;
        assert!(c.validate().is_err(), "air_setup without a gamma is a pure takeoff penalty");
        c.win_prob_gamma = 0.9954;
        assert!(c.validate().is_ok());
    }

    #[test]
    fn compute_terms_sums_to_compute() {
        // E9's contract: the telemetry is a decomposition of the SAME number, not
        // a parallel re-implementation that can drift.
        let cfg = v9_cfg();
        let (prev, cur, i) = contact_state(
            [0.0, -220.0, 1000.0], [0.0, 1900.0, 0.0], [0.0, 0.0, 1000.0], [0.0, 0.0, 0.0], 16);
        let mut terms = [0.0f64; N_TERMS];
        let r = compute_terms(&prev, &cur, i, Some(Team::Blue), &cfg, &mut terms);
        let plain = compute(&prev, &cur, i, Some(Team::Blue), &cfg);
        assert_eq!(r, plain, "compute() must be compute_terms() with the counters discarded");
        let sum: f64 = terms[T_GOAL..=T_AERIAL_TOUCH].iter().sum();
        assert!((sum - r as f64).abs() < 1e-5, "terms {sum} != total {r}");
    }
}
