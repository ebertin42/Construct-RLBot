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
    /// Height floor for the `gated_aerial_touch_events` COUNTER only, never for
    /// a payout. 0.0 (the default, and every tape written before 2026-07-27)
    /// means "inherit `aerial_z_lo`", so existing configs are bit-identical.
    ///
    /// It exists because `aerial_z_lo` is read by three things at once -- the
    /// `aerial_touch` payout, `air_setup`'s potential, and the counter -- so
    /// moving the ramp to make the reward reachable ALSO moves the instrument
    /// that would say whether it worked. Any post-change rise in air_gate_frac
    /// would be partly definitional and the baseline it is measured against
    /// (1,786 iterations, 312M-491M steps, mean 2.76e-4) would be destroyed.
    /// Pin this at the historical 400.0 before touching `aerial_z_lo` and the
    /// counter keeps meaning what it meant. Same discipline as never rebuilding
    /// the gate `.so` mid-measurement.
    #[serde(default)]
    pub aerial_meas_z_lo: f32,
    /// Separate height ramp for `air_setup`'s potential. Both 0.0 (the default)
    /// means "inherit `aerial_z_lo`/`aerial_z_hi`", so existing tapes are
    /// bit-identical.
    ///
    /// The shared ramp has a structural flaw that no hypothesis about the aerial
    /// deficit had noticed: PHI is identically 0 whenever ball_z <= aerial_z_lo,
    /// REGARDLESS of what the car is doing. So the one term whose stated job is
    /// teaching the takeoff decision -- "the ball is going up, jump NOW", which
    /// `aerial_touch` structurally cannot teach -- has exactly zero gradient in
    /// the states where takeoff is decided. It can only pay for already being
    /// airborne near an already-high ball. Splitting the ramps is what makes
    /// that fixable as a single variable, without touching the payout.
    #[serde(default)]
    pub air_setup_z_lo: f32,
    #[serde(default)]
    pub air_setup_z_hi: f32,
    /// Reward per boost pickup, scaled by `(sqrt(b1) - sqrt(b0))/10` so a pad pays
    /// in proportion to how much the boost was NEEDED: 0->100 pays 1.0x this
    /// coefficient, 0->12 pays 0.346x, 88->100 pays 0.062x, 100->100 pays 0.
    /// NOT linear `gained/100`, which prices 0->20 and 80->100 the same.
    ///
    /// 0.2 means a big pad from empty is worth ~4 decisions of forgone ball
    /// approach at `vel_to_ball = 0.05`, i.e. roughly a 0.27s detour -- so a pad
    /// already on the path is always worth taking and a real detour never is.
    /// That framing prices ONE pickup; it does not bound the loop. See
    /// `T_BOOST_PICKUP` for the measured spend-and-refill farm (up to 57.7 per
    /// 300s match at 0.2, against a 4-10 shaping budget) before choosing a value.
    /// `serde(default)` -> 0.0, which is what keeps every existing tape
    /// reward-identical (`v0_behavior_unchanged_by_new_fields` is the gate).
    #[serde(default)]
    pub boost_pickup: f32,
    /// Flat reward, once per air phase, when the car's POST-JUMP hang clock
    /// crosses `AIR_HANG_T`. SHIPPED AT 0.0 AND ABSENT FROM EVERY TAPE, on
    /// purpose -- see `T_AIR_HANG` for the review that says why, and read it
    /// before setting this to anything.
    ///
    /// The one-line version: the gate is sound (the cheapest hop spam qualifies
    /// exactly zero times, measured) but the coefficient is not pinnable. It
    /// cannot be sized without `sigma_raw`, which nothing logs, and the money it
    /// unlocks is mostly not its own -- raising P(jump edge) at all opens a
    /// PRE-EXISTING +5.8/match `vel_to_ball` pogo ratchet that this coefficient
    /// cannot cap and that setting it back to 0.0 cannot unlearn.
    #[serde(default)]
    pub air_hang: f32,
}

/// Per-term reward telemetry (E9). Index space for the `[f64; N_TERMS]`
/// counter array `EpisodeArena` accumulates and `Engine.reward_terms()`
/// exposes. Existing in this form because "is the new term actually firing,
/// and is it being farmed?" must be answerable from the log without a probe
/// run -- the 2026-07-20 confound was an arm that ran fully INERT while every
/// log line looked normal.
///
/// Indices 0-8 are reward contributions and 9-19 are COUNTS, not reward: the
/// counts are what the §8.3 farming tripwires (touches/min/car, airborne-touch
/// fraction, reward per goal) are computed from.
///
/// That reward-prefix/count-suffix split stopped being a clean partition at
/// index 20: `boost_pickup` is a REWARD appended AFTER the counters, because
/// renumbering is forbidden (this index space is positional and the telemetry
/// baselines are read against a fixed order). Any consumer that sums "the reward
/// slice" must therefore sum `T_GOAL..=T_AERIAL_TOUCH` PLUS `T_BOOST_PICKUP`
/// PLUS `T_AIR_HANG` (index 23, appended 2026-07-30 for the same reason) --
/// see `compute_terms_sums_to_compute`, and `reward_keys` in
/// `tests/python/test_engine_v9.py`, which still hardcodes the nine-name prefix.
/// Both appended rewards are 0.0 on every shipped tape, so the hardcoded prefix
/// is currently still exact; that is a coincidence of the coefficients, not a
/// property of the layout.
pub const N_TERMS: usize = 26;
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
/// Touches that pass `aerial_touch`'s OWN gate: airborne AND on a ball above
/// `aerial_z_lo`. APPENDED at the end (2026-07-26) so every existing index,
/// and the `T_GOAL..=T_AERIAL_TOUCH` reward slice, is untouched.
///
/// It exists because `airborne_touch_events` is an ANTI-CORRELATED instrument
/// and was being read as if it measured air play. Measured: a RANDOM policy
/// scores 0.87-0.93 on it -- the curriculum's random resets start half the cars
/// airborne, and a tumbling car's contacts are all "airborne" -- while the
/// champion, which makes 63x more real aerial touches than v9, scores 0.0117.
/// The missing half of the gate is the height: the champion's airborne touches
/// are ALL under the z_lo = 400 ramp, which is exactly why its own share of
/// r_aerial_touch is 0.0000. This counter applies the height too, so it moves
/// with the behaviour the reward pays for.
///
/// The touch COOLDOWN is deliberately not applied: this is a volume signal for
/// "is the manoeuvre happening", not a payout, and gating it would make a
/// carry read as one aerial. Stays 0 on any tape that does not configure the
/// ramp (reward_v0's zeros would otherwise make `height_ramp` a 0/0 inf and
/// count every airborne touch).
pub const T_AERIAL_TOUCH_EVENTS: usize = 13;
/// Six-bucket histogram of BALL HEIGHT at an AIRBORNE learner touch, appended
/// 2026-07-27 so every existing index survives.
///
/// THE measurement the whole aerial debate turned on and that nobody had ever
/// made. `air_gate_frac` says how many airborne touches clear `aerial_z_lo`; it
/// cannot say how far short the rest fall, so it cannot distinguish "the ramp
/// sits just above the distribution" (lower it) from "the distribution is on the
/// floor" (the z-floor is not the problem and lowering it just builds a hop-tap
/// farm in front of the behaviour you want). Every z_lo proposed so far was
/// anchored to a figure -- "91.6% of touches below 400uu, median 109.8uu" -- with
/// no source in the repo, no tool that computes it, and, worse, over ALL touches
/// when only AIRBORNE ones can ever be paid. This is the right population.
///
/// Same gate as `T_AIRBORNE_TOUCH_EVENTS` and deliberately NO cooldown and NO
/// height test, so the six buckets sum EXACTLY to it -- that identity is the
/// self-check that says the histogram is wired to the population it claims.
/// Edges are ball-centre z: a resting ball is 93.15, a hop reaches ~360 and a
/// double jump ~525, so the 300-500 bucket is "jumped but not really flying".
pub const T_AIR_TOUCH_Z0: usize = 14; // < 150
pub const T_AIR_TOUCH_Z1: usize = 15; // 150 - 300
pub const T_AIR_TOUCH_Z2: usize = 16; // 300 - 500
pub const T_AIR_TOUCH_Z3: usize = 17; // 500 - 800
pub const T_AIR_TOUCH_Z4: usize = 18; // 800 - 1200
pub const T_AIR_TOUCH_Z5: usize = 19; // >= 1200
/// Upper edges for buckets 0..4; anything at or above the last lands in Z5.
const AIR_TOUCH_Z_EDGES: [f32; 5] = [150.0, 300.0, 500.0, 800.0, 1200.0];
/// REWARD (not a count, despite sitting after the counters -- see `N_TERMS`):
/// boost collected, priced on the SQRT of the amount gained so a pad pays in
/// proportion to how much the boost was NEEDED. Measured in f32: 0->100 pays
/// 1.000x the coefficient, 0->12 pays 0.346x, 88->100 pays 0.0619x, and a full
/// car gains nothing so it pays exactly 0. Linear `gained/100` would price 0->20
/// and 80->100 identically, which is wrong -- the first is desperate, the second
/// worthless. An EVENT, not a potential: a potential on boost AMOUNT penalises
/// SPENDING it and yields a hoarder, the same gamma-drag that left `air_setup`
/// net negative for 2B steps.
///
/// MEASURED HAZARD, recorded here so nobody has to rediscover it. Because
/// spending is deliberately never charged, the term is a RATCHET and the payout
/// is PATH-DEPENDENT. Exact telescoping -- splitting a climb into pads earns
/// nothing extra, pinned by `a_monotone_climb_pays_the_same_however_it_is_split`
/// -- holds ONLY for a MONOTONE climb. A spend-and-refill loop resets the
/// baseline and re-earns at the higher marginal rate: 12-from-empty chunks pay
/// 0.005774 per boost unit against 0.002000 for one 0->100, i.e. 2.89x. The burn
/// rate (`BOOST_USED_PER_SECOND` = 33.33) is the only cap, at 0.1925 reward/s =
/// 57.7 per 300s match at coefficient 0.2 (a measured 6-pad routing loop reaches
/// 31/match; a big-pad grand tour 30.6/match, and the six big pads are served to
/// the net as labelled `obs_v1` entity rows with an availability flag, so that
/// one is directly learnable). Against a stated shaping budget of 4-10 per match
/// that makes 0.2 a SECOND touch term in magnitude, earned with zero touches --
/// so the coefficient is the entire safety margin, and `T_BOOST_GAINED` (raw
/// units, coefficient-free) is the tripwire that says whether it is being farmed.
/// Pin the champion's boost_gained/min/car BEFORE shipping a nonzero coefficient.
pub const T_BOOST_PICKUP: usize = 20;
/// Instrument: raw boost units gained, independent of `cfg.boost_pickup`, so
/// retuning the payout does not move the measurement. Same payout/instrument
/// split as `aerial_z_lo` vs `aerial_meas_z_lo`. THE farm tripwire for
/// `T_BOOST_PICKUP`, because it is denominated in the resource, not the reward.
///
/// A NET delta cannot hide a pad at `tick_skip = 8` -- but only because the
/// smallest pad (12) exceeds one decision's maximum burn (33.33 * 8/120 = 2.22).
/// That property breaks at `tick_skip >= 44`, where same-window spending could
/// mask a pickup and this instrument would under-count.
pub const T_BOOST_GAINED: usize = 21;
/// Instrument: flips consumed, counted on the `has_flipped` RISING EDGE. Flips
/// were previously unmeasurable -- "does the bot flip?" could only be answered by
/// watching the viewer.
///
/// Reads FLIPS ONLY, which is narrower than it sounds. RocketSim sets
/// `hasFlipped` in exactly one place, the dodge branch of
/// `Car::_UpdateDoubleJumpOrFlip`, and clears it unconditionally on any tick the
/// car is on the ground. So this counts dodges AND stalls (a stall is a flip with
/// a cancelling dodge impulse; both v1 stall rows clear the 0.5 dodge deadzone),
/// but NOT double jumps, which set `hasDoubleJumped` and leave `hasFlipped`
/// alone. The two no-direction jump rows the action table gives the most mass to
/// are DOUBLE jumps, so this counter reads 0 for that lever -- `p_jump` and
/// `air_z` are its readouts, not this. Pinned by `a_double_jump_is_not_a_flip`.
///
/// It is a LOWER BOUND, not an exact count: `compute_terms` runs once per
/// decision against a state 8 physics ticks old, so a flip that starts and lands
/// inside one 67ms window is invisible and two flips separated by a landing in
/// one window count as one. Comparable across runs; not a true flip total.
pub const T_FLIP_EVENTS: usize = 22;
/// REWARD (a second non-prefix reward slot -- see `N_TERMS`), and the one term
/// in this file that ships DELIBERATELY INERT: `air_hang` is absent from every
/// tape, so this index reads exactly 0.0 on every run. `T_AIR_HANG_EVENTS` and
/// `T_AIR_DECISIONS` are the deliverable; this slot exists so that the payout
/// can be exercised by a test and priced against a measured event rate without
/// anyone having to re-derive the plumbing later.
///
/// WHAT IT WOULD PAY FOR. One flat payment per air phase, on the RISING crossing
/// of `air_time_since_jump` through `AIR_HANG_T` (1.2 s). It exists because the
/// learner will not take off: P(jump | previous action was NOT a jump) -- the
/// rising edge RocketSim needs to start a jump (`jumpPressed = jump &&
/// !lastControls.jump`, Car.cpp:107) -- measures 6.655e-4 (mean over 714 live
/// iterations, 2.088-2.153B steps, log10 sd 0.286), against ~1.9e-2 implied by
/// the foreign opponents' airborne fraction. Entropy cannot bridge it: x1.42
/// entropy bought x1.38 edge, so the target needs ~52x baseline.
///
/// WHY THE CROSSING, AND WHY IT IS AT MOST ONCE PER AIR PHASE. This is a physics
/// property, not a cooldown. `air_time_since_jump` accumulates only on an
/// airborne tick with `has_jumped && !is_jumping`, and is set to 0 on every
/// other tick including every grounded one (Car.cpp:688-701); `is_jumping` can
/// only be armed from `is_on_ground && jumpPressed` (Car.cpp:569-570), so it can
/// never be re-armed mid-air. Within one air phase the clock is therefore
/// strictly monotone from 0, so a rising crossing fires exactly once per phase,
/// and only for a phase a real jump impulse from a surface started.
///
/// That last clause is the whole false-fire story, and it is why this reads the
/// post-jump clock rather than a `is_on_ground` transition. On the raw
/// ground->air proxy, MEASURED false fires with zero jump input: 7-9/min from
/// driving off the curved corner (RocketSim's `is_on_ground` is
/// `numWheelsInContact >= 3` with no surface test, so a wall reads GROUNDED --
/// see `wall_riding_is_not_airborne`), one per demo respawn, and 2 per 40
/// decisions for a bumped car. All of them leave `has_jumped` false, so all of
/// them read `air_time_since_jump == 0` forever. Same for the ~28% of
/// car-episodes that SPAWN airborne (`random_reset`'s 50% coin x curriculum_v3's
/// 0.6 random weight): the kickoff reset that precedes every reset branch
/// zeroes the jump flags, so a spawned car's fall is unpaid no matter how long
/// it lasts -- pinned by
/// `an_airborne_spawn_pays_no_hang_bonus_and_the_reset_launders_the_jump_state`.
///
/// MEASURED, `engine/examples/hang_gate_probe.rs`, 200 s per tape at tick_skip
/// 8, qualifying crossings/min at T = 1.2:
///
/// * hop, jump released after 1 decision -- **0.0/min**, 93.2% airborne, apex
///   121.5 uu, no boost. This is the maximising tape for ANY flat per-takeoff
///   payment (60.6-69.6 takeoffs/min) and the hang gate pays it NOTHING: its
///   air phase is 1.13 s, and of the 200 phases measured, 98% peaked below 1.0 s
///   of post-jump clock and NONE reached 1.2 s.
/// * hop, jump held 2 / 3 / 4+ decisions -- 5.7 / **28.2** / 27.3/min, apex 263.
/// * hop then double jump -- 22.5/min, apex 567.3.
/// * jump + pitch back + boost (a real aerial) -- 0.3/min, 36.9 boost per phase.
/// * drive into the corner, no jump; drive flat, no jump; sit still -- 0.0/min.
/// * wall climb then jump off at z>300 -- 9.6/min.
///
/// THE VERDICT (2026-07-30 review), recorded here so it is not re-litigated from
/// scratch. Three things kill the coefficient and none of them are fixable from
/// inside this term:
///
/// 1. The ranking is still inverted. 28.2/min for a 263 uu hop against 0.3/min
///    for a real aerial is 94:1, because a flat per-air-phase payment pays
///    1/phase-duration and a real aerial hangs 6 s and costs a full tank. The
///    gate removes the FREE minimum hop; it does not make air play the best way
///    to collect. Raising T does not fix it either: at T = 1.4/2.0 the held hop
///    falls to 10.5/1.8 but hop-then-double-jump only falls to 21.9/18.9, i.e.
///    every threshold selects a farm rather than eliminating one.
/// 2. The coefficient is unpinnable. The entropy-regularised-optimum model gives
///    r = beta * sigma_raw * ln(odds ratio); the live beta is 0.006
///    (checkpoints_v9/train_remote.log:3811) and `sigma_raw` -- the sd of the RAW
///    per-row advantage -- is NOT logged, because `standardise_advantages` runs
///    before the `|A|` print, so the logged number is 1 by construction. The
///    honest range is 0.0134-0.0282 per event, a 2.1x band. At the measured
///    28.2/min that is 141 events per 300 s match = 1.9-4.0/match of this term's
///    own money, for a car that never touches the ball.
/// 3. The money it unlocks is mostly not its own. `vel_to_ball` (see the `proj`
///    line in `compute_terms`) is a full 3-D projection clamped at 0 below, so a
///    hop's +v_z under an overhead ball is PAID and the descent is free rather
///    than charged. Measured +5.79/match for a stationary pogo under a high
///    ball, zero ball contact. The only thing suppressing that today is the
///    6.655e-4 jump edge -- exactly the barrier this term exists to remove. So
///    total exposure is ~7.7-9.8/match, of which ~5.8 is pre-existing income
///    that this coefficient cannot cap and that setting it back to 0.0 cannot
///    unlearn. The tape is removable; the learned ratchet is not.
///
/// PREREQUISITES, in order, before this is worth a second look: (a) log
/// `sigma_raw`; (b) fix `vel_to_ball` to drop the vertical component while
/// airborne (a regime change -- it moves a term paying ~37/match on two live
/// runs, so it needs a ~50% contested start); (c) pin the champion's and the
/// foreign bots' `air_hang_events`/min/car with THIS instrument, which is what
/// would say whether an outcome gate (one payment per air phase whose peak car z
/// exceeds ~450, above the measured 393.6 double-jump ceiling) can bootstrap off
/// the 1.67% of airborne touches already above 300 uu.
pub const T_AIR_HANG: usize = 23;
/// Instrument: qualifying hang crossings, independent of `cfg.air_hang`, so
/// retuning the payout cannot move the ruler it would be judged by. Same split
/// as `T_BOOST_PICKUP`/`T_BOOST_GAINED` and `aerial_z_lo`/`aerial_meas_z_lo`.
///
/// A LOWER BOUND on real qualifying phases, like `T_FLIP_EVENTS`: `compute_terms`
/// runs once per decision against a state 8 ticks old, so the crossing is
/// observed on the decision boundary after it happened. It cannot over-count --
/// the clock is monotone within a phase and 0 across a landing.
///
/// This is the counter to read, because nothing else in the log can see this
/// behaviour class. `flips/min/car` reads exactly 0.00 through every hop-spam
/// tape measured (a minimum hop sets neither `has_flipped` nor
/// `has_double_jumped`; see `T_FLIP_EVENTS`), and `airborne_touch_events` scores
/// 0.87-0.93 for a RANDOM policy.
pub const T_AIR_HANG_EVENTS: usize = 24;
/// Instrument: decisions on which the car was airborne. Duty cycle, against the
/// `T_AGENT_STEPS` denominator this array already carries.
///
/// The crossing count alone cannot separate a pogo from air play -- a
/// 93.2%-airborne hop farm and a bot that flies both raise it. Duty is the axis
/// that separates them: measured 92.1-93.2% for every spam tape, against a
/// learner baseline of 3.8-15% and foreign opponents at 19-23%. Read the pair,
/// never either alone.
pub const T_AIR_DECISIONS: usize = 25;
/// Seconds of POST-JUMP hang time that define a qualifying air phase.
///
/// 1.2 sits in the gap between the two hop families, which is the only reason a
/// duration gate discriminates at all: the cheapest hop (jump released after one
/// 8-tick decision) has a 1.13 s air phase, and over 200 measured phases 98% of
/// its post-jump clocks peaked below 1.0 s and none reached 1.2 s, while a
/// full-hold hop (0.2 s = `JUMP_MAX_TIME`) measures 1.40 s on its first phase.
/// A RAMP is not available here -- the quantity is a one-shot crossing -- so the
/// threshold IS load-bearing, unlike `aerial_z_lo`.
///
/// It is also a WEAK discriminator, and the margin is one decision wide: 1.2 s is
/// 18 decisions, and the two families are 4-6 decisions apart. Anything that
/// changes `tick_skip`, gravity, or the car config moves both sides of that gap
/// and this number has to be re-measured (`engine/examples/hang_gate_probe.rs`
/// prints the per-phase distribution for every T in one run).
///
/// Deliberately a constant and not a config field: it is the instrument's
/// definition, and the project's own rule (`aerial_meas_z_lo`) is that the ruler
/// must not move when the payout is retuned. Changing this number invalidates
/// every `air_hang_events` baseline ever recorded.
pub const AIR_HANG_T: f32 = 1.2;
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
    "gated_aerial_touch_events",
    "air_touch_z_lt150",
    "air_touch_z_150_300",
    "air_touch_z_300_500",
    "air_touch_z_500_800",
    "air_touch_z_800_1200",
    "air_touch_z_ge1200",
    "boost_pickup",
    "boost_gained",
    "flip_events",
    "air_hang",
    "air_hang_events",
    "air_decisions",
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

/// The floor the `gated_aerial_touch_events` COUNTER uses. Falls back to the
/// payout's own floor, so a tape that does not set it reads exactly as before.
#[inline]
fn meas_z_lo(cfg: &RewardConfig) -> f32 {
    if cfg.aerial_meas_z_lo > 0.0 { cfg.aerial_meas_z_lo } else { cfg.aerial_z_lo }
}

/// The ramp `air_setup`'s potential uses. Falls back to the shared
/// `height_ramp`, so a tape that does not set the pair is bit-identical.
/// `validate_aerial` guarantees `hi > lo` whenever the override is live, for
/// the same inf/NaN reason `height_ramp` documents.
#[inline]
fn setup_ramp(ball_z: f32, cfg: &RewardConfig) -> f32 {
    if cfg.air_setup_z_hi > cfg.air_setup_z_lo {
        ((ball_z - cfg.air_setup_z_lo) / (cfg.air_setup_z_hi - cfg.air_setup_z_lo))
            .clamp(0.0, 1.0)
    } else {
        height_ramp(ball_z, cfg)
    }
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
        // The measurement floor is allowed to sit ANYWHERE at or above the ball
        // radius, including above `aerial_z_hi` (a deliberately strict ruler is
        // a legitimate thing to want). Below the radius it would count a ball
        // rolling on the floor as an aerial, which is the same failure the
        // payout floor is guarded against -- and an instrument that lies is
        // worse than a payout that leaks, because every later decision is made
        // through it.
        if self.aerial_meas_z_lo != 0.0 && self.aerial_meas_z_lo < BALL_RADIUS {
            return Err(format!(
                "aerial_meas_z_lo={} is below the ball radius ({BALL_RADIUS}): the \
                 gated-aerial COUNTER would score a ball resting on the floor as an \
                 aerial touch",
                self.aerial_meas_z_lo));
        }
        // Half-configured air_setup ramp: same inf/NaN trap as the shared ramp,
        // and `setup_ramp` only takes the override when `hi > lo`, so a lone
        // `air_setup_z_lo` would silently fall back to the shared ramp and the
        // run would report a split that is not happening.
        let setup_ramp_set = self.air_setup_z_lo != 0.0 || self.air_setup_z_hi != 0.0;
        if setup_ramp_set && !(self.air_setup_z_hi > self.air_setup_z_lo) {
            return Err(format!(
                "air_setup_z_lo={} / air_setup_z_hi={}: setting either demands \
                 hi > lo. A lone bound does not error into a NaN here -- it falls \
                 back to the SHARED ramp, so the run would look split and not be",
                self.air_setup_z_lo, self.air_setup_z_hi));
        }
        if setup_ramp_set && self.air_setup_z_lo < BALL_RADIUS {
            return Err(format!(
                "air_setup_z_lo={} is below the ball radius ({BALL_RADIUS}): PHI \
                 would be nonzero for a car airborne over a ball on the floor",
                self.air_setup_z_lo));
        }
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

    // Boost pickup + flip instrument. Both read `prev`'s matching car, so they
    // sit here rather than in the `touched` block -- neither depends on contact.
    {
        let was = &prev.cars[car_idx];
        // A DEMO RESPAWN is not a pad, and this guard is the only thing that says
        // so. `Car::Demolish` leaves boost untouched, then `Car::Respawn` writes
        // `carSpawnBoostAmount` = BOOST_SPAWN_AMOUNT = 33.333 MID-EPISODE from
        // `_PreTickUpdate` with no reset -- and `prev_state` is only re-latched
        // inside `reset_episode`, so that +33.3 jump lands inside an ordinary
        // (prev, cur) pair. Unguarded, a car demoed at 0 boost is paid
        // 0.2*sqrt(33.333)/10 = 0.115, i.e. 58% of a big pad, FOR BEING
        // DEMOLISHED, and `boost_gained` over-reads by up to 33.3 units per demo.
        // Demos are on by default (DemoMode::NORMAL) and this engine never
        // overrides the mutator config. On the respawn window `prev` is demoed and
        // `cur` is not, so this excludes exactly that window and nothing else: the
        // demo window itself has `gained == 0` because Demolish preserves boost.
        if !was.state.is_demoed {
            let gained = me.state.boost - was.state.boost;
            if gained > 0.0 {
                terms[T_BOOST_GAINED] += gained as f64;
                if cfg.boost_pickup != 0.0 {
                    // SQRT, not linear: boost has diminishing marginal value, so a
                    // pad must pay in proportion to how much it was NEEDED. A pad
                    // taken at 88 boost pays ~1/16 of the same pad taken empty, and
                    // a full car gains 0 so it pays 0. Linear `gained/100` prices
                    // 0->20 and 80->100 identically, which is wrong.
                    // /10 normalises sqrt(100) = 10 to a [0,1] range.
                    //
                    // `max(0.0)` is NaN containment, not decoration:
                    // `apply_replay_state` writes `cs.boost = sp.boost * 100.0`
                    // unclamped and the replay filter (`reset_pool` F6) checks
                    // FINITENESS only, so a negative replay boost reaches here.
                    // sqrt of it is NaN, and since `prev_state` is latched to the
                    // post-reset state the NaN would sit in BOTH states and be paid
                    // every step for the whole episode -- silent poisoning, the
                    // failure `height_ramp`'s doc comment was written against.
                    let g = (me.state.boost.max(0.0).sqrt()
                        - was.state.boost.max(0.0).sqrt())
                        / 10.0;
                    if g > 0.0 {
                        // Not dead code despite `gained > 0.0`: near b = 100 a gain
                        // of 7.6e-6 is sqrt-equal in f32 and yields g == 0.0.
                        let b = cfg.boost_pickup * g;
                        r += b;
                        terms[T_BOOST_PICKUP] += b as f64;
                    }
                }
            }
        }
        // RISING edge only: `has_flipped` stays true until >=3 wheels contact a
        // surface (floor, wall or ceiling), which is the only thing that clears
        // it, so testing the LEVEL would count one flip every decision of the
        // air time. A demo cannot fabricate an edge: Demolish freezes the flags
        // and Respawn rebuilds a default `CarState` with has_flipped = false.
        if me.state.has_flipped && !was.state.has_flipped {
            terms[T_FLIP_EVENTS] += 1.0;
        }
        // Airborne duty (instrument). Denominator is T_AGENT_STEPS, incremented
        // once per call above.
        //
        // The `!is_demoed` guard is LOAD-BEARING and was missing on first write.
        // A demoed car reads is_on_ground = false for the whole ~3 s respawn
        // delay, so without it every airborne demolition injects ~45 decisions
        // of fake flight at tick_skip = 8 -- and duty is the DENOMINATOR that
        // decides whether a hang rate means "flies" or "hops constantly", so
        // inflating it silently deflates the thing it exists to normalise.
        // Demos are on by default (DemoMode::NORMAL) and this engine never
        // overrides the mutator config.
        //
        // KNOWN RESIDUAL, not fixed here: a car that lands on its ROOF also
        // reads is_on_ground = false indefinitely, so it still counts as
        // airborne. Detecting it needs an orientation or wheel-contact test I
        // cannot validate against the frozen gate .so, so it is documented
        // rather than guessed at. It inflates duty upward, which makes any hang
        // RATE computed against it conservative (too low) -- the safe direction
        // for a farm tripwire.
        if !me.state.is_on_ground && !me.state.is_demoed {
            terms[T_AIR_DECISIONS] += 1.0;
        }
        // HANG CROSSING. RISING edge of the POST-JUMP clock through AIR_HANG_T,
        // which fires at most once per air phase as a physics property -- see
        // T_AIR_HANG for the proof and for why the coefficient is 0.0.
        //
        // `!me.state.is_on_ground` is NOT redundant. RocketSim increments this
        // clock in `_PreTickUpdate` (from the previous tick's is_on_ground) and
        // recomputes is_on_ground in `_PostTickUpdate`, so the LANDING tick can
        // read is_on_ground = true with a still-positive clock. Without the
        // guard, a hop whose clock crosses on exactly the tick it touches down
        // would be paid for hang time it no longer has.
        //
        // No demo guard is needed, unlike `boost_pickup` above. A demoed car
        // early-returns from both tick-update halves, so its clock FREEZES; a
        // frozen value cannot rise, and `Car::Respawn` rebuilds a default
        // CarState (clock 0), which can only produce a falling edge.
        if !me.state.is_on_ground
            && was.state.air_time_since_jump < AIR_HANG_T
            && me.state.air_time_since_jump >= AIR_HANG_T
        {
            terms[T_AIR_HANG_EVENTS] += 1.0;
            if cfg.air_hang != 0.0 {
                r += cfg.air_hang;
                terms[T_AIR_HANG] += cfg.air_hang as f64;
            }
        }
    }

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
            // Ball-height histogram over EXACTLY this population, so the six
            // buckets sum to T_AIRBORNE_TOUCH_EVENTS -- see T_AIR_TOUCH_Z0.
            // Unconditional on the ramp: this is the distribution you consult
            // to CHOOSE a floor, so it must not presuppose one.
            let b = AIR_TOUCH_Z_EDGES.iter().filter(|e| cur.ball.pos.z >= **e).count();
            terms[T_AIR_TOUCH_Z0 + b] += 1.0;
            // `aerial_touch`'s own gate, minus the cooldown -- see
            // T_AERIAL_TOUCH_EVENTS. The `hi > lo` test is what keeps this at
            // zero on a tape with no ramp configured rather than counting every
            // airborne touch through a 0/0 height_ramp. The THRESHOLD is
            // `meas_z_lo`, not `aerial_z_lo`, so the payout ramp can move
            // without moving the instrument that scores it.
            if cfg.aerial_z_hi > cfg.aerial_z_lo && cur.ball.pos.z > meas_z_lo(cfg) {
                terms[T_AERIAL_TOUCH_EVENTS] += 1.0;
            }
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
    setup_ramp(ball_pos[2], cfg) * (1.0 - dist / 2500.0).clamp(0.0, 1.0)
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

    /// `[ppo] gamma` read out of a TRAIN config. The reward tape and the trainer
    /// are two files that have to agree on one number and nothing in the loader
    /// makes them: `air_setup`'s potential is discounted by `win_prob_gamma`, so
    /// if that drifts from the discount GAE actually uses, the shaping no longer
    /// telescopes against the return it is shaping and the run optimises a
    /// different objective with every log line looking normal.
    fn trainer_gamma(path: &str) -> f32 {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let v: toml::Value = text.parse().unwrap_or_else(|e| panic!("{path}: {e}"));
        v.get("ppo")
            .and_then(|p| p.get("gamma"))
            .and_then(|g| g.as_float())
            .unwrap_or_else(|| panic!("{path}: no [ppo] gamma")) as f32
    }

    /// The `air_setup` CO-REQUISITE, as a predicate so the design test and its
    /// own negative case share one implementation rather than two copies that
    /// can drift apart.
    ///
    /// A weight of zero needs nothing (`air_shaping` returns a literal 0.0). A
    /// NONZERO weight needs a gamma that is both present and the trainer's:
    /// `validate()` already refuses gamma <= 0 outright, but it cannot see the
    /// train config, so "0.99 next to a trainer running 0.9954" is the failure
    /// mode that survives every existing check.
    fn air_setup_is_consistent(c: &RewardConfig, trainer_gamma: f32) -> bool {
        c.air_setup == 0.0
            || (c.win_prob_gamma > 0.0 && (c.win_prob_gamma - trainer_gamma).abs() < 1e-6)
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
        assert_eq!(c.aerial_meas_z_lo, 400.0,
                   "the counter's ruler is pinned to where the payout ramp is TODAY, \
                    so a later move of aerial_z_lo cannot silently move it too");
        // The PHI ramp reaches down to where the ball actually is at an airborne
        // touch (76.8% under 150uu, measured); the PAYOUT ramp deliberately does
        // not, because paying there is a hop-tap farm.
        assert_eq!(c.air_setup_z_lo, 100.0);
        assert_eq!(c.air_setup_z_hi, 1400.0);
        assert!(c.air_setup_z_lo < c.aerial_z_lo,
                "the whole point: PHI must have gradient below the payout floor");
        assert_eq!(c.team_spirit, 0.3);
        assert_eq!(c.opp_spirit, 0.0);
        // `air_setup` USED TO BE PINNED AT 0.0 here. That assertion encoded a
        // schedule, not a design: the term is a STAGED lever, documented in the
        // config to switch on when the airborne-touch fraction stalls below ~5%
        // at 150M steps, and it was pulled on that trigger at 222M (0.0 -> 0.3).
        // A literal that has to be edited every time the value legitimately
        // moves teaches the next reader to edit the number instead of checking
        // the design -- the opposite of what a guard is for.
        //
        // What must NEVER drift is the co-requisite, so pin THAT instead: a
        // nonzero air_setup requires a positive win_prob_gamma (validate() gets
        // that far) that also EQUALS the trainer's [ppo] gamma. Nothing else in
        // the repo compares those two files, and the failure is silent -- the
        // potential telescopes against the wrong discount and the run optimises
        // something other than the return it is shaping.
        let g = trainer_gamma("../configs/train_v9_fromscratch.toml");
        assert!(
            air_setup_is_consistent(&c, g),
            "air_setup={} needs win_prob_gamma == the trainer's [ppo] gamma {g}, got {}",
            c.air_setup, c.win_prob_gamma
        );
    }

    #[test]
    fn air_setup_consistency_check_catches_a_gamma_drift() {
        // Teeth for the assertion above: it must reject the two ways the pair
        // can disagree, and stay out of the way when the lever is staged off.
        let trainer = 0.9954f32;
        let mut c = v9_cfg();
        c.air_setup = 0.3;
        c.win_prob_gamma = trainer;
        assert!(air_setup_is_consistent(&c, trainer), "the shipped pairing must pass");
        c.win_prob_gamma = 0.99;
        assert!(!air_setup_is_consistent(&c, trainer),
                "a gamma that is merely DIFFERENT is the silent failure -- must be caught");
        c.win_prob_gamma = 0.0;
        assert!(!air_setup_is_consistent(&c, trainer), "a missing gamma must be caught");
        // staged off: the gamma is then nobody's business
        c.air_setup = 0.0;
        assert!(air_setup_is_consistent(&c, trainer));
    }

    /// One car, airborne, forged contact, ball at `ball_z`. Returns the counters.
    fn airborne_touch_terms(cfg: &RewardConfig, ball_z: f32) -> [f64; N_TERMS] {
        let (p, c) = synth(
            [0.0, 0.0, ball_z], [0.0, 0.0, 0.0], [0.0, 1800.0, 0.0],
            &[(Team::Blue, [0.0, -200.0, ball_z - 100.0], true)],
        );
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&p, &c, 0, None, cfg, &mut t);
        assert!(t[T_TOUCH_EVENTS] > 0.0, "precondition: the forged contact registers");
        t
    }

    #[test]
    fn the_gated_aerial_counter_separates_real_aerials_from_airborne_contact() {
        // WHY THIS COUNTER EXISTS. `airborne_touch_events` reads 0.87-0.93 on a
        // RANDOM policy (the curriculum starts half the cars in the air, and a
        // tumbling car's contacts are all "airborne") and 0.0117 on the
        // champion, which makes 63x more real aerial touches -- it is
        // ANTI-CORRELATED with the skill it was being read as measuring. The
        // missing half of the gate is the height.
        let cfg = v9_cfg();          // aerial_z_lo = 400
        let high = airborne_touch_terms(&cfg, 1000.0);
        assert_eq!(high[T_AIRBORNE_TOUCH_EVENTS], 1.0);
        assert_eq!(high[T_AERIAL_TOUCH_EVENTS], 1.0, "a real high strike must count");
        assert!(high[T_AERIAL_TOUCH] > 0.0, "and the term itself pays for it");

        // The champion's ENTIRE airborne-touch population: off the ground, but
        // on a ball under the ramp. The old counter says 1; the new one says 0,
        // which is the number that matches its measured r_aerial_touch share of
        // exactly 0.0000 over 24,000 learner steps.
        let low = airborne_touch_terms(&cfg, 300.0);
        assert_eq!(low[T_AIRBORNE_TOUCH_EVENTS], 1.0,
                   "the old counter cannot tell these two apart -- that is the bug");
        assert_eq!(low[T_AERIAL_TOUCH_EVENTS], 0.0);
        assert_eq!(low[T_AERIAL_TOUCH], 0.0, "the reward gate agrees: h = 0 here");
    }

    /// The six buckets must partition the airborne-touch population exactly.
    /// That identity IS the instrument's self-check: if they ever stop summing
    /// to `T_AIRBORNE_TOUCH_EVENTS`, the histogram is measuring some other set
    /// of touches than the one its readers think, and a z-floor chosen from it
    /// would be anchored to the wrong distribution -- the precise failure that
    /// made the unsourced "91.6% below 400uu" figure worthless (it was over ALL
    /// touches when only airborne ones can ever be paid).
    fn z_hist(t: &[f64; N_TERMS]) -> [f64; 6] {
        let mut h = [0.0; 6];
        h.copy_from_slice(&t[T_AIR_TOUCH_Z0..=T_AIR_TOUCH_Z5]);
        h
    }

    #[test]
    fn the_touch_height_histogram_partitions_the_airborne_population() {
        let cfg = v9_cfg();
        // One touch per bucket, at heights that are unambiguously inside it.
        for (z, want) in [(100.0, 0), (200.0, 1), (400.0, 2),
                          (600.0, 3), (1000.0, 4), (1500.0, 5)] {
            let t = airborne_touch_terms(&cfg, z);
            let h = z_hist(&t);
            assert_eq!(h[want], 1.0, "ball z={z} belongs in bucket {want}, got {h:?}");
            assert_eq!(h.iter().sum::<f64>(), t[T_AIRBORNE_TOUCH_EVENTS],
                       "the buckets must sum to the population they partition");
        }
    }

    #[test]
    fn the_histogram_bucket_edges_are_half_open_from_below() {
        // Exactly ON an edge lands in the HIGHER bucket, so the edges read as
        // [lo, hi) and no touch is double-counted or dropped.
        let cfg = v9_cfg();
        for (z, want) in [(149.9, 0), (150.0, 1), (299.9, 1), (300.0, 2),
                          (1199.9, 4), (1200.0, 5)] {
            assert_eq!(z_hist(&airborne_touch_terms(&cfg, z))[want], 1.0,
                       "ball z={z} must land in bucket {want}");
        }
    }

    #[test]
    fn the_histogram_counts_touches_the_aerial_gate_rejects() {
        // The whole point: air_gate_frac says how MANY airborne touches clear
        // the ramp, and is silent about how far the rest fall short. A 300uu
        // touch pays nothing and gates nothing, and must still be visible --
        // otherwise the distribution you consult to choose a floor is censored
        // by the floor you already have.
        let cfg = v9_cfg();                       // aerial_z_lo = 400
        let t = airborne_touch_terms(&cfg, 300.0);
        assert_eq!(t[T_AERIAL_TOUCH_EVENTS], 0.0, "precondition: gated out");
        assert_eq!(t[T_AERIAL_TOUCH], 0.0, "precondition: paid nothing");
        assert_eq!(z_hist(&t)[2], 1.0, "and yet it is counted, in 300-500");
    }

    #[test]
    fn the_histogram_is_live_on_a_tape_with_no_ramp() {
        // Unlike the gated counter, this one does NOT need a configured ramp:
        // it presupposes no floor, so it is exactly the instrument a zero-weight
        // probe tape needs. reward_v0 is what every gate and bench scores on.
        let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
        let t = airborne_touch_terms(&cfg, 1000.0);
        assert_eq!(t[T_AERIAL_TOUCH_EVENTS], 0.0, "no ramp, no gate -- unchanged");
        assert_eq!(z_hist(&t)[4], 1.0, "but the distribution is still measurable");
    }

    #[test]
    fn the_measurement_floor_defaults_to_the_payout_floor() {
        // Every tape written before 2026-07-27 leaves aerial_meas_z_lo at 0.0
        // and MUST read bit-identically -- the 1,786-iteration air_gate_frac
        // baseline is the only thing a future ramp change can be judged against.
        // Zeroed HERE rather than read off the shipped toml: this pins the
        // INHERIT behaviour, which must hold for every historical tape whatever
        // configs/reward_v9_aerial.toml happens to say today.
        let mut cfg = v9_cfg();
        cfg.aerial_meas_z_lo = 0.0;
        let before = airborne_touch_terms(&cfg, 500.0)[T_AERIAL_TOUCH_EVENTS];
        cfg.aerial_meas_z_lo = cfg.aerial_z_lo;     // stating the default explicitly
        assert_eq!(airborne_touch_terms(&cfg, 500.0)[T_AERIAL_TOUCH_EVENTS], before);
    }

    #[test]
    fn the_measurement_floor_survives_a_payout_ramp_move() {
        // THE reason the field exists. Drop the payout floor to 250 -- the
        // change under consideration -- and the counter pinned at 400 must not
        // move, or a rise in air_gate_frac would be partly definitional and the
        // baseline would be destroyed.
        let mut cfg = v9_cfg();
        cfg.aerial_meas_z_lo = 400.0;
        let pinned = airborne_touch_terms(&cfg, 300.0)[T_AERIAL_TOUCH_EVENTS];
        assert_eq!(pinned, 0.0, "300 is under the pinned ruler");

        cfg.aerial_z_lo = 250.0;                    // the payout ramp moves...
        let t = airborne_touch_terms(&cfg, 300.0);
        assert!(t[T_AERIAL_TOUCH] != 0.0, "...so the term now pays at 300uu...");
        assert_eq!(t[T_AERIAL_TOUCH_EVENTS], 0.0,
                   "...and the INSTRUMENT must be exactly where it was");
    }

    #[test]
    fn the_air_setup_ramp_defaults_to_the_shared_ramp() {
        // Zeroed here, not read off the shipped toml -- see the measurement-floor
        // test above. Every tape written before 2026-07-27 leaves these unset and
        // must keep reading through the shared ramp bit-for-bit.
        let mut cfg = v9_cfg();
        cfg.air_setup_z_lo = 0.0;
        cfg.air_setup_z_hi = 0.0;
        for z in [0.0, 200.0, 400.0, 700.0, 1400.0, 2000.0] {
            assert_eq!(setup_ramp(z, &cfg), height_ramp(z, &cfg),
                       "unset means inherit, bit-for-bit, at z={z}");
        }
    }

    #[test]
    fn the_air_setup_ramp_can_reach_below_the_payout_floor() {
        // The structural finding no aerial hypothesis had: with the shared ramp,
        // PHI is identically 0 below aerial_z_lo, so the term whose job is
        // teaching the TAKEOFF DECISION has zero gradient in every state where
        // takeoff is decided. Splitting the ramps is what makes that reachable
        // -- without touching the payout, so the two are separable variables.
        let mut cfg = v9_cfg();
        assert_eq!(height_ramp(200.0, &cfg), 0.0, "precondition: flat at takeoff height");
        cfg.air_setup_z_lo = 100.0;
        cfg.air_setup_z_hi = 1400.0;
        assert!(setup_ramp(200.0, &cfg) > 0.0, "now there is a gradient to climb");
        assert_eq!(height_ramp(200.0, &cfg), 0.0, "and the payout ramp did not move");
    }

    #[test]
    fn a_half_configured_air_setup_ramp_is_rejected() {
        // It would not NaN -- it would silently fall back to the shared ramp,
        // so the run would report a split that is not happening. That is worse
        // than a crash: it is a config whose log looks like the experiment you
        // meant to run.
        let mut cfg = v9_cfg();
        cfg.air_setup_z_hi = 0.0;                   // start from unset
        cfg.air_setup_z_lo = 300.0;                 // ...and set only the low bound
        assert!(cfg.validate().is_err(), "a lone bound must not sail through");

        cfg.air_setup_z_hi = 1400.0;
        assert!(cfg.validate().is_ok());

        cfg.air_setup_z_lo = 50.0;                  // under the ball radius
        assert!(cfg.validate().is_err(), "PHI must not pay for a ball on the floor");
    }

    #[test]
    fn a_measurement_floor_under_the_ball_radius_is_rejected() {
        let mut cfg = v9_cfg();
        cfg.aerial_meas_z_lo = 50.0;
        assert!(cfg.validate().is_err(),
                "an instrument that scores a resting ball as an aerial poisons every \
                 decision made through it");
    }

    #[test]
    fn the_gated_aerial_counter_is_zero_on_a_tape_with_no_ramp() {
        // reward_v0 -- the tape the gate, the benches, the league and the
        // auto-curriculum eval all score with -- leaves aerial_z_lo/hi at 0,
        // where `height_ramp` is a 0/0 inf that clamps to 1.0. Without the
        // `hi > lo` test this counter would silently equal the airborne one on
        // every instrument in the repo.
        let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
        assert_eq!((cfg.aerial_z_lo, cfg.aerial_z_hi), (0.0, 0.0), "precondition");
        let t = airborne_touch_terms(&cfg, 1000.0);
        assert_eq!(t[T_AIRBORNE_TOUCH_EVENTS], 1.0);
        assert_eq!(t[T_AERIAL_TOUCH_EVENTS], 0.0,
                   "a tape with no ramp defines no aerial gate; it must not guess one");
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
        // The episode-level call site is UNCONDITIONAL, so this exact-zero
        // return is what keeps the per-step reward of every config that does not
        // set air_setup byte-identical -- which is every config but
        // reward_v9_aerial.toml, and v9 itself for its first 222M steps.
        // The weight is FORCED here rather than read off the shipped file: v9's
        // is a staged lever and is 0.3 today, so asserting the file was zero
        // tested the schedule instead of the property.
        let mut cfg = v9_cfg();
        cfg.air_setup = 0.0;
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

        // The paid terms STOPPED being the contiguous `T_GOAL..=T_AERIAL_TOUCH`
        // prefix when `boost_pickup` was appended at index 20 (renumbering is
        // forbidden -- TERM_NAMES is positional). Exercise it with a NONZERO
        // coefficient and a real boost delta, or this identity would hold for the
        // new term only because the term is 0 on every shipped tape.
        let bcfg = RewardConfig { boost_pickup: 0.2, ..v9_cfg() };
        let (mut bprev, mut bcur) = boost_pair();
        bprev.cars[0].state.boost = 0.0;
        bcur.cars[0].state.boost = 60.0;
        let mut bt = [0.0f64; N_TERMS];
        let br = compute_terms(&bprev, &bcur, 0, Some(Team::Blue), &bcfg, &mut bt);
        assert!(bt[T_BOOST_PICKUP] > 0.0,
                "precondition: the appended reward term must actually pay in this fixture");
        let bsum: f64 =
            bt[T_GOAL..=T_AERIAL_TOUCH].iter().sum::<f64>() + bt[T_BOOST_PICKUP];
        assert!((bsum - br as f64).abs() < 1e-5, "terms {bsum} != total {br}");

        // Same again for the SECOND non-prefix reward slot, `air_hang` at index
        // 23. Exercised with a nonzero coefficient for the same reason: on every
        // shipped tape it is 0.0, so the identity would hold vacuously.
        let hcfg = RewardConfig { air_hang: 0.008, ..v9_cfg() };
        let (hprev, hcur) = hang_pair(AIR_HANG_T - 0.1, AIR_HANG_T + 0.05);
        let mut ht = [0.0f64; N_TERMS];
        let hr = compute_terms(&hprev, &hcur, 0, Some(Team::Blue), &hcfg, &mut ht);
        assert!(ht[T_AIR_HANG] > 0.0,
                "precondition: the appended reward term must actually pay here");
        let hsum: f64 = ht[T_GOAL..=T_AERIAL_TOUCH].iter().sum::<f64>()
            + ht[T_BOOST_PICKUP]
            + ht[T_AIR_HANG];
        assert!((hsum - hr as f64).abs() < 1e-5, "terms {hsum} != total {hr}");
    }

    // ------------------------------------------------------ boost / flip (v10) --

    /// Zero-arg `(prev, cur)` fixture for the boost/flip tests: one AIRBORNE blue
    /// car with a forged contact, ball centre at 800uu. A thin wrapper over
    /// `synth` -- these tests are about a car whose `state` they can write, not
    /// about physics, and a stepped arena cannot hold a chosen boost value across
    /// `step()`. NOTE: `synth` writes `cur = prev.clone()` AFTER setting up prev,
    /// so a test must write BOTH `prev.cars[0].state.x` and `cur.cars[0].state.x`.
    fn boost_pair() -> (GameState, GameState) {
        synth(
            [0.0, 0.0, 800.0], [0.0, 0.0, 0.0], [0.0, 1800.0, 0.0],
            &[(Team::Blue, [0.0, -200.0, 700.0], true)],
        )
    }

    /// `boost_pickup` alone for one `b0 -> b1` boost step at coefficient 0.2.
    /// `RewardConfig::default()` is the real "all coefficients zero" tape (the
    /// struct derives Default over f32/u32 only), so this isolates the new term:
    /// `touch`/`touch_accel`/`aerial_touch` are all 0 even though `synth` forges a
    /// live airborne contact.
    fn boost_pay(b0: f32, b1: f32) -> f64 {
        let (mut prev, mut cur) = boost_pair();
        prev.cars[0].state.boost = b0;
        cur.cars[0].state.boost = b1;
        let cfg = RewardConfig { boost_pickup: 0.2, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        t[T_BOOST_PICKUP]
    }

    /// A pad pickup is a POSITIVE boost delta, scaled by SQRT so it pays only
    /// in proportion to how much the boost was NEEDED.
    #[test]
    fn boost_pickup_pays_on_sqrt_of_amount_gained() {
        let (mut prev, mut cur) = boost_pair();
        prev.cars[0].state.boost = 0.0;
        cur.cars[0].state.boost = 100.0;
        let cfg = RewardConfig { boost_pickup: 0.2, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert!((t[T_BOOST_PICKUP] - 0.2).abs() < 1e-6,
                "big pad from empty pays the full coefficient; got {}", t[T_BOOST_PICKUP]);
        assert!((t[T_BOOST_GAINED] - 100.0).abs() < 1e-6,
                "instrument counts RAW units, not the sqrt-shaped reward");

        // small pad from empty: (sqrt(12) - 0)/10 = 0.3464
        let mut t2 = [0.0f64; N_TERMS];
        cur.cars[0].state.boost = 12.0;
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t2);
        assert!((t2[T_BOOST_PICKUP] - 0.2 * 0.34641).abs() < 1e-5,
                "small pad from empty; got {}", t2[T_BOOST_PICKUP]);
    }

    /// The whole point of sqrt: a pad taken when nearly full must pay almost
    /// nothing, and a pad taken when FULL must pay exactly nothing. A linear
    /// `gained/100` gets the full case right but prices 0->20 and 80->100 the
    /// same, which is wrong -- the first is desperate, the second worthless.
    #[test]
    fn a_pad_pays_only_to_the_extent_the_boost_was_needed() {
        // ALREADY FULL: driving over a pad gains nothing, so it pays nothing.
        assert_eq!(boost_pay(100.0, 100.0), 0.0, "a full car gains nothing from a pad");

        // NEARLY full: a big pad is worth ~1/16 of the same pad taken empty
        // (measured in f32: g(88,100) = 0.061917, g(0,100) = 1.0 -> 1/16.15).
        let empty_big = boost_pay(0.0, 100.0);
        let nearly_full_big = boost_pay(88.0, 100.0);
        assert!(nearly_full_big > 0.0, "12 units gained is still worth something");
        assert!(nearly_full_big < empty_big / 10.0,
                "a pad at 88 boost must pay <1/10 of the same pad at 0: {nearly_full_big} vs {empty_big}");

        // A SMALL pad while empty must beat a BIG pad while nearly full --
        // this is the ordering a linear scaling gets backwards.
        assert!(boost_pay(0.0, 12.0) > nearly_full_big,
                "need beats quantity: small-pad-when-empty must outpay big-pad-when-full");
    }

    /// Spending boost must NOT be penalised. This is the whole reason the term
    /// is an EVENT and not a potential: a potential on boost amount yields a
    /// hoarder, the same gamma-drag trap that left air_setup net negative.
    #[test]
    fn spending_boost_is_never_penalised() {
        let (mut prev, mut cur) = boost_pair();
        prev.cars[0].state.boost = 100.0;
        cur.cars[0].state.boost = 40.0;
        let cfg = RewardConfig { boost_pickup: 0.2, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_BOOST_PICKUP], 0.0, "burning boost must cost nothing");
        assert_eq!(t[T_BOOST_GAINED], 0.0, "and must not register as a pickup");
    }

    /// A tape that does not set boost_pickup is bit-identical to today.
    #[test]
    fn boost_pickup_defaults_to_zero_reward() {
        let (mut prev, mut cur) = boost_pair();
        prev.cars[0].state.boost = 0.0;
        cur.cars[0].state.boost = 100.0;
        let cfg = RewardConfig::default(); // boost_pickup untouched -> 0.0
        let mut t = [0.0f64; N_TERMS];
        let r = compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_BOOST_PICKUP], 0.0, "unset coefficient pays nothing");
        assert_eq!(r, 0.0, "an all-zero tape stays byte-identical across a pickup");
        assert!((t[T_BOOST_GAINED] - 100.0).abs() < 1e-6,
                "but the INSTRUMENT still records, so v9 telemetry is comparable");
    }

    /// SPLITTING a monotone climb earns nothing extra: the sqrt deltas telescope,
    /// so nine small pads from empty to full pay exactly what one big pad pays.
    /// This is the anti-exploit property that makes the sqrt form safe for the
    /// only path it is safe on.
    #[test]
    fn a_monotone_climb_pays_the_same_however_it_is_split() {
        let ladder = [0.0f32, 12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 96.0, 100.0];
        let split: f64 = ladder.windows(2).map(|w| boost_pay(w[0], w[1])).sum();
        let single = boost_pay(0.0, 100.0);
        assert!((split - single).abs() < 1e-6,
                "pad-splitting on a monotone climb must be worth nothing: {split} vs {single}");
    }

    /// PINNED HAZARD, not a desirable property. Because spending is deliberately
    /// never charged, the term is a RATCHET: burning boost resets the baseline, so
    /// the same 100 units re-collected in 12-unit chunks from empty pay 2.89x what
    /// one 0->100 refill pays. The burn rate (33.33 boost/s) caps this at
    /// 0.1925 reward/s = 57.7 per 300s match at coefficient 0.2, against a stated
    /// shaping budget of 4-10 per match. If this test ever changes, the farm
    /// arithmetic in `T_BOOST_PICKUP`'s doc comment must be recomputed with it.
    #[test]
    fn burn_and_recollect_out_earns_one_full_refill_per_boost_unit() {
        let per_unit_small = boost_pay(0.0, 12.0) / 12.0;
        let per_unit_big = boost_pay(0.0, 100.0) / 100.0;
        assert!((per_unit_small / per_unit_big - 2.887).abs() < 0.01,
                "the spend-and-refill rate advantage is sqrt(100/12) = 2.887; got {}",
                per_unit_small / per_unit_big);
        assert!(boost_pay(0.0, 12.0) * 8.0 > boost_pay(0.0, 100.0) * 2.0,
                "eight refills from empty out-earn two full tanks -- the farm is real");
    }

    /// A DEMO RESPAWN is not a pad. `Car::Respawn` writes boost = 33.333
    /// mid-episode with no reset, so the (prev, cur) pair straddles a +33.3 jump
    /// with no pad involved -- worth 58% of a big pad, for being demolished.
    #[test]
    fn a_demo_respawn_is_not_a_pad_pickup() {
        let (mut prev, mut cur) = boost_pair();
        prev.cars[0].state.is_demoed = true;
        prev.cars[0].state.boost = 0.0;
        cur.cars[0].state.is_demoed = false;
        cur.cars[0].state.boost = 100.0 / 3.0; // BOOST_SPAWN_AMOUNT
        let cfg = RewardConfig { boost_pickup: 0.2, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_BOOST_PICKUP], 0.0, "respawn boost must not pay -- no pad was taken");
        assert_eq!(t[T_BOOST_GAINED], 0.0, "and must not inflate the instrument by 33.3 units");
    }

    /// A negative boost value must not poison the run with a NaN. `apply_replay_state`
    /// writes `cs.boost = sp.boost * 100.0` unclamped and the replay filter checks
    /// FINITENESS only, so a negative boost reaches here; `sqrt` of it is NaN, and
    /// because `prev_state` is latched to the post-reset state the NaN would be
    /// paid every step for the whole episode while every log line looked normal.
    #[test]
    fn a_negative_boost_cannot_produce_a_nan() {
        let (mut prev, mut cur) = boost_pair();
        prev.cars[0].state.boost = -5.0;
        cur.cars[0].state.boost = 25.0;
        let cfg = RewardConfig { boost_pickup: 0.2, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        let r = compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert!(r.is_finite(), "reward must stay finite; got {r}");
        assert!(t.iter().all(|x| x.is_finite()), "no term may be NaN: {t:?}");
        assert!((t[T_BOOST_PICKUP] - 0.2 * 0.5).abs() < 1e-6,
                "a negative baseline is clamped to empty; got {}", t[T_BOOST_PICKUP]);
    }

    /// The flip instrument fires on the has_flipped rising edge -- that is the
    /// flip (a dodge, or a stall, which the sim also records as a flip), not the
    /// first jump.
    #[test]
    fn flip_events_counts_the_flip_rising_edge() {
        let (mut prev, mut cur) = boost_pair();
        prev.cars[0].state.has_flipped = false;
        cur.cars[0].state.has_flipped = true;
        let cfg = RewardConfig::default();
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_FLIP_EVENTS], 1.0, "rising edge is one flip");

        // Still flipped on the next tick is the SAME flip, not a new one.
        let mut t2 = [0.0f64; N_TERMS];
        prev.cars[0].state.has_flipped = true;
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t2);
        assert_eq!(t2[T_FLIP_EVENTS], 0.0, "held-high must not double count");
    }

    /// A DOUBLE JUMP is not a flip, and this matters for reading the experiment:
    /// the two action rows the table doubles the mass of have zero directional
    /// input, so their airborne rising edge is a double jump (`hasDoubleJumped`),
    /// which RocketSim records WITHOUT touching `hasFlipped`. So `flip_events`
    /// reads exactly 0 for the clean-air lever -- `p_jump` and `air_z` are its
    /// readouts, not this counter.
    ///
    /// Written by mutating `has_double_jumped` directly: `test_support::build_state`
    /// forces it false, so no fixture built through that helper can express this.
    #[test]
    fn a_double_jump_is_not_a_flip() {
        let (mut prev, mut cur) = boost_pair();
        prev.cars[0].state.has_flipped = false;
        prev.cars[0].state.has_double_jumped = false;
        cur.cars[0].state.has_flipped = false;
        cur.cars[0].state.has_double_jumped = true;
        let cfg = RewardConfig::default();
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_FLIP_EVENTS], 0.0, "a double jump must not count as a flip");
    }

    /// TERM_NAMES is positional and consumed by name on the Python side.
    #[test]
    fn term_names_covers_every_index_and_is_unique() {
        assert_eq!(TERM_NAMES[T_BOOST_PICKUP], "boost_pickup");
        assert_eq!(TERM_NAMES[T_BOOST_GAINED], "boost_gained");
        assert_eq!(TERM_NAMES[T_FLIP_EVENTS], "flip_events");
        assert_eq!(TERM_NAMES[T_AIR_HANG], "air_hang");
        assert_eq!(TERM_NAMES[T_AIR_HANG_EVENTS], "air_hang_events");
        assert_eq!(TERM_NAMES[T_AIR_DECISIONS], "air_decisions");
        // The indices are FROZEN, not "the last three": appending is safe only
        // because nothing may ever be renumbered. `flip_events` stopped being
        // last when the air-hang trio was appended on 2026-07-30.
        assert_eq!(T_FLIP_EVENTS, 22, "index 22 is flip_events forever");
        assert_eq!(T_AIR_DECISIONS, N_TERMS - 1, "the air-hang trio is the LAST three");
        let mut seen = std::collections::HashSet::new();
        for n in TERM_NAMES { assert!(seen.insert(n), "duplicate term name {n}"); }
    }

    // ------------------------------------------------- air hang (2026-07-30) --
    //
    // The instrument for the takeoff-exploration question, shipped at weight
    // 0.0. See `T_AIR_HANG` for the verdict these tests encode: the gate is
    // sound and the payout is not, so the EVENT counters are the deliverable
    // and the coefficient is the thing nobody may set without new evidence.

    /// A (prev, cur) pair whose single AIRBORNE car's `air_time_since_jump`
    /// moves `t0` -> `t1`, carrying the flag combination a real post-jump air
    /// phase carries: `has_jumped` set (a surface applied a jump impulse) and
    /// `is_jumping` cleared (the hold is over). That is the ONLY combination
    /// under which RocketSim accumulates the clock at all -- every other
    /// combination zeroes it (Car.cpp:697-701).
    fn hang_pair(t0: f32, t1: f32) -> (GameState, GameState) {
        let (mut prev, mut cur) = boost_pair();
        for gs in [&mut prev, &mut cur] {
            gs.cars[0].state.has_jumped = true;
            gs.cars[0].state.is_jumping = false;
        }
        prev.cars[0].state.air_time_since_jump = t0;
        cur.cars[0].state.air_time_since_jump = t1;
        (prev, cur)
    }

    /// The term array for one `hang_pair` at coefficient `w`.
    fn hang_terms(t0: f32, t1: f32, w: f32) -> [f64; N_TERMS] {
        let (prev, cur) = hang_pair(t0, t1);
        let cfg = RewardConfig { air_hang: w, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        t
    }

    /// Run one control tape on a single-car arena for `decs` decisions and
    /// accumulate `compute_terms` over every one of them, exactly as
    /// `EpisodeArena::step` does: ONE call per decision against a `prev` latched
    /// 8 physics ticks earlier. Returns (terms, air phases), where a phase is a
    /// maximal run of airborne decisions -- the denominator the "at most one
    /// payment per air phase" bound is stated against.
    ///
    /// Car start and ball placement copy `engine/examples/hang_gate_probe.rs`
    /// exactly, so the rates asserted here are the rates measured there.
    fn air_tape(
        decs: u32,
        cfg: &RewardConfig,
        tape: &dyn Fn(&rocketsim_rs::sim::CarState, bool) -> CarControls,
    ) -> ([f64; N_TERMS], u32) {
        use rocketsim_rs::math::{RotMat, Vec3};
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        arena.pin_mut().reset_to_random_kickoff(Some(1));
        let id = arena.pin_mut().get_game_state().cars[0].id;
        let mut cs = arena.pin_mut().get_car(id);
        cs.pos = Vec3::new(0.0, -2000.0, 17.0);
        cs.vel = Vec3::new(0.0, 0.0, 0.0);
        cs.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        cs.rot_mat = RotMat {
            forward: Vec3::new(0.0, 1.0, 0.0),
            right: Vec3::new(-1.0, 0.0, 0.0),
            up: Vec3::new(0.0, 0.0, 1.0),
        };
        cs.boost = 100.0;
        arena.pin_mut().set_car(id, cs).unwrap();
        // Ball parked far away and at rest: this is about the car only, and a
        // contact would drag `touch`/`vel_to_ball` into the numbers.
        let mut b = arena.pin_mut().get_ball();
        b.pos = Vec3::new(0.0, 4000.0, 93.15);
        b.vel = Vec3::new(0.0, 0.0, 0.0);
        arena.pin_mut().set_ball(b);

        let mut t = [0.0f64; N_TERMS];
        let mut prev = arena.pin_mut().get_game_state();
        let (mut phases, mut in_air, mut last_jump) = (0u32, false, false);
        for _ in 0..decs {
            let st = arena.pin_mut().get_car(id);
            let c = tape(&st, last_jump);
            last_jump = c.jump;
            arena.pin_mut().set_car_controls(id, c).unwrap();
            arena.pin_mut().step(8);
            let cur = arena.pin_mut().get_game_state();
            compute_terms(&prev, &cur, 0, None, cfg, &mut t);
            let air = !cur.cars[0].state.is_on_ground;
            if air && !in_air {
                phases += 1;
            }
            in_air = air;
            prev = cur;
        }
        (t, phases)
    }

    /// The whole mechanism: a RISING crossing of the post-jump hang clock, once
    /// per air phase. A LEVEL test on the same quantity would pay every decision
    /// of the hang (15x/s), which is the flat-touch farm in a new costume.
    #[test]
    fn the_hang_gate_fires_once_on_the_crossing_and_never_on_the_level() {
        let w = 0.008;
        let t = hang_terms(AIR_HANG_T - 0.1, AIR_HANG_T + 0.05, w);
        assert_eq!(t[T_AIR_HANG_EVENTS], 1.0, "the crossing is one event");
        assert!((t[T_AIR_HANG] - w as f64).abs() < 1e-9,
                "the payout is the flat coefficient; got {}", t[T_AIR_HANG]);

        // ALREADY above, one decision later in the SAME air phase.
        let held = hang_terms(AIR_HANG_T + 0.05, AIR_HANG_T + 0.12, w);
        assert_eq!(held[T_AIR_HANG_EVENTS], 0.0, "held-high must not double count");
        assert_eq!(held[T_AIR_HANG], 0.0);

        // Never reached: a hop that lands before the threshold.
        let short = hang_terms(0.5, AIR_HANG_T - 0.01, w);
        assert_eq!(short[T_AIR_HANG_EVENTS], 0.0, "below the threshold pays nothing");

        // The `cur` side is inclusive, so landing exactly on T fires.
        let exact = hang_terms(AIR_HANG_T - 0.1, AIR_HANG_T, w);
        assert_eq!(exact[T_AIR_HANG_EVENTS], 1.0, "cur == T is a crossing");
    }

    /// THE false-fire immunity, and the reason the gate reads
    /// `air_time_since_jump` rather than `air_time` or a ground->air transition.
    /// Measured on the raw transition proxy: 7-9 fires/min from driving off the
    /// curved corner with ZERO jump input, one per demo respawn, and two per 40
    /// decisions for a bumped car. All of them read `air_time_since_jump == 0`,
    /// because RocketSim only accumulates that clock while `has_jumped`
    /// (Car.cpp:697-701) and only ever sets `has_jumped` from
    /// `is_on_ground && jumpPressed` (Car.cpp:569-578).
    #[test]
    fn the_hang_gate_reads_the_post_jump_clock_not_raw_air_time() {
        let (mut prev, mut cur) = boost_pair();
        for (gs, at) in [(&mut prev, 5.0f32), (&mut cur, 5.07f32)] {
            gs.cars[0].state.has_jumped = false;
            gs.cars[0].state.is_jumping = false;
            gs.cars[0].state.air_time = at;
            gs.cars[0].state.air_time_since_jump = 0.0;
        }
        let cfg = RewardConfig { air_hang: 0.008, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        let r = compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_AIR_HANG_EVENTS], 0.0,
                   "5 s of air the car never jumped for must qualify zero times");
        assert_eq!(r, 0.0, "and must not move the reward at all");
        assert_eq!(t[T_AIR_DECISIONS], 1.0, "but the duty instrument still counts it");
    }

    /// A crossing that lands on the TOUCHDOWN decision is not hang time the car
    /// still has. RocketSim increments the clock in `_PreTickUpdate` off the
    /// previous tick's `is_on_ground` and recomputes `is_on_ground` in
    /// `_PostTickUpdate`, so a landing state can carry a positive clock -- which
    /// is why the payout carries an explicit airborne guard rather than relying
    /// on the clock's own grounded zeroing. Not caught by any of the measured
    /// tapes; pinned here so the guard cannot be "simplified" away.
    #[test]
    fn a_crossing_on_the_landing_decision_is_not_paid() {
        let (mut prev, mut cur) = hang_pair(AIR_HANG_T - 0.1, AIR_HANG_T + 0.05);
        cur.cars[0].state.is_on_ground = true; // wheels down on this decision
        let cfg = RewardConfig { air_hang: 0.008, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_AIR_HANG_EVENTS], 0.0, "a landed car is not hanging");
        assert_eq!(t[T_AIR_HANG], 0.0);
        assert_eq!(t[T_AIR_DECISIONS], 0.0, "and the duty instrument agrees");

        // Sanity: the SAME pair with the wheels still off the ground does fire,
        // so the assertion above is about the guard and not about the clock.
        prev.cars[0].state.is_on_ground = false;
        cur.cars[0].state.is_on_ground = false;
        let mut air = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut air);
        assert_eq!(air[T_AIR_HANG_EVENTS], 1.0);
    }

    /// Shipped state: the coefficient is absent from every tape, so the term is
    /// inert, while the RULER still records. Same payout/instrument split as
    /// `boost_pickup`/`boost_gained` -- retuning the payout must never move the
    /// measurement the retune would be judged by.
    #[test]
    fn air_hang_defaults_to_zero_reward_but_the_instrument_still_records() {
        let (prev, cur) = hang_pair(AIR_HANG_T - 0.1, AIR_HANG_T + 0.05);
        let mut t = [0.0f64; N_TERMS];
        let r = compute_terms(&prev, &cur, 0, None, &RewardConfig::default(), &mut t);
        assert_eq!(r, 0.0, "an all-zero tape stays byte-identical across a crossing");
        assert_eq!(t[T_AIR_HANG], 0.0, "unset coefficient pays nothing");
        assert_eq!(t[T_AIR_HANG_EVENTS], 1.0, "but the coefficient-free ruler records");
    }

    /// The duty instrument, whose denominator is the existing `agent_steps`.
    /// It is the second half of the pair: a crossing count alone cannot tell a
    /// 93%-airborne pogo from real air play, and `flip_events` reads exactly
    /// 0.00 through every hop-spam tape measured (a minimum hop sets neither
    /// `has_flipped` nor `has_double_jumped`), so it is structurally blind here.
    #[test]
    fn air_decisions_counts_airborne_decisions_and_nothing_else() {
        let (mut prev, mut cur) = boost_pair(); // `synth` builds AIRBORNE cars
        let cfg = RewardConfig::default();
        let mut t = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
        assert_eq!(t[T_AIR_DECISIONS], 1.0, "an airborne decision counts");
        assert_eq!(t[T_AGENT_STEPS], 1.0, "the duty denominator is agent_steps");

        prev.cars[0].state.is_on_ground = true;
        cur.cars[0].state.is_on_ground = true;
        let mut g = [0.0f64; N_TERMS];
        compute_terms(&prev, &cur, 0, None, &cfg, &mut g);
        assert_eq!(g[T_AIR_DECISIONS], 0.0, "a grounded decision is not air time");
        assert_eq!(g[T_AGENT_STEPS], 1.0);
    }

    /// THE farm test. The maximising tape for any flat per-takeoff payment is
    /// the CHEAPEST hop: release the jump after one decision and re-press the
    /// instant the wheels touch. Measured 60.6-69.6 takeoffs/min, 93.2%
    /// airborne, apex 121.5 uu, zero boost -- and it qualifies for the hang gate
    /// exactly ZERO times: its air phase is 1.13 s, and over 200 measured phases
    /// 98% of its post-jump clocks peaked below 1.0 s and none reached 1.2 s.
    ///
    /// This is the property the whole design rests on: the tape that maximises a
    /// per-transition bonus is the tape this gate pays nothing for.
    #[test]
    fn the_maximum_rate_pogo_earns_no_hang_payment_at_all() {
        let secs = 60.0f64;
        let cfg = RewardConfig { air_hang: 0.008, ..Default::default() };
        let (t, phases) = air_tape(900, &cfg, &|st, lj| CarControls {
            jump: if lj {
                st.is_jumping && st.jump_time < 1.0 / 15.0 - 1e-4
            } else {
                st.is_on_ground
            },
            ..Default::default()
        });
        assert!(phases >= 40,
                "precondition: this tape must BE a hop farm; got {phases} phases in {secs}s");
        let duty = t[T_AIR_DECISIONS] / t[T_AGENT_STEPS];
        assert!(duty > 0.85, "precondition: the pogo is ~93% airborne; got {duty}");
        assert_eq!(t[T_AIR_HANG_EVENTS], 0.0,
                   "the fastest, cheapest, highest-duty hop spam must qualify ZERO times");
        assert_eq!(t[T_AIR_HANG], 0.0, "so it earns nothing even at a nonzero coefficient");
    }

    /// PINNED FARM CEILING, not a desirable property. The best QUALIFYING hop
    /// tape holds the jump the full 0.2 s (`JUMP_MAX_TIME`) and reaches
    /// 28.2 crossings/min over 200 s -- 141 per 300 s match, apex ~263 uu, zero
    /// boost, zero ball contact. That is the farm any nonzero coefficient buys,
    /// and it is why the coefficient is 0.0. If this number moves, the arithmetic
    /// in `T_AIR_HANG`'s doc comment must be recomputed with it.
    ///
    /// The structural half of the bound is a physics property, not a cooldown:
    /// `air_time_since_jump` is zeroed on every grounded tick and `is_jumping`
    /// can only be re-armed from the ground, so within one air phase the clock
    /// is strictly monotone and the crossing fires AT MOST ONCE per phase.
    #[test]
    fn the_hang_gate_bounds_the_hop_farm_at_one_payment_per_air_phase() {
        let secs = 60.0f64;
        let cfg = RewardConfig::default();
        let (t, phases) = air_tape(900, &cfg, &|st, lj| CarControls {
            jump: if lj { st.is_jumping && st.jump_time < 0.1999 } else { st.is_on_ground },
            ..Default::default()
        });
        let ev = t[T_AIR_HANG_EVENTS];
        assert!(ev > 0.0, "precondition: the qualifying tape must actually qualify");
        assert!(ev <= phases as f64,
                "at most ONE payment per air phase: {ev} payments in {phases} phases");
        let per_min = 60.0 * ev / secs;
        assert!(per_min <= 60.0 / AIR_HANG_T as f64,
                "physics ceiling is one crossing per {AIR_HANG_T}s of hang; got {per_min}/min");
        assert!(per_min <= 35.0,
                "measured hop-farm ceiling is 28.2/min over 200 s; got {per_min}/min");
    }

    /// An episode that SPAWNS a car airborne must collect nothing for the fall it
    /// was handed. ~46% of `random_reset` cars start in the air (measured
    /// 184/400), and at curriculum_v3's 0.6 random weight that is ~28% of
    /// car-episodes.
    ///
    /// Also pins the state the 2026-07-30 review called a live physics bug: that
    /// neither reset path writes `has_jumped`/`is_jumping`/`has_double_jumped`/
    /// `air_time`/`air_time_since_jump`, so "flip availability is inherited from
    /// an unrelated episode". MEASURED FALSE, and this is the test that says so:
    /// `reset_episode` calls `reset_to_random_kickoff` FIRST on every branch
    /// (episode.rs:518) and RocketSim's kickoff does
    /// `CarState spawnState; ... SetState(spawnState)` (Arena.cpp:180-192) with
    /// `_internalState = state` (Car.cpp:34), so the get->mutate->set idiom in
    /// `random_reset`/`apply_replay_state` inherits FRESH ZEROS, not stale flags.
    #[test]
    fn an_airborne_spawn_pays_no_hang_bonus_and_the_reset_launders_the_jump_state() {
        ensure_init(None);
        let mut arena = Arena::default_standard();
        arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
        let bounds = crate::curriculum::RandomStateBounds::default();
        // Drive the REAL reset path until it hands back a car high enough in the
        // air to out-hang the threshold on the way down.
        let mut spawn = None;
        for seed in 0..400u64 {
            arena.pin_mut().reset_to_random_kickoff(Some(1));
            let mut rng = crate::sampler::Pcg32::new(seed);
            crate::curriculum::random_reset(arena.pin_mut(), &mut rng, &bounds);
            let st = arena.pin_mut().get_game_state().cars[0].state;
            if !st.is_on_ground && st.pos.z >= 1000.0 && st.vel.z > -200.0 {
                spawn = Some(st);
                break;
            }
        }
        let st = spawn.expect("random_reset must give a high airborne spawn within 400 seeds");
        assert!(!st.has_jumped, "the kickoff reset must launder has_jumped");
        assert!(!st.is_jumping, "... and is_jumping");
        assert!(!st.has_double_jumped, "... and has_double_jumped");
        assert!(!st.has_flipped, "... and has_flipped");
        assert_eq!(st.air_time, 0.0, "... and air_time");
        assert_eq!(st.air_time_since_jump, 0.0, "... and air_time_since_jump");

        let id = arena.pin_mut().get_game_state().cars[0].id;
        let cfg = RewardConfig { air_hang: 0.008, ..Default::default() };
        let mut t = [0.0f64; N_TERMS];
        let mut prev = arena.pin_mut().get_game_state();
        let mut max_air = 0.0f32;
        for _ in 0..45 {
            arena.pin_mut().set_car_controls(id, CarControls::default()).unwrap();
            arena.pin_mut().step(8);
            let cur = arena.pin_mut().get_game_state();
            compute_terms(&prev, &cur, 0, None, &cfg, &mut t);
            max_air = max_air.max(cur.cars[0].state.air_time);
            prev = cur;
        }
        assert!(max_air > AIR_HANG_T,
                "precondition: the spawn must hang past the threshold; got {max_air}s");
        assert!(t[T_AIR_DECISIONS] >= 18.0,
                "precondition: >= 1.2 s of airborne decisions; got {}", t[T_AIR_DECISIONS]);
        assert_eq!(t[T_AIR_HANG_EVENTS], 0.0,
                   "a car handed its air phase by a RESET must qualify zero times");
        assert_eq!(t[T_AIR_HANG], 0.0, "and must be paid nothing");
    }
}
