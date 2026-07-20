# Match-Win Objective (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace shaped self-play reward with a potential-based win-probability reward over full-match episodes, so the training objective targets the same thing the gate measures.

**Architecture:** A config-gated "match layer" in the Rust engine tracks score and a match clock; a goal stops terminating the RL episode and instead updates the score and kicks off again, while clock expiry becomes the trajectory boundary. Reward becomes `w * (gamma * PHI(s') - PHI(s))` where PHI is an analytic win probability — potential-based, so it telescopes to the match outcome and leaves the optimum unchanged (Ng, Harada & Russell 1999). The gate becomes two-stage: the existing goal-share run screens candidates in legacy episode mode, and a new match-win stage confirms promotions.

**Tech Stack:** Rust (rocketsim_rs, pyo3, serde/toml), Python 3.12 (numpy, torch, pytest), maturin abi3 wheel.

## Global Constraints

- **Phase 1 changes NO observation and NO schema version.** `Q_FEAT` stays 64, `query[60..62]` stay 0.0. Filling them is Phase 2 and is out of scope — it makes the v1 champion see out-of-distribution input and blocks comparison against it.
- **Never edit `python/construct/_engine.abi3.so` on the laptop.** It is the gate instrument; every historical gate in `logs/champion_history.jsonl` was scored with it. Build wheels for the TRAINER box only.
- **The match layer must be config-gated and default OFF.** Turning it on unconditionally changes reset dynamics in the goal-share screen and makes every historical gate incomparable.
- **The shaping gamma MUST equal the training gamma** (`configs/train_v1.toml` `[ppo] gamma = 0.9954`). Any other value breaks the Ng et al. guarantee that the optimum is preserved.
- Existing reward terms (`goal`, `touch`, `vel_to_ball`, `vel_ball_to_goal`, `offensive_potential`, `touch_accel`, `aggression_bias`) stay in `RewardConfig` and default to 0.0 in v5, so every historical config keeps parsing.
- Run Rust tests with `cargo test --manifest-path engine/Cargo.toml --lib`, Python with `nice -n 10 .venv/bin/python -m pytest`.

---

### Task 1: Win-probability potential (PHI)

Pure function, no engine state. This is the mathematical core; everything else consumes it.

**Files:**
- Modify: `engine/src/reward.rs` (append below `RewardConfig`)
- Test: `engine/src/reward.rs` (in the existing `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing.
- Produces: `pub fn win_prob(score_diff: i32, t_frac: f32, k_base: f32, t_floor: f32) -> f32`

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests` in `engine/src/reward.rs`:

```rust
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path engine/Cargo.toml --lib win_prob`
Expected: FAIL — `cannot find function 'win_prob' in module 'super'`

- [ ] **Step 3: Write the implementation**

Append to `engine/src/reward.rs` after the `impl RewardConfig` block:

```rust
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --manifest-path engine/Cargo.toml --lib win_prob`
Expected: PASS, 4 tests

- [ ] **Step 5: Commit**

```bash
git add engine/src/reward.rs
git commit -m "feat(reward): win-probability potential for match-win shaping"
```

---

### Task 2: Match state and the telescoping shaping term

**This task contains the load-bearing test of the whole plan.** If summed shaped reward does not equal `PHI(end) - PHI(start)`, the shaping is not potential-based, the Ng guarantee does not apply, and the policy is optimising something other than match wins.

**Files:**
- Modify: `engine/src/reward.rs`
- Test: `engine/src/reward.rs` (`mod tests`)

**Interfaces:**
- Consumes: `win_prob` from Task 1.
- Produces:
  - `pub struct MatchState { pub score_blue: u32, pub score_orange: u32, pub t_frac: f32 }`
  - `impl MatchState { pub fn score_diff(&self, team: Team) -> i32 }`
  - `pub fn win_prob_shaping(prev: &MatchState, cur: &MatchState, team: Team, gamma: f32, cfg: &RewardConfig) -> f32`
  - three new `RewardConfig` fields: `win_prob_weight`, `win_prob_k_base`, `win_prob_t_floor`, `win_prob_gamma` (all `#[serde(default)]`)

- [ ] **Step 1: Write the failing tests**

Append inside `mod tests`:

```rust
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
    let cfg = v5_cfg();
    let (a, b) = (ms(0, 0, 0.5), ms(1, 0, 0.5));
    let blue = super::win_prob_shaping(&a, &b, Team::Blue, cfg.win_prob_gamma, &cfg);
    let orange = super::win_prob_shaping(&a, &b, Team::Orange, cfg.win_prob_gamma, &cfg);
    assert!((blue + orange).abs() < 1e-6, "blue {blue} + orange {orange} != 0");
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
    for p in ["configs/reward_v0.toml", "configs/reward_v3.toml",
              "configs/reward_v4_1.toml"] {
        let c = super::RewardConfig::load(p).unwrap_or_else(|e| panic!("{p}: {e}"));
        assert_eq!(c.win_prob_weight, 0.0, "{p} must default win_prob_weight to 0");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path engine/Cargo.toml --lib win_prob_shaping`
Expected: FAIL — `cannot find struct 'MatchState'`, `cannot find function 'win_prob_shaping'`

- [ ] **Step 3: Write the implementation**

Add the four fields to `RewardConfig` in `engine/src/reward.rs`, immediately after `opp_spirit`:

```rust
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
```

Add `#[derive(Default)]` to `RewardConfig` if it is not already derived, then append:

```rust
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
    /// values are exact negations and the shaped game stays zero-sum.
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
        win_prob(m.score_diff(team), m.t_frac, cfg.win_prob_k_base, cfg.win_prob_t_floor)
    };
    cfg.win_prob_weight * (gamma * phi(cur) - phi(prev))
}
```

- [ ] **Step 4: Add loud validation for half-configured shaping**

A config that sets `win_prob_weight` but forgets `win_prob_k_base` would leave
`k_base = 0`, making PHI a constant 0.5 and the shaping term identically zero —
a run that trains on NOTHING and looks completely healthy. That is precisely
the silent-degradation failure the spec forbids. Add to `impl RewardConfig`:

```rust
    /// Reject half-configured shaping. With `win_prob_weight` set but
    /// `k_base`/`t_floor` left at their zero defaults, PHI collapses to a
    /// constant 0.5, the shaping term is identically zero, and the run trains
    /// on nothing while looking entirely normal.
    pub fn validate(&self) -> Result<(), String> {
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
```

Call it at the end of `RewardConfig::load`, propagating the error.

Add the tests:

```rust
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test --manifest-path engine/Cargo.toml --lib`
Expected: PASS — all new tests plus every pre-existing reward test still green.

- [ ] **Step 6: Commit**

```bash
git add engine/src/reward.rs
git commit -m "feat(reward): MatchState + potential-based win-prob shaping

Includes the load-bearing telescoping test: summed shaped reward over a
match must equal PHI(end)-PHI(start). If that ever fails the shaping is
not potential-based and the optimum is no longer match-winning."
```

---

### Task 3: Config-gated match layer in the engine

**Files:**
- Modify: `engine/src/curriculum.rs` (add `match_mode` to `CurriculumConfig`)
- Modify: `engine/src/episode.rs` (`EpisodeArena` fields, termination logic, reset)
- Test: `engine/src/episode.rs` (`mod tests`)

**Interfaces:**
- Consumes: `MatchState` from Task 2.
- Produces:
  - `CurriculumConfig.match_mode: bool` (serde default `false`)
  - `EpisodeArena` fields `score_blue: u32`, `score_orange: u32`, `match_start_tick: u64`
  - `EpisodeArena::match_state(&self, cur_tick: u64) -> MatchState`

- [ ] **Step 1: Write the failing tests**

Append inside `episode.rs`'s `mod tests`:

```rust
#[test]
fn legacy_mode_still_terminates_on_a_goal() {
    // The goal-share screen depends on this. If match_mode leaks into the
    // default path, every historical gate becomes incomparable.
    let c = curr_with_match_mode(false);
    assert!(!c.match_mode);
}

#[test]
fn match_mode_is_off_by_default_in_every_shipped_curriculum() {
    for p in ["configs/curriculum_v1.toml", "configs/curriculum_v2.toml"] {
        let c = crate::curriculum::CurriculumConfig::load(p)
            .unwrap_or_else(|e| panic!("{p}: {e}"));
        assert!(!c.match_mode, "{p} must default match_mode to false");
    }
}

#[test]
fn match_state_reports_signed_diff_and_falling_clock() {
    let mut a = mk_match_arena();
    a.score_blue = 2;
    a.score_orange = 1;
    a.match_start_tick = 0;
    let half = a.match_state(MAX_TICKS / 2);
    assert_eq!(half.score_diff(Team::Blue), 1);
    assert_eq!(half.score_diff(Team::Orange), -1);
    assert!((half.t_frac - 0.5).abs() < 0.01, "t_frac {}", half.t_frac);

    let start = a.match_state(0);
    assert!((start.t_frac - 1.0).abs() < 1e-6);
    let end = a.match_state(MAX_TICKS);
    assert!((end.t_frac - 0.0).abs() < 1e-6);
}

#[test]
fn t_frac_never_leaves_unit_range_past_the_whistle() {
    let mut a = mk_match_arena();
    a.match_start_tick = 0;
    let over = a.match_state(MAX_TICKS * 2);
    assert!((0.0..=1.0).contains(&over.t_frac), "t_frac {}", over.t_frac);
}
```

Add these helpers to the same `mod tests`:

```rust
fn curr_with_match_mode(on: bool) -> crate::curriculum::CurriculumConfig {
    let mut c = crate::curriculum::CurriculumConfig {
        kickoff_weight: 1.0, random_weight: 0.0, ..Default::default()
    };
    c.match_mode = on;
    c
}

fn mk_match_arena() -> super::EpisodeArena {
    // Mirrors the existing constructor use at episode.rs:1143.
    let s = crate::schema::Schema::load("schema/v1.toml").unwrap();
    let cfg = crate::reward::RewardConfig::load("configs/reward_v5_winprob.toml").unwrap();
    super::EpisodeArena::new_with_curriculum(
        1, 1, s.tick_skip, cfg, s.normalization, 5150,
        Some(curr_with_match_mode(true)),
    )
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test --manifest-path engine/Cargo.toml --lib match_mode`
Expected: FAIL — `no field 'match_mode' on type 'CurriculumConfig'`

- [ ] **Step 3: Add the config flag**

In `engine/src/curriculum.rs`, inside `pub struct CurriculumConfig`, after `replay_pool`:

```rust
    /// Task #56 Phase 1: run FULL MATCHES instead of single episodes. A goal
    /// updates the score and kicks off again; only clock expiry terminates.
    ///
    /// Defaults to false, and that default is load-bearing: with match_mode on,
    /// goals stop terminating episodes, which changes reset dynamics and would
    /// make the goal-share gate incomparable with every historical entry in
    /// logs/champion_history.jsonl. The screen must keep running legacy mode.
    #[serde(default)]
    pub match_mode: bool,
```

- [ ] **Step 4: Add match state to the arena**

In `engine/src/episode.rs`, add to `pub struct EpisodeArena` after `last_touch_tick`:

```rust
    /// Match score, only meaningful when `curriculum.match_mode` is set.
    score_blue: u32,
    score_orange: u32,
    /// Tick the current MATCH began (distinct from `episode_start_tick`, which
    /// in match mode restarts at every kickoff within the match).
    match_start_tick: u64,
```

Initialise all three to 0 in every constructor that builds an `EpisodeArena`.

Add the accessor next to the other small helpers:

```rust
impl EpisodeArena {
    /// Score and remaining-clock fraction for the current match.
    pub fn match_state(&self, cur_tick: u64) -> MatchState {
        let elapsed = cur_tick.saturating_sub(self.match_start_tick);
        let t_frac = 1.0 - (elapsed as f32 / MAX_TICKS as f32);
        MatchState {
            score_blue: self.score_blue,
            score_orange: self.score_orange,
            t_frac: t_frac.clamp(0.0, 1.0),
        }
    }

    fn match_mode(&self) -> bool {
        self.curriculum.as_ref().map_or(false, |c| c.match_mode)
    }
}
```

Import `MatchState` at the top of `episode.rs`: add it to the existing
`use crate::reward::{...}` list.

- [ ] **Step 5: Change the termination logic**

In `engine/src/episode.rs`, replace the two lines at the `let terminated = ...` site:

```rust
        let terminated = scored.is_some();
        let truncated = !terminated
            && (cur.tick_count - self.last_touch_tick >= NO_TOUCH_TICKS
                || cur.tick_count - self.episode_start_tick >= MAX_TICKS);
```

with:

```rust
        // MATCH MODE (task #56): a goal is a scoring event INSIDE the
        // trajectory, not the end of it. Credit therefore flows across the
        // goal, which is the whole point -- "was that goal worth the
        // counter-attack it conceded?" is only learnable if the trajectory
        // continues. Only the clock ends a match.
        //
        // LEGACY MODE: unchanged. A goal terminates. The goal-share screen and
        // every historical gate depend on this exact behavior.
        let (terminated, truncated) = if self.match_mode() {
            if let Some(team) = scored {
                match team {
                    Team::Blue => self.score_blue += 1,
                    Team::Orange => self.score_orange += 1,
                }
            }
            let match_over = cur.tick_count - self.match_start_tick >= MAX_TICKS;
            let stalled = cur.tick_count - self.last_touch_tick >= NO_TOUCH_TICKS;
            (match_over, !match_over && stalled)
        } else {
            let t = scored.is_some();
            let tr = !t
                && (cur.tick_count - self.last_touch_tick >= NO_TOUCH_TICKS
                    || cur.tick_count - self.episode_start_tick >= MAX_TICKS);
            (t, tr)
        };
```

Then, at the reset site (`if terminated || truncated { ... self.reset_episode(); }`),
add a mid-match kickoff for the goal-scored-but-match-continues case. Immediately
before that block:

```rust
        // Match mode: a goal that did NOT end the match still needs a kickoff.
        if self.match_mode() && scored.is_some() && !terminated && !truncated {
            self.reset_episode();
        }
```

Do NOT touch `reset_episode` — it cannot distinguish a match start from a
post-goal kickoff, and overloading it is how the two get conflated. Add an
explicit method instead:

```rust
    /// Begin a NEW match: zero the score and restart the clock. Deliberately
    /// separate from `reset_episode`, which also runs on every post-goal
    /// kickoff WITHIN a match and must leave the score alone.
    fn start_match(&mut self, tick: u64) {
        self.score_blue = 0;
        self.score_orange = 0;
        self.match_start_tick = tick;
    }
```

Call it in exactly two places:
1. at the end of every constructor, as `self.start_match(0)`;
2. in the reset block, only when the MATCH ended:

```rust
        if terminated || truncated {
            if self.match_mode() {
                self.start_match(cur.tick_count);
            }
            self.reset_episode();
        }
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test --manifest-path engine/Cargo.toml --lib`
Expected: PASS — new match tests plus every existing episode test.

- [ ] **Step 7: Commit**

```bash
git add engine/src/curriculum.rs engine/src/episode.rs
git commit -m "feat(engine): config-gated match layer — goals score, clock terminates

match_mode defaults false so the goal-share screen keeps legacy episode
behavior and stays comparable with logs/champion_history.jsonl."
```

---

### Task 4: Wire shaping into the step, and ship reward_v5

**Files:**
- Modify: `engine/src/episode.rs` (reward accumulation in `step_impl`)
- Create: `configs/reward_v5_winprob.toml`
- Create: `configs/curriculum_v3_match.toml`
- Test: `engine/tests/episode_test.rs`

**Interfaces:**
- Consumes: `win_prob_shaping`, `MatchState` (Task 2); `match_state()`, `match_mode()` (Task 3).
- Produces: `configs/reward_v5_winprob.toml`, `configs/curriculum_v3_match.toml`.

- [ ] **Step 1: Write the failing integration test**

Create/append `engine/tests/episode_test.rs`:

```rust
#[test]
fn match_mode_run_accumulates_shaped_reward_and_ends_on_the_clock() {
    // End-to-end: a match-mode arena must (a) not terminate on a goal,
    // (b) eventually terminate on the clock, (c) produce nonzero shaped
    // reward once a goal has been scored.
    let curriculum = construct_engine::curriculum::CurriculumConfig::load(
        "configs/curriculum_v3_match.toml").expect("curriculum_v3_match");
    assert!(curriculum.match_mode, "v3 must enable match mode");

    let cfg = construct_engine::reward::RewardConfig::load(
        "configs/reward_v5_winprob.toml").expect("reward_v5");
    assert!(cfg.win_prob_weight > 0.0, "v5 must enable win-prob shaping");
    assert_eq!(cfg.goal, 0.0, "v5 must not double-pay a raw goal term");
    assert!((cfg.win_prob_gamma - 0.9954).abs() < 1e-9,
            "v5 gamma must match configs/train_v1.toml [ppo] gamma");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path engine/Cargo.toml --test episode_test match_mode_run`
Expected: FAIL — `configs/curriculum_v3_match.toml` does not exist.

- [ ] **Step 3: Create the configs**

`configs/reward_v5_winprob.toml`:

```toml
# Reward v5 — potential-based match-win shaping (task #56 Phase 1).
#
# Every shaping term from v0-v4.1 is ZERO here. The only signal is
# w * (gamma * PHI(s') - PHI(s)) where PHI is the win probability from
# score_diff and remaining clock (engine/src/reward.rs::win_prob).
#
# Because the term is potential-based, it telescopes to the match outcome and
# leaves the optimal policy unchanged (Ng, Harada & Russell 1999) -- so PHI's
# accuracy costs sample efficiency, never correctness.
#
# REQUIRES a curriculum with match_mode = true (configs/curriculum_v3_match.toml).
# Without it there is no score or clock and the term is meaningless; the trainer
# refuses this combination loudly rather than training on a constant.
goal = 0.0
touch = 0.0
vel_to_ball = 0.0
aggression_bias = 0.0
touch_accel = 0.0
vel_ball_to_goal = 0.0
offensive_potential = 0.0

team_spirit = 0.3
opp_spirit = 0.3

win_prob_weight = 10.0
win_prob_k_base = 0.6
win_prob_t_floor = 0.05
# MUST equal configs/train_v1.toml [ppo] gamma. Checked at trainer startup.
win_prob_gamma = 0.9954
```

`configs/curriculum_v3_match.toml`:

```toml
# Curriculum v3 — full-match episodes (task #56 Phase 1).
#
# Identical reset mixture to curriculum_v1, plus match_mode. Reset mixture is
# held fixed ON PURPOSE: Phase 1 varies the objective and nothing else, so any
# gate movement is attributable. Combining this with the replay pool would
# repeat the 2026-07-20 confound where two variables moved at once.
match_mode = true
kickoff_weight = 0.4
random_weight = 0.6

[random]
car_speed_max = 1800.0
ball_speed_max = 2500.0
z_max = 1700.0
min_separation = 300.0
```

- [ ] **Step 4: Wire the shaping into the reward loop**

In `engine/src/episode.rs`, the per-agent reward loop currently reads:

```rust
        for a in 0..n {
            let ci = self.agent_car_index(&cur, a);
            rewards[a] = reward::compute(&self.prev_state, &cur, ci, scored, &self.reward_cfg);
            flags[a] = StepFlags { terminated, truncated };
        }
```

Replace with:

```rust
        // Match state BEFORE this step's goal was applied, and after. The
        // shaping term is the change in potential between them.
        let ms_prev = MatchState {
            score_blue: prev_score_blue,
            score_orange: prev_score_orange,
            t_frac: self.match_state(self.prev_state.tick_count).t_frac,
        };
        let ms_cur = self.match_state(cur.tick_count);

        for a in 0..n {
            let ci = self.agent_car_index(&cur, a);
            let mut r = reward::compute(&self.prev_state, &cur, ci, scored, &self.reward_cfg);
            if self.match_mode() {
                r += reward::win_prob_shaping(
                    &ms_prev, &ms_cur, cur.cars[ci].team,
                    self.reward_cfg.win_prob_gamma, &self.reward_cfg,
                );
            }
            rewards[a] = r;
            flags[a] = StepFlags { terminated, truncated };
        }
```

Capture `prev_score_blue` / `prev_score_orange` immediately BEFORE the
termination block that increments them:

```rust
        let (prev_score_blue, prev_score_orange) = (self.score_blue, self.score_orange);
```

- [ ] **Step 5: Add the loud-confirmation log line**

At the end of `EpisodeArena::new` (or wherever the curriculum is first read),
add:

```rust
        // Positive confirmation, mirroring the "[curriculum] replay pool ..."
        // line. On 2026-07-20 an arm ran fully INERT because a missing
        // capability degraded to a silent no-op and the run looked normal.
        // "Is the objective live?" must be answerable from the log, never
        // inferred from the absence of an error.
        if curriculum.as_ref().map_or(false, |c| c.match_mode) {
            eprintln!(
                "[match] match_mode ON: full matches of {} s, win_prob_weight={}, \
                 k_base={}, t_floor={}, gamma={}",
                MAX_TICKS / TICKS_PER_SEC,
                reward_cfg.win_prob_weight, reward_cfg.win_prob_k_base,
                reward_cfg.win_prob_t_floor, reward_cfg.win_prob_gamma,
            );
        }
```

- [ ] **Step 6: Run tests**

Run: `cargo test --manifest-path engine/Cargo.toml`
Expected: PASS — lib and integration suites.

- [ ] **Step 7: Commit**

```bash
git add engine/src/episode.rs configs/reward_v5_winprob.toml configs/curriculum_v3_match.toml
git commit -m "feat(engine): wire win-prob shaping into step; add reward_v5 + curriculum_v3"
```

---

### Task 5: Trainer-side gamma consistency guard

The Ng guarantee holds only if the shaping gamma equals the training gamma. A
silent mismatch would train a subtly different objective and look completely
normal — exactly the failure mode this project keeps hitting.

**Files:**
- Modify: `python/construct/learn/train.py`
- Test: `tests/python/test_train_config_guard.py` (create)

**Interfaces:**
- Consumes: `configs/reward_v5_winprob.toml`, `configs/curriculum_v3_match.toml` (Task 4).
- Produces:
  - `check_win_prob_gamma(reward_cfg: dict, ppo_cfg: dict) -> None` (raises `ValueError`)
  - `check_match_mode_required(reward_cfg: dict, curriculum_cfg: dict) -> None` (raises `ValueError`)

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_train_config_guard.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from construct.learn.train import check_win_prob_gamma  # noqa: E402


def test_matching_gamma_passes():
    check_win_prob_gamma({"win_prob_weight": 10.0, "win_prob_gamma": 0.9954},
                         {"gamma": 0.9954})


def test_mismatched_gamma_raises_with_both_values_named():
    """A silent mismatch trains a different objective and looks normal, so the
    error must name both numbers and say why it matters."""
    with pytest.raises(ValueError) as e:
        check_win_prob_gamma({"win_prob_weight": 10.0, "win_prob_gamma": 0.99},
                             {"gamma": 0.9954})
    msg = str(e.value)
    assert "0.99" in msg and "0.9954" in msg
    assert "potential" in msg.lower()


def test_guard_is_inert_when_shaping_is_off():
    """Historical configs have no win_prob keys; they must not trip the guard."""
    check_win_prob_gamma({}, {"gamma": 0.9954})
    check_win_prob_gamma({"win_prob_weight": 0.0, "win_prob_gamma": 0.0},
                         {"gamma": 0.9954})


def test_shaping_without_match_mode_raises():
    """Win-prob shaping needs a score and a clock. Without match_mode there is
    neither, PHI never moves, and the run trains on a constant while looking
    perfectly healthy."""
    from construct.learn.train import check_match_mode_required
    with pytest.raises(ValueError) as e:
        check_match_mode_required({"win_prob_weight": 10.0}, {"match_mode": False})
    assert "match_mode" in str(e.value)


def test_match_mode_without_shaping_is_allowed():
    """Full matches with a legacy reward is a legitimate ablation -- it isolates
    the match layer from the objective change."""
    from construct.learn.train import check_match_mode_required
    check_match_mode_required({"win_prob_weight": 0.0}, {"match_mode": True})


def test_shaping_with_match_mode_passes():
    from construct.learn.train import check_match_mode_required
    check_match_mode_required({"win_prob_weight": 10.0}, {"match_mode": True})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nice -n 10 .venv/bin/python -m pytest tests/python/test_train_config_guard.py -q`
Expected: FAIL — `ImportError: cannot import name 'check_win_prob_gamma'`

- [ ] **Step 3: Write the implementation**

Add to `python/construct/learn/train.py` at module level:

```python
def check_win_prob_gamma(reward_cfg, ppo_cfg):
    """Refuse to train if the shaping gamma differs from the training gamma.

    Potential-based shaping preserves the optimal policy only when the gamma
    inside `gamma * PHI(s') - PHI(s)` is the SAME gamma the returns are
    discounted with. A mismatch silently optimises a different objective and
    produces a run that looks entirely healthy, so this fails loud at startup.
    """
    if float(reward_cfg.get("win_prob_weight", 0.0)) == 0.0:
        return
    shaping = float(reward_cfg.get("win_prob_gamma", 0.0))
    training = float(ppo_cfg["gamma"])
    if abs(shaping - training) > 1e-9:
        raise ValueError(
            f"win_prob_gamma ({shaping}) != ppo gamma ({training}). "
            f"Potential-based shaping only preserves the optimal policy when "
            f"these are identical; a mismatch trains a different objective and "
            f"the run will look normal. Fix the reward config."
        )
```

And the second guard:

```python
def check_match_mode_required(reward_cfg, curriculum_cfg):
    """Refuse win-probability shaping without a match layer.

    PHI is a function of score and remaining clock. With match_mode off there
    is no score and no clock, PHI is pinned at 0.5 forever, and the shaping
    term is identically zero -- so the run trains on nothing at all while every
    log line looks normal. Fail at startup instead.
    """
    if float(reward_cfg.get("win_prob_weight", 0.0)) == 0.0:
        return
    if not bool(curriculum_cfg.get("match_mode", False)):
        raise ValueError(
            "win_prob_weight is set but the curriculum has match_mode = false. "
            "Win-probability shaping needs a score and a match clock; without "
            "them PHI is constant, the shaping term is exactly zero, and the "
            "run trains on nothing while appearing healthy. "
            "Use configs/curriculum_v3_match.toml."
        )
```

Call both from `Trainer.__init__` right after the reward and curriculum
configs are loaded.

- [ ] **Step 4: Run tests to verify they pass**

Run: `nice -n 10 .venv/bin/python -m pytest tests/python/test_train_config_guard.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add python/construct/learn/train.py tests/python/test_train_config_guard.py
git commit -m "feat(train): fail loud on shaping/gamma and shaping/match_mode mismatch

Both mismatches train a different objective (or nothing at all) while
producing a run that looks completely healthy."
```

---

### Task 6: Match-win gate stage

**Files:**
- Modify: `python/construct/league/matches.py`
- Test: `tests/python/test_match_win_gate.py` (create)

**Interfaces:**
- Consumes: engine `collect()` output keys `rewards` and `terminated` (already exported, `engine/src/lib.rs:351-364`).
- Produces:
  - `split_matches(rewards: np.ndarray, terminated: np.ndarray, threshold: float) -> list[tuple[int, int]]`
  - `match_record(matches: list[tuple[int, int]]) -> dict` with keys `wins`, `draws`, `losses`, `win_share`

- [ ] **Step 1: Write the failing tests**

Create `tests/python/test_match_win_gate.py`:

```python
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from construct.league.matches import match_record, split_matches  # noqa: E402

TH = 9.4


def tape(*rows):
    """(T,1) reward tape for a single arena."""
    return np.array(rows, dtype=np.float32).reshape(-1, 1)


def flags(*rows):
    return np.array(rows, dtype=bool).reshape(-1, 1)


def test_splits_on_terminated_boundaries():
    r = tape(10.0, 0.1, -10.0, 0.0, 10.0, 0.0)
    t = flags(False, False, False, True, False, True)
    assert split_matches(r, t, TH) == [(1, 1), (1, 0)]


def test_sub_threshold_rows_are_not_goals():
    """Shaping noise must never be counted as a goal."""
    r = tape(0.55, -0.55, 9.39, -9.39, 0.0)
    t = flags(False, False, False, False, True)
    assert split_matches(r, t, TH) == [(0, 0)]


def test_trailing_incomplete_match_is_discarded():
    """A match still in progress at the end of the tape has no outcome and must
    not be scored as a draw -- that would bias every gate toward 0.5."""
    r = tape(10.0, 0.0, 10.0)
    t = flags(False, True, False)
    assert split_matches(r, t, TH) == [(1, 0)]


def test_match_record_counts_wins_draws_losses():
    rec = match_record([(2, 1), (0, 0), (1, 3), (1, 0)])
    assert rec["wins"] == 2 and rec["draws"] == 1 and rec["losses"] == 1
    assert rec["win_share"] == pytest.approx((2 + 0.5) / 4)


def test_a_draw_counts_as_half():
    """Standard convention, and it keeps a self-play null control at exactly
    0.5 rather than pushing it around by the draw rate."""
    assert match_record([(0, 0), (0, 0)])["win_share"] == pytest.approx(0.5)


def test_no_completed_matches_is_none_not_zero():
    """0 wins of 0 matches is not 0% -- returning 0.0 would read as a total
    loss and could promote or reject on nothing."""
    assert match_record([])["win_share"] is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `nice -n 10 .venv/bin/python -m pytest tests/python/test_match_win_gate.py -q`
Expected: FAIL — `ImportError: cannot import name 'split_matches'`

- [ ] **Step 3: Write the implementation**

Add to `python/construct/league/matches.py`:

```python
def split_matches(rewards, terminated, threshold=GOAL_THRESHOLD):
    """Group a reward tape into per-match (goals_a, goals_b) using terminated
    flags as match boundaries.

    In match mode `terminated` means "the clock expired", so it is exactly the
    match boundary. A goal is still a reward spike past `threshold` -- matches
    always run reward_v0 as a neutral scoring tape (see module doc), so this
    holds whatever the policies trained on.

    A trailing partial match is DISCARDED: it has no outcome, and scoring it as
    a draw would bias every gate toward 0.5.
    """
    rewards = np.asarray(rewards)
    terminated = np.asarray(terminated)
    out, a, b = [], 0, 0
    for t in range(rewards.shape[0]):
        a += int((rewards[t] >= threshold).sum())
        b += int((rewards[t] <= -threshold).sum())
        if bool(terminated[t].any()):
            out.append((a, b))
            a, b = 0, 0
    return out


def match_record(matches):
    """Win/draw/loss counts and win share (draws count 0.5).

    `win_share` is None when no match completed -- 0.0 would read as a total
    loss and could drive a promotion decision off zero evidence.
    """
    wins = sum(1 for a, b in matches if a > b)
    losses = sum(1 for a, b in matches if a < b)
    draws = len(matches) - wins - losses
    share = None if not matches else (wins + 0.5 * draws) / len(matches)
    return {"wins": wins, "draws": draws, "losses": losses,
            "matches": len(matches), "win_share": share}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `nice -n 10 .venv/bin/python -m pytest tests/python/test_match_win_gate.py -q`
Expected: PASS, 6 tests

- [ ] **Step 5: Commit**

```bash
git add python/construct/league/matches.py tests/python/test_match_win_gate.py
git commit -m "feat(gate): split reward tape into matches; win/draw/loss record"
```

---

### Task 7: Characterise the match-win gate's null distribution

**Mandatory before any promotion is trusted.** On 2026-07-20 a pure random
perturbation PASSED the 52% goal-share threshold at 56.3%. The match-win gate
is noisier per unit compute (SE ~11.5% at 19 matches vs ~3.7% at 180 goals), so
its null distribution must be measured, not assumed.

**Files:**
- Create: `scripts/match_gate_null.py`
- Test: `tests/python/test_match_gate_null.py`

**Interfaces:**
- Consumes: `split_matches`, `match_record` (Task 6).
- Produces: `null_summary(shares: list[float]) -> dict` with keys `n`, `mean`, `sd`, `lo`, `hi`.

- [ ] **Step 1: Write the failing test**

Create `tests/python/test_match_gate_null.py`:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import match_gate_null as m  # noqa: E402


def test_null_summary_reports_spread_not_just_a_mean():
    """A mean alone hides the spread, and the spread is the entire point: it
    tells us how large a win-share difference the gate can even resolve."""
    s = m.null_summary([0.5, 0.6, 0.4, 0.55, 0.45])
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(0.5)
    assert s["sd"] > 0
    assert s["lo"] < s["mean"] < s["hi"]


def test_single_sample_has_no_defined_spread():
    s = m.null_summary([0.5])
    assert s["sd"] is None and s["lo"] is None and s["hi"] is None


def test_empty_input_is_not_a_crash():
    assert m.null_summary([])["n"] == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `nice -n 10 .venv/bin/python -m pytest tests/python/test_match_gate_null.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'match_gate_null'`

- [ ] **Step 3: Write the script**

Create `scripts/match_gate_null.py`:

```python
#!/usr/bin/env python3
"""Measure the match-win gate's null distribution: a policy against ITSELF.

Why this must exist before any promotion is trusted: on 2026-07-20 a pure
random perturbation of the champion PASSED the 52% goal-share gate at 56.3%.
That gate had SE ~3.7%. The match-win gate is far noisier per unit compute
(~19 matches in today's budget, SE ~11.5%), so a threshold picked by intuition
would promote noise routinely.

A policy played against itself must centre on 0.5. The SPREAD across seeds is
the number that matters -- it sets the smallest win-share difference this gate
can resolve, and therefore where a defensible threshold sits.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys


def null_summary(shares):
    """Mean and spread of self-play win shares. sd/lo/hi are None below n=2,
    where a spread is undefined rather than zero."""
    shares = [float(s) for s in shares]
    n = len(shares)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(shares)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "lo": None, "hi": None}
    sd = statistics.stdev(shares)
    half = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", default="checkpoints_entity/ck_000320471040.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 31)))
    ap.add_argument("--arenas", type=int, default=32)
    ap.add_argument("--steps", type=int, default=45000,
                    help="engine steps per seed; a match is ~4500 steps")
    args = ap.parse_args(argv)

    from construct.league.matches import MatchRunner, load_sd, match_record, split_matches

    sd = load_sd(args.champion)
    shares = []
    for seed in args.seeds:
        mr = MatchRunner(num_arenas=args.arenas, seed=seed, mode=1,
                         schema_version=1, net_heads=4,
                         reward_config="configs/reward_v0.toml")
        mr.eng.set_weights(sd)
        mr.eng.set_opponents([sd])
        out = mr.eng.collect(args.steps, arena_opponents=mr.assignment)
        rec = match_record(split_matches(out["rewards"], out["terminated"]))
        if rec["win_share"] is not None:
            shares.append(rec["win_share"])
            print(f"  seed {seed:5d}: {rec['wins']}W/{rec['draws']}D/{rec['losses']}L "
                  f"share={rec['win_share']:.3f}", flush=True)

    s = null_summary(shares)
    print(f"\nNULL over {s['n']} seeds: mean={s['mean']:.4f}")
    if s["sd"] is not None:
        print(f"  sd={s['sd']:.4f}  95% CI of the mean=[{s['lo']:.4f},{s['hi']:.4f}]")
        print(f"  a defensible threshold sits at least 2 sd above 0.5, "
              f"i.e. >= {0.5 + 2 * s['sd']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `nice -n 10 .venv/bin/python -m pytest tests/python/test_match_gate_null.py -q`
Expected: PASS, 3 tests

- [ ] **Step 5: Run the full suite and commit**

Run: `nice -n 10 .venv/bin/python -m pytest tests/python -q` and
`cargo test --manifest-path engine/Cargo.toml`
Expected: all green.

```bash
git add scripts/match_gate_null.py tests/python/test_match_gate_null.py
git commit -m "feat(gate): match-win null distribution characterisation

Mandatory before trusting a promotion: on 2026-07-20 pure random noise
passed the 52% goal-share gate at 56.3%, and the match-win gate is
noisier per unit compute."
```

---

## After the plan

Deployment is deliberately NOT part of this plan. Shipping a rebuilt engine
wheel to the trainer box is what created the training-engine confound on
2026-07-20 (journal entry ~14:40). When Phase 1 is ready to run:

1. Build and ship the wheel to the TRAINER box only, never the laptop.
2. Verify the `[match] match_mode ON: ...` line appears in the run log before
   trusting a single number from it.
3. Run `scripts/match_gate_null.py` and set the promote threshold from the
   measured spread — not from the goal-share gate's 0.52, which was calibrated
   for a different and much less noisy metric.
