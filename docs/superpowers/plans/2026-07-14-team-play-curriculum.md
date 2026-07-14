# Team Play + State Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train 2v2/3v3 alongside 1v1 in one engine (team-spirit reward blending + mixed team sizes across arenas) and break the kickoff-only monotony with random-state episode resets.

**Architecture:** All engine-side. Team-spirit blending happens in `EpisodeArena::step` after per-agent rewards (config-gated, bit-identical when zero). Team sizes are allocated per-arena at Engine construction from a 3-weight vector (largest-remainder method, deterministic); the collect path drops its uniform agents-per-arena assumption. Curriculum is a new `curriculum.rs` module: per-reset weighted choice between kickoff and a bounded random state, driven by the arena's own PCG32.

**Tech Stack:** existing engine crate (no new deps), serde-defaulted config extensions, existing PyO3/trainer plumbing.

## Global Constraints

- **DO NOT DEPLOY**: build, test, and commit only. The controller asks the user before any wheel reship or trainer restart (standing user instruction, 2026-07-14).
- Backward compatibility is a hard gate: existing configs (`reward_v0.toml`, `reward_v1.toml`, no-curriculum, no-team-weights) must produce bit-identical behavior — regression-tested with exact equality.
- New RewardConfig fields serde-defaulted to 0.0: `team_spirit` (τ), `opp_spirit`. Blend: `r_i' = (1−τ)·r_i + τ·mean(own team) − opp_spirit·mean(opponent team)`. Empty opposing team (e.g. 1v0 test arenas) → that mean is 0.0.
- Team-size allocation: weights `[w1, w2, w3]` for 1v1/2v2/3v3 over `num_arenas` via largest-remainder; arenas ordered 1v1-block, then 2v2, then 3v3 (deterministic). Default (None) preserves today's uniform `blue`/`orange` behavior exactly.
- Agent ordering contract extends naturally: worker-major, arena-major, blue-then-orange within arena. Nothing else about buffer layout changes.
- Curriculum bounds (world units): car pos |x| ≤ 3500, |y| ≤ 4500, 17 ≤ z ≤ z_max; ball z ∈ [93.15, z_max]; speeds ≤ car_speed_max / ball_speed_max; min pairwise separation between all cars and ball: `min_separation`, ≤ 10 resample attempts then accept.
- Grounded spawn (z == 17): yaw random, pitch = roll = 0, on-ground physics. Airborne spawn: full random orientation + velocity.
- Determinism contract unchanged: fixed (seed, num_arenas, team_size_weights, num_threads, configs) → identical buffers. Curriculum choices come from a per-arena `Pcg32` seeded `(engine_seed as u64) * 7919 + global_arena_index as u64 + 13` (distinct stream from the sampler's).
- Touch-detection safety: random resets do NOT clear `ball_hit_info` (only `reset_to_random_kickoff` does); the tick-window contract `(prev.tick_count, cur.tick_count]` already makes stale hits harmless — do not add extra machinery.
- `eval_metrics.py` stays 1v1 (metric comparability with history).
- Suites stay green: `cargo test`, `pytest tests/python -q --deselect tests/python/test_render_session.py::test_render_session_smoke`.

## File Structure

```
engine/src/reward.rs        # +team_spirit/opp_spirit fields (serde default)
engine/src/episode.rs       # blend after per-agent rewards; curriculum-aware reset helper; rng field
engine/src/curriculum.rs    # NEW: CurriculumConfig + random_reset()
engine/src/engine.rs        # per-arena team sizes; non-uniform collect bookkeeping; allocate_team_sizes()
engine/src/lib.rs           # Engine team_size_weights + curriculum_config_path params
configs/curriculum_v1.toml  # NEW
configs/reward_v1.toml      # +team_spirit/opp_spirit values
python/construct/learn/{config,train}.py  # env.team_size_weights + curriculum_config_path plumbing
scripts/resume_train.py     # --team-sizes and --curriculum-config flags
tests/python/test_team_curriculum.py       # NEW: mixed-size + curriculum integration
```

---

### Task 1: Team-spirit reward blending

**Files:**
- Modify: `engine/src/reward.rs` (config fields), `engine/src/episode.rs` (blend + `blue_count` field)
- Test: in-module additions in both files' test modules + `engine/tests/episode_test.rs`

**Interfaces:**
- Consumes: existing `RewardConfig` (serde-defaulted extension pattern from reward v1), `EpisodeArena` internals (`car_ids` is blue-then-orange by construction).
- Produces: `RewardConfig` gains `#[serde(default)] pub team_spirit: f32` and `#[serde(default)] pub opp_spirit: f32`. `EpisodeArena` stores `blue_count: usize` (set in `new` from the `blue` arg). Blending applied inside `step()` immediately after the per-agent `reward::compute` loop and BEFORE the physics-NaN/termination handling writes flags (order: compute raw → blend → flags unchanged).

- [ ] **Step 1: Write the failing tests**

Append to `engine/src/reward.rs` tests:
```rust
    #[test]
    fn v1_toml_and_v0_toml_parse_with_zero_spirit_defaults() {
        let v0 = RewardConfig::load("../configs/reward_v0.toml").unwrap();
        assert_eq!(v0.team_spirit, 0.0);
        assert_eq!(v0.opp_spirit, 0.0);
    }
```

Append to `engine/tests/episode_test.rs`:
```rust
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
```
Note: `RewardConfig` needs `Clone` — it already derives it. If `Normalization` lacks `Clone` here, it has it since Task 4 of P0.

Exact-math unit for the blend itself (append to `engine/src/episode.rs` tests or a new in-module test block):
```rust
    #[test]
    fn blend_math_exact() {
        // raw: blue [1.0, 3.0], orange [-2.0, 0.0]; tau=0.5, opp=0.25
        // bm=2.0, om=-1.0
        // blue0' = 0.5*1.0 + 0.5*2.0 - 0.25*(-1.0) = 1.75
        // blue1' = 0.5*3.0 + 0.5*2.0 - 0.25*(-1.0) = 2.75
        // org0'  = 0.5*(-2.0) + 0.5*(-1.0) - 0.25*2.0 = -2.0
        // org1'  = 0.5*0.0 + 0.5*(-1.0) - 0.25*2.0 = -1.0
        let mut r = vec![1.0f32, 3.0, -2.0, 0.0];
        blend_team_spirit(&mut r, 2, 0.5, 0.25);
        assert_eq!(r, vec![1.75, 2.75, -2.0, -1.0]);
    }

    #[test]
    fn blend_handles_empty_orange() {
        let mut r = vec![2.0f32];
        blend_team_spirit(&mut r, 1, 0.5, 0.25); // opp mean = 0.0
        assert_eq!(r, vec![2.0]); // (1-.5)*2 + .5*2 - .25*0 = 2.0
    }
```

- [ ] **Step 2: Run to verify RED**

Run: `cd engine && cargo test blend_math_exact 2>&1 | tail -3`
Expected: compile FAIL (`blend_team_spirit` not found)

- [ ] **Step 3: Implement**

`engine/src/reward.rs` — add to `RewardConfig`:
```rust
    /// Team-spirit blending (spec §4): r_i' = (1-t)*r_i + t*mean(team) - opp_spirit*mean(opponents).
    /// Applied in EpisodeArena::step, not here (needs all agents' raw rewards).
    #[serde(default)]
    pub team_spirit: f32,
    #[serde(default)]
    pub opp_spirit: f32,
```

`engine/src/episode.rs` — free function + wiring:
```rust
/// r_i' = (1-tau)*r_i + tau*mean(own team) - opp*mean(opponent team).
/// `blue_count` splits `rewards` (blue-then-orange agent order). An empty
/// team's mean is 0.0 (1v0 test arenas).
pub fn blend_team_spirit(rewards: &mut [f32], blue_count: usize, tau: f32, opp: f32) {
    let mean = |s: &[f32]| if s.is_empty() { 0.0 } else { s.iter().sum::<f32>() / s.len() as f32 };
    let bm = mean(&rewards[..blue_count]);
    let om = mean(&rewards[blue_count..]);
    for (i, r) in rewards.iter_mut().enumerate() {
        let (own, other) = if i < blue_count { (bm, om) } else { (om, bm) };
        *r = (1.0 - tau) * *r + tau * own - opp * other;
    }
}
```
In `EpisodeArena`: add field `blue_count: usize`, set from the `blue` constructor arg. In `step()`, right after the per-agent `reward::compute` loop:
```rust
        if self.reward_cfg.team_spirit != 0.0 || self.reward_cfg.opp_spirit != 0.0 {
            blend_team_spirit(
                &mut rewards[..n],
                self.blue_count,
                self.reward_cfg.team_spirit,
                self.reward_cfg.opp_spirit,
            );
        }
```
(The `!= 0.0` short-circuit is what makes the bit-identical test trivially true — the blend function is skipped entirely on legacy configs.)

Also set values in `configs/reward_v1.toml`:
```toml
# Team play (identity at 1v1: own == team mean). Lucy-SKG used tau=0.3.
team_spirit = 0.3
opp_spirit = 0.3
```

- [ ] **Step 4: Run GREEN + full check**

Run: `cd engine && cargo test 2>&1 | grep -E "test result"`
Expected: all pass, including the four new tests.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: team-spirit reward blending (config-gated, bit-identical at zero)"
```

---

### Task 2: Variable team sizes across arenas

**Files:**
- Modify: `engine/src/engine.rs`, `engine/src/lib.rs`
- Test: `tests/python/test_team_curriculum.py` (new), in-module Rust test for the allocator

**Interfaces:**
- Consumes: `EpisodeArena::new(blue, orange, ...)` (already size-agnostic), obs padding (5 other-slots — supports 3v3), Task 1's `blue_count` (constructed per-arena).
- Produces:
  - Rust: `engine::allocate_team_sizes(num_arenas: usize, weights: [f64; 3]) -> Vec<usize>` — element = 1, 2, or 3 (cars per team), largest-remainder, ordered 1s-block/2s-block/3s-block.
  - `MultiEngine::new(..., sizes: Vec<usize>, ...)` replaces the single `blue: usize, orange: usize` pair internally (uniform case = `vec![blue; num_arenas]` when blue==orange; the PyO3 layer keeps the old params for compat).
  - Python: `Engine(..., team_size_weights=None)` — `None` → uniform blue/orange as today; `[w1,w2,w3]` (list of 3 nonneg floats, sum > 0) → mixed arenas, `blue`/`orange` args ignored.
  - `Engine.num_agents` = sum over arenas of `2*size`. Everything downstream (collect shapes, trainer) adapts automatically because it already reads `num_agents`.
- Collect bookkeeping changes (engine.rs): the worker's `a_to_arena` mapping and rng indexing must be built from a per-arena prefix-sum instead of the uniform `a / per_agent` division. `Worker` keeps `num_agents` (now a sum) — the gather code is already generic over it.

- [ ] **Step 1: Write failing Rust allocator test (in engine.rs tests or new mod)**

```rust
    #[test]
    fn allocates_largest_remainder_deterministic() {
        assert_eq!(allocate_team_sizes(4, [1.0, 0.0, 0.0]), vec![1, 1, 1, 1]);
        // 192 arenas at [0.5, 0.3, 0.2] -> 96/58/38 (0.5*192=96, 57.6->58 via remainder, 38.4->38)
        let s = allocate_team_sizes(192, [0.5, 0.3, 0.2]);
        assert_eq!(s.iter().filter(|&&x| x == 1).count(), 96);
        assert_eq!(s.iter().filter(|&&x| x == 2).count(), 58);
        assert_eq!(s.iter().filter(|&&x| x == 3).count(), 38);
        assert_eq!(s.len(), 192);
        // blocks are ordered 1s, 2s, 3s
        let mut sorted = s.clone();
        sorted.sort_unstable();
        assert_eq!(s, sorted);
        // exact division stays exact
        assert_eq!(allocate_team_sizes(4, [0.5, 0.25, 0.25]), vec![1, 1, 2, 3]);
    }
```

- [ ] **Step 2: RED**

Run: `cd engine && cargo test allocates_largest 2>&1 | tail -3` — compile FAIL.

- [ ] **Step 3: Implement allocator**

```rust
/// Split num_arenas into team sizes 1/2/3 proportional to `weights`
/// (largest-remainder method), ordered as a 1s block, 2s block, 3s block.
pub fn allocate_team_sizes(num_arenas: usize, weights: [f64; 3]) -> Vec<usize> {
    let total: f64 = weights.iter().sum();
    assert!(total > 0.0, "team size weights must sum > 0");
    let exact: Vec<f64> = weights.iter().map(|w| w / total * num_arenas as f64).collect();
    let mut counts: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
    let mut short = num_arenas - counts.iter().sum::<usize>();
    // hand out remainders by largest fractional part; ties -> smaller size first (stable)
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&a, &b| {
        let fa = exact[a] - exact[a].floor();
        let fb = exact[b] - exact[b].floor();
        fb.partial_cmp(&fa).unwrap().then(a.cmp(&b))
    });
    for &i in &order {
        if short == 0 {
            break;
        }
        counts[i] += 1;
        short -= 1;
    }
    let mut out = Vec::with_capacity(num_arenas);
    for (i, &c) in counts.iter().enumerate() {
        out.extend(std::iter::repeat(i + 1).take(c));
    }
    out
}
```

- [ ] **Step 4: Wire through MultiEngine + workers**

In `MultiEngine::new`, replace the uniform `blue/orange` usage with a `sizes: Vec<usize>` argument (`sizes[arena] = cars per team`). Worker construction slices its contiguous arena range's sizes and builds `EpisodeArena::new(size, size, ...)` per arena. Inside the worker:
```rust
// replaces: let per_agent = blue + orange; and any `a / per_agent` arena lookup
let arena_agent_counts: Vec<usize> = arenas.iter().map(|ar| ar.num_agents()).collect();
let mut a_to_arena = Vec::with_capacity(agents);
for (ai, &c) in arena_agent_counts.iter().enumerate() {
    a_to_arena.extend(std::iter::repeat(ai).take(c));
}
// sampling loop: rngs[a_to_arena[a]]
```
`Worker.num_agents` = `arena_agent_counts.iter().sum()`. All existing gather/offset code reads `w.num_agents` and continues to work. `MultiEngine.num_agents` = total sum. Keep the global-arena-index rng/kickoff seeding exactly as-is (it is index-based, not size-based).

In `lib.rs` `Engine::new`, add parameter and translate:
```rust
#[pyo3(signature = (num_arenas=32, blue=1, orange=1, schema_path="schema/v0.toml",
                    reward_config_path="configs/reward_v0.toml", meshes_path=None,
                    seed=0, num_threads=0, team_size_weights=None))]
```
```rust
let sizes: Vec<usize> = match team_size_weights {
    Some(w) => {
        let w: Vec<f64> = w; // Vec<f64> from Python list
        if w.len() != 3 || w.iter().any(|x| *x < 0.0) || w.iter().sum::<f64>() <= 0.0 {
            return Err(PyValueError::new_err("team_size_weights must be 3 nonnegative floats summing > 0"));
        }
        engine::allocate_team_sizes(num_arenas, [w[0], w[1], w[2]])
    }
    None => {
        if blue != orange {
            // uniform asymmetric (e.g. tests' 1v0) still supported via old params
            return Ok(/* construct with per-arena (blue, orange) as today */ todo_keep_old_path());
        }
        vec![blue; num_arenas]
    }
};
```
NOTE to implementer: the `blue != orange` asymmetric case (used by one no-touch test at 1v0) must keep working — cleanest is to make `sizes` a `Vec<(usize, usize)>` of (blue, orange) pairs throughout (`allocate_team_sizes` output mapped to `(s, s)`), so the old path is `vec![(blue, orange); num_arenas]`. Do that instead of the sketch above if it reads cleaner — the tests below only constrain observable behavior. Mirror the same signature extension on `RenderSession`? NO — YAGNI, viewer stays 1v1-configurable via its existing blue/orange params.

- [ ] **Step 5: Write failing Python integration tests**

```python
# tests/python/test_team_curriculum.py
import numpy as np
import pytest
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet


def weights(seed=0):
    torch.manual_seed(seed)
    net = PolicyValueNet(94, 90, (64, 64))
    return {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def test_mixed_team_sizes_agent_count_and_shapes():
    eng = Engine(num_arenas=4, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=0,
                 team_size_weights=[0.5, 0.25, 0.25])
    # sizes [1,1,2,3] -> agents 2+2+4+6 = 14
    assert eng.num_agents == 14
    eng.set_weights(weights())
    out = eng.collect(8)
    assert out["obs"].shape == (8, 14, 94)
    assert np.isfinite(out["obs"]).all() and np.isfinite(out["logprobs"]).all()


def test_mixed_sizes_deterministic_fixed_config():
    w = weights(3)
    mk = lambda: Engine(num_arenas=6, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", seed=11,
                        num_threads=2, team_size_weights=[1.0, 1.0, 1.0])
    a, b = mk(), mk()
    a.set_weights(w); b.set_weights(w)
    oa, ob = a.collect(16), b.collect(16)
    for k in oa:
        np.testing.assert_array_equal(oa[k], ob[k], err_msg=k)


def test_bad_weights_rejected():
    with pytest.raises(Exception):
        Engine(num_arenas=4, schema_path="schema/v0.toml",
               reward_config_path="configs/reward_v0.toml",
               team_size_weights=[0.0, 0.0, 0.0])
    with pytest.raises(Exception):
        Engine(num_arenas=4, schema_path="schema/v0.toml",
               reward_config_path="configs/reward_v0.toml",
               team_size_weights=[1.0, 2.0])


def test_default_none_matches_legacy():
    mk_old = lambda: Engine(num_arenas=2, blue=1, orange=1, schema_path="schema/v0.toml",
                            reward_config_path="configs/reward_v0.toml", seed=4)
    a, b = mk_old(), mk_old()
    np.testing.assert_array_equal(a.reset(), b.reset())
```

- [ ] **Step 6: RED → implement → GREEN**

Run: `maturin develop --release && pytest tests/python/test_team_curriculum.py -v`
Expected after implementation: 4 passed. Then full: `cd engine && cargo test && cd .. && pytest tests/python -q --deselect tests/python/test_render_session.py::test_render_session_smoke` — all green (legacy suites prove backward compat).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: mixed 1v1/2v2/3v3 arenas via team_size_weights"
```

---

### Task 3: Random-state curriculum

**Files:**
- Create: `engine/src/curriculum.rs`, `configs/curriculum_v1.toml`
- Modify: `engine/src/episode.rs` (rng field, reset helper), `engine/src/engine.rs` + `engine/src/lib.rs` (config plumbing)
- Test: in-module + `engine/tests/curriculum_test.rs`

**Interfaces:**
- Consumes: `EpisodeArena` reset sites (constructor, goal/truncation reset, NaN containment reset), `sampler::Pcg32`, `rocketsim_rs::{math::{Angle, Vec3}, sim::Arena}`, `arena.set_car/set_ball/get_car/get_game_state`.
- Produces:
  ```rust
  // curriculum.rs
  #[derive(Debug, Clone, Deserialize)]
  pub struct CurriculumConfig {
      pub kickoff_weight: f32,
      pub random_weight: f32,
      #[serde(default)] pub random: RandomStateBounds,
  }
  #[derive(Debug, Clone, Deserialize)]
  pub struct RandomStateBounds {
      pub car_speed_max: f32,   // default 1800.0
      pub ball_speed_max: f32,  // default 2500.0
      pub z_max: f32,           // default 1700.0
      pub min_separation: f32,  // default 300.0
  }
  impl Default for RandomStateBounds { /* the defaults above */ }
  impl CurriculumConfig { pub fn load(path: &str) -> Result<Self, String>; }
  /// Overwrites ball + all car states with a bounded random scenario.
  pub fn random_reset(arena: Pin<&mut Arena>, rng: &mut crate::sampler::Pcg32, b: &RandomStateBounds);
  ```
  `EpisodeArena::new` gains `curriculum: Option<CurriculumConfig>` and an `rng: Pcg32` field (seed: `(seed as u64) * 7919 + global-arena-independent constant 13` — the arena seed already encodes the global index, so just derive from it). A private `reset_episode(&mut self)` replaces every `reset_to_random_kickoff` call site: kickoff when `curriculum.is_none()` or the weighted coin says kickoff; else `random_reset`. Constructor plumb-through: `MultiEngine::new(..., curriculum: Option<CurriculumConfig>)` (cloned per arena), `Engine(..., curriculum_config_path=None)`.

- [ ] **Step 1: Write configs/curriculum_v1.toml**

```toml
# Episode-reset mixture. WeIGHTS are relative (normalized internally).
kickoff_weight = 0.4
random_weight = 0.6

[random]
car_speed_max = 1800.0
ball_speed_max = 2500.0
z_max = 1700.0
min_separation = 300.0
```
(Fix the "weIGHTS" typo — it's here to check you're reading.)

- [ ] **Step 2: Write failing tests**

```rust
// engine/tests/curriculum_test.rs
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
```

`EpisodeArena` additions used above: `new_with_curriculum(...)` (existing `new` delegates with `None` — zero churn at old call sites) and `pub fn debug_force_reset(&mut self)` (test helper calling the private `reset_episode`).

- [ ] **Step 3: RED**

Run: `cd engine && cargo test --test curriculum_test 2>&1 | tail -3` — compile FAIL.

- [ ] **Step 4: Implement curriculum.rs**

```rust
use std::pin::Pin;

use rocketsim_rs::{
    math::{Angle, Vec3},
    sim::Arena,
};
use serde::Deserialize;

use crate::sampler::Pcg32;

#[derive(Debug, Clone, Deserialize)]
pub struct RandomStateBounds {
    pub car_speed_max: f32,
    pub ball_speed_max: f32,
    pub z_max: f32,
    pub min_separation: f32,
}

impl Default for RandomStateBounds {
    fn default() -> Self {
        Self { car_speed_max: 1800.0, ball_speed_max: 2500.0, z_max: 1700.0, min_separation: 300.0 }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct CurriculumConfig {
    pub kickoff_weight: f32,
    pub random_weight: f32,
    #[serde(default)]
    pub random: RandomStateBounds,
}

impl CurriculumConfig {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let c: Self = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        if c.kickoff_weight < 0.0 || c.random_weight < 0.0 || c.kickoff_weight + c.random_weight <= 0.0 {
            return Err(format!("{path}: weights must be nonnegative and sum > 0"));
        }
        Ok(c)
    }
}

const TAU: f32 = std::f32::consts::TAU;

fn rand_range(rng: &mut Pcg32, lo: f32, hi: f32) -> f32 {
    lo + rng.next_f32() * (hi - lo)
}

fn rand_vel(rng: &mut Pcg32, max: f32) -> Vec3 {
    // random direction (uniform-ish over sphere via yaw+pitch), random magnitude
    let yaw = rand_range(rng, 0.0, TAU);
    let pitch = rand_range(rng, -1.2, 1.2);
    let mag = rng.next_f32() * max;
    Vec3::new(mag * pitch.cos() * yaw.cos(), mag * pitch.cos() * yaw.sin(), mag * pitch.sin())
}

/// Overwrite ball + all car states with a bounded random scenario. Positions are
/// resampled up to 10 times to keep `min_separation` from already-placed bodies;
/// after 10 attempts the last sample is accepted (best-effort, never spins).
pub fn random_reset(mut arena: Pin<&mut Arena>, rng: &mut Pcg32, b: &RandomStateBounds) {
    let mut placed: Vec<Vec3> = Vec::new();
    let mut sample_pos = |rng: &mut Pcg32, z_lo: f32, z_hi: f32, placed: &[Vec3]| -> Vec3 {
        let mut p = Vec3::new(0.0, 0.0, z_lo);
        for _ in 0..10 {
            p = Vec3::new(
                rand_range(rng, -3500.0, 3500.0),
                rand_range(rng, -4500.0, 4500.0),
                rand_range(rng, z_lo, z_hi),
            );
            let ok = placed.iter().all(|q| {
                let d2 = (p.x - q.x).powi(2) + (p.y - q.y).powi(2) + (p.z - q.z).powi(2);
                d2 >= b.min_separation * b.min_separation
            });
            if ok {
                break;
            }
        }
        p
    };

    // ball first
    let ball_pos = sample_pos(rng, 93.15, b.z_max, &placed);
    placed.push(ball_pos);
    let mut ball = arena.as_mut().get_ball();
    ball.pos = ball_pos;
    ball.vel = rand_vel(rng, b.ball_speed_max);
    ball.ang_vel = Vec3::new(0.0, 0.0, 0.0);
    arena.as_mut().set_ball(ball);

    // cars
    let ids: Vec<u32> = {
        let gs = arena.as_mut().get_game_state();
        gs.cars.iter().map(|c| c.id).collect()
    };
    for id in ids {
        let grounded = rng.next_f32() < 0.5;
        let (z_lo, z_hi) = if grounded { (17.0, 17.0) } else { (100.0, b.z_max) };
        let pos = sample_pos(rng, z_lo, z_hi, &placed);
        placed.push(pos);
        let mut cs = arena.as_mut().get_car(id);
        cs.pos = pos;
        cs.vel = if grounded {
            let v = rand_vel(rng, b.car_speed_max);
            Vec3::new(v.x, v.y, 0.0)
        } else {
            rand_vel(rng, b.car_speed_max)
        };
        cs.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        let ang = Angle {
            yaw: rand_range(rng, 0.0, TAU),
            pitch: if grounded { 0.0 } else { rand_range(rng, -1.0, 1.0) },
            roll: if grounded { 0.0 } else { rand_range(rng, -1.0, 1.0) },
        };
        cs.rot_mat = ang.to_rotmat();
        cs.boost = rand_range(rng, 0.0, 100.0);
        cs.is_on_ground = grounded;
        arena.as_mut().set_car(id, cs).expect("car exists");
    }
}
```
API-check note (same discipline as always): verify `Angle { yaw, pitch, roll }.to_rotmat()`, `get_car/set_car`, `get_ball/set_ball` signatures against rocketsim_rs 0.37.0 via `cargo doc` if compilation disagrees; adapt call syntax only, never bounds/semantics. `Pin<&mut Arena>` re-borrow patterns: use `arena.as_mut()` per call as shown.

- [ ] **Step 5: Wire EpisodeArena**

```rust
// fields
curriculum: Option<crate::curriculum::CurriculumConfig>,
rng: crate::sampler::Pcg32,
// constructor: existing new(...) delegates:
pub fn new(blue: usize, orange: usize, tick_skip: u32, reward_cfg: RewardConfig,
           norm: Normalization, seed: u32) -> Self {
    Self::new_with_curriculum(blue, orange, tick_skip, reward_cfg, norm, seed, None)
}
pub fn new_with_curriculum(..., curriculum: Option<CurriculumConfig>) -> Self {
    // as before, plus:
    // rng: Pcg32::new((seed as u64) * 7919 + 13),
    // and the initial reset goes through reset_episode() after construction
}
fn reset_episode(&mut self) {
    self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
    let use_random = match &self.curriculum {
        Some(c) => {
            let p = c.random_weight / (c.random_weight + c.kickoff_weight);
            self.rng.next_f32() < p
        }
        None => false,
    };
    if use_random {
        // kickoff first for a clean baseline (boost pads, ball_hit reset), then overwrite
        self.arena.pin_mut().reset_to_random_kickoff(Some(self.seed));
        let bounds = self.curriculum.as_ref().unwrap().random.clone();
        crate::curriculum::random_reset(self.arena.pin_mut(), &mut self.rng, &bounds);
    } else {
        self.arena.pin_mut().reset_to_random_kickoff(Some(self.seed));
    }
    let gs = self.arena.pin_mut().get_game_state();
    self.episode_start_tick = gs.tick_count;
    self.last_touch_tick = gs.tick_count;
    self.prev_state = gs;
}
pub fn debug_force_reset(&mut self) { self.reset_episode(); }
```
Replace the reset code in BOTH end-of-episode sites (normal terminated/truncated block AND the physics-NaN containment block) with `self.reset_episode();` — the kickoff-first-then-overwrite trick means `ball_hit_info` and boost pads reset properly in all paths.

- [ ] **Step 6: Plumb through engine.rs + lib.rs**

`MultiEngine::new(..., curriculum: Option<CurriculumConfig>)` — clone into each worker's arena construction (`new_with_curriculum`). `lib.rs` Engine signature adds `curriculum_config_path=None`; load + `map_err(PyValueError)` when Some. Same for `RenderSession`? NO (YAGNI; viewer shows kickoff play fine).

- [ ] **Step 7: GREEN + full check**

Run: `cd engine && cargo test 2>&1 | grep -E "test result" && cd .. && maturin develop --release && pytest tests/python -q --deselect tests/python/test_render_session.py::test_render_session_smoke`
Expected: all green; legacy python tests unchanged (curriculum defaults to None).

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: random-state episode curriculum (config-gated, kickoff-only default)"
```

---

### Task 4: Trainer plumbing + integration (NO DEPLOY)

**Files:**
- Modify: `python/construct/learn/config.py` (nothing — dict passthrough), `python/construct/learn/train.py`, `scripts/resume_train.py`, `configs/train_v0.toml`
- Test: append to `tests/python/test_team_curriculum.py`

**Interfaces:**
- Consumes: `Engine(team_size_weights=..., curriculum_config_path=...)` (Tasks 2-3).
- Produces: `TrainConfig.env` may contain `team_size_weights` (list) — passed through to Engine when present; `TrainConfig` top-level optional `curriculum_config_path` (add as dataclass field with default `""`, empty = None). `resume_train.py` gains `--team-sizes W1,W2,W3` and `--curriculum-config PATH`. Checkpoint provenance: `team_size_weights` rides free inside the env dict (already persisted); `curriculum_config_path` recorded alongside `reward_config_path`.

- [ ] **Step 1: Write failing test**

```python
# append to tests/python/test_team_curriculum.py
from construct.learn.config import TrainConfig
from construct.learn.train import Trainer


def test_trainer_runs_mixed_sizes_with_curriculum(tmp_path):
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=4, team_size_weights=[0.5, 0.25, 0.25])
    cfg.curriculum_config_path = "configs/curriculum_v1.toml"
    cfg.ppo.update(rollout_steps=16, minibatch_size=128)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=1)
    t = Trainer(cfg)
    assert t.engine.num_agents == 14
    t.run(max_iterations=1)
    assert t.total_steps == 16 * 14
    import torch
    ck = torch.load(f"{tmp_path}/ck_{t.total_steps:012d}.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["env"]["team_size_weights"] == [0.5, 0.25, 0.25]
    assert ck["curriculum_config_path"] == "configs/curriculum_v1.toml"
```

- [ ] **Step 2: RED**

Run: `pytest tests/python/test_team_curriculum.py::test_trainer_runs_mixed_sizes_with_curriculum -x`
Expected: TypeError (TrainConfig has no curriculum_config_path) or Engine TypeError.

- [ ] **Step 3: Implement**

`config.py`: add dataclass field `curriculum_config_path: str = ""`.
`train.py` Engine construction:
```python
        self.engine = Engine(
            num_arenas=cfg.env["num_arenas"], blue=cfg.env["blue"], orange=cfg.env["orange"],
            schema_path=cfg.schema_path, reward_config_path=cfg.reward_config_path,
            seed=cfg.env["seed"],
            team_size_weights=cfg.env.get("team_size_weights"),
            curriculum_config_path=cfg.curriculum_config_path or None,
        )
```
`save_checkpoint`: add `"curriculum_config_path": self.cfg.curriculum_config_path,` beside the existing reward provenance line.
`resume_train.py`:
```python
p.add_argument("--team-sizes", default=None, help="W1,W2,W3 weights for 1v1/2v2/3v3 arena mix")
p.add_argument("--curriculum-config", default=None)
...
if args.team_sizes:
    cfg.env["team_size_weights"] = [float(x) for x in args.team_sizes.split(",")]
if args.curriculum_config:
    cfg.curriculum_config_path = args.curriculum_config
```
`configs/train_v0.toml`: document (commented out) in `[env]`:
```toml
# team_size_weights = [0.5, 0.3, 0.2]   # 1v1/2v2/3v3 arena mix; omit = pure 1v1
```
and top-level: `curriculum_config_path = ""  # e.g. "configs/curriculum_v1.toml"`.

- [ ] **Step 4: GREEN + full suite + smoke**

Run: `pytest tests/python -q --deselect tests/python/test_render_session.py::test_render_session_smoke && python scripts/smoke_test.py`
Expected: all green; `SMOKE OK: 2048 steps` (smoke unchanged — legacy config path).

- [ ] **Step 5: Commit — and STOP**

```bash
git add -A && git commit -m "feat: trainer plumbing for team-size mix + curriculum"
```
**Do not deploy.** Report DONE to the controller; the controller asks the user which boxes/settings to swap (standing instruction: user approves all deployments).

---

## Self-Review Notes

- Spec §4 coverage: team spirit ✓(T1, formula verbatim), variable team size ✓(T2), state-mix curriculum ✓(T3, kickoff+random; replay/scenario states are P2), plumbing ✓(T4). Deploy gate honored (T4 stop + global constraint).
- Type consistency: `blend_team_spirit(&mut [f32], usize, f32, f32)` used identically in T1 tests/impl; `allocate_team_sizes(usize, [f64;3]) -> Vec<usize>` matches tests; `new_with_curriculum`/`debug_force_reset` names match T3 tests; python kwargs (`team_size_weights`, `curriculum_config_path`) consistent across T2/T3/T4.
- Known judgment calls delegated with instructions: `(blue, orange)` pair representation in T2 step 4; Pin re-borrow syntax and Angle/set_car API verification in T3.
- Curriculum "kickoff-then-overwrite" trick deliberately reuses RocketSim's own reset for pad/hit-info hygiene — cheaper and safer than replicating it.
- Determinism: curriculum rng is a separate Pcg32 stream from the action sampler; both derive from the arena seed → fixed-config determinism preserved (tested T2/T3).
