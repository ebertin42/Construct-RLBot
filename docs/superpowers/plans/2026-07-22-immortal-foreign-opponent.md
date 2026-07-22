# Immortal Foreign Opponent — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the community bot **Immortal** (a foreign-architecture torch policy) as an in-engine league *training opponent*, so PPO trains against a strong externally-derived opponent instead of only our own checkpoints.

**Architecture:** Immortal is ported into the Rust engine as a new **"foreign" opponent kind** that operates at the arena level: for each orange car of a foreign-assigned arena it builds Immortal's own 107-float AdvancedObs from the live `GameState`, runs Immortal's MLP (candle), argmaxes over its 126-row action table, and applies the resulting controls-8 directly to that car (a per-car override of the shared action-index→controls step). Native self-play and `EntityPolicy` opponents are untouched. Fidelity is enforced by golden tests against the Python reference at every layer.

**Tech Stack:** Rust (engine crate, `construct_engine`), candle-core/candle-nn 0.11, PyO3 0.29 (abi3-py311), maturin, rocketsim_rs 0.37, Python 3.11 + torch (reference only), pytest.

## Global Constraints

- **Naming:** the in-engine kind is **`foreign`** / `ForeignPolicy`. Do NOT call it "external" — `scripts/bench_external.py` + `docs/external-bench.md` already use "external" for the real-game RLBot benchmark path; overloading the name will confuse.
- **Immortal weights are unlicensed** (RLMarlbot, no stated license). NEVER commit `jit.pt` or the extracted weights (`*.npz`/`*.safetensors`) to git. Store on the training box / local cache only; pin SHA256 in a manifest. Private-research training use only.
- **Source pin:** RLMarlbot @ commit `9963b3bd328267d424ebac4a63429422df053e9c`, files `rlmarlbot/immortal/{jit.pt, agent.py, action/actionparser.py, obs/advanced_obs.py, bot.py}`.
- **Immortal MLP (verified from jit.pt):** `net` = `nn.Sequential`, 7 `Linear` layers `107→512→512→512→512→512→512→126`, 6 `LeakyReLU(negative_slope=0.01)` between them, single 126-wide head. Param names `net.{0,2,4,6,8,10,12}.{weight,bias}` (torch `[out,in]` weight, `[out]` bias). No buffers. Deterministic play = `argmax` over the 126 logits.
- **AdvancedObs normalizers:** `POS_STD = 2300` (pos & vel), `ANG_STD = π` (angular velocity). NOT the engine's `ang_vel_norm = 1/5.5`.
- **Cadence:** engine `tick_skip = 8` (15 Hz); Immortal trained at 6 (20 Hz). v1 accepts the 8-tick cadence (foreign opponent decides on our clock). Documented, not fixed here.
- **Build loop after any Rust edit:** `cargo test --manifest-path engine/Cargo.toml` → `maturin develop --release` → `pytest tests/python -q`.
- **Determinism:** keep the existing per-agent `sample_categorical` rng draw for every agent (stream intact); the foreign path OVERRIDES foreign-car output deterministically after sampling. Never remove an rng draw.

---

## File Structure

- Create `engine/src/obs_advanced.rs` — `build_advanced_obs(state, car_idx, prev_action, out)` → 107 floats. One responsibility: reproduce Immortal's AdvancedObs byte-exact.
- Create `engine/src/foreign.rs` — `ForeignKind` enum, `ForeignPolicy` struct (candle MLP + action table + per-car prev_action), `ForeignPolicy::new(dict, kind)`, `ForeignPolicy::decide(state, car_idx) -> [f32;8]`.
- Modify `engine/src/actions.rs` — add `make_immortal_table() -> Vec<[f32;8]>` (126 rows).
- Modify `engine/src/engine.rs` — `NetWeights::Foreign` variant; per-worker `opponents_foreign` storage; route foreign-assigned orange cars in `collect_v1_worker`.
- Modify `engine/src/episode.rs` — accept a per-car controls override in the v1 step.
- Modify `engine/src/lib.rs` — `set_foreign_opponents(...)` PyO3 method + parse/validate.
- Create `scripts/gen_immortal_obs_fixture.py` — vendored AdvancedObs reference → `engine/tests/fixtures/immortal_obs.json`.
- Create `scripts/gen_immortal_action_fixture.py` — reference 126-table → `engine/tests/fixtures/immortal_action.json`.
- Create `scripts/extract_immortal_weights.py` — `jit.pt` → `immortal_weights.npz` (out of git) + SHA manifest.
- Create `engine/tests/immortal_obs_test.rs`, `engine/tests/immortal_action_test.rs`, `engine/tests/foreign_mlp_test.rs`.
- Create `tests/python/test_foreign_opponent.py` — end-to-end + non-regression.
- Modify `python/construct/learn/train.py` — allow a foreign opponent id in the pool.
- Create `docs/foreign-opponents.md` — how to extract weights + run.

---

## Phase 1 — Pure Rust primitives (no engine wiring)

Each task here is fully testable in isolation and carries the highest silent-bug risk.

### Task 1: Immortal 126-row action table (Rust)

**Files:**
- Modify: `engine/src/actions.rs` (append after `make_lookup_table_v1`, ~line 58)
- Create: `scripts/gen_immortal_action_fixture.py`
- Create: `engine/tests/fixtures/immortal_action.json`
- Create: `engine/tests/immortal_action_test.rs`

**Interfaces:**
- Produces: `pub fn make_immortal_table() -> Vec<[f32; 8]>` (exactly 126 rows).

- [ ] **Step 1: Write the reference fixture generator**

Create `scripts/gen_immortal_action_fixture.py`:

```python
"""Emit Immortal's 126-row action lookup table as a fixture for the Rust golden
test. Reproduces rlmarlbot/immortal/action/actionparser.py::ImmortalAction
exactly (verified 126x8). Weights-free -- safe to commit."""
import json, pathlib

def make_table():
    actions = []
    # Ground (36 rows)
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    # Aerial (90 rows)
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if pitch == roll == jump == 0:
                            continue
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, 1])
    return actions

def main():
    table = make_table()
    assert len(table) == 126, len(table)
    out = pathlib.Path("engine/tests/fixtures/immortal_action.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"table": table}))
    print(f"wrote {out} ({len(table)} rows)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the fixture**

Run: `cd /home/steamo/Construct-RLBot && .venv/bin/python scripts/gen_immortal_action_fixture.py`
Expected: `wrote engine/tests/fixtures/immortal_action.json (126 rows)`

- [ ] **Step 3: Write the failing Rust test**

Create `engine/tests/immortal_action_test.rs`:

```rust
use construct_engine::actions::make_immortal_table;

#[test]
fn immortal_table_matches_python_fixture() {
    let raw = std::fs::read_to_string("tests/fixtures/immortal_action.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let rows = v["table"].as_array().unwrap();
    let table = make_immortal_table();
    assert_eq!(table.len(), 126, "row count");
    assert_eq!(rows.len(), 126, "fixture row count");
    for (i, row) in rows.iter().enumerate() {
        for j in 0..8 {
            let expect = row[j].as_f64().unwrap() as f32;
            assert_eq!(table[i][j], expect, "row {i} col {j}");
        }
    }
}
```

Note: `make_immortal_table` must be reachable — ensure `actions` is a `pub mod` in `engine/src/lib.rs` (it is; `actions::to_controls` is already used cross-module). `serde_json` is already a dev-dependency (used by other fixture tests); confirm in `engine/Cargo.toml` `[dev-dependencies]` and add `serde_json = "1"` there if missing.

- [ ] **Step 4: Run test to verify it fails**

Run: `cargo test --manifest-path engine/Cargo.toml --test immortal_action_test`
Expected: FAIL — `cannot find function make_immortal_table`.

- [ ] **Step 5: Implement `make_immortal_table`**

Append to `engine/src/actions.rs`:

```rust
/// Immortal's (RLMarlbot) 126-row action table. Distinct from `make_lookup_table`
/// (our 90-row v0 table): different layout quirks -- `throttle or boost` uses
/// Python truthy-or semantics (0 -> boost, else throttle), `steer` is written into
/// BOTH the steer(1) and yaw(3) slots on ground, aerial rows force handbrake=1.
/// Reproduces rlmarlbot/immortal/action/actionparser.py::ImmortalAction byte-exact.
/// Row layout: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
pub fn make_immortal_table() -> Vec<[f32; 8]> {
    let mut actions: Vec<[f32; 8]> = Vec::with_capacity(126);
    // Ground (36)
    for throttle in [-1.0f32, 0.0, 1.0] {
        for steer in [-1.0f32, 0.0, 1.0] {
            for boost in [0.0f32, 1.0] {
                for handbrake in [0.0f32, 1.0] {
                    if boost == 1.0 && throttle != 1.0 {
                        continue;
                    }
                    // Python `throttle or boost`: throttle if nonzero else boost
                    let t = if throttle != 0.0 { throttle } else { boost };
                    actions.push([t, steer, 0.0, steer, 0.0, 0.0, boost, handbrake]);
                }
            }
        }
    }
    // Aerial (90)
    for pitch in [-1.0f32, 0.0, 1.0] {
        for yaw in [-1.0f32, 0.0, 1.0] {
            for roll in [-1.0f32, 0.0, 1.0] {
                for jump in [0.0f32, 1.0] {
                    for boost in [0.0f32, 1.0] {
                        if pitch == 0.0 && roll == 0.0 && jump == 0.0 {
                            continue;
                        }
                        actions.push([boost, yaw, pitch, yaw, roll, jump, boost, 1.0]);
                    }
                }
            }
        }
    }
    debug_assert_eq!(actions.len(), 126);
    actions
}
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test --manifest-path engine/Cargo.toml --test immortal_action_test`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add engine/src/actions.rs scripts/gen_immortal_action_fixture.py \
        engine/tests/fixtures/immortal_action.json engine/tests/immortal_action_test.rs
git commit -m "feat(engine): Immortal 126-row action table + golden fixture test"
```

---

### Task 2: AdvancedObs-107 builder (Rust)

**Files:**
- Create: `engine/src/obs_advanced.rs`
- Modify: `engine/src/lib.rs` (add `pub mod obs_advanced;` next to `pub mod obs;`, ~line 18)
- Create: `scripts/gen_immortal_obs_fixture.py`
- Create: `engine/tests/fixtures/immortal_obs.json`
- Create: `engine/tests/immortal_obs_test.rs`

**Interfaces:**
- Consumes: `rocketsim_rs::GameState`, `obs::mir` (make `mir` `pub(crate)` — it already is).
- Produces: `pub fn build_advanced_obs(state: &rocketsim_rs::GameState, car_idx: usize, prev_action: &[f32; 8], out: &mut [f32])` — writes 107 floats; `pub const ADV_OBS_SIZE: usize = 107;`.

**Layout (verified, 1v1), concatenation order:**
`ball.pos/2300 (3), ball.vel/2300 (3), ball.angvel/π (3), prev_action (8), pads (34),
SELF[rel_pos/2300(3), rel_vel/2300(3), pos/2300(3), forward(3), up(3), vel/2300(3), angvel/π(3), boost, on_ground, has_flip, is_demoed],
OPP[same 25] + (opp.pos−self.pos)/2300 (3) + (opp.vel−self.vel)/2300 (3)`.
Orange (`team==Orange`) mirrors ball, pads, and every car via `mir` (negate x,y keep z). `boost` here is `car.boost/100`. `has_flip` = `has_flip_or_jump()`. Pads = `pad.state.is_active as f32`, in engine pad order, mirrored for orange (see Step 1 note).

- [ ] **Step 1: Write the reference fixture generator (vendored AdvancedObs)**

`rlgym_compat` is not installed and pulling it in is heavy; vendor the exact math instead. Create `scripts/gen_immortal_obs_fixture.py`:

```python
"""Generate (state, expected_107_obs) fixtures for the Rust AdvancedObs golden
test. Vendors rlmarlbot/immortal/obs/advanced_obs.py math against a minimal mock
state, so no rlgym_compat dependency. Weights-free -- safe to commit.

The mock state uses the SAME field semantics the Rust builder reads from
rocketsim_rs: car.pos/vel/ang_vel (Vec3), rot forward+up (columns of rot_mat),
boost (0..100), on_ground/has_flip/is_demoed (bool), team (0 blue / 1 orange);
ball pos/vel/ang_vel; 34 pad activity flags. Orange inversion = negate x,y keep z."""
import json, math, pathlib, random

POS_STD, ANG_STD = 2300.0, math.pi

def mir(v, m):
    return [-v[0], -v[1], v[2]] if m else [v[0], v[1], v[2]]

def div(v, k):
    return [v[0]/k, v[1]/k, v[2]/k]

def sub(a, b):
    return [a[0]-b[0], a[1]-b[1], a[2]-b[2]]

def add_player(obs, car, ball_pos, ball_vel, m):
    pos = mir(car["pos"], m); vel = mir(car["vel"], m)
    fwd = mir(car["forward"], m); up = mir(car["up"], m); av = mir(car["ang_vel"], m)
    bp = mir(ball_pos, m); bv = mir(ball_vel, m)
    obs += div(sub(bp, pos), POS_STD)          # rel_pos
    obs += div(sub(bv, vel), POS_STD)          # rel_vel
    obs += div(pos, POS_STD)
    obs += fwd
    obs += up
    obs += div(vel, POS_STD)
    obs += div(av, ANG_STD)
    obs += [car["boost"], float(car["on_ground"]), float(car["has_flip"]), float(car["is_demoed"])]
    return pos, vel

def build_obs(state, self_idx):
    self_car = state["cars"][self_idx]
    m = self_car["team"] == 1
    obs = []
    obs += div(mir(state["ball"]["pos"], m), POS_STD)
    obs += div(mir(state["ball"]["vel"], m), POS_STD)
    obs += div(mir(state["ball"]["ang_vel"], m), ANG_STD)
    obs += state["prev_action"]
    pads = state["pads"]
    obs += [float(x) for x in (pads[::-1] if m else pads)]   # inverted_boost_pads = reversed
    self_pos, self_vel = add_player(obs, self_car, state["ball"]["pos"], state["ball"]["vel"], m)
    for i, car in enumerate(state["cars"]):
        if i == self_idx:
            continue
        # allies would precede enemies; 1v1 has only the enemy
        opos, ovel = add_player(obs, car, state["ball"]["pos"], state["ball"]["vel"], m)
        obs += div(sub(mir(car["pos"], m), self_pos), POS_STD)
        obs += div(sub(mir(car["vel"], m), self_vel), POS_STD)
    return obs

def rnd_vec(r, s=1.0):
    return [r.uniform(-s, s), r.uniform(-s, s), r.uniform(-s, s)]

def rnd_car(r, team):
    return {"pos": rnd_vec(r, 4000), "vel": rnd_vec(r, 2000), "ang_vel": rnd_vec(r, 5),
            "forward": rnd_vec(r, 1), "up": rnd_vec(r, 1),
            "boost": r.uniform(0, 1), "on_ground": r.random() > 0.5,
            "has_flip": r.random() > 0.5, "is_demoed": r.random() > 0.5, "team": team}

def main():
    r = random.Random(1234)
    cases = []
    for _ in range(16):
        for self_idx, self_team in ((0, 0), (1, 1)):   # exercise blue + orange perspective
            state = {
                "cars": [rnd_car(r, 0), rnd_car(r, 1)],
                "ball": {"pos": rnd_vec(r, 4000), "vel": rnd_vec(r, 2000), "ang_vel": rnd_vec(r, 5)},
                "pads": [1 if r.random() > 0.5 else 0 for _ in range(34)],
                "prev_action": [r.uniform(-1, 1) for _ in range(8)],
            }
            obs = build_obs(state, self_idx)
            assert len(obs) == 107, len(obs)
            cases.append({"state": state, "self_idx": self_idx, "obs": obs})
    out = pathlib.Path("engine/tests/fixtures/immortal_obs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}))
    print(f"wrote {out} ({len(cases)} cases, 107 floats each)")

if __name__ == "__main__":
    main()
```

**CRITICAL parity note on pads:** the reference above models `inverted_boost_pads` as the plain reverse of the pad array (`pads[::-1]`). This MUST match how the Rust builder mirrors pads. In rlgym_compat, `inverted_boost_pads` reverses the 34-length pad vector; the Rust builder must reproduce whatever that convention is. Before trusting this test, verify the real convention: `pip install rlgym-compat` in a throwaway venv and print `GameState.inverted_boost_pads` semantics, OR cross-check against `deploy/obs.py`'s pad handling. If the real convention is a positional mirror (nearest pad at `-x,-y`) rather than array-reverse, update BOTH this generator and the Rust builder together. Do not proceed past the golden test on an assumed convention.

- [ ] **Step 2: Generate the fixture**

Run: `cd /home/steamo/Construct-RLBot && .venv/bin/python scripts/gen_immortal_obs_fixture.py`
Expected: `wrote engine/tests/fixtures/immortal_obs.json (32 cases, 107 floats each)`

- [ ] **Step 3: Write the failing Rust test**

Create `engine/tests/immortal_obs_test.rs`:

```rust
use construct_engine::obs_advanced::{build_advanced_obs, ADV_OBS_SIZE};
use construct_engine::test_support::state_from_fixture;   // helper added in Step 5

#[test]
fn advanced_obs_matches_python_fixture() {
    let raw = std::fs::read_to_string("tests/fixtures/immortal_obs.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    for (ci, case) in v["cases"].as_array().unwrap().iter().enumerate() {
        let (state, prev) = state_from_fixture(&case["state"]);
        let self_idx = case["self_idx"].as_u64().unwrap() as usize;
        let mut out = vec![0.0f32; ADV_OBS_SIZE];
        build_advanced_obs(&state, self_idx, &prev, &mut out);
        let expect = case["obs"].as_array().unwrap();
        for k in 0..ADV_OBS_SIZE {
            let e = expect[k].as_f64().unwrap() as f32;
            assert!((out[k] - e).abs() < 1e-5, "case {ci} idx {k}: rust {} py {}", out[k], e);
        }
    }
}
```

The `state_from_fixture` helper (constructing a `rocketsim_rs::GameState` from the JSON `state` object) is non-trivial because `CarState`/`BallState` have many fields. Add it as a small `pub mod test_support` gated on `#[cfg(any(test, feature = "test-support"))]` in the engine crate (Step 5) so both this integration test and the fixture share one constructor. It sets only the fields the obs reads (pos/vel/ang_vel/rot_mat.forward+up/boost/is_on_ground/is_demoed/team, ball pos/vel/ang_vel, pads is_active) and leaves the rest `Default`.

- [ ] **Step 4: Run test to verify it fails**

Run: `cargo test --manifest-path engine/Cargo.toml --test immortal_obs_test`
Expected: FAIL — module `obs_advanced` / `test_support` not found.

- [ ] **Step 5: Implement `build_advanced_obs` + `test_support`**

Create `engine/src/obs_advanced.rs`:

```rust
//! Immortal's RLGym AdvancedObs (107 floats, 1v1) built from the engine GameState.
//! Distinct from `obs.rs` (our 94-float v0 obs): different normalizers
//! (pos/vel /2300, ang_vel /PI), includes 34 boost-pad states and previous action,
//! and uses the forward/up basis (not Euler). Reproduces
//! rlmarlbot/immortal/obs/advanced_obs.py byte-exact -- see immortal_obs_test.rs.
use crate::obs::mir;
use rocketsim_rs::math::Vec3;
use rocketsim_rs::GameState;
use rocketsim_rs::sim::Team;

pub const ADV_OBS_SIZE: usize = 107;
const POS_STD: f32 = 2300.0;
const ANG_STD: f32 = std::f32::consts::PI;

struct W<'a> { out: &'a mut [f32], i: usize }
impl W<'_> {
    #[inline]
    fn v3(&mut self, v: [f32; 3], k: f32) {
        self.out[self.i] = v[0] * k;
        self.out[self.i + 1] = v[1] * k;
        self.out[self.i + 2] = v[2] * k;
        self.i += 3;
    }
    #[inline]
    fn f(&mut self, x: f32) { self.out[self.i] = x; self.i += 1; }
}

#[inline]
fn sub(a: Vec3, b: Vec3) -> Vec3 { Vec3::new(a.x - b.x, a.y - b.y, a.z - b.z) }

/// Append one player's 25-float block (rel-to-ball, absolutes, flags). Returns the
/// mirrored (pos, vel) so the caller can compute the enemy rel-extras.
fn add_player(w: &mut W, car: &rocketsim_rs::CarInfo, ball_p: Vec3, ball_v: Vec3, m: bool)
    -> (Vec3, Vec3)
{
    let s = &car.state;
    let pos = mir_vec(s.pos, m);
    let vel = mir_vec(s.vel, m);
    w.v3(mir(sub(ball_p, s.pos), m), 1.0 / POS_STD);   // rel_pos
    w.v3(mir(sub(ball_v, s.vel), m), 1.0 / POS_STD);   // rel_vel
    w.v3(mir(s.pos, m), 1.0 / POS_STD);
    w.v3(mir(s.rot_mat.forward, m), 1.0);
    w.v3(mir(s.rot_mat.up, m), 1.0);
    w.v3(mir(s.vel, m), 1.0 / POS_STD);
    w.v3(mir(s.ang_vel, m), 1.0 / ANG_STD);
    w.f(s.boost / 100.0);
    w.f(s.is_on_ground as u8 as f32);
    w.f(s.has_flip_or_jump() as u8 as f32);
    w.f(s.is_demoed as u8 as f32);
    (pos, vel)
}

#[inline]
fn mir_vec(v: Vec3, m: bool) -> Vec3 { let a = mir(v, m); Vec3::new(a[0], a[1], a[2]) }

pub fn build_advanced_obs(state: &GameState, car_idx: usize, prev_action: &[f32; 8], out: &mut [f32]) {
    assert_eq!(out.len(), ADV_OBS_SIZE);
    let me = &state.cars[car_idx];
    let m = me.team == Team::Orange;
    let mut w = W { out, i: 0 };

    let b = &state.ball;
    w.v3(mir(b.pos, m), 1.0 / POS_STD);
    w.v3(mir(b.vel, m), 1.0 / POS_STD);
    w.v3(mir(b.ang_vel, m), 1.0 / ANG_STD);
    for k in 0..8 { w.f(prev_action[k]); }

    // pads: is_active flags. Orange uses inverted_boost_pads. Reproduce whatever
    // rlgym_compat's inversion is (array reverse per the fixture generator's note);
    // if the verified convention is positional-mirror, swap to the permutation used
    // by obs_v1.rs::mirror_pad_perm and update the generator to match.
    let n_pads = state.pads.len();
    for idx in 0..n_pads {
        let src = if m { n_pads - 1 - idx } else { idx };
        w.f(state.pads[src].state.is_active as u8 as f32);
    }

    let (self_pos, self_vel) = add_player(&mut w, me, b.pos, b.vel, m);

    // allies (same team, ascending id) then enemies -- 1v1 has only the enemy.
    let mut others: Vec<&rocketsim_rs::CarInfo> = state.cars.iter()
        .enumerate().filter(|(i, _)| *i != car_idx).map(|(_, c)| c).collect();
    others.sort_by_key(|c| (c.team != me.team, c.id));
    for c in others {
        let (opos, ovel) = add_player(&mut w, c, b.pos, b.vel, m);
        w.v3([ (opos.x - self_pos.x), (opos.y - self_pos.y), (opos.z - self_pos.z) ], 1.0 / POS_STD);
        w.v3([ (ovel.x - self_vel.x), (ovel.y - self_vel.y), (ovel.z - self_vel.z) ], 1.0 / POS_STD);
    }
    debug_assert_eq!(w.i, ADV_OBS_SIZE);
}
```

Add `pub mod obs_advanced;` to `engine/src/lib.rs` beside `pub mod obs;`. Make `obs::mir` `pub(crate)` (already is). Add a `test_support` module (new file `engine/src/test_support.rs`, `pub mod test_support;` gated `#[cfg(any(test, feature = "test-support"))]`) exposing `pub fn state_from_fixture(v: &serde_json::Value) -> (GameState, [f32;8])` that builds a `GameState` with two `CarInfo`s from the JSON, setting only the read fields and `..Default::default()` for the rest. (Consult `rocketsim_rs` `CarState`/`BallState`/`BoostPad` field lists mapped in the design doc; `RotMat { forward, right, up }` — set `right` to any unit vector, obs ignores it.)

- [ ] **Step 6: Run test to verify it passes**

Run: `cargo test --manifest-path engine/Cargo.toml --test immortal_obs_test`
Expected: PASS. If it fails only on the pad block, resolve the `inverted_boost_pads` convention per the Step 1 CRITICAL note before anything else.

- [ ] **Step 7: Commit**

```bash
git add engine/src/obs_advanced.rs engine/src/test_support.rs engine/src/lib.rs \
        scripts/gen_immortal_obs_fixture.py engine/tests/fixtures/immortal_obs.json \
        engine/tests/immortal_obs_test.rs
git commit -m "feat(engine): AdvancedObs-107 builder + golden fixture (blue+orange)"
```

---

### Task 3: Foreign MLP forward (candle, LeakyReLU) — logic golden vs torch

Test the forward LOGIC (Linear stack + LeakyReLU(0.01)) with a SMALL synthetic net so the test is CI-safe (no unlicensed weights). Real-Immortal parity is verified later (Task 4, weights-gated).

**Files:**
- Create: `engine/src/foreign.rs` (MLP portion only this task)
- Modify: `engine/src/lib.rs` (`pub mod foreign;`)
- Create: `scripts/gen_foreign_mlp_fixture.py`
- Create: `engine/tests/fixtures/foreign_mlp.json`
- Create: `engine/tests/foreign_mlp_test.rs`

**Interfaces:**
- Produces: `pub struct ForeignMlp { layers: Vec<candle_nn::Linear> }`, `ForeignMlp::from_named(&HashMap<String, (Vec<f32>, Vec<usize>)>, in_dim, hidden, out_dim, n_hidden) -> Result<Self, String>`, `ForeignMlp::forward(&self, obs: &[f32], batch: usize, in_dim: usize) -> Result<Vec<f32>, String>` (leaky_relu 0.01 between Linears, none after the last; returns flat `batch*out_dim` logits).

- [ ] **Step 1: Write the synthetic fixture generator**

Create `scripts/gen_foreign_mlp_fixture.py`:

```python
"""Synthetic MLP fixture: a small net (in=5, hidden=8, out=4, 2 hidden layers) with
LeakyReLU(0.01), torch forward on random inputs -> expected logits. Verifies the
candle forward LOGIC (not Immortal's weights). Safe to commit."""
import json, pathlib, torch

torch.manual_seed(7)
IN, HID, OUT, NH = 5, 8, 4, 2
sizes = [IN] + [HID]*NH + [OUT]
lins = [torch.nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]
sd = {}
for i, lin in enumerate(lins):
    sd[f"net.{2*i}.weight"] = lin.weight.detach().numpy().tolist()
    sd[f"net.{2*i}.bias"] = lin.bias.detach().numpy().tolist()

def fwd(x):
    for i, lin in enumerate(lins):
        x = lin(x)
        if i < len(lins) - 1:
            x = torch.nn.functional.leaky_relu(x, 0.01)
    return x

inputs = torch.randn(6, IN)
with torch.no_grad():
    logits = fwd(inputs)
out = pathlib.Path("engine/tests/fixtures/foreign_mlp.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "in_dim": IN, "hidden": HID, "out_dim": OUT, "n_hidden": NH,
    "weights": sd, "inputs": inputs.tolist(), "logits": logits.tolist(),
}))
print(f"wrote {out}")
```

Run: `cd /home/steamo/Construct-RLBot && .venv/bin/python scripts/gen_foreign_mlp_fixture.py`
Expected: `wrote engine/tests/fixtures/foreign_mlp.json`

- [ ] **Step 2: Write the failing Rust test**

Create `engine/tests/foreign_mlp_test.rs`:

```rust
use std::collections::HashMap;
use construct_engine::foreign::ForeignMlp;

#[test]
fn foreign_mlp_forward_matches_torch() {
    let raw = std::fs::read_to_string("tests/fixtures/foreign_mlp.json").unwrap();
    let v: serde_json::Value = serde_json::from_str(&raw).unwrap();
    let in_dim = v["in_dim"].as_u64().unwrap() as usize;
    let hidden = v["hidden"].as_u64().unwrap() as usize;
    let out_dim = v["out_dim"].as_u64().unwrap() as usize;
    let n_hidden = v["n_hidden"].as_u64().unwrap() as usize;

    let mut w: HashMap<String, (Vec<f32>, Vec<usize>)> = HashMap::new();
    for (k, val) in v["weights"].as_object().unwrap() {
        // 2D weight -> flatten row-major [out,in]; 1D bias -> [out]
        let arr = val.as_array().unwrap();
        if arr[0].is_array() {
            let rows = arr.len();
            let cols = arr[0].as_array().unwrap().len();
            let mut flat = Vec::with_capacity(rows * cols);
            for r in arr { for c in r.as_array().unwrap() { flat.push(c.as_f64().unwrap() as f32); } }
            w.insert(k.clone(), (flat, vec![rows, cols]));
        } else {
            let flat: Vec<f32> = arr.iter().map(|x| x.as_f64().unwrap() as f32).collect();
            let n = flat.len();
            w.insert(k.clone(), (flat, vec![n]));
        }
    }
    let mlp = ForeignMlp::from_named(&w, in_dim, hidden, out_dim, n_hidden).unwrap();

    let inputs = v["inputs"].as_array().unwrap();
    let expect = v["logits"].as_array().unwrap();
    for (bi, row) in inputs.iter().enumerate() {
        let obs: Vec<f32> = row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect();
        let got = mlp.forward(&obs, 1, in_dim).unwrap();
        let exp = expect[bi].as_array().unwrap();
        for j in 0..out_dim {
            let e = exp[j].as_f64().unwrap() as f32;
            assert!((got[j] - e).abs() < 1e-5, "b{bi} j{j}: {} vs {}", got[j], e);
        }
    }
}
```

Run: `cargo test --manifest-path engine/Cargo.toml --test foreign_mlp_test`
Expected: FAIL — `foreign` module missing.

- [ ] **Step 3: Implement `ForeignMlp`**

Create `engine/src/foreign.rs` (MLP portion; the `ForeignPolicy` wrapper is added in Task 4):

```rust
//! Foreign (ported community-bot) opponent primitives. This module holds the
//! candle MLP + (Task 4) the obs+table+prev-action wrapper. Kept separate from
//! policy.rs (our v0 MLP) because the activation (LeakyReLU) and I/O contract differ.
use std::collections::HashMap;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};

const LEAKY_SLOPE: f64 = 0.01;

fn linear(w: &(Vec<f32>, Vec<usize>), b: &(Vec<f32>, Vec<usize>), dev: &Device)
    -> Result<Linear, String>
{
    let (out_dim, in_dim) = (w.1[0], w.1[1]);
    let wt = Tensor::from_vec(w.0.clone(), (out_dim, in_dim), dev).map_err(|e| e.to_string())?;
    let bt = Tensor::from_vec(b.0.clone(), out_dim, dev).map_err(|e| e.to_string())?;
    Ok(Linear::new(wt, Some(bt)))
}

pub struct ForeignMlp { layers: Vec<Linear> }

impl ForeignMlp {
    /// Build from a state_dict keyed `net.0.weight/bias, net.2..., net.{2*(n_hidden)}...`.
    /// Validates the chain dims: in_dim -> hidden (x n_hidden) -> out_dim.
    pub fn from_named(
        w: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        in_dim: usize, hidden: usize, out_dim: usize, n_hidden: usize,
    ) -> Result<Self, String> {
        crate::policy::ensure_single_thread_gemm();
        let dev = Device::Cpu;
        let n_layers = n_hidden + 1;
        let mut layers = Vec::with_capacity(n_layers);
        for li in 0..n_layers {
            let key = format!("net.{}", 2 * li);
            let wk = format!("{key}.weight");
            let bk = format!("{key}.bias");
            let wv = w.get(&wk).ok_or_else(|| format!("missing {wk}"))?;
            let bv = w.get(&bk).ok_or_else(|| format!("missing {bk}"))?;
            let expect_in = if li == 0 { in_dim } else { hidden };
            let expect_out = if li == n_layers - 1 { out_dim } else { hidden };
            if wv.1 != vec![expect_out, expect_in] {
                return Err(format!("{wk} shape {:?} != [{expect_out},{expect_in}]", wv.1));
            }
            layers.push(linear(wv, bv, &dev)?);
        }
        Ok(Self { layers })
    }

    /// Forward `batch` rows of `in_dim` -> flat `batch*out_dim` logits.
    /// LeakyReLU(0.01) after every layer except the last.
    pub fn forward(&self, obs: &[f32], batch: usize, in_dim: usize) -> Result<Vec<f32>, String> {
        let dev = Device::Cpu;
        let mut x = Tensor::from_vec(obs.to_vec(), (batch, in_dim), &dev).map_err(|e| e.to_string())?;
        let last = self.layers.len() - 1;
        for (i, l) in self.layers.iter().enumerate() {
            x = l.forward(&x).map_err(|e| e.to_string())?;
            if i != last {
                // candle has no leaky_relu helper on Tensor in 0.11; compute manually:
                // max(x,0) + slope*min(x,0)
                let pos = x.relu().map_err(|e| e.to_string())?;
                let neg = x.minimum(0.0).map_err(|e| e.to_string())?;
                let neg = (neg * LEAKY_SLOPE).map_err(|e| e.to_string())?;
                x = (pos + neg).map_err(|e| e.to_string())?;
            }
        }
        x.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())
    }
}
```

Add `pub mod foreign;` to `engine/src/lib.rs`. Make `policy::ensure_single_thread_gemm` `pub(crate)` (it is `fn` today — promote visibility). If candle 0.11's `Tensor::minimum(f64)` signature differs, use `x.affine(...)`/`broadcast` equivalents; confirm against `policy_v1.rs` which already uses candle ops. The `max(x,0)+slope*min(x,0)` identity must equal torch `leaky_relu`; the golden test is the guard.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path engine/Cargo.toml --test foreign_mlp_test`
Expected: PASS (max-abs-diff < 1e-5).

- [ ] **Step 5: Commit**

```bash
git add engine/src/foreign.rs engine/src/lib.rs scripts/gen_foreign_mlp_fixture.py \
        engine/tests/fixtures/foreign_mlp.json engine/tests/foreign_mlp_test.rs
git commit -m "feat(engine): ForeignMlp candle forward (LeakyReLU) + synthetic golden test"
```

---

## Phase 2 — ForeignPolicy assembly + real-Immortal parity

### Task 4: `ForeignPolicy` (obs + MLP + table + prev-action) and weights-gated e2e parity

**Files:**
- Modify: `engine/src/foreign.rs`
- Create: `scripts/extract_immortal_weights.py`
- Create: `tests/python/test_foreign_immortal_parity.py` (weights-gated skip)

**Interfaces:**
- Consumes: `ForeignMlp`, `build_advanced_obs`, `actions::make_immortal_table`.
- Produces:
  - `pub enum ForeignKind { Immortal }`
  - `pub struct ForeignPolicy { kind, mlp: ForeignMlp, table: Vec<[f32;8]>, prev: HashMap<u32,[f32;8]> }`
  - `ForeignPolicy::new(w: &HashMap<String,(Vec<f32>,Vec<usize>)>, kind: ForeignKind) -> Result<Self,String>`
  - `ForeignPolicy::decide(&mut self, state: &GameState, car_idx: usize) -> [f32;8]` (builds obs with the car's stored prev_action, forwards, argmax, looks up controls, stores the controls as the new prev_action, returns controls-8)
  - `ForeignPolicy::reset(&mut self)` (clear prev on episode reset)

- [ ] **Step 1: Write the weight-extraction script**

Create `scripts/extract_immortal_weights.py`:

```python
"""Extract Immortal's MLP weights from jit.pt into an npz keyed by state_dict name,
for loading as a foreign opponent. OUTPUT IS UNLICENSED -- writes outside the repo
tree by default and prints a SHA256 for the manifest. Never commit the npz."""
import argparse, hashlib, pathlib, numpy as np, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jit_pt", help="path to rlmarlbot/immortal/jit.pt")
    ap.add_argument("--out", default=str(pathlib.Path.home() / ".cache/construct/immortal_weights.npz"))
    args = ap.parse_args()
    actor = torch.jit.load(args.jit_pt); actor.eval()
    named = dict(actor.named_parameters())
    sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in named.items()}
    expect = [f"net.{i}.{p}" for i in (0,2,4,6,8,10,12) for p in ("weight","bias")]
    assert sorted(sd) == sorted(expect), f"unexpected keys: {sorted(sd)}"
    assert sd["net.0.weight"].shape == (512, 107)
    assert sd["net.12.weight"].shape == (126, 512)
    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **sd)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out}\nsha256 {sha}\nAdd to external_weights.manifest; DO NOT commit the npz.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Extend `foreign.rs` with `ForeignPolicy`**

Append to `engine/src/foreign.rs`:

```rust
use rocketsim_rs::GameState;
use crate::obs_advanced::{build_advanced_obs, ADV_OBS_SIZE};

#[derive(Clone, Copy)]
pub enum ForeignKind { Immortal }

pub struct ForeignPolicy {
    mlp: ForeignMlp,
    table: Vec<[f32; 8]>,
    prev: std::collections::HashMap<u32, [f32; 8]>,
}

impl ForeignPolicy {
    pub fn new(w: &HashMap<String, (Vec<f32>, Vec<usize>)>, kind: ForeignKind) -> Result<Self, String> {
        let (mlp, table) = match kind {
            ForeignKind::Immortal => (
                ForeignMlp::from_named(w, ADV_OBS_SIZE, 512, 126, 6)?,
                crate::actions::make_immortal_table(),
            ),
        };
        Ok(Self { mlp, table, prev: HashMap::new() })
    }

    pub fn reset(&mut self) { self.prev.clear(); }

    /// One decision for `state.cars[car_idx]`: obs (with stored prev action) ->
    /// argmax -> controls; stores the chosen controls as the next prev action.
    pub fn decide(&mut self, state: &GameState, car_idx: usize) -> [f32; 8] {
        let car_id = state.cars[car_idx].id;
        let prev = *self.prev.get(&car_id).unwrap_or(&[0.0; 8]);
        let mut obs = vec![0.0f32; ADV_OBS_SIZE];
        build_advanced_obs(state, car_idx, &prev, &mut obs);
        let logits = self.mlp.forward(&obs, 1, ADV_OBS_SIZE).unwrap_or_else(|_| vec![0.0; self.table.len()]);
        let mut best = 0usize; let mut bv = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() { if v > bv { bv = v; best = i; } }
        let controls = self.table[best];
        self.prev.insert(car_id, controls);
        controls
    }
}
```

Note the prev-action semantics: Immortal feeds the *previous chosen action vector* (the 8-float controls) back into the obs. On the very first decision of an episode `prev = [0;8]` (matches `bot.py` `self.action = np.zeros(8)`). `reset()` is called on episode reset (wired in Task 6).

- [ ] **Step 3: Write the weights-gated parity test**

Create `tests/python/test_foreign_immortal_parity.py`:

```python
"""End-to-end parity: for a batch of random game states, the Rust ForeignPolicy
(exposed via a debug hook) picks the SAME action index as Python Immortal.act.
SKIPS if the extracted weights / reference bot aren't present (unlicensed)."""
import os, pathlib, numpy as np, pytest

WEIGHTS = pathlib.Path.home() / ".cache/construct/immortal_weights.npz"
REF = os.environ.get("IMMORTAL_SRC")   # path to rlmarlbot/immortal

pytestmark = pytest.mark.skipif(
    not WEIGHTS.exists() or not REF,
    reason="Immortal weights (npz) or IMMORTAL_SRC not present (unlicensed).",
)

def test_rust_foreign_matches_python_immortal():
    # Load weights dict, construct the foreign opponent, and for N synthetic states
    # compare argmax index from the Rust debug hook vs Python agent.act.
    # (Debug hook `debug_foreign_action(state_json)` added in Task 5.)
    ...
```

Leave the body as a scaffold to be completed in Task 5 once the debug hook exists (it references `debug_foreign_action`). The skip guard means CI stays green without the unlicensed assets.

- [ ] **Step 4: Build + run the Rust unit tests (no Python yet)**

Run: `cargo test --manifest-path engine/Cargo.toml`
Expected: PASS (all engine tests incl. the three fixtures; `ForeignPolicy` compiles).

- [ ] **Step 5: Commit**

```bash
git add engine/src/foreign.rs scripts/extract_immortal_weights.py tests/python/test_foreign_immortal_parity.py
git commit -m "feat(engine): ForeignPolicy (obs+mlp+table+prev-action) + weight extraction + parity scaffold"
```

---

## Phase 3 — Engine integration (v1 collect path)

### Task 5: `set_foreign_opponents` PyO3 API + `NetWeights::Foreign` + debug hook

**Files:**
- Modify: `engine/src/engine.rs` (`NetWeights` enum ~27; `MultiEngine` fields; `Cmd`)
- Modify: `engine/src/lib.rs` (new `set_foreign_opponents` method ~after 285; `debug_foreign_action` hook)

**Interfaces:**
- Produces (Python): `Engine.set_foreign_opponents(dicts: list[dict[str, np.ndarray]], kinds: list[str])` — loads foreign opponents into slots parallel to `set_opponents`; `kinds[i]` in `{"immortal"}`. And `Engine.debug_foreign_action(kind: str, weights: dict, state_json: str) -> int` for the parity test.
- Produces (Rust): `NetWeights::Foreign { raw: RawStateDict, kind: ForeignKind }`; `MultiEngine.foreign_slots: usize`; per-worker `opponents_foreign: Vec<ForeignPolicy>`.

- [ ] **Step 1: Add the `Foreign` weights variant + storage**

In `engine/src/engine.rs`, extend the `NetWeights` enum (line 27):

```rust
pub enum NetWeights {
    V0(PolicyWeights),
    V1 { raw: RawStateDict, heads: usize },
    Foreign { raw: RawStateDict, kind: crate::foreign::ForeignKind },
}
```

Add a `MultiEngine` field `foreign_slots: usize` (init 0, beside `opponent_slots` ~557) and per-worker storage `opponents_foreign: Vec<ForeignPolicy>` (beside `opponents`/`opponents_v1` ~656). Add a `Cmd::SetForeignOpponents(Vec<NetWeights>)` variant + arm that builds `ForeignPolicy::new(raw, kind)` per slot (mirror the `Cmd::SetOpponents` arm ~746-772). Store the count in `foreign_slots`.

- [ ] **Step 2: Add the PyO3 loader + debug hook**

In `engine/src/lib.rs`, after `set_opponents` (~285) add:

```rust
/// Load foreign (ported community-bot) opponents into slots parallel to
/// `set_opponents`. Each dict is a state_dict of `net.{even}.weight/bias`;
/// `kinds[i]` selects the obs+table (currently only "immortal").
fn set_foreign_opponents(
    &mut self,
    opponents: Vec<HashMap<String, PyReadonlyArrayDyn<'_, f32>>>,
    kinds: Vec<String>,
) -> PyResult<()> {
    if opponents.len() != kinds.len() {
        return Err(PyValueError::new_err("opponents/kinds length mismatch"));
    }
    if opponents.len() > 8 {
        return Err(PyValueError::new_err("at most 8 foreign slots"));
    }
    let mut parsed = Vec::with_capacity(opponents.len());
    for (dict, kind) in opponents.into_iter().zip(kinds) {
        let arrays: RawStateDict = dict.into_iter()
            .map(|(k, a)| (k, (a.as_slice().unwrap().to_vec(), a.shape().to_vec())))
            .collect();
        let fk = match kind.as_str() {
            "immortal" => crate::foreign::ForeignKind::Immortal,
            other => return Err(PyValueError::new_err(format!("unknown foreign kind {other}"))),
        };
        parsed.push(engine::NetWeights::Foreign { raw: arrays, kind: fk });
    }
    self.inner.set_foreign_opponents(parsed).map_err(PyValueError::new_err)
}
```

Add `MultiEngine::set_foreign_opponents` (mirror `set_opponents` ~1177-1210: build one `ForeignPolicy` per slot to validate shapes eagerly, set `foreign_slots`, broadcast `Cmd::SetForeignOpponents`). Also add a `debug_foreign_action` PyO3 method that builds a single `ForeignPolicy` + a `GameState` (via `test_support::state_from_fixture`, promoted to `feature="test-support"` compiled into the wheel for debug builds) and returns the argmax index — used only by the parity test.

- [ ] **Step 3: Rebuild the wheel + smoke-test the API**

Run:
```bash
cargo test --manifest-path engine/Cargo.toml && maturin develop --release
python -c "from construct._engine import Engine; print(hasattr(Engine, 'set_foreign_opponents'))"
```
Expected: `True`.

- [ ] **Step 4: Complete + run the parity test (if weights present)**

Fill `tests/python/test_foreign_immortal_parity.py` body to call `Engine.debug_foreign_action("immortal", weights, state_json)` and compare to Python `agent.Agent().act(obs)` argmax on the same state (build the reference obs with the vendored math from Task 2). If weights absent, it skips.

Run: `pytest tests/python/test_foreign_immortal_parity.py -v`
Expected: PASS or SKIP (skip if no weights).

- [ ] **Step 5: Commit**

```bash
git add engine/src/engine.rs engine/src/lib.rs tests/python/test_foreign_immortal_parity.py
git commit -m "feat(engine): set_foreign_opponents API + NetWeights::Foreign + debug hook"
```

---

### Task 6: Route foreign opponents in the v1 collect loop + per-car controls override

**Files:**
- Modify: `engine/src/engine.rs` (`collect_v1_worker` ~270-495; assignment validation ~1226-1245)
- Modify: `engine/src/episode.rs` (`step_v1` / `step_impl` ~676-706 — accept controls override; call `ForeignPolicy::reset` on episode reset)
- Create/extend: `tests/python/test_foreign_opponent.py` (non-regression + smoke)

**Interfaces:**
- Consumes: `opponents_foreign: Vec<ForeignPolicy>`, `foreign` assignment (see below).
- The v1 collect path gains a parallel assignment for foreign slots. **Decision:** reuse the existing `arena_opponents: Vec<i32>` space by encoding foreign slots as a separate argument `arena_foreign: Option<Vec<i32>>` to `collect` (cleanest — keeps native `-1/k` semantics intact). An arena is foreign-driven when `arena_foreign[i] >= 0`; its orange cars are driven by `opponents_foreign[arena_foreign[i]]` instead of the native path. An arena may not be both a native opponent and a foreign opponent (validate mutually exclusive).

- [ ] **Step 1: Write the non-regression test first**

Create `tests/python/test_foreign_opponent.py`:

```python
"""Foreign opponent: (a) self-play collect is byte-identical whether or not the
foreign machinery is compiled/loaded (non-regression), (b) a foreign-assigned
arena runs without error and produces learner rows for its blue cars."""
import numpy as np
from construct._engine import Engine

def _fresh_engine():
    # Mirror how tests/python/test_rust_collect.py builds an engine (schema v0/v1,
    # num_arenas, set_weights with a real checkpoint or random dict). Reuse that helper.
    ...

def test_selfplay_unchanged_with_foreign_compiled():
    eng = _fresh_engine()
    # ... set_weights(sd); collect(steps=32) with arena_opponents=None ...
    out_a = eng.collect(32)
    eng2 = _fresh_engine()
    out_b = eng2.collect(32)
    assert np.array_equal(out_a["actions"], out_b["actions"])   # deterministic, foreign off
```

(Model the engine construction on `tests/python/test_rust_collect.py`. The key assertion is that adding the foreign code path does NOT change self-play collect output when no foreign slot is assigned.)

- [ ] **Step 2: Run it (expect it to pass BEFORE wiring — guards non-regression)**

Run: `maturin develop --release && pytest tests/python/test_foreign_opponent.py::test_selfplay_unchanged_with_foreign_compiled -v`
Expected: PASS now (baseline). Re-run after Step 3-4 to confirm still PASS.

- [ ] **Step 3: Thread `arena_foreign` through `collect`**

In `engine/src/lib.rs` `collect` (~302), add `arena_foreign: Option<Vec<i32>>` to the signature; pass to `MultiEngine::collect`. Validate (engine.rs ~1226): same length as `num_arenas`; each entry `-1` or in `[0, foreign_slots)`; an arena with `arena_foreign[i] >= 0` must have `arena_opponents[i] == -1` (mutual exclusion); a foreign arena's blue cars are learner rows (must be > 0 total).

- [ ] **Step 4: Drive foreign orange cars in `collect_v1_worker`**

In `collect_v1_worker` (engine.rs ~270-495): after the native learner+opponent logits are gathered and actions sampled for all agents (preserving the rng draw), for each arena `li` with `my_foreign[li] = k >= 0`, override its orange cars' controls: call `opponents_foreign[k].decide(&arena_state, orange_car_idx)` → controls-8, and pass these as a per-car override into the step. Because `decide` needs the live `GameState`, fetch it from the arena (the same source obs is built from) BEFORE stepping. Blue cars of that arena keep their sampled learner actions (recorded as learner rows for training).

Concretely, extend `EpisodeArena::step_v1` to take `overrides: &[(u32 /*car_id*/, [f32;8] /*controls*/)]`. In `episode.rs` (~699-705), build controls from `self.table[idx]` for non-overridden cars and from the override controls for overridden car_ids:

```rust
let controls: Vec<(u32, rocketsim_rs::sim::CarControls)> = (0..n).map(|a| {
    let cid = self.car_ids[a];
    if let Some((_, ctrl)) = overrides.iter().find(|(oid, _)| *oid == cid) {
        (cid, actions::to_controls(ctrl))
    } else {
        (cid, actions::to_controls(&self.table[action_idx[a] as usize]))
    }
}).collect();
```

Call `ForeignPolicy::reset()` for a slot when any arena it drives resets its episode (so `prev_action` restarts at zero) — hook into the episode-reset branch the worker already has.

- [ ] **Step 5: Add a foreign-arena smoke test**

Extend `tests/python/test_foreign_opponent.py` with `test_foreign_arena_runs` that loads a foreign opponent (skip if no weights) via `set_foreign_opponents`, assigns one arena `arena_foreign=[0, -1, ...]`, runs `collect(64)`, and asserts it returns learner rows for the blue cars and does not error.

- [ ] **Step 6: Full test loop**

Run:
```bash
cargo test --manifest-path engine/Cargo.toml && maturin develop --release && pytest tests/python -q
```
Expected: PASS (non-regression holds; foreign smoke passes or skips).

- [ ] **Step 7: Commit**

```bash
git add engine/src/engine.rs engine/src/episode.rs engine/src/lib.rs tests/python/test_foreign_opponent.py
git commit -m "feat(engine): route foreign opponents in v1 collect via per-car controls override"
```

---

## Phase 4 — Training integration

### Task 7: League/train.py foreign-opponent pool entry + docs

**Files:**
- Modify: `python/construct/learn/train.py` (opponent pool build ~316-327)
- Create: `docs/foreign-opponents.md`
- Create: `external_weights.manifest` (SHA pins; the npz itself stays out of git)

**Interfaces:**
- Consumes: `Engine.set_foreign_opponents`.
- A training config may list a foreign opponent (e.g. `{"kind": "immortal", "weights": "~/.cache/construct/immortal_weights.npz"}`) in the league pool; `train.py` loads its npz → dict → `set_foreign_opponents`, and assigns some fraction of arenas to it via `arena_foreign`.

- [ ] **Step 1: Load foreign weights in the pool builder**

In `python/construct/learn/train.py` where the opponent pool is assembled (~316-327), add: if a pool entry is a foreign spec, load its npz (`dict(np.load(path))`) into a `{name: f32 array}` dict and collect `(dict, kind)`; call `self.engine.set_foreign_opponents(foreign_dicts, foreign_kinds)` once. Build the `arena_foreign` assignment alongside `arena_opponents` so a configurable fraction of opponent arenas are foreign-driven; pass both to `collect`.

Show the exact code once the current pool-build block is read (it constructs `sds` and calls `set_opponents(sds)` at 327). Keep native and foreign assignments mutually exclusive per arena (Task 6 validation enforces it).

- [ ] **Step 2: Write the ops doc**

Create `docs/foreign-opponents.md`: how to fetch RLMarlbot @ `9963b3bd`, run `scripts/extract_immortal_weights.py jit.pt`, the SHA to record in `external_weights.manifest`, the license caveat (never commit weights), and the config snippet to add Immortal to a league pool.

- [ ] **Step 3: Dry-run a short foreign-league collect**

Run a tiny smoke (2 arenas, 1 foreign, 64 steps) via a scratch script or an extended pytest, confirming the trainer steps without error and logs the foreign opponent in the pool. (Full training runs go on the remote box, not the laptop — per project policy.)

- [ ] **Step 4: Commit**

```bash
git add python/construct/learn/train.py docs/foreign-opponents.md external_weights.manifest
git commit -m "feat(train): allow foreign (Immortal) opponent in the league pool"
```

---

## Self-Review

**Spec coverage:** goal (Immortal as in-engine training opponent) → Tasks 1-7; AdvancedObs-107 → Task 2; candle MLP (LeakyReLU) → Task 3; 126-action lookup → Task 1; direct controls-8 application → Task 6; new `External`/`foreign` opponent kind → Tasks 5-6; determinism (rng draw preserved, argmax override) → Task 6 Step 4; cadence (8-tick accepted) → Global Constraints; weights out of git / license → Tasks 4,7 + Global Constraints; golden tests (obs, net, e2e, non-regression) → Tasks 2,3,4,6. Nexto/Necto/Element reuse → the `ForeignKind` enum + `ForeignPolicy` framework leave room (out of scope here, per spec).

**Open item flagged, not hidden:** the `inverted_boost_pads` convention (array-reverse vs positional-mirror) is verified inside Task 2 Step 1/Step 6 — the golden test will fail loudly if the assumption is wrong, and the note says fix generator + builder together before proceeding.

**Type consistency:** `make_immortal_table` (Task 1) ← used in `ForeignPolicy::new` (Task 4). `build_advanced_obs`/`ADV_OBS_SIZE` (Task 2) ← used in `ForeignPolicy::decide` (Task 4). `ForeignMlp::from_named/forward` (Task 3) ← used in `ForeignPolicy` (Task 4). `NetWeights::Foreign` (Task 5) ← built in the `Cmd::SetForeignOpponents` arm (Task 5) and consumed in `collect_v1_worker` (Task 6). `set_foreign_opponents`/`arena_foreign` names consistent across Tasks 5-7.
