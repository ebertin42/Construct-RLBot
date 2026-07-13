# Construct P0 — Walking Skeleton Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the whole Construct loop end-to-end: Rust multi-arena RocketSim engine → PyO3 → minimal PPO trainer → RLViser rendering → RLBot v5 deployment, exiting with a ball-chasing bot visible in the real game.

**Architecture:** Rust crate `construct-engine` (rocketsim-rs 0.37.0) owns sim stepping, obs building, rewards, and episode logic across N worker threads, exposed to Python as `construct._engine` via PyO3/maturin with numpy arrays at the boundary. Python package `construct.learn` implements a minimal-but-correct PPO (GAE, clipping, entropy). A standalone `deploy/` folder holds the Windows-side RLBot v5 bot with a duplicated pure-numpy obs builder kept honest by a parity test.

**Tech Stack:** Rust stable (edition 2024), rocketsim_rs 0.37.0, pyo3 0.29 + rust-numpy 0.29, maturin ≥1.14, Python 3.11, torch ≥2.12, uv, pytest.

## Global Constraints

- Python **3.11** everywhere (ecosystem sweet spot: rlbot ≥3.11, rlgym-compat ≥3.11).
- Pin `rocketsim_rs = "=0.37.0"` — 0.37 changed the RLViser protocol to flatbuffers (`flat_ext::PacketCodec`); older versions use an incompatible byte format.
- `pyo3 = { version = "0.29", features = ["abi3-py311"] }` and `numpy = "0.29"` — these two crates are version-locked to each other. Do NOT add the `extension-module` feature (deprecated in 0.29; maturin sets `PYO3_BUILD_EXTENSION_MODULE` itself).
- Collision meshes (`assets/collision_meshes/`) are **never committed** — RocketSim meshes are dumped game assets ("Do NOT redistribute"). Fetched locally by script.
- Trained weights are **never published** (Psyonix request, see spec §1).
- `tick_skip = 8` (15 Hz decisions). Obs schema **v0**: size **94**. Action table: RLGym-standard **90 rows**.
- Windows deploy only: `rlbot>=2.0.0b52` (must be the v5 beta — plain `pip install rlbot` yields v4), `rlgym_compat` from git (PyPI is stale at 1.1.0).
- Reward/config values come from TOML; no magic numbers in Rust code.
- Spec deviation (documented): deploy uses `state_dict` + rebuilt `nn.Module`, not TorchScript — `torch.jit` is deprecated/maintenance-mode in torch 2.13; state_dict is what working community bots use.

## File Structure

```
pyproject.toml                     # maturin backend; module construct._engine; python-source python/
engine/Cargo.toml                  # crate construct-engine
engine/src/lib.rs                  # #[pymodule] construct._engine: Engine pyclass
engine/src/actions.rs              # 90-row lookup table + CarControls conversion
engine/src/obs.rs                  # obs v0 builder + mirroring (94 floats)
engine/src/reward.rs               # RewardConfig (TOML) + goal/touch/vel-to-ball components
engine/src/episode.rs              # EpisodeArena: one arena + episode/reset/reward logic
engine/src/engine.rs               # multi-arena worker threads + batch step
engine/src/viser.rs                # RLViser UDP streaming (flat_ext PacketCodec)
engine/src/schema.rs               # schema/v0.toml loader + validation
python/construct/__init__.py
python/construct/learn/model.py    # PolicyValueNet MLP
python/construct/learn/gae.py      # GAE computation
python/construct/learn/buffer.py   # rollout buffer
python/construct/learn/ppo.py      # PPO update
python/construct/learn/train.py    # training loop + checkpoints
python/construct/learn/config.py   # TrainConfig from TOML
schema/v0.toml                     # obs size, action table id, tick_skip, norm constants
configs/reward_v0.toml
configs/train_v0.toml
scripts/fetch_meshes.sh            # pulls .cmf meshes locally
scripts/smoke_test.py              # 1k-step end-to-end training smoke test
scripts/watch.py                   # render a checkpoint in RLViser
scripts/eval_metrics.py            # touches/min, dist-to-ball eval
deploy/bot.py                      # RLBot v5 bot (standalone, Windows)
deploy/obs.py                      # pure-numpy obs v0 (duplicate by design; parity-tested)
deploy/actions.py                  # lookup table (duplicate by design; parity-tested)
deploy/model.py                    # PolicyValueNet duplicate for state_dict load
deploy/bot.toml  deploy/loadout.toml  deploy/match.toml  deploy/README.md
tests/python/                      # pytest suite
docs/superpowers/...               # spec + this plan
.gitignore
```

Rust unit tests live in each module (`#[cfg(test)]`); integration tests needing meshes in `engine/tests/`.

---

### Task 1: Repo scaffolding + maturin hello world

**Files:**
- Create: `pyproject.toml`, `engine/Cargo.toml`, `engine/src/lib.rs`, `python/construct/__init__.py`, `tests/python/test_import.py`, `.gitignore`

**Interfaces:**
- Produces: importable module `construct._engine` with function `version() -> str`; repo-wide build commands (`maturin develop --release`, `pytest`, `cargo test`).

- [ ] **Step 1: Write .gitignore**

```gitignore
/target
engine/target
.venv
__pycache__/
*.pyc
.pytest_cache/
dist/
*.so
assets/collision_meshes/
checkpoints/
wandb/
*.pt
```

- [ ] **Step 2: Write engine/Cargo.toml**

```toml
[package]
name = "construct-engine"
version = "0.1.0"
edition = "2021"

[lib]
name = "construct_engine"
crate-type = ["cdylib", "rlib"]

[dependencies]
pyo3 = { version = "0.29", features = ["abi3-py311"] }
numpy = "0.29"
rocketsim_rs = "=0.37.0"
cxx = "1"
serde = { version = "1", features = ["derive"] }
toml = "0.8"
serde_json = "1"
```

(`rlib` in crate-type so `cargo test` can run unit tests against the lib.)

- [ ] **Step 3: Write pyproject.toml**

```toml
[build-system]
requires = ["maturin>=1.14,<2"]
build-backend = "maturin"

[project]
name = "construct"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy>=1.26", "torch>=2.12"]

[project.optional-dependencies]
dev = ["pytest>=8", "maturin>=1.14"]

[tool.maturin]
manifest-path = "engine/Cargo.toml"
module-name = "construct._engine"
python-source = "python"

[tool.pytest.ini_options]
testpaths = ["tests/python"]
```

- [ ] **Step 4: Write minimal lib.rs**

```rust
use pyo3::prelude::*;

#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(version, m)?)?;
    Ok(())
}
```

- [ ] **Step 5: Write python/construct/__init__.py**

```python
from construct._engine import version

__all__ = ["version"]
```

- [ ] **Step 6: Write failing import test**

```python
# tests/python/test_import.py
def test_engine_importable():
    import construct
    assert construct.version() == "0.1.0"
```

- [ ] **Step 7: Create venv, build, run test**

```bash
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install maturin pytest "numpy>=1.26"
maturin develop --release
pytest tests/python/test_import.py -v
```
Expected: `test_engine_importable PASSED`. (torch install deferred to Task 9 — big download.)

- [ ] **Step 8: Commit**

```bash
git add -A && git commit -m "feat: scaffold maturin workspace with construct._engine hello module"
```

---

### Task 2: Collision meshes + RocketSim init

**Files:**
- Create: `scripts/fetch_meshes.sh`, `engine/src/sim_init.rs`
- Modify: `engine/src/lib.rs` (add `mod sim_init;`)
- Test: `engine/tests/arena_basics.rs`

**Interfaces:**
- Produces: `sim_init::ensure_init(meshes_path: Option<&str>)` — idempotent global RocketSim init used by every later task. Meshes land in `assets/collision_meshes/soccar/mesh_*.cmf` (16 files).

- [ ] **Step 1: Write fetch script**

```bash
#!/usr/bin/env bash
# scripts/fetch_meshes.sh — meshes are game-derived assets; keep local, never commit.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets/collision_meshes
tmp=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Martico2432/Rlgym-v2-to-rlbot-v5 "$tmp"
git -C "$tmp" sparse-checkout set src/collision_meshes
cp -r "$tmp/src/collision_meshes/soccar" assets/collision_meshes/
rm -rf "$tmp"
ls assets/collision_meshes/soccar/ | head -3
echo "OK: $(ls assets/collision_meshes/soccar | wc -l) mesh files"
```

- [ ] **Step 2: Run it**

Run: `bash scripts/fetch_meshes.sh`
Expected: `OK: 16 mesh files`

- [ ] **Step 3: Write sim_init.rs**

```rust
use std::sync::Once;

static INIT: Once = Once::new();

/// Idempotent RocketSim global init. Default path works from repo root and from engine/.
pub fn ensure_init(meshes_path: Option<&str>) {
    INIT.call_once(|| {
        let path = meshes_path.map(String::from).unwrap_or_else(|| {
            for candidate in ["assets/collision_meshes", "../assets/collision_meshes"] {
                if std::path::Path::new(candidate).join("soccar").exists() {
                    return candidate.to_string();
                }
            }
            "assets/collision_meshes".to_string()
        });
        rocketsim_rs::init(Some(&path), true);
    });
}
```

- [ ] **Step 4: Write failing integration test**

```rust
// engine/tests/arena_basics.rs
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
```

Also add to `lib.rs`: `pub mod sim_init;`

- [ ] **Step 5: Run test — fails first (module missing), then passes**

Run: `cd engine && cargo test --test arena_basics`
Expected: compile error before `mod` line added; after: `test arena_steps_and_ball_rests_at_spawn_height ... ok`
Note: first build compiles RocketSim C++ — takes minutes.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: rocketsim init with local collision meshes + fetch script"
```

---

### Task 3: 90-action lookup table

**Files:**
- Create: `engine/src/actions.rs`
- Modify: `engine/src/lib.rs` (`pub mod actions;`)

**Interfaces:**
- Produces: `actions::make_lookup_table() -> Vec<[f32; 8]>` (row layout `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`), `actions::to_controls(&[f32; 8]) -> CarControls`, `actions::TABLE_SIZE == 90`. This exact table is reproduced in Python in Task 13 (`deploy/actions.py`) — the two must match row-for-row.

- [ ] **Step 1: Write failing tests (in-module)**

```rust
// engine/src/actions.rs (tests at bottom)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_has_90_rows() {
        assert_eq!(make_lookup_table().len(), TABLE_SIZE);
    }

    #[test]
    fn first_ground_row_matches_rlgym_reference() {
        // throttle=-1, steer=-1, boost=0, handbrake=0 -> [-1,-1,0,-1,0,0,0,0]
        assert_eq!(make_lookup_table()[0], [-1., -1., 0., -1., 0., 0., 0., 0.]);
    }

    #[test]
    fn ground_rows_count_24() {
        // rows with pitch==roll==jump==0 produced by the ground loop
        let n = make_lookup_table().iter().take(24).count();
        assert_eq!(n, 24);
        // 25th row is the first aerial row
        let t = make_lookup_table();
        assert!(t[24][5] != 0.0 || t[24][2] != 0.0 || t[24][4] != 0.0);
    }

    #[test]
    fn to_controls_maps_booleans() {
        let c = to_controls(&[1., 0., 0., 0., 0., 1., 1., 0.]);
        assert_eq!(c.throttle, 1.0);
        assert!(c.jump && c.boost && !c.handbrake);
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run: `cd engine && cargo test actions`
Expected: compile FAIL (`make_lookup_table` not found)

- [ ] **Step 3: Implement — direct port of RLGym 2.0 `LookupTableAction.make_lookup_table` (semantics preserved exactly, incl. Python's `throttle or boost`)**

```rust
use rocketsim_rs::sim::CarControls;

pub const TABLE_SIZE: usize = 90;

/// Row layout: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake]
pub fn make_lookup_table() -> Vec<[f32; 8]> {
    let mut actions: Vec<[f32; 8]> = Vec::with_capacity(TABLE_SIZE);
    // Ground
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
    // Aerial
    for pitch in [-1.0f32, 0.0, 1.0] {
        for yaw in [-1.0f32, 0.0, 1.0] {
            for roll in [-1.0f32, 0.0, 1.0] {
                for jump in [0.0f32, 1.0] {
                    for boost in [0.0f32, 1.0] {
                        if jump == 1.0 && yaw != 0.0 {
                            continue; // Only need roll for sideflip
                        }
                        if pitch == 0.0 && roll == 0.0 && jump == 0.0 {
                            continue; // Duplicate with ground
                        }
                        // Enable handbrake for potential wavedashes
                        let handbrake =
                            (jump == 1.0 && (pitch != 0.0 || yaw != 0.0 || roll != 0.0)) as u8 as f32;
                        actions.push([boost, yaw, pitch, yaw, roll, jump, boost, handbrake]);
                    }
                }
            }
        }
    }
    debug_assert_eq!(actions.len(), TABLE_SIZE);
    actions
}

pub fn to_controls(row: &[f32; 8]) -> CarControls {
    CarControls {
        throttle: row[0],
        steer: row[1],
        pitch: row[2],
        yaw: row[3],
        roll: row[4],
        jump: row[5] != 0.0,
        boost: row[6] != 0.0,
        handbrake: row[7] != 0.0,
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cd engine && cargo test actions`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: RLGym-standard 90-action lookup table in Rust"
```

---

### Task 4: Versioned schema file + loader

**Files:**
- Create: `schema/v0.toml`, `engine/src/schema.rs`
- Modify: `engine/src/lib.rs` (`pub mod schema;` + expose to Python)
- Test: `tests/python/test_schema.py`, Rust in-module tests

**Interfaces:**
- Produces: `schema/v0.toml` as single source of truth; Rust `schema::Schema { version, obs_size, action_table, action_count, tick_skip, pos_norm, vel_norm, ang_vel_norm }` with `Schema::load(path)`; Python-visible `construct._engine.schema_dict() -> dict`. Every checkpoint (Task 10) and the deploy bot (Task 13) records/validates `version`.

- [ ] **Step 1: Write schema/v0.toml**

```toml
version = 0
obs_size = 94
action_table = "rlgym_lookup_90"
action_count = 90
tick_skip = 8

[normalization]
pos_norm = 0.00043478260869565216      # 1/2300
vel_norm = 0.00043478260869565216      # 1/2300
ang_vel_norm = 0.18181818181818182     # 1/5.5
```

- [ ] **Step 2: Write failing Rust test**

```rust
// in engine/src/schema.rs tests
#[test]
fn loads_v0() {
    let s = Schema::load("../schema/v0.toml").unwrap();
    assert_eq!(s.version, 0);
    assert_eq!(s.obs_size, 94);
    assert_eq!(s.action_count, 90);
    assert_eq!(s.tick_skip, 8);
    assert!((s.normalization.pos_norm - 1.0 / 2300.0).abs() < 1e-12);
}
```

- [ ] **Step 3: Implement schema.rs**

```rust
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
pub struct Normalization {
    pub pos_norm: f64,
    pub vel_norm: f64,
    pub ang_vel_norm: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Schema {
    pub version: u32,
    pub obs_size: usize,
    pub action_table: String,
    pub action_count: usize,
    pub tick_skip: u32,
    pub normalization: Normalization,
}

impl Schema {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        toml::from_str(&text).map_err(|e| format!("{path}: {e}"))
    }
}
```

- [ ] **Step 4: Expose to Python in lib.rs**

```rust
use pyo3::types::PyDict;

#[pyfunction]
fn schema_dict<'py>(py: Python<'py>, path: &str) -> PyResult<Bound<'py, PyDict>> {
    let s = crate::schema::Schema::load(path)
        .map_err(pyo3::exceptions::PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("version", s.version)?;
    d.set_item("obs_size", s.obs_size)?;
    d.set_item("action_table", s.action_table)?;
    d.set_item("action_count", s.action_count)?;
    d.set_item("tick_skip", s.tick_skip)?;
    d.set_item("pos_norm", s.normalization.pos_norm)?;
    d.set_item("vel_norm", s.normalization.vel_norm)?;
    d.set_item("ang_vel_norm", s.normalization.ang_vel_norm)?;
    Ok(d)
}
```
(register with `m.add_function(wrap_pyfunction!(schema_dict, m)?)?;`)

- [ ] **Step 5: Python round-trip test**

```python
# tests/python/test_schema.py
import tomllib
from construct._engine import schema_dict

def test_rust_and_python_read_same_schema():
    rust = schema_dict("schema/v0.toml")
    with open("schema/v0.toml", "rb") as f:
        py = tomllib.load(f)
    assert rust["obs_size"] == py["obs_size"] == 94
    assert rust["action_count"] == py["action_count"] == 90
    assert rust["tick_skip"] == py["tick_skip"] == 8
    assert abs(rust["pos_norm"] - py["normalization"]["pos_norm"]) < 1e-15
```

- [ ] **Step 6: Run both**

Run: `cd engine && cargo test schema && cd .. && maturin develop --release && pytest tests/python/test_schema.py -v`
Expected: all pass

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: versioned obs/action schema shared by Rust and Python"
```

---

### Task 5: Obs builder v0 (94 floats, mirrored)

**Files:**
- Create: `engine/src/obs.rs`
- Modify: `engine/src/lib.rs` (`pub mod obs;`)
- Test: in-module + `engine/tests/obs_integration.rs`

**Interfaces:**
- Consumes: `schema::Schema`, rocketsim `GameState`.
- Produces: `obs::OBS_SIZE = 94`, `obs::build_obs(state: &GameState, car_idx: usize, norm: &Normalization, out: &mut [f32])`. Layout (indices) — **this exact layout is duplicated in `deploy/obs.py` (Task 12) and parity-tested**:
  - `[0:3]` self pos · pos_norm (mirrored for Orange)
  - `[3:6]` self forward, `[6:9]` self up (mirrored)
  - `[9:12]` self vel · vel_norm, `[12:15]` self ang_vel · ang_vel_norm (mirrored)
  - `[15]` boost/100, `[16]` is_on_ground, `[17]` has_flip_or_jump, `[18]` is_demoed
  - `[19:22]` ball pos, `[22:25]` ball vel, `[25:28]` ball ang_vel (mirrored, normalized)
  - `[28:31]` (ball pos − self pos) · pos_norm, `[31:34]` (ball vel − self vel) · vel_norm (mirrored)
  - `[34:94]` 5 other-car slots × 12: pos·pos_norm(3), vel·vel_norm(3), forward(3), boost/100, same_team, alive — teammates first (ascending car id), then opponents (ascending car id); zero-padded.
  - Mirroring = 180° rotation about Z for Orange agents: `(x, y, z) -> (-x, -y, z)` applied to every positional/velocity/direction vector (ang_vel transforms identically under this rotation).

- [ ] **Step 1: Write failing integration test**

```rust
// engine/tests/obs_integration.rs
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
```

- [ ] **Step 2: Run to verify failure**

Run: `cd engine && cargo test --test obs_integration`
Expected: compile FAIL (`obs` module missing)

- [ ] **Step 3: Implement obs.rs**

```rust
use crate::schema::Normalization;
use rocketsim_rs::{
    math::Vec3,
    sim::Team,
    GameState,
};

pub const OBS_SIZE: usize = 94;
pub const MAX_OTHERS: usize = 5;

#[inline]
fn mir(v: Vec3, mirror: bool) -> [f32; 3] {
    if mirror { [-v.x, -v.y, v.z] } else { [v.x, v.y, v.z] }
}

struct W<'a> {
    out: &'a mut [f32],
    i: usize,
}
impl W<'_> {
    #[inline]
    fn v3(&mut self, v: [f32; 3], k: f32) {
        self.out[self.i] = v[0] * k;
        self.out[self.i + 1] = v[1] * k;
        self.out[self.i + 2] = v[2] * k;
        self.i += 3;
    }
    #[inline]
    fn f(&mut self, x: f32) {
        self.out[self.i] = x;
        self.i += 1;
    }
}

pub fn build_obs(state: &GameState, car_idx: usize, n: &Normalization, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_SIZE);
    out.fill(0.0);
    let me = &state.cars[car_idx];
    let mirror = me.team == Team::Orange;
    let (pk, vk, ak) = (n.pos_norm as f32, n.vel_norm as f32, n.ang_vel_norm as f32);
    let ms = &me.state;
    let mut w = W { out, i: 0 };

    // self [0:19]
    w.v3(mir(ms.pos, mirror), pk);
    w.v3(mir(ms.rot_mat.forward, mirror), 1.0);
    w.v3(mir(ms.rot_mat.up, mirror), 1.0);
    w.v3(mir(ms.vel, mirror), vk);
    w.v3(mir(ms.ang_vel, mirror), ak);
    w.f(ms.boost / 100.0);
    w.f(ms.is_on_ground as u8 as f32);
    w.f(ms.has_flip_or_jump() as u8 as f32);
    w.f(ms.is_demoed as u8 as f32);

    // ball [19:34]
    let b = &state.ball;
    w.v3(mir(b.pos, mirror), pk);
    w.v3(mir(b.vel, mirror), vk);
    w.v3(mir(b.ang_vel, mirror), ak);
    let rel_p = Vec3::new(b.pos.x - ms.pos.x, b.pos.y - ms.pos.y, b.pos.z - ms.pos.z);
    let rel_v = Vec3::new(b.vel.x - ms.vel.x, b.vel.y - ms.vel.y, b.vel.z - ms.vel.z);
    w.v3(mir(rel_p, mirror), pk);
    w.v3(mir(rel_v, mirror), vk);

    // others [34:94] — teammates (asc id) then opponents (asc id)
    let mut others: Vec<&rocketsim_rs::CarInfo> = state
        .cars
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != car_idx)
        .map(|(_, c)| c)
        .collect();
    others.sort_by_key(|c| (c.team != me.team, c.id));
    for c in others.into_iter().take(MAX_OTHERS) {
        w.v3(mir(c.state.pos, mirror), pk);
        w.v3(mir(c.state.vel, mirror), vk);
        w.v3(mir(c.state.rot_mat.forward, mirror), 1.0);
        w.f(c.state.boost / 100.0);
        w.f((c.team == me.team) as u8 as f32);
        w.f((!c.state.is_demoed) as u8 as f32);
    }
    // remaining slots stay zero (out.fill above)
}
```
Note: if `GameState`/`CarInfo` import paths differ in 0.37.0, check `rocketsim_rs` docs (`cargo doc --open`) — they are re-exported at crate root per the examples.

- [ ] **Step 4: Run tests**

Run: `cd engine && cargo test --test obs_integration`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: obs v0 builder (94 floats) with orange mirroring"
```

---

### Task 6: Reward v0 (TOML-configured)

**Files:**
- Create: `engine/src/reward.rs`, `configs/reward_v0.toml`
- Modify: `engine/src/lib.rs` (`pub mod reward;`)
- Test: in-module tests

**Interfaces:**
- Consumes: rocketsim `GameState`.
- Produces: `reward::RewardConfig::load(path)`, `reward::compute(prev: &GameState, cur: &GameState, car_idx: usize, scored: Option<Team>, cfg: &RewardConfig) -> f32`. Touch detection contract: a car touched the ball during the last step iff `cur.cars[i].state.ball_hit_info.tick_count_when_hit > prev tick_count` (verify exact `BallHitInfo` field names via `cargo doc -p rocketsim_rs`; if absent, fallback: `ball_hit_info.is_valid` + tick comparison).

- [ ] **Step 1: Write configs/reward_v0.toml**

```toml
# P0 ball-chaser rewards. Bounded components, weights set magnitude.
goal = 10.0          # +goal for scoring team, -goal for conceding team
touch = 0.5          # flat, on new ball contact
vel_to_ball = 0.05   # max(0, dot(vel, unit(ball-car))) / 2300, in [0,1]
```

- [ ] **Step 2: Write failing tests**

```rust
// engine/src/reward.rs tests
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
        let blue = compute(&gs, &gs, 0, Some(Team::Blue), &cfg());
        let orange = compute(&gs, &gs, 1, Some(Team::Blue), &cfg());
        assert_eq!(blue, 10.0 + expected_shaping(&gs, 0, &cfg()));
        assert_eq!(orange, -10.0 + expected_shaping(&gs, 1, &cfg()));
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
}
```

- [ ] **Step 3: Run to verify failure**

Run: `cd engine && cargo test reward`
Expected: compile FAIL

- [ ] **Step 4: Implement reward.rs**

```rust
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

    // touch: ball_hit_info advanced during this step
    let hit_now = me.state.ball_hit_info.tick_count_when_hit;
    let hit_before = prev.cars[car_idx].state.ball_hit_info.tick_count_when_hit;
    if hit_now > hit_before && hit_now > prev.tick_count {
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
```
If `BallHitInfo` lacks `tick_count_when_hit` in 0.37.0, run `cargo doc -p rocketsim_rs` and use the equivalent tick field; last resort fallback is `is_valid` + comparing `ball_hit_info.tick_count_when_extra_impulse_applied`.

- [ ] **Step 5: Run tests**

Run: `cd engine && cargo test reward`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: TOML-configured reward v0 (goal/touch/vel-to-ball)"
```

---

### Task 7: EpisodeArena — episode logic + auto-reset

**Files:**
- Create: `engine/src/episode.rs`
- Modify: `engine/src/lib.rs` (`pub mod episode;`)
- Test: `engine/tests/episode_test.rs`

**Interfaces:**
- Consumes: `actions`, `obs`, `reward`, `schema`.
- Produces:
  ```rust
  pub struct StepFlags { pub terminated: bool, pub truncated: bool }
  pub struct EpisodeArena { /* private */ }
  impl EpisodeArena {
      pub fn new(blue: usize, orange: usize, tick_skip: u32, reward_cfg: RewardConfig,
                 norm: Normalization, seed: u32) -> Self;
      pub fn num_agents(&self) -> usize;                      // blue + orange
      /// obs written contiguously: agent a -> out[a*OBS_SIZE..(a+1)*OBS_SIZE]
      pub fn write_obs(&mut self, out: &mut [f32]);
      /// Steps tick_skip ticks with the given action indices (one per agent).
      /// On episode end: writes final-state obs into final_obs, auto-resets,
      /// then next write_obs returns the new episode's first obs.
      pub fn step(&mut self, action_idx: &[i64], rewards: &mut [f32],
                  flags: &mut [StepFlags], final_obs: &mut [f32]);
      pub fn game_state(&mut self) -> GameState;              // for rendering/debug
  }
  ```
- Episode rules: goal → `terminated` for all agents; no ball touch for 30 s (3600 ticks) or episode length 300 s (36000 ticks) → `truncated`. Goal team inferred from `is_ball_scored()` + ball y-sign (`y > 0` = ball in Orange's net = Blue scored). Agent order: blue cars (ascending id) then orange cars — this order is the engine-wide agent indexing contract.

- [ ] **Step 1: Write failing tests**

```rust
// engine/tests/episode_test.rs
use construct_engine::episode::{EpisodeArena, StepFlags};
use construct_engine::{obs::OBS_SIZE, reward::RewardConfig, schema::Schema, sim_init::ensure_init};

fn mk(blue: usize, orange: usize, seed: u32) -> EpisodeArena {
    ensure_init(None);
    let s = Schema::load("../schema/v0.toml").unwrap();
    let cfg = RewardConfig::load("../configs/reward_v0.toml").unwrap();
    EpisodeArena::new(blue, orange, s.tick_skip, cfg, s.normalization, seed)
}

#[test]
fn deterministic_given_seed_and_actions() {
    let (mut a, mut b) = (mk(1, 1, 123), mk(1, 1, 123));
    let mut oa = vec![0.0; 2 * OBS_SIZE];
    let mut ob = vec![0.0; 2 * OBS_SIZE];
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    for step in 0..50 {
        let acts = [(step % 90) as i64, ((step * 7) % 90) as i64];
        a.step(&acts, &mut r, &mut f, &mut fo);
        let ra = r.clone();
        b.step(&acts, &mut r, &mut f, &mut fo);
        assert_eq!(ra, r, "step {step}");
    }
    a.write_obs(&mut oa);
    b.write_obs(&mut ob);
    assert_eq!(oa, ob);
}

#[test]
fn scored_ball_terminates_and_pays_goal() {
    let mut a = mk(1, 1, 5);
    // Warp ball into orange net with velocity (test helper below)
    a.debug_place_ball([0.0, 5200.0, 320.0], [0.0, 2000.0, 0.0]);
    let mut r = vec![0.0; 2];
    let mut f = vec![StepFlags::default(); 2];
    let mut fo = vec![0.0; 2 * OBS_SIZE];
    let mut terminated = false;
    for _ in 0..30 {
        a.step(&[0, 0], &mut r, &mut f, &mut fo);
        if f[0].terminated {
            terminated = true;
            assert!(r[0] > 5.0, "blue agent gets goal reward, got {}", r[0]);
            assert!(r[1] < -5.0, "orange agent concedes, got {}", r[1]);
            break;
        }
    }
    assert!(terminated, "ball placed in goal mouth must score within 30 steps");
}

#[test]
fn no_touch_truncates_after_30s() {
    let mut a = mk(1, 0, 5);
    let mut r = vec![0.0; 1];
    let mut f = vec![StepFlags::default(); 1];
    let mut fo = vec![0.0; OBS_SIZE];
    let mut truncated_at = None;
    for step in 0..600 {
        a.step(&[2], &mut r, &mut f, &mut fo); // action 2 = straight reverse: drives away from ball
        if f[0].truncated {
            truncated_at = Some(step);
            break;
        }
    }
    // 30s at 15 steps/s = step 449 (0-indexed step count 450)
    let t = truncated_at.expect("must truncate");
    assert!((445..=455).contains(&t), "truncated at {t}");
}
```

Add `#[derive(Default, Clone, Copy)]` to `StepFlags`, plus a `debug_place_ball(pos: [f32;3], vel: [f32;3])` test helper method on `EpisodeArena`.

- [ ] **Step 2: Run to verify failure**

Run: `cd engine && cargo test --test episode_test`
Expected: compile FAIL

- [ ] **Step 3: Implement episode.rs**

```rust
use crate::{
    actions,
    obs::{self, OBS_SIZE},
    reward::{self, RewardConfig},
    schema::Normalization,
};
use cxx::UniquePtr;
use rocketsim_rs::{
    math::Vec3,
    sim::{Arena, CarConfig, Team},
    GameState,
};

const TICKS_PER_SEC: u64 = 120;
const NO_TOUCH_TICKS: u64 = 30 * TICKS_PER_SEC;
const MAX_TICKS: u64 = 300 * TICKS_PER_SEC;

#[derive(Debug, Default, Clone, Copy)]
pub struct StepFlags {
    pub terminated: bool,
    pub truncated: bool,
}

pub struct EpisodeArena {
    arena: UniquePtr<Arena>,
    table: Vec<[f32; 8]>,
    car_ids: Vec<u32>, // blue asc, then orange asc — agent index order
    tick_skip: u32,
    reward_cfg: RewardConfig,
    norm: Normalization,
    seed: u32,
    episode_start_tick: u64,
    last_touch_tick: u64,
    prev_state: GameState,
}

impl EpisodeArena {
    pub fn new(
        blue: usize,
        orange: usize,
        tick_skip: u32,
        reward_cfg: RewardConfig,
        norm: Normalization,
        seed: u32,
    ) -> Self {
        let mut arena = Arena::default_standard();
        let mut car_ids = Vec::with_capacity(blue + orange);
        for _ in 0..blue {
            car_ids.push(arena.pin_mut().add_car(Team::Blue, CarConfig::octane()));
        }
        for _ in 0..orange {
            car_ids.push(arena.pin_mut().add_car(Team::Orange, CarConfig::octane()));
        }
        arena.pin_mut().reset_to_random_kickoff(Some(seed));
        let prev_state = arena.pin_mut().get_game_state();
        let start = prev_state.tick_count;
        Self {
            arena,
            table: actions::make_lookup_table(),
            car_ids,
            tick_skip,
            reward_cfg,
            norm,
            seed,
            episode_start_tick: start,
            last_touch_tick: start,
            prev_state,
        }
    }

    pub fn num_agents(&self) -> usize {
        self.car_ids.len()
    }

    fn agent_car_index(&self, state: &GameState, agent: usize) -> usize {
        let id = self.car_ids[agent];
        state.cars.iter().position(|c| c.id == id).expect("car exists")
    }

    pub fn write_obs(&mut self, out: &mut [f32]) {
        let gs = self.arena.pin_mut().get_game_state();
        for a in 0..self.num_agents() {
            let ci = self.agent_car_index(&gs, a);
            obs::build_obs(&gs, ci, &self.norm, &mut out[a * OBS_SIZE..(a + 1) * OBS_SIZE]);
        }
    }

    pub fn step(
        &mut self,
        action_idx: &[i64],
        rewards: &mut [f32],
        flags: &mut [StepFlags],
        final_obs: &mut [f32],
    ) {
        let n = self.num_agents();
        assert_eq!(action_idx.len(), n);

        let controls: Vec<(u32, rocketsim_rs::sim::CarControls)> = (0..n)
            .map(|a| {
                let row = &self.table[action_idx[a] as usize];
                (self.car_ids[a], actions::to_controls(row))
            })
            .collect();
        self.arena.pin_mut().set_all_controls(&controls).expect("valid car ids");
        self.arena.pin_mut().step(self.tick_skip);

        let cur = self.arena.pin_mut().get_game_state();
        let scored = if self.arena.is_ball_scored() {
            Some(if cur.ball.pos.y > 0.0 { Team::Blue } else { Team::Orange })
        } else {
            None
        };

        // touch tracking for no-touch truncation
        for a in 0..n {
            let ci = self.agent_car_index(&cur, a);
            let hit = cur.cars[ci].state.ball_hit_info.tick_count_when_hit;
            if hit > self.last_touch_tick && hit <= cur.tick_count {
                self.last_touch_tick = hit;
            }
        }

        let terminated = scored.is_some();
        let truncated = !terminated
            && (cur.tick_count - self.last_touch_tick >= NO_TOUCH_TICKS
                || cur.tick_count - self.episode_start_tick >= MAX_TICKS);

        for a in 0..n {
            let ci = self.agent_car_index(&cur, a);
            rewards[a] = reward::compute(&self.prev_state, &cur, ci, scored, &self.reward_cfg);
            flags[a] = StepFlags { terminated, truncated };
        }

        if terminated || truncated {
            // capture final obs, then reset
            for a in 0..n {
                let ci = self.agent_car_index(&cur, a);
                obs::build_obs(&cur, ci, &self.norm, &mut final_obs[a * OBS_SIZE..(a + 1) * OBS_SIZE]);
            }
            self.seed = self.seed.wrapping_mul(747796405).wrapping_add(2891336453);
            self.arena.pin_mut().reset_to_random_kickoff(Some(self.seed));
            let gs = self.arena.pin_mut().get_game_state();
            self.episode_start_tick = gs.tick_count;
            self.last_touch_tick = gs.tick_count;
            self.prev_state = gs;
        } else {
            self.prev_state = cur;
        }
    }

    pub fn game_state(&mut self) -> GameState {
        self.arena.pin_mut().get_game_state()
    }

    /// Test/debug helper: warp the ball.
    pub fn debug_place_ball(&mut self, pos: [f32; 3], vel: [f32; 3]) {
        let mut ball = self.arena.pin_mut().get_ball();
        ball.pos = Vec3::new(pos[0], pos[1], pos[2]);
        ball.vel = Vec3::new(vel[0], vel[1], vel[2]);
        self.arena.pin_mut().set_ball(ball);
    }
}
```
Note: `is_ball_scored()` takes `&Arena` (no pin) per the API report. Goal-y sign check happens on the state captured after stepping, before RocketSim's own reset (RocketSim does not auto-reset without a goal callback — we never register one, we reset ourselves).

- [ ] **Step 4: Run tests**

Run: `cd engine && cargo test --test episode_test`
Expected: 3 passed. If `scored_ball_terminates_and_pays_goal` flakes on ball placement, place ball deeper (y=5300) or raise step allowance.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: EpisodeArena with goal/no-touch/timeout episode logic and auto-reset"
```

---

### Task 8: Multi-arena Engine + PyO3 batch API

**Files:**
- Create: `engine/src/engine.rs`
- Modify: `engine/src/lib.rs` (register `Engine` pyclass)
- Test: `tests/python/test_engine.py`

**Interfaces:**
- Consumes: `EpisodeArena` (Task 7), schema (Task 4).
- Produces (Python API — the trainer contract for Task 10):
  ```python
  from construct._engine import Engine
  eng = Engine(num_arenas=32, blue=1, orange=1,
               schema_path="schema/v0.toml",
               reward_config_path="configs/reward_v0.toml",
               meshes_path=None, seed=0, num_threads=0)  # 0 = auto
  eng.num_agents   # num_arenas * (blue + orange)
  eng.obs_size     # 94
  eng.action_count # 90
  obs = eng.reset()                      # np.float32 (num_agents, obs_size)
  obs, rew, term, trunc, final_obs = eng.step(actions)
  # actions: np.int64 (num_agents,) in [0, 90)
  # rew: f32 (num_agents,) ; term/trunc: bool (num_agents,)
  # final_obs: f32 (num_agents, obs_size) — valid only in rows where term|trunc
  ```
- Threading: worker threads own their arenas (arenas are created inside the worker thread; no cross-thread `UniquePtr<Arena>` movement assumptions). Command/response over `std::sync::mpsc` channels. GIL released during stepping via `py.detach(...)` (pyo3 0.29 name for `allow_threads` — verify at implementation, use whichever compiles).

- [ ] **Step 1: Write failing Python test**

```python
# tests/python/test_engine.py
import numpy as np
import pytest
from construct._engine import Engine

def mk(n=4, seed=0):
    return Engine(num_arenas=n, blue=1, orange=1, schema_path="schema/v0.toml",
                  reward_config_path="configs/reward_v0.toml", seed=seed)

def test_shapes_and_dtypes():
    eng = mk()
    assert eng.num_agents == 8 and eng.obs_size == 94 and eng.action_count == 90
    obs = eng.reset()
    assert obs.shape == (8, 94) and obs.dtype == np.float32
    acts = np.zeros(8, dtype=np.int64)
    obs, rew, term, trunc, final_obs = eng.step(acts)
    assert obs.shape == (8, 94) and rew.shape == (8,)
    assert term.dtype == np.bool_ and trunc.dtype == np.bool_
    assert final_obs.shape == (8, 94)

def test_deterministic_with_seed():
    a, b = mk(seed=7), mk(seed=7)
    a.reset(); b.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):
        acts = rng.integers(0, 90, size=8).astype(np.int64)
        oa, ra, *_ = a.step(acts)
        ob, rb, *_ = b.step(acts)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)

def test_rejects_bad_actions():
    eng = mk()
    eng.reset()
    with pytest.raises(Exception):
        eng.step(np.full(8, 90, dtype=np.int64))  # out of range
    with pytest.raises(Exception):
        eng.step(np.zeros(3, dtype=np.int64))     # wrong length

def test_episodes_eventually_end():
    eng = mk(n=2)
    eng.reset()
    acts = np.full(4, 4, dtype=np.int64)  # idle-ish action -> no-touch truncation
    done_seen = False
    for _ in range(500):
        _, _, term, trunc, _ = eng.step(acts)
        if term.any() or trunc.any():
            done_seen = True
            break
    assert done_seen
```

- [ ] **Step 2: Run to verify failure**

Run: `pytest tests/python/test_engine.py -x`
Expected: ImportError (`Engine` not found)

- [ ] **Step 3: Implement engine.rs**

```rust
use crate::{
    episode::{EpisodeArena, StepFlags},
    obs::OBS_SIZE,
    reward::RewardConfig,
    schema::Schema,
};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::thread::JoinHandle;

enum Cmd {
    Reset,
    Step(Vec<i64>),
    Shutdown,
}

struct WorkerOut {
    obs: Vec<f32>,
    rewards: Vec<f32>,
    terminated: Vec<bool>,
    truncated: Vec<bool>,
    final_obs: Vec<f32>,
}

struct Worker {
    tx: Sender<Cmd>,
    rx: Receiver<WorkerOut>,
    handle: Option<JoinHandle<()>>,
    num_agents: usize,
}

pub struct MultiEngine {
    workers: Vec<Worker>,
    pub num_agents: usize,
    pub obs_size: usize,
    pub action_count: usize,
}

impl MultiEngine {
    pub fn new(
        num_arenas: usize,
        blue: usize,
        orange: usize,
        schema: Schema,
        reward_cfg: RewardConfig,
        seed: u32,
        num_threads: usize,
    ) -> Self {
        let threads = if num_threads == 0 {
            std::thread::available_parallelism().map(|n| n.get().saturating_sub(2).max(1)).unwrap_or(4)
        } else {
            num_threads
        }
        .min(num_arenas);
        let per_agent = blue + orange;

        // distribute arenas round-robin-contiguously over threads
        let mut workers = Vec::with_capacity(threads);
        let mut assigned = 0usize;
        for t in 0..threads {
            let count = (num_arenas - assigned) / (threads - t); // even split
            assigned += count;
            let (ctx, crx) = channel::<Cmd>();
            let (otx, orx) = channel::<WorkerOut>();
            let (sch, cfg) = (schema.clone(), reward_cfg.clone());
            let wseed = seed.wrapping_add((t as u32) << 16);
            let handle = std::thread::spawn(move || {
                // arenas created inside the worker thread
                let mut arenas: Vec<EpisodeArena> = (0..count)
                    .map(|i| {
                        EpisodeArena::new(blue, orange, sch.tick_skip, cfg.clone(),
                                          sch.normalization.clone(), wseed.wrapping_add(i as u32))
                    })
                    .collect();
                let agents = count * per_agent;
                while let Ok(cmd) = crx.recv() {
                    match cmd {
                        Cmd::Shutdown => break,
                        Cmd::Reset => {
                            let mut out = WorkerOut {
                                obs: vec![0.0; agents * OBS_SIZE],
                                rewards: vec![0.0; agents],
                                terminated: vec![false; agents],
                                truncated: vec![false; agents],
                                final_obs: vec![0.0; agents * OBS_SIZE],
                            };
                            let mut off = 0;
                            for ar in arenas.iter_mut() {
                                let n = ar.num_agents() * OBS_SIZE;
                                ar.write_obs(&mut out.obs[off..off + n]);
                                off += n;
                            }
                            let _ = otx.send(out);
                        }
                        Cmd::Step(acts) => {
                            let mut out = WorkerOut {
                                obs: vec![0.0; agents * OBS_SIZE],
                                rewards: vec![0.0; agents],
                                terminated: vec![false; agents],
                                truncated: vec![false; agents],
                                final_obs: vec![0.0; agents * OBS_SIZE],
                            };
                            let mut a_off = 0;
                            let mut flags = vec![StepFlags::default(); per_agent];
                            for ar in arenas.iter_mut() {
                                let n = ar.num_agents();
                                ar.step(
                                    &acts[a_off..a_off + n],
                                    &mut out.rewards[a_off..a_off + n],
                                    &mut flags[..n],
                                    &mut out.final_obs[a_off * OBS_SIZE..(a_off + n) * OBS_SIZE],
                                );
                                for (i, f) in flags[..n].iter().enumerate() {
                                    out.terminated[a_off + i] = f.terminated;
                                    out.truncated[a_off + i] = f.truncated;
                                }
                                ar.write_obs(&mut out.obs[a_off * OBS_SIZE..(a_off + n) * OBS_SIZE]);
                                a_off += n;
                            }
                            let _ = otx.send(out);
                        }
                    }
                }
            });
            workers.push(Worker { tx: ctx, rx: orx, handle: Some(handle), num_agents: count * per_agent });
        }

        MultiEngine {
            num_agents: num_arenas * per_agent,
            obs_size: OBS_SIZE,
            action_count: crate::actions::TABLE_SIZE,
            workers,
        }
    }

    pub fn reset_into(&mut self, obs: &mut [f32]) {
        for w in &self.workers {
            w.tx.send(Cmd::Reset).unwrap();
        }
        let mut off = 0;
        for w in &self.workers {
            let out = w.rx.recv().unwrap();
            obs[off * OBS_SIZE..(off + w.num_agents) * OBS_SIZE].copy_from_slice(&out.obs);
            off += w.num_agents;
        }
    }

    pub fn step_into(
        &mut self,
        actions: &[i64],
        obs: &mut [f32],
        rewards: &mut [f32],
        terminated: &mut [bool],
        truncated: &mut [bool],
        final_obs: &mut [f32],
    ) -> Result<(), String> {
        if actions.len() != self.num_agents {
            return Err(format!("expected {} actions, got {}", self.num_agents, actions.len()));
        }
        if let Some(bad) = actions.iter().find(|&&a| a < 0 || a as usize >= self.action_count) {
            return Err(format!("action index {bad} out of range [0, {})", self.action_count));
        }
        let mut off = 0;
        for w in &self.workers {
            w.tx.send(Cmd::Step(actions[off..off + w.num_agents].to_vec())).unwrap();
            off += w.num_agents;
        }
        off = 0;
        for w in &self.workers {
            let out = w.rx.recv().unwrap();
            let (s, e) = (off, off + w.num_agents);
            obs[s * OBS_SIZE..e * OBS_SIZE].copy_from_slice(&out.obs);
            rewards[s..e].copy_from_slice(&out.rewards);
            terminated[s..e].copy_from_slice(&out.terminated);
            truncated[s..e].copy_from_slice(&out.truncated);
            final_obs[s * OBS_SIZE..e * OBS_SIZE].copy_from_slice(&out.final_obs);
            off += w.num_agents;
        }
        Ok(())
    }
}

impl Drop for MultiEngine {
    fn drop(&mut self) {
        for w in &self.workers {
            let _ = w.tx.send(Cmd::Shutdown);
        }
        for w in &mut self.workers {
            if let Some(h) = w.handle.take() {
                let _ = h.join();
            }
        }
    }
}
```

- [ ] **Step 4: Add the pyclass wrapper in lib.rs**

```rust
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub mod actions;
pub mod engine;
pub mod episode;
pub mod obs;
pub mod reward;
pub mod schema;
pub mod sim_init;

#[pyclass]
struct Engine {
    inner: engine::MultiEngine,
}

#[pymethods]
impl Engine {
    #[new]
    #[pyo3(signature = (num_arenas=32, blue=1, orange=1, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", meshes_path=None,
                        seed=0, num_threads=0))]
    fn new(
        num_arenas: usize,
        blue: usize,
        orange: usize,
        schema_path: &str,
        reward_config_path: &str,
        meshes_path: Option<&str>,
        seed: u32,
        num_threads: usize,
    ) -> PyResult<Self> {
        sim_init::ensure_init(meshes_path);
        let sch = schema::Schema::load(schema_path).map_err(PyValueError::new_err)?;
        let cfg = reward::RewardConfig::load(reward_config_path).map_err(PyValueError::new_err)?;
        Ok(Engine { inner: engine::MultiEngine::new(num_arenas, blue, orange, sch, cfg, seed, num_threads) })
    }

    #[getter]
    fn num_agents(&self) -> usize { self.inner.num_agents }
    #[getter]
    fn obs_size(&self) -> usize { self.inner.obs_size }
    #[getter]
    fn action_count(&self) -> usize { self.inner.action_count }

    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let (n, d) = (self.inner.num_agents, self.inner.obs_size);
        let mut obs = vec![0.0f32; n * d];
        py.detach(|| self.inner.reset_into(&mut obs));
        numpy::ndarray::Array2::from_shape_vec((n, d), obs).unwrap().into_pyarray(py)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let acts = actions.as_slice()?.to_vec();
        let (n, d) = (self.inner.num_agents, self.inner.obs_size);
        let (mut obs, mut fin) = (vec![0.0f32; n * d], vec![0.0f32; n * d]);
        let mut rew = vec![0.0f32; n];
        let (mut term, mut trunc) = (vec![false; n], vec![false; n]);
        py.detach(|| {
            self.inner.step_into(&acts, &mut obs, &mut rew, &mut term, &mut trunc, &mut fin)
        })
        .map_err(PyValueError::new_err)?;
        Ok((
            numpy::ndarray::Array2::from_shape_vec((n, d), obs).unwrap().into_pyarray(py),
            rew.into_pyarray(py),
            term.into_pyarray(py),
            trunc.into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((n, d), fin).unwrap().into_pyarray(py),
        ))
    }
}
```
Register in the pymodule: `m.add_class::<Engine>()?;`. If `py.detach` doesn't exist in pyo3 0.29, use `py.allow_threads(...)` — one of the two compiles; they are the same function across the rename.

- [ ] **Step 5: Build and run Python tests**

Run: `maturin develop --release && pytest tests/python/test_engine.py -v`
Expected: 4 passed

- [ ] **Step 6: Benchmark throughput (informational, not a test)**

```python
# scripts/bench_engine.py
import time
import numpy as np
from construct._engine import Engine

eng = Engine(num_arenas=64, blue=1, orange=1)
eng.reset()
acts = np.random.default_rng(0).integers(0, 90, size=eng.num_agents).astype(np.int64)
t0 = time.perf_counter()
N = 2000
for _ in range(N):
    eng.step(acts)
dt = time.perf_counter() - t0
print(f"{N * eng.num_agents / dt:,.0f} agent-steps/sec ({N * 64 / dt:,.0f} env-steps/sec)")
```
Run: `python scripts/bench_engine.py` — record the number in the commit message. Target: >30k env-steps/sec on 16 threads (no NN inference yet).

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: multi-arena threaded Engine with numpy batch API (bench: <N> sps)"
```

---

### Task 9: Python — policy net, GAE, PPO update

**Files:**
- Create: `python/construct/learn/__init__.py`, `python/construct/learn/model.py`, `python/construct/learn/gae.py`, `python/construct/learn/ppo.py`
- Test: `tests/python/test_gae.py`, `tests/python/test_ppo.py`

**Interfaces:**
- Produces:
  - `model.PolicyValueNet(obs_size: int, action_count: int, hidden: tuple[int, ...] = (512, 512))` — `forward(obs) -> (logits, value)`; `act(obs) -> (action, logprob, value)` (sampling); `evaluate(obs, actions) -> (logprobs, entropy, values)`.
  - `gae.compute_gae(rewards, values, final_values, terminated, truncated, gamma, lam) -> (advantages, returns)` — all inputs `np.ndarray` shaped `(T, N)`.
  - `ppo.ppo_update(net, optimizer, batch, clip=0.2, entropy_coef=0.01, value_coef=1.0, epochs=3, minibatch_size=4096, max_grad_norm=0.5) -> dict` (loss stats).

- [ ] **Step 1: Install torch**

Run: `uv pip install "torch>=2.12"`
Expected: installs CUDA-enabled build under WSL2 (verify `python -c "import torch; print(torch.cuda.is_available())"` → `True`; CPU-only is acceptable fallback, training just runs slower).

- [ ] **Step 2: Write failing GAE test (hand-computed fixture)**

```python
# tests/python/test_gae.py
import numpy as np
from construct.learn.gae import compute_gae

def test_gae_matches_hand_computation():
    # T=3, N=1, gamma=0.5, lam=1.0 (plain discounted advantage), no dones
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    final_values = np.zeros((3, 1), dtype=np.float32)
    term = np.zeros((3, 1), dtype=bool)
    trunc = np.zeros((3, 1), dtype=bool)
    # bootstrap value after last step = 0 (values of step T would be needed;
    # convention: caller appends next_value row to values -> shape (T+1, N))
    values_ext = np.vstack([values, np.zeros((1, 1), dtype=np.float32)])
    adv, ret = compute_gae(rewards, values_ext, final_values, term, trunc, gamma=0.5, lam=1.0)
    # deltas: r + 0.5*V' - V = [1,1,1]; adv_2=1, adv_1=1+0.5*1=1.5, adv_0=1+0.5*1.5=1.75
    np.testing.assert_allclose(adv[:, 0], [1.75, 1.5, 1.0], atol=1e-6)
    np.testing.assert_allclose(ret[:, 0], adv[:, 0] + values[:, 0], atol=1e-6)

def test_truncation_bootstraps_final_obs_value_and_blocks_next_episode():
    rewards = np.array([[1.0], [1.0]], dtype=np.float32)
    values_ext = np.array([[0.0], [0.0], [9.0]], dtype=np.float32)  # V(post-reset obs) = 9
    final_values = np.array([[0.0], [5.0]], dtype=np.float32)       # V(final obs of step 1) = 5
    term = np.array([[False], [False]])
    trunc = np.array([[False], [True]])
    adv, _ = compute_gae(rewards, values_ext, final_values, term, trunc, gamma=1.0, lam=1.0)
    # step 1 (truncated): delta_1 = 1 + V(final_obs)=5 - 0 = 6; done_1 blocks flow
    #   from the NEXT episode (post-reset V=9 must not leak in) -> adv_1 = 6
    # step 0 (same episode as step 1): delta_0 = 1 + values[1]=0 - 0 = 1;
    #   done_0=False so adv_1 flows back: adv_0 = 1 + 1*1*6 = 7
    np.testing.assert_allclose(adv[:, 0], [7.0, 6.0], atol=1e-6)
```
(The second test pins the rules: truncated → bootstrap from `final_values`; done at step t blocks advantage flow from the *next episode* into step t, while step t's own advantage still flows to earlier steps of its episode.)

- [ ] **Step 3: Implement gae.py**

```python
import numpy as np


def compute_gae(
    rewards: np.ndarray,      # (T, N)
    values: np.ndarray,       # (T+1, N) — includes V(s_{T}) bootstrap row
    final_values: np.ndarray, # (T, N) — V(final_obs) rows valid where truncated
    terminated: np.ndarray,   # (T, N) bool
    truncated: np.ndarray,    # (T, N) bool
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        done = terminated[t] | truncated[t]
        # value after this transition:
        #  - terminated: 0
        #  - truncated: V(final_obs) (pre-reset state)
        #  - otherwise: V(next obs) = values[t+1]
        next_v = np.where(terminated[t], 0.0, np.where(truncated[t], final_values[t], values[t + 1]))
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * (~done) * last
        adv[t] = last
    returns = adv + values[:T]
    return adv, returns
```

- [ ] **Step 4: Run GAE tests**

Run: `pytest tests/python/test_gae.py -v`
Expected: 2 passed

- [ ] **Step 5: Write model.py**

```python
import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden: tuple[int, ...] = (512, 512)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_size
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_count)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value
```

- [ ] **Step 6: Write failing PPO test (bandit overfit)**

```python
# tests/python/test_ppo.py
import numpy as np
import torch
from construct.learn.model import PolicyValueNet
from construct.learn.ppo import ppo_update

def test_ppo_solves_two_armed_bandit():
    torch.manual_seed(0)
    net = PolicyValueNet(obs_size=4, action_count=2, hidden=(32,))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    obs = torch.ones(512, 4)
    for _ in range(40):
        with torch.no_grad():
            actions, logprobs, values = net.act(obs)
        rewards = (actions == 0).float()          # arm 0 pays 1
        adv = rewards - rewards.mean()
        batch = {
            "obs": obs, "actions": actions, "logprobs": logprobs,
            "advantages": adv, "returns": rewards, "values": values,
        }
        ppo_update(net, opt, batch, epochs=2, minibatch_size=256)
    with torch.no_grad():
        logits, _ = net(obs[:1])
        p0 = torch.softmax(logits, -1)[0, 0].item()
    assert p0 > 0.9, f"P(arm0)={p0}"
```

- [ ] **Step 7: Implement ppo.py**

```python
import torch


def ppo_update(
    net,
    optimizer,
    batch: dict,          # obs, actions, logprobs, advantages, returns, values (flat tensors)
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 1.0,
    epochs: int = 3,
    minibatch_size: int = 4096,
    max_grad_norm: float = 0.5,
) -> dict:
    n = batch["obs"].shape[0]
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "updates": 0}
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["obs"].device)
        for s in range(0, n, minibatch_size):
            idx = perm[s : s + minibatch_size]
            logprobs, entropy, values = net.evaluate(batch["obs"][idx], batch["actions"][idx])
            ratio = torch.exp(logprobs - batch["logprobs"][idx])
            a = adv[idx]
            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, batch["returns"][idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.mean().item()
            stats["clip_frac"] += ((ratio - 1).abs() > clip).float().mean().item()
            stats["updates"] += 1
    for k in ("policy_loss", "value_loss", "entropy", "clip_frac"):
        stats[k] /= max(stats["updates"], 1)
    return stats
```

- [ ] **Step 8: Run PPO tests**

Run: `pytest tests/python/test_ppo.py tests/python/test_gae.py -v`
Expected: all pass (bandit test takes seconds on CPU)

- [ ] **Step 9: Commit**

```bash
git add -A && git commit -m "feat: PolicyValueNet, GAE with truncation bootstrap, clipped PPO update"
```

---

### Task 10: Training loop + checkpoints + smoke test

**Files:**
- Create: `python/construct/learn/train.py`, `python/construct/learn/config.py`, `configs/train_v0.toml`, `scripts/smoke_test.py`
- Test: `tests/python/test_train.py`

**Interfaces:**
- Consumes: `Engine` (Task 8), `PolicyValueNet`/`compute_gae`/`ppo_update` (Task 9), schema (Task 4).
- Produces:
  - `config.TrainConfig.load(path) -> TrainConfig` (dataclass mirroring `configs/train_v0.toml`).
  - `train.Trainer(cfg: TrainConfig)` with `.run(max_iterations: int | None = None)`, `.save_checkpoint(path)`, `Trainer.load_checkpoint(path) -> Trainer` (resumes step count + optimizer).
  - Checkpoint format: single `torch.save` dict — `{"model": state_dict, "optimizer": state_dict, "total_steps": int, "schema_version": 0, "config": dict}`. **Deploy (Task 13) reads `model` + `schema_version` from this exact format.**

- [ ] **Step 1: Write configs/train_v0.toml**

```toml
schema_path = "schema/v0.toml"
reward_config_path = "configs/reward_v0.toml"

[env]
num_arenas = 64
blue = 1
orange = 1
seed = 0

[net]
hidden = [512, 512]

[ppo]
rollout_steps = 256        # T per iteration -> batch = T * num_agents = 32768
gamma = 0.9954             # ~10s half-life at 15 Hz
lam = 0.95
lr = 2e-4
clip = 0.2
entropy_coef = 0.01
value_coef = 1.0
epochs = 3
minibatch_size = 8192

[run]
device = "cuda"            # falls back to cpu if unavailable
checkpoint_dir = "checkpoints"
save_every_iters = 20
log_every_iters = 1
```

- [ ] **Step 2: Write config.py**

```python
import tomllib
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    schema_path: str
    reward_config_path: str
    env: dict = field(default_factory=dict)
    net: dict = field(default_factory=dict)
    ppo: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(**raw)
```

- [ ] **Step 3: Write failing trainer test**

```python
# tests/python/test_train.py
import numpy as np
import torch
from construct.learn.config import TrainConfig
from construct.learn.train import Trainer

def small_cfg(tmp_path):
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=4)
    cfg.ppo.update(rollout_steps=16, minibatch_size=128)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=1)
    return cfg

def test_one_iteration_runs_and_steps_counted(tmp_path):
    t = Trainer(small_cfg(tmp_path))
    t.run(max_iterations=1)
    assert t.total_steps == 16 * 8  # T * num_agents

def test_checkpoint_roundtrip_resumes(tmp_path):
    t = Trainer(small_cfg(tmp_path))
    t.run(max_iterations=1)
    p = f"{tmp_path}/ck.pt"
    t.save_checkpoint(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["schema_version"] == 0 and ck["total_steps"] == 128
    t2 = Trainer.load_checkpoint(p)
    assert t2.total_steps == 128
    before = [x.clone() for x in t2.net.parameters()]
    t2.run(max_iterations=1)
    assert t2.total_steps == 256
    assert any(not torch.equal(a, b) for a, b in zip(before, t2.net.parameters()))
```

- [ ] **Step 4: Implement train.py**

```python
import os
import time

import numpy as np
import torch

from construct._engine import Engine, schema_dict
from construct.learn.config import TrainConfig
from construct.learn.gae import compute_gae
from construct.learn.model import PolicyValueNet
from construct.learn.ppo import ppo_update


class Trainer:
    def __init__(self, cfg: TrainConfig, _state: dict | None = None):
        self.cfg = cfg
        self.schema = schema_dict(cfg.schema_path)
        self.engine = Engine(
            num_arenas=cfg.env["num_arenas"], blue=cfg.env["blue"], orange=cfg.env["orange"],
            schema_path=cfg.schema_path, reward_config_path=cfg.reward_config_path,
            seed=cfg.env["seed"],
        )
        dev = cfg.run.get("device", "cuda")
        self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
        self.net = PolicyValueNet(
            self.engine.obs_size, self.engine.action_count, tuple(cfg.net["hidden"])
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo["lr"])
        self.total_steps = 0
        if _state:
            self.net.load_state_dict(_state["model"])
            self.opt.load_state_dict(_state["optimizer"])
            self.total_steps = _state["total_steps"]
        self.obs = torch.as_tensor(self.engine.reset(), device=self.device)

    def collect(self, T: int) -> dict:
        N, D = self.engine.num_agents, self.engine.obs_size
        obs_b = torch.zeros((T, N, D), device=self.device)
        act_b = torch.zeros((T, N), dtype=torch.long, device=self.device)
        logp_b = torch.zeros((T, N), device=self.device)
        val_b = torch.zeros((T + 1, N), device=self.device)
        rew_b = np.zeros((T, N), dtype=np.float32)
        term_b = np.zeros((T, N), dtype=bool)
        trunc_b = np.zeros((T, N), dtype=bool)
        finv_b = torch.zeros((T, N), device=self.device)

        for t in range(T):
            action, logp, value = self.net.act(self.obs)
            obs_b[t], act_b[t], logp_b[t], val_b[t] = self.obs, action, logp, value
            nobs, rew, term, trunc, fin = self.engine.step(action.cpu().numpy().astype(np.int64))
            rew_b[t], term_b[t], trunc_b[t] = rew, term, trunc
            done = term | trunc
            if done.any():
                with torch.no_grad():
                    fv = self.net(torch.as_tensor(fin[done], device=self.device))[1]
                finv_b[t, torch.as_tensor(done, device=self.device)] = fv
            self.obs = torch.as_tensor(nobs, device=self.device)
        with torch.no_grad():
            val_b[T] = self.net(self.obs)[1]

        adv, ret = compute_gae(
            rew_b, val_b.cpu().numpy(), finv_b.cpu().numpy(), term_b, trunc_b,
            self.cfg.ppo["gamma"], self.cfg.ppo["lam"],
        )
        flat = lambda x: x.reshape(-1, *x.shape[2:])
        return {
            "obs": flat(obs_b), "actions": flat(act_b), "logprobs": flat(logp_b),
            "advantages": torch.as_tensor(adv, device=self.device).reshape(-1),
            "returns": torch.as_tensor(ret, device=self.device).reshape(-1),
            "values": flat(val_b[:T]),
            "ep_reward_mean": float(rew_b.sum() / max(1, (term_b | trunc_b).sum())),
        }

    def run(self, max_iterations: int | None = None):
        it = 0
        p = self.cfg.ppo
        while max_iterations is None or it < max_iterations:
            t0 = time.perf_counter()
            batch = self.collect(p["rollout_steps"])
            stats = ppo_update(
                self.net, self.opt, batch, clip=p["clip"], entropy_coef=p["entropy_coef"],
                value_coef=p["value_coef"], epochs=p["epochs"], minibatch_size=p["minibatch_size"],
            )
            n = p["rollout_steps"] * self.engine.num_agents
            self.total_steps += n
            it += 1
            if it % self.cfg.run.get("log_every_iters", 1) == 0:
                sps = n / (time.perf_counter() - t0)
                print(
                    f"iter {it} steps {self.total_steps:,} sps {sps:,.0f} "
                    f"ep_rew {batch['ep_reward_mean']:.3f} "
                    f"pi_loss {stats['policy_loss']:.4f} v_loss {stats['value_loss']:.4f} "
                    f"ent {stats['entropy']:.3f} clip {stats['clip_frac']:.3f}",
                    flush=True,
                )
            if it % self.cfg.run.get("save_every_iters", 20) == 0:
                os.makedirs(self.cfg.run["checkpoint_dir"], exist_ok=True)
                self.save_checkpoint(
                    os.path.join(self.cfg.run["checkpoint_dir"], f"ck_{self.total_steps:012d}.pt")
                )

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.opt.state_dict(),
                "total_steps": self.total_steps,
                "schema_version": self.schema["version"],
                "config": {"net": self.cfg.net, "ppo": self.cfg.ppo, "env": self.cfg.env},
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, cfg_path: str = "configs/train_v0.toml") -> "Trainer":
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TrainConfig.load(cfg_path)
        cfg.net = state["config"]["net"]
        return cls(cfg, _state=state)


if __name__ == "__main__":
    import sys

    cfg = TrainConfig.load(sys.argv[1] if len(sys.argv) > 1 else "configs/train_v0.toml")
    Trainer(cfg).run()
```
(Deviation guard: `test_train.py` overrides device to cpu so CI-ish runs don't need CUDA.)

- [ ] **Step 5: Run trainer tests**

Run: `pytest tests/python/test_train.py -v`
Expected: 2 passed (~1-2 min: engine + small rollouts on CPU)

- [ ] **Step 6: Write scripts/smoke_test.py**

```python
"""End-to-end smoke: 1k env-steps of real training must run without error
and produce a loadable checkpoint. Run after any engine/learn change."""
import tempfile

from construct.learn.config import TrainConfig
from construct.learn.train import Trainer

with tempfile.TemporaryDirectory() as d:
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=8)
    cfg.ppo.update(rollout_steps=64, minibatch_size=512)
    cfg.run.update(checkpoint_dir=d, save_every_iters=2, device="cpu")
    t = Trainer(cfg)
    t.run(max_iterations=2)
    assert t.total_steps == 2 * 64 * 16
    print("SMOKE OK:", t.total_steps, "steps")
```

Run: `python scripts/smoke_test.py`
Expected: `SMOKE OK: 2048 steps`

- [ ] **Step 7: Commit**

```bash
git add -A && git commit -m "feat: PPO training loop with resumable checkpoints + smoke test"
```

---

### Task 11: RLViser rendering

**Files:**
- Create: `engine/src/viser.rs`, `scripts/watch.py`
- Modify: `engine/src/lib.rs`, `engine/Cargo.toml` (ensure `rocketsim_rs` `bin` feature — it's in defaults)
- Test: `engine/tests/viser_test.rs`

**Interfaces:**
- Consumes: `EpisodeArena::game_state()` (Task 7).
- Produces: pyclass `RenderSession` — single-arena env with the same step API (obs/step for 1 arena) that streams each state to RLViser over UDP and paces to realtime. Python: `RenderSession(blue=1, orange=1, ...)`, `.reset()`, `.step(actions) -> same 5-tuple`, `.close()` (sends Quit). `scripts/watch.py <checkpoint>` drives it with a trained policy.
- Protocol (rocketsim-rs 0.37.0 `flat_ext`): bind UDP `0.0.0.0:34254`, send `RlviserMessage::Connection` to `127.0.0.1:45243`, then one `RlviserMessage::GameState(...)` per step via `PacketCodec::encode`; `Quit` on close. RLViser binary launched manually by the user (`./rlviser` from any dir; Windows binary exists for running on the host while training in WSL2 — point it at the WSL IP if so, or run everything Linux-side with WSLg).

- [ ] **Step 1: Write failing codec round-trip test**

```rust
// engine/tests/viser_test.rs
use construct_engine::sim_init::ensure_init;
use rocketsim_rs::flat_ext::{PacketCodec, RlviserMessage, PACKET_SIZE_BYTES};
use rocketsim_rs::sim::{Arena, CarConfig, Team};

#[test]
fn gamestate_packet_roundtrips_through_codec() {
    ensure_init(None);
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(3));
    let gs = arena.pin_mut().get_game_state();

    let mut codec = PacketCodec::new();
    let bytes = codec.encode(RlviserMessage::GameState(Box::new(gs.clone()))).to_vec();
    assert!(bytes.len() > PACKET_SIZE_BYTES);
    let decoded = PacketCodec::decode_payload(&bytes[PACKET_SIZE_BYTES..]).unwrap().unwrap();
    match decoded {
        RlviserMessage::GameState(g) => {
            assert_eq!(g.cars.len(), 1);
            assert_eq!(g.tick_count, gs.tick_count);
        }
        other => panic!("wrong message decoded: {other:?}"),
    }
}
```
(If `GameState` isn't `Clone` or enum variants differ, adapt to the actual 0.37.0 `flat_ext` surface — the constants/types were verified from source, but check `cargo doc -p rocketsim_rs` for exact derive set.)

- [ ] **Step 2: Run to verify it compiles/fails appropriately, fix until green**

Run: `cd engine && cargo test --test viser_test`
Expected: pass once imports match

- [ ] **Step 3: Implement viser.rs**

```rust
use crate::episode::{EpisodeArena, StepFlags};
use rocketsim_rs::flat_ext::{PacketCodec, RlviserMessage};
use std::net::UdpSocket;
use std::time::{Duration, Instant};

pub struct ViserStream {
    socket: UdpSocket,
    codec: PacketCodec,
    target: &'static str,
}

impl ViserStream {
    pub fn new() -> std::io::Result<Self> {
        let socket = UdpSocket::bind(("0.0.0.0", 34254))?;
        socket.set_nonblocking(true)?;
        let mut s = Self { socket, codec: PacketCodec::new(), target: "127.0.0.1:45243" };
        s.send(RlviserMessage::Connection)?;
        Ok(s)
    }

    pub fn send(&mut self, msg: RlviserMessage) -> std::io::Result<()> {
        let bytes = self.codec.encode(msg);
        self.socket.send_to(bytes, self.target)?;
        Ok(())
    }

    pub fn send_state(&mut self, gs: rocketsim_rs::GameState) -> std::io::Result<()> {
        self.send(RlviserMessage::GameState(Box::new(gs)))
    }

    pub fn quit(&mut self) {
        let _ = self.send(RlviserMessage::Quit);
    }
}

/// Realtime pacing helper: sleep so each tick_skip step takes tick_skip/120 s.
pub struct Pacer {
    next: Instant,
    step_dur: Duration,
}

impl Pacer {
    pub fn new(tick_skip: u32) -> Self {
        Self { next: Instant::now(), step_dur: Duration::from_secs_f64(tick_skip as f64 / 120.0) }
    }
    pub fn pace(&mut self) {
        let now = Instant::now();
        if self.next > now {
            std::thread::sleep(self.next - now);
        }
        self.next = Instant::now().max(self.next) + self.step_dur;
    }
}
```
Then add pyclass `RenderSession` in `lib.rs` (single arena — no worker threads needed):

```rust
#[pyclass]
struct RenderSession {
    arena: episode::EpisodeArena,
    stream: viser::ViserStream,
    pacer: viser::Pacer,
    num_agents: usize,
}

#[pymethods]
impl RenderSession {
    #[new]
    #[pyo3(signature = (blue=1, orange=1, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", meshes_path=None, seed=0))]
    fn new(blue: usize, orange: usize, schema_path: &str, reward_config_path: &str,
           meshes_path: Option<&str>, seed: u32) -> PyResult<Self> {
        sim_init::ensure_init(meshes_path);
        let sch = schema::Schema::load(schema_path).map_err(PyValueError::new_err)?;
        let cfg = reward::RewardConfig::load(reward_config_path).map_err(PyValueError::new_err)?;
        let tick_skip = sch.tick_skip;
        Ok(RenderSession {
            arena: episode::EpisodeArena::new(blue, orange, tick_skip, cfg, sch.normalization, seed),
            stream: viser::ViserStream::new().map_err(|e| PyValueError::new_err(e.to_string()))?,
            pacer: viser::Pacer::new(tick_skip),
            num_agents: blue + orange,
        })
    }

    #[getter]
    fn obs_size(&self) -> usize { obs::OBS_SIZE }
    #[getter]
    fn action_count(&self) -> usize { actions::TABLE_SIZE }
    #[getter]
    fn num_agents(&self) -> usize { self.num_agents }

    fn reset<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray2<f32>> {
        let (n, d) = (self.num_agents, obs::OBS_SIZE);
        let mut o = vec![0.0f32; n * d];
        self.arena.write_obs(&mut o);
        numpy::ndarray::Array2::from_shape_vec((n, d), o).unwrap().into_pyarray(py)
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions_in: PyReadonlyArray1<'py, i64>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let acts = actions_in.as_slice()?.to_vec();
        let (n, d) = (self.num_agents, obs::OBS_SIZE);
        let (mut o, mut fin) = (vec![0.0f32; n * d], vec![0.0f32; n * d]);
        let mut rew = vec![0.0f32; n];
        let mut flags = vec![episode::StepFlags::default(); n];
        self.arena.step(&acts, &mut rew, &mut flags, &mut fin);
        self.arena.write_obs(&mut o);
        let gs = self.arena.game_state();
        let _ = self.stream.send_state(gs);
        self.pacer.pace();
        let term: Vec<bool> = flags.iter().map(|f| f.terminated).collect();
        let trunc: Vec<bool> = flags.iter().map(|f| f.truncated).collect();
        Ok((
            numpy::ndarray::Array2::from_shape_vec((n, d), o).unwrap().into_pyarray(py),
            rew.into_pyarray(py),
            term.into_pyarray(py),
            trunc.into_pyarray(py),
            numpy::ndarray::Array2::from_shape_vec((n, d), fin).unwrap().into_pyarray(py),
        ))
    }

    fn close(&mut self) {
        self.stream.quit();
    }
}
```
Register with `m.add_class::<RenderSession>()?;`.

- [ ] **Step 4: Write scripts/watch.py**

```python
"""Watch a checkpoint play in RLViser.
1) Download rlviser binary (github.com/VirxEC/rlviser/releases) into repo root
2) ./rlviser   (Linux/WSLg)  — or run rlviser on Windows and adjust target IP
3) python scripts/watch.py checkpoints/ck_XXXX.pt
"""
import sys

import numpy as np
import torch

from construct._engine import RenderSession
from construct.learn.model import PolicyValueNet

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sess = RenderSession(blue=1, orange=1, schema_path="schema/v0.toml",
                     reward_config_path="configs/reward_v0.toml", seed=42)
net = PolicyValueNet(sess.obs_size, sess.action_count, tuple(ck["config"]["net"]["hidden"]))
net.load_state_dict(ck["model"])
net.eval()

obs = torch.as_tensor(sess.reset())
try:
    while True:
        with torch.no_grad():
            actions = net(obs)[0].argmax(-1)
        nobs, *_ = sess.step(actions.numpy().astype(np.int64))
        obs = torch.as_tensor(nobs)
except KeyboardInterrupt:
    sess.close()
```

- [ ] **Step 5: Build + manual verification**

Run: `maturin develop --release && cargo test --test viser_test --manifest-path engine/Cargo.toml`
Expected: test passes. Manual check deferred to Task 14 (needs a trained checkpoint to be interesting); a random-policy sanity run works now: `python -c "..."` variant with random actions — cars should visibly drive in RLViser if the binary is running.

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: RLViser streaming RenderSession with realtime pacing"
```

---

### Task 12: Deploy obs/actions duplicates + parity test

**Files:**
- Create: `deploy/obs.py`, `deploy/actions.py`
- Modify: `engine/src/lib.rs` (add debug state dump)
- Test: `tests/python/test_parity.py`

**Interfaces:**
- Consumes: obs layout contract (Task 5), action table (Task 3).
- Produces:
  - `deploy/actions.py`: `make_lookup_table() -> np.ndarray (90, 8)` — verbatim RLGym reference implementation.
  - `deploy/obs.py`: `build_obs(state: dict, agent_key, pos_norm, vel_norm, ang_vel_norm) -> np.ndarray (94,)` where `state` is a plain dict (documented below) so the same function serves both the parity test and the RLBot bot (Task 13 adapts rlgym-compat objects into this dict).
  - Rust: `Engine.debug_state_and_obs(arena_idx: int) -> (str, np.ndarray)` — JSON dump of arena state + the engine-built obs for all agents of that arena. JSON schema:
    ```json
    {"ball": {"pos": [x,y,z], "vel": [...], "ang_vel": [...]},
     "cars": [{"id": 1, "team": 0, "pos": [...], "vel": [...], "ang_vel": [...],
               "forward": [...], "up": [...], "boost": 33.3,
               "is_on_ground": true, "has_flip": true, "is_demoed": false}]}
    ```

- [ ] **Step 1: Write deploy/actions.py (verbatim RLGym port — must match Rust Task 3)**

```python
import numpy as np


def make_lookup_table() -> np.ndarray:
    actions = []
    # Ground
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    # Aerial
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    return np.array(actions, dtype=np.float32)
```

- [ ] **Step 2: Write deploy/obs.py (mirror of Task 5 layout, pure numpy, f32 throughout)**

```python
import numpy as np

OBS_SIZE = 94
MAX_OTHERS = 5


def _mir(v: np.ndarray, mirror: bool) -> np.ndarray:
    return np.array([-v[0], -v[1], v[2]], dtype=np.float32) if mirror else v.astype(np.float32)


def build_obs(state: dict, car_id: int, pos_norm: float, vel_norm: float, ang_vel_norm: float) -> np.ndarray:
    pk, vk, ak = np.float32(pos_norm), np.float32(vel_norm), np.float32(ang_vel_norm)
    out = np.zeros(OBS_SIZE, dtype=np.float32)
    cars = {c["id"]: c for c in state["cars"]}
    me = cars[car_id]
    mirror = me["team"] == 1  # orange
    i = 0

    def put3(v, k):
        nonlocal i
        out[i : i + 3] = _mir(np.asarray(v, dtype=np.float32), mirror) * k
        i += 3

    def put(x):
        nonlocal i
        out[i] = np.float32(x)
        i += 1

    put3(me["pos"], pk); put3(me["forward"], np.float32(1.0)); put3(me["up"], np.float32(1.0))
    put3(me["vel"], vk); put3(me["ang_vel"], ak)
    put(me["boost"] / 100.0); put(float(me["is_on_ground"]))
    put(float(me["has_flip"])); put(float(me["is_demoed"]))

    b = state["ball"]
    put3(b["pos"], pk); put3(b["vel"], vk); put3(b["ang_vel"], ak)
    rel_p = np.asarray(b["pos"], np.float32) - np.asarray(me["pos"], np.float32)
    rel_v = np.asarray(b["vel"], np.float32) - np.asarray(me["vel"], np.float32)
    put3(rel_p, pk); put3(rel_v, vk)

    others = [c for c in state["cars"] if c["id"] != car_id]
    others.sort(key=lambda c: (c["team"] != me["team"], c["id"]))
    for c in others[:MAX_OTHERS]:
        put3(c["pos"], pk); put3(c["vel"], vk); put3(c["forward"], np.float32(1.0))
        put(c["boost"] / 100.0); put(float(c["team"] == me["team"])); put(float(not c["is_demoed"]))
    return out
```

- [ ] **Step 3: Add `debug_state_and_obs` to the Rust Engine pyclass**

In `episode.rs`, add a JSON dump method (serde_json is already a dependency):

```rust
impl EpisodeArena {
    /// JSON state dump matching the deploy/obs.py dict contract (Task 12 interfaces).
    pub fn debug_state_json(&mut self) -> String {
        let gs = self.arena.pin_mut().get_game_state();
        let v3 = |v: &Vec3| serde_json::json!([v.x, v.y, v.z]);
        let cars: Vec<serde_json::Value> = self
            .car_ids
            .iter()
            .map(|&id| {
                let c = gs.cars.iter().find(|c| c.id == id).expect("car exists");
                serde_json::json!({
                    "id": c.id,
                    "team": c.team as u8,
                    "pos": v3(&c.state.pos),
                    "vel": v3(&c.state.vel),
                    "ang_vel": v3(&c.state.ang_vel),
                    "forward": v3(&c.state.rot_mat.forward),
                    "up": v3(&c.state.rot_mat.up),
                    "boost": c.state.boost,
                    "is_on_ground": c.state.is_on_ground,
                    "has_flip": c.state.has_flip_or_jump(),
                    "is_demoed": c.state.is_demoed,
                })
            })
            .collect();
        serde_json::json!({
            "ball": {
                "pos": v3(&gs.ball.pos),
                "vel": v3(&gs.ball.vel),
                "ang_vel": v3(&gs.ball.ang_vel),
            },
            "cars": cars,
        })
        .to_string()
    }
}
```

In `engine.rs`, add a `Cmd::Debug { local_idx: usize }` variant; the worker replies with a new one-field response enum (or reuse `WorkerOut` with `obs` holding the arena's obs and stash the JSON in a `debug_json: Option<String>` field added to `WorkerOut`):

```rust
// worker match arm:
Cmd::Debug { local_idx } => {
    let ar = &mut arenas[local_idx];
    let n = ar.num_agents();
    let mut out = WorkerOut {
        obs: vec![0.0; n * OBS_SIZE],
        rewards: vec![], terminated: vec![], truncated: vec![], final_obs: vec![],
        debug_json: None,
    };
    ar.write_obs(&mut out.obs);
    out.debug_json = Some(ar.debug_state_json());
    let _ = otx.send(out);
}
```
(Add `debug_json: Option<String>` to `WorkerOut`, defaulting to `None` in the Reset/Step arms.)

`MultiEngine` maps the global `arena_idx` to `(worker, local_idx)` using the same even-split arithmetic as construction (store per-worker arena counts in `Worker` at build time). Then in `lib.rs`:

```rust
fn debug_state_and_obs<'py>(
    &mut self,
    py: Python<'py>,
    arena_idx: usize,
) -> PyResult<(String, Bound<'py, PyArray2<f32>>)> {
    let (json, obs, agents) = self
        .inner
        .debug_arena(arena_idx)
        .map_err(PyValueError::new_err)?;
    Ok((
        json,
        numpy::ndarray::Array2::from_shape_vec((agents, self.inner.obs_size), obs)
            .unwrap()
            .into_pyarray(py),
    ))
}
```
with `MultiEngine::debug_arena(&mut self, arena_idx) -> Result<(String, Vec<f32>, usize), String>` doing the send/recv round-trip and returning the worker's `debug_json` + obs + agent count.

- [ ] **Step 4: Write the parity test**

```python
# tests/python/test_parity.py
import json
import sys

import numpy as np

sys.path.insert(0, "deploy")
from obs import build_obs  # deploy/obs.py
from construct._engine import Engine, schema_dict


def test_deploy_obs_matches_engine_obs_exactly():
    eng = Engine(num_arenas=2, blue=1, orange=1, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=99)
    eng.reset()
    rng = np.random.default_rng(1)
    s = schema_dict("schema/v0.toml")
    for _ in range(30):  # step to varied states
        eng.step(rng.integers(0, 90, size=eng.num_agents).astype(np.int64))
    for arena in range(2):
        state_json, rust_obs = eng.debug_state_and_obs(arena)
        state = json.loads(state_json)
        for i, car in enumerate(state["cars"]):
            py_obs = build_obs(state, car["id"], s["pos_norm"], s["vel_norm"], s["ang_vel_norm"])
            diff = np.max(np.abs(py_obs - rust_obs[i]))
            assert diff < 1e-5, f"arena {arena} car {car['id']}: max diff {diff}"


def test_action_tables_match():
    from actions import make_lookup_table  # deploy/actions.py
    t = make_lookup_table()
    assert t.shape == (90, 8)
    # spot-check against the Rust-side contract rows (see Task 3 tests)
    np.testing.assert_array_equal(t[0], [-1, -1, 0, -1, 0, 0, 0, 0])
```
Tolerance note: 1e-5, not bit-identical — f32 JSON round-trip costs ulps. If diff exceeds this, the layouts genuinely diverge; fix the code, don't loosen the tolerance.

- [ ] **Step 5: Run tests**

Run: `maturin develop --release && pytest tests/python/test_parity.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: deploy-side obs/action duplicates with Rust parity tests"
```

---

### Task 13: RLBot v5 bot package

**Files:**
- Create: `deploy/bot.py`, `deploy/model.py`, `deploy/bot.toml`, `deploy/loadout.toml`, `deploy/match.toml`, `deploy/requirements.txt`, `deploy/README.md`
- Test: `tests/python/test_bot_logic.py`

**Interfaces:**
- Consumes: checkpoint format (Task 10), `deploy/obs.py` + `deploy/actions.py` (Task 12).
- Produces: a folder the user copies to Windows and registers in the RLBot v5 GUI. `deploy/model.py` re-declares `PolicyValueNet` (no `construct` package dependency on Windows). Bot logic split so it's testable without RLBot installed: `deploy/bot.py` contains `compat_to_state_dict(game_state, packet) -> dict` and `ConstructBot(Bot)`.

- [ ] **Step 1: Write deploy/model.py**

Copy of `python/construct/learn/model.py`'s `PolicyValueNet` class verbatim (18 lines), plus:

```python
import torch


def load_policy(checkpoint_path: str, obs_size: int, action_count: int) -> "PolicyValueNet":
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert ck["schema_version"] == 0, f"schema mismatch: {ck['schema_version']}"
    net = PolicyValueNet(obs_size, action_count, tuple(ck["config"]["net"]["hidden"]))
    net.load_state_dict(ck["model"])
    net.eval()
    torch.set_num_threads(1)
    return net
```

- [ ] **Step 2: Write deploy/bot.py**

```python
import os

import numpy as np
import torch

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot
from rlgym_compat import GameState
from rlgym_compat.sim_extra_info import SimExtraInfo

from actions import make_lookup_table
from model import load_policy
from obs import build_obs

TICK_SKIP = 8
POS_NORM = 1.0 / 2300.0
VEL_NORM = 1.0 / 2300.0
ANG_VEL_NORM = 1.0 / 5.5


def compat_to_state_dict(game_state: GameState) -> dict:
    """Adapt rlgym-compat GameState to the deploy/obs.py dict contract."""
    cars = []
    for agent_id, car in game_state.cars.items():
        phys = car.physics
        cars.append({
            "id": int(agent_id),
            "team": int(car.team_num),
            "pos": phys.position.tolist(),
            "vel": phys.linear_velocity.tolist(),
            "ang_vel": phys.angular_velocity.tolist(),
            "forward": phys.forward.tolist(),
            "up": phys.up.tolist(),
            "boost": float(car.boost_amount * 100.0),   # compat stores 0..1; engine uses 0..100
            "is_on_ground": bool(car.on_ground),
            "has_flip": bool(car.can_flip),
            "is_demoed": bool(car.is_demoed),
        })
    ball = game_state.ball
    return {
        "ball": {
            "pos": ball.position.tolist(),
            "vel": ball.linear_velocity.tolist(),
            "ang_vel": ball.angular_velocity.tolist(),
        },
        "cars": cars,
    }


class ConstructBot(Bot):
    def initialize(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.table = make_lookup_table()
        self.net = load_policy(os.path.join(here, "checkpoint.pt"), obs_size=94, action_count=90)
        self.extra_info = SimExtraInfo(self.field_info, tick_skip=TICK_SKIP)
        self.game_state = GameState.create_compat_game_state(self.field_info)
        self.ticks = TICK_SKIP  # act on first packet
        self.prev_control = ControllerState()
        self.prev_frame = -1

    def get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            return self.prev_control  # replay/kickoff countdown frames
        frame = packet.match_info.frame_num
        self.ticks += max(0, frame - self.prev_frame) if self.prev_frame >= 0 else TICK_SKIP
        self.prev_frame = frame
        if self.ticks < TICK_SKIP:
            return self.prev_control
        self.ticks = 0

        extra = self.extra_info.get_extra_info(packet)
        self.game_state.update(packet, extra_info=extra)
        state = compat_to_state_dict(self.game_state)
        obs = build_obs(state, int(self.player_id), POS_NORM, VEL_NORM, ANG_VEL_NORM)
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(obs).unsqueeze(0))
        row = self.table[int(logits.argmax(-1))]

        c = ControllerState()
        c.throttle, c.steer = float(row[0]), float(row[1])
        c.pitch, c.yaw, c.roll = float(row[2]), float(row[3]), float(row[4])
        c.jump, c.boost, c.handbrake = bool(row[5]), bool(row[6]), bool(row[7])
        self.prev_control = c
        return c


if __name__ == "__main__":
    ConstructBot("construct/construct_v0").run()
```
Implementation caveats to verify against installed rlgym-compat (field names drifted between versions): `car.physics.forward` vs rotation-matrix accessor, `car.boost_amount` scale (0-1 vs 0-100), `player_id` vs agent key type. The parity test can't cover this seam — verify by printing one obs in a real match and sanity-checking ranges.

- [ ] **Step 3: Write configs + requirements + README**

`deploy/bot.toml`:
```toml
#:schema https://rlbot.org/schemas/agent.json
[settings]
name = "Construct"
loadout_file = "loadout.toml"
root_dir = ""
run_command = "..\\venv\\Scripts\\python bot.py"
run_command_linux = "../venv/bin/python bot.py"
agent_id = "construct/construct_v0"

[details]
description = "RLGym-trained PPO bot (Construct P0)"
developer = "steamo"
language = "Python"
tags = []
```

`deploy/loadout.toml` (minimal valid loadout — octane body, default everything else):
```toml
[blue_loadout]
team_color_id = 0
custom_color_id = 0
car_id = 23          # Octane
decal_id = 0
wheels_id = 0
boost_id = 0
antenna_id = 0
hat_id = 0

[orange_loadout]
team_color_id = 0
custom_color_id = 0
car_id = 23
decal_id = 0
wheels_id = 0
boost_id = 0
antenna_id = 0
hat_id = 0
```
(If the RLBot GUI rejects fields, regenerate with its loadout editor — the file is cosmetic only.)

`deploy/match.toml`:
```toml
[rlbot]
launcher = "steam"

[match]
game_mode = "Soccar"
game_map_upk = "Stadium_P"

[[cars]]
config_file = "bot.toml"
team = 0

[[cars]]
type = "psyonix"
skill = "allstar"
team = 1

[mutators]
match_length = "five_minutes"
```

`deploy/requirements.txt`:
```
rlbot>=2.0.0b52
torch
numpy
rlgym_compat@git+https://github.com/JPK314/rlgym-compat.git
```

`deploy/README.md` — Windows deploy steps:
```markdown
# Deploying Construct to Rocket League (Windows)

1. Install RLBot v5 launcher: https://rlbot.org/v5/
2. Install Python 3.11 (python.org, add to PATH)
3. In this folder (copied to Windows, e.g. from \\wsl$\...\Construct-RLBot\deploy):
   py -3.11 -m venv ..\venv
   ..\venv\Scripts\pip install -r requirements.txt
4. Copy a trained checkpoint here as checkpoint.pt
5. RLBot GUI -> Add -> Load Folder -> select this folder -> start a match
   (or: server headless route — run RLBotServer, then `python bot.py` with
   RLBOT_AGENT_ID=construct/construct_v0 set)
RLBot only works in local/offline matches; it launches the game with -rlbot.
```

- [ ] **Step 4: Write bot-logic unit test (no rlbot install needed in WSL)**

```python
# tests/python/test_bot_logic.py
import sys

import numpy as np

sys.path.insert(0, "deploy")
from actions import make_lookup_table


def test_action_row_to_controller_semantics():
    t = make_lookup_table()
    # row 0: full reverse + left, no boost/jump/handbrake
    row = t[0]
    assert row[0] == -1 and row[1] == -1
    assert not any(row[5:8])
    # every row: bounded controls
    assert np.all(np.abs(t[:, :5]) <= 1.0)
    assert set(np.unique(t[:, 5:8])) <= {0.0, 1.0}


def test_state_dict_obs_smoke():
    from obs import build_obs
    state = {
        "ball": {"pos": [0, 0, 93.15], "vel": [0, 0, 0], "ang_vel": [0, 0, 0]},
        "cars": [
            {"id": 1, "team": 0, "pos": [0, -4608, 17], "vel": [0, 0, 0], "ang_vel": [0, 0, 0],
             "forward": [0, 1, 0], "up": [0, 0, 1], "boost": 33.3,
             "is_on_ground": True, "has_flip": True, "is_demoed": False},
            {"id": 2, "team": 1, "pos": [0, 4608, 17], "vel": [0, 0, 0], "ang_vel": [0, 0, 0],
             "forward": [0, -1, 0], "up": [0, 0, 1], "boost": 33.3,
             "is_on_ground": True, "has_flip": True, "is_demoed": False},
        ],
    }
    o1 = build_obs(state, 1, 1 / 2300, 1 / 2300, 1 / 5.5)
    o2 = build_obs(state, 2, 1 / 2300, 1 / 2300, 1 / 5.5)
    assert o1.shape == (94,)
    np.testing.assert_allclose(o1, o2, atol=1e-6)  # mirrored symmetric state
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/python/test_bot_logic.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add -A && git commit -m "feat: RLBot v5 deploy package with standalone bot + configs"
```

---

### Task 14: P0 exit — train, evaluate, deploy

**Files:**
- Create: `scripts/eval_metrics.py`
- No new library code — this task runs the system.

**Interfaces:**
- Consumes: everything.
- Produces: a checkpoint demonstrating ball-chasing; P0 exit evidence.

- [ ] **Step 1: Write scripts/eval_metrics.py**

```python
"""Headless behavior eval: touches/min and mean dist-to-ball over N eval steps."""
import sys

import numpy as np
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
eng = Engine(num_arenas=16, blue=1, orange=1, schema_path="schema/v0.toml",
             reward_config_path="configs/reward_v0.toml", seed=1234)
net = PolicyValueNet(eng.obs_size, eng.action_count, tuple(ck["config"]["net"]["hidden"]))
net.load_state_dict(ck["model"])
net.eval()

obs = torch.as_tensor(eng.reset())
STEPS = 4500  # 5 min of game time
touches = 0
dist_sum = 0.0
for _ in range(STEPS):
    with torch.no_grad():
        acts = net(obs)[0].argmax(-1).numpy().astype(np.int64)
    nobs, rew, term, trunc, _ = eng.step(acts)
    touches += int((rew >= 0.5).sum())  # touch weight fires at >= 0.5
    # obs[28:31] is (ball - car) * pos_norm
    dist_sum += float(np.linalg.norm(nobs[:, 28:31], axis=1).mean() / (1 / 2300))
    obs = torch.as_tensor(nobs)

minutes = STEPS / 15 / 60 * eng.num_agents
print(f"touches/min/agent: {touches / minutes:.2f}")
print(f"mean dist to ball: {dist_sum / STEPS:.0f} uu")
```

- [ ] **Step 2: Baseline eval with random weights**

Run: `python -c "from construct.learn.config import TrainConfig; from construct.learn.train import Trainer; t=Trainer(TrainConfig.load('configs/train_v0.toml')); t.save_checkpoint('checkpoints/random.pt')" && python scripts/eval_metrics.py checkpoints/random.pt`
Expected: touches/min near 0, dist-to-ball ~3000+ uu. Record numbers.

- [ ] **Step 3: Train for real**

Run: `python -m construct.learn.train configs/train_v0.toml` (leave running; checkpoints land in `checkpoints/`)
Expected: `sps` printed per iter (target >15k with GPU inference); `ep_rew` climbing within ~20-50M steps. Stop (Ctrl-C) when eval improves decisively or after a few hours.

- [ ] **Step 4: Eval trained checkpoint**

Run: `python scripts/eval_metrics.py checkpoints/ck_<latest>.pt`
Exit gate: **touches/min ≥ 3× random baseline AND mean dist-to-ball < half of random baseline.**

- [ ] **Step 5: Watch it in RLViser**

Download rlviser binary to repo root, run `./rlviser`, then `python scripts/watch.py checkpoints/ck_<latest>.pt`.
Expected: car visibly pursues and hits the ball. (WSLg required; else run rlviser on Windows and adjust the UDP target IP in `viser.rs` — documented there.)

- [ ] **Step 6: Windows deploy (manual, user-assisted)**

Follow `deploy/README.md` on the Windows host with `checkpoint.pt` copied in. Start a 1v1 vs Psyonix Allstar via the RLBot GUI.
Expected: Construct drives at the ball in the real game. **This is the P0 exit criterion from the spec.**

- [ ] **Step 7: Record results + commit**

Append actual numbers (baseline vs trained eval, sps, subjective deploy notes) to `docs/superpowers/specs/2026-07-13-construct-rlbot-design.md` under a "P0 Results" heading.

```bash
git add -A && git commit -m "chore: P0 exit — trained ball-chaser, eval metrics, deploy verified"
```

---

## Self-Review Notes

- Spec coverage (P0 scope): engine ✓ (T2-8), PyO3 ✓ (T8), minimal PPO ✓ (T9-10), RLViser ✓ (T11), RLBot deploy ✓ (T13-14), schema versioning ✓ (T4), parity test ✓ (T12), `rlbot_delay` — **deferred note:** rocketsim-rs (unlike RLGym's `RocketSimEngine`) has no `rlbot_delay` flag; the 1-tick action delay emulation is a P1 item (add a 1-action queue in `EpisodeArena`), noted here so it isn't lost.
- Known P0 simplifications (deliberate): no WandB (stdout logging; P1), no self-play pool (opponent = own policy symmetric control; that IS naive self-play), single reward stage, MLP not transformer, no replay data. All arrive in P1/P2 per spec.
- API risk flags carried from research: `BallHitInfo` field names (T6 fallback documented), pyo3 `detach` vs `allow_threads` (T8 note), rlgym-compat car field names (T13 caveat) — each has an in-task verification step.
