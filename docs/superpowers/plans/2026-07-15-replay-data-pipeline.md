# Replay Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Acquire high-rank human Rocket League replays and parse them into normalized 120 Hz physics-state frame shards (+ ReplayMutator reset-state pools) that later IDM/BC tasks consume.

**Architecture:** A new standalone Rust crate `replay/` (tied to the existing `engine/` crate by a new cargo workspace root) parses `.replay` files with **boxcars 0.11.5** + **subtr-actor 1.2.0**, reconstructs 120 Hz-consistent arena states by stepping **rocketsim_rs 0.37.0** between the ~30 Hz replay frames (porting VirxEC/replay-to-rocketsim's tick-stepping), estimates the missing pitch/yaw/roll analytically, and writes frame shards + reset-state pools via a `replay-parse` CLI binary (rayon-parallel). A thin Python package `python/construct/data/` orchestrates acquisition (HuggingFace bulk dataset + ballchasing.com API for SSL finetune data), batch parsing, and shard indexing.

**Tech Stack:** Rust (boxcars 0.11.5, subtr-actor 1.2.0, rocketsim_rs =0.37.0, ndarray, serde, rayon, clap), Python (huggingface_hub, httpx, numpy), Parquet/npz shard format.

## Global Constraints

- **Replays are NEVER committed** (like meshes): raw `.replay` files and parsed shards live under gitignored `data/`. Only one tiny fixture replay is committed, under `replay/tests/fixtures/`.
- **rocketsim_rs pinned `=0.37.0`** (matches engine; mesh assets fetched via existing `scripts/fetch_meshes.sh`, never bundled).
- **Parser output is RAW physics (uu, uu/s, rad/s, quaternions), NOT normalized obs** — obs normalization happens later at obs-build time. Keeps the pipeline independent of obs v1.
- **Every shard records a `schema_version` integer** so downstream loaders fail loud on format drift.
- **Reproducibility:** fixed seeds for any sampling; parse output for a given replay + parser version is deterministic.
- **Local box only** (801 GB free); remote 192.168.86.117 is the sole v2 training run and must not be touched.
- Python deps go through `uv pip` (no `pip` binary in the venv).

---

### Task 1: Cargo workspace + `replay` crate skeleton + header parse

**Files:**
- Create: `Cargo.toml` (workspace root listing `engine` and `replay`)
- Create: `replay/Cargo.toml`, `replay/src/lib.rs`, `replay/src/meta.rs`
- Create: `replay/tests/fixtures/sample.replay` (one small public replay, fetched — see step 1)
- Test: `replay/tests/meta_test.rs`

**Interfaces:**
- Produces: `pub struct ReplayMeta { pub playlist: String, pub team_size: u8, pub duration_secs: u32, pub net_version: i32, pub num_frames: usize }` and `pub fn parse_meta(bytes: &[u8]) -> Result<ReplayMeta, String>`.

- [ ] **Step 1: Fetch a fixture replay** (no API key needed — pull one file from the public HF dataset):
```bash
mkdir -p replay/tests/fixtures
uv run python -c "from huggingface_hub import hf_hub_download; import glob, shutil; \
  p=hf_hub_download('chrisrca/rocket-league-replays', filename=None, repo_type='dataset', allow_patterns='**/*.replay')" 2>/dev/null || true
# Fallback: any single ranked-duels .replay (~200 KB) copied to replay/tests/fixtures/sample.replay
```
If the dataset layout blocks a single-file pull, download one small `.replay` manually and place it at `replay/tests/fixtures/sample.replay`. Confirm it is < 1 MB.

- [ ] **Step 2: Write the failing test** in `replay/tests/meta_test.rs`:
```rust
use construct_replay::meta::parse_meta;

#[test]
fn parses_header_of_fixture() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let m = parse_meta(&bytes).unwrap();
    assert!(m.num_frames > 100, "expected real network frames, got {}", m.num_frames);
    assert!(m.team_size >= 1 && m.team_size <= 3);
    assert!(m.duration_secs > 0);
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p construct-replay parses_header_of_fixture`
Expected: FAIL — crate/function does not exist.

- [ ] **Step 4: Create the workspace root `Cargo.toml`:**
```toml
[workspace]
members = ["engine", "replay"]
resolver = "2"
```

- [ ] **Step 5: Create `replay/Cargo.toml`:**
```toml
[package]
name = "construct-replay"
version = "0.1.0"
edition = "2021"

[lib]
name = "construct_replay"

[[bin]]
name = "replay-parse"
path = "src/bin/replay_parse.rs"

[dependencies]
boxcars = "0.11.5"
subtr-actor = "1.2.0"
rocketsim_rs = "=0.37.0"
ndarray = "0.16"
ndarray-npy = "0.9"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
rayon = "1"
clap = { version = "4", features = ["derive"] }
```

- [ ] **Step 6: Implement `replay/src/meta.rs`:**
```rust
use boxcars::{HeaderProp, ParserBuilder};

#[derive(Debug, Clone)]
pub struct ReplayMeta {
    pub playlist: String,
    pub team_size: u8,
    pub duration_secs: u32,
    pub net_version: i32,
    pub num_frames: usize,
}

fn prop_i32(props: &[(String, HeaderProp)], key: &str) -> Option<i32> {
    props.iter().find(|(k, _)| k == key).and_then(|(_, v)| match v {
        HeaderProp::Int(i) => Some(*i),
        _ => None,
    })
}

pub fn parse_meta(bytes: &[u8]) -> Result<ReplayMeta, String> {
    let replay = ParserBuilder::new(bytes)
        .must_parse_network_data()
        .parse()
        .map_err(|e| format!("parse: {e}"))?;
    let props = &replay.properties;
    let team_size = prop_i32(props, "TeamSize").unwrap_or(0) as u8;
    let record_fps = prop_i32(props, "RecordFPS").unwrap_or(30).max(1);
    let frames = replay
        .network_frames
        .as_ref()
        .map(|nf| nf.frames.len())
        .unwrap_or(0);
    Ok(ReplayMeta {
        playlist: prop_i32(props, "MatchType").map(|_| "ranked".into()).unwrap_or_else(|| "unknown".into()),
        team_size,
        duration_secs: (frames as u32) / record_fps as u32,
        net_version: replay.net_version.unwrap_or(0),
        num_frames: frames,
    })
}
```

- [ ] **Step 7: Create `replay/src/lib.rs`:**
```rust
pub mod meta;
```
Create a stub `replay/src/bin/replay_parse.rs` with an empty `fn main() {}` so the crate builds.

- [ ] **Step 8: Run test to verify it passes**

Run: `cargo test -p construct-replay parses_header_of_fixture`
Expected: PASS.

- [ ] **Step 9: Gitignore + commit**

Add to `.gitignore`: `/data/` and `replay/target/`. Confirm `replay/tests/fixtures/sample.replay` is force-added (it is under the ignored-nothing path; verify `git status` shows it).
```bash
git add Cargo.toml replay/Cargo.toml replay/src replay/tests .gitignore
git add -f replay/tests/fixtures/sample.replay
git commit -m "feat: replay crate skeleton with boxcars header parse"
```

---

### Task 2: Frame extraction via subtr-actor (ball + cars + partial inputs + events)

**Files:**
- Create: `replay/src/frames.rs`
- Modify: `replay/src/lib.rs` (add `pub mod frames;`)
- Test: `replay/tests/frames_test.rs`

**Interfaces:**
- Consumes: nothing from Task 1 beyond the fixture.
- Produces:
```rust
pub struct RigidFrame { pub pos: [f32; 3], pub vel: [f32; 3], pub ang_vel: [f32; 3], pub quat: [f32; 4] }
pub struct CarFrame {
    pub rb: RigidFrame, pub boost: f32, pub team: u8,
    pub throttle: f32, pub steer: f32, pub handbrake: bool,
    pub jump_active: bool, pub dodge_active: bool, pub on_ground: bool, pub demoed: bool,
}
pub struct ReplayFrames { pub fps: u32, pub ball: Vec<RigidFrame>, pub cars: Vec<Vec<CarFrame>>, pub player_teams: Vec<u8> }
pub fn extract_frames(bytes: &[u8], target_fps: u32) -> Result<ReplayFrames, String>;
```
(`cars[t]` is the per-player vector at frame `t`; `player_teams` is stable per player index.)

- [ ] **Step 1: Write the failing test** in `replay/tests/frames_test.rs`:
```rust
use construct_replay::frames::extract_frames;

#[test]
fn extracts_resampled_frames() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let f = extract_frames(&bytes, 30).unwrap();
    assert_eq!(f.fps, 30);
    assert!(f.ball.len() > 100);
    assert_eq!(f.cars.len(), f.ball.len(), "one car-set per ball frame");
    assert!(!f.player_teams.is_empty());
    // ball must move (not all-zero) and stay finite
    let moved = f.ball.iter().any(|r| r.pos[0].abs() + r.pos[1].abs() > 1.0);
    assert!(moved && f.ball.iter().all(|r| r.pos.iter().all(|x| x.is_finite())));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p construct-replay extracts_resampled_frames`
Expected: FAIL — `frames` module missing.

- [ ] **Step 3: Implement `replay/src/frames.rs`** using subtr-actor's `ReplayProcessor` + `ReplayDataCollector` with `FrameRateDecorator::new_from_fps`. Map subtr-actor's per-player `PlayerData` (rigid_body, boost_amount, boost_active, powerslide_active, jump/dodge_active, team, input{throttle,steer}) into `CarFrame`; ball `rigid_body` into `RigidFrame`. Boost normalized to 0..1 (subtr-actor gives 0..255 → divide 255). Missing players in a frame (pre-spawn/demoed) → carry the last known frame and set `demoed`/absent flag. Return `ReplayFrames`.
```rust
use subtr_actor::*;

pub struct RigidFrame { pub pos: [f32; 3], pub vel: [f32; 3], pub ang_vel: [f32; 3], pub quat: [f32; 4] }
// ... CarFrame, ReplayFrames as in Interfaces ...

pub fn extract_frames(bytes: &[u8], target_fps: u32) -> Result<ReplayFrames, String> {
    let replay = boxcars::ParserBuilder::new(bytes)
        .must_parse_network_data().parse().map_err(|e| e.to_string())?;
    let mut collector = NDArrayCollector::<f32>::from_strings(
        &["RigidBody", "PlayerBoost", "PlayerAnyJump"], &[]).map_err(|e| e.to_string())?;
    let mut processor = ReplayProcessor::new(&replay).map_err(|e| e.to_string())?;
    let mut decorated = FrameRateDecorator::new_from_fps(target_fps as f32, &mut collector);
    processor.process(&mut decorated).map_err(|e| e.to_string())?;
    // Convert the collected ndarray + processor player metadata into ReplayFrames.
    // (Exact column mapping follows subtr-actor's feature registry order; assert
    //  column count matches the registry at runtime and error loudly otherwise.)
    unimplemented!("map collector output -> ReplayFrames")
}
```
NOTE to implementer: subtr-actor exposes two paths — the typed `ReplayDataCollector` (`FrameData { ball_data, players }`) is easier to map field-by-field than `NDArrayCollector`; prefer it. Use whichever compiles against 1.2.0's actual API; pin the exact call by reading `subtr-actor 1.2.0` docs.rs. The `unimplemented!` above is a scaffold — replace with the real mapping; the test is the contract.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p construct-replay extracts_resampled_frames`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add replay/src/frames.rs replay/src/lib.rs replay/tests/frames_test.rs
git commit -m "feat: subtr-actor frame extraction to typed ReplayFrames"
```

---

### Task 3: Analytic pitch/yaw/roll estimation (fill missing action channels)

**Files:**
- Create: `replay/src/pyr.rs`
- Modify: `replay/src/lib.rs` (add `pub mod pyr;`)
- Test: `replay/tests/pyr_test.rs`

**Interfaces:**
- Consumes: `RigidFrame` from Task 2.
- Produces: `pub fn estimate_pyr(prev: &RigidFrame, cur: &RigidFrame, dt: f32, on_ground: bool) -> [f32; 3]` returning `[pitch, yaw, roll]` each in `[-1, 1]`.

- [ ] **Step 1: Write the failing test** in `replay/tests/pyr_test.rs`:
```rust
use construct_replay::pyr::estimate_pyr;
use construct_replay::frames::RigidFrame;

#[test]
fn zero_angvel_change_gives_zero_torque_inputs() {
    let r = RigidFrame { pos: [0.0; 3], vel: [0.0; 3], ang_vel: [0.0; 3], quat: [0.0, 0.0, 0.0, 1.0] };
    let out = estimate_pyr(&r, &r, 1.0 / 120.0, false);
    assert!(out.iter().all(|x| x.abs() < 1e-3), "no rotation -> ~zero inputs, got {out:?}");
    assert!(out.iter().all(|x| (-1.0..=1.0).contains(x)));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p construct-replay zero_angvel_change_gives_zero_torque_inputs`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `replay/src/pyr.rs`** porting rlgym-tools' `inverse_aerial_controls` (torque constants `T_r=-36.07956616966136, T_p=-12.146176938276769, T_y=8.91962804287785`, drag `D_r=-4.47, D_p=-2.798194258050845, D_y=-1.886491900437232`). Convert `ang_vel` from world to car-local using `quat`, apply the inverse-dynamics formula per axis over `dt`, clamp to `[-1, 1]`. When `on_ground`, return `[0,0,0]` (aerial controls do nothing grounded). Reference: rlgym-tools `replays/` and Rolv-Arild/replay-pretraining `inverse_aerial_controls.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p construct-replay zero_angvel_change_gives_zero_torque_inputs`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add replay/src/pyr.rs replay/src/lib.rs replay/tests/pyr_test.rs
git commit -m "feat: analytic pitch/yaw/roll inversion for replay actions"
```

---

### Task 4: 120 Hz reconstruction via RocketSim tick-stepping

**Files:**
- Create: `replay/src/reconstruct.rs`
- Modify: `replay/src/lib.rs` (add `pub mod reconstruct;`)
- Test: `replay/tests/reconstruct_test.rs`

**Interfaces:**
- Consumes: `ReplayFrames` (Task 2).
- Produces:
```rust
pub struct Tick { pub ball: RigidFrame, pub cars: Vec<CarFrame> }
pub struct Reconstructed { pub ticks: Vec<Tick>, pub player_teams: Vec<u8> }
pub fn reconstruct_120hz(frames: &ReplayFrames) -> Result<Reconstructed, String>;
```

- [ ] **Step 1: Write the failing test** in `replay/tests/reconstruct_test.rs`:
```rust
use construct_replay::{frames::extract_frames, reconstruct::reconstruct_120hz};

#[test]
fn reconstructs_dense_120hz_finite() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let f = extract_frames(&bytes, 30).unwrap();
    let r = reconstruct_120hz(&f).unwrap();
    // ~4 ticks per 30 Hz frame
    assert!(r.ticks.len() >= f.ball.len() * 3, "expected ~4x densification");
    assert!(r.ticks.iter().all(|t| t.ball.pos.iter().all(|x| x.is_finite())));
    assert!(r.ticks.iter().all(|t| t.ball.pos.iter().all(|x| x.abs() < 20_000.0)),
        "reconstructed states must be physically sane (matches engine state_is_sane bounds)");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p construct-replay reconstructs_dense_120hz_finite`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `replay/src/reconstruct.rs`** porting VirxEC/replay-to-rocketsim: create an `Arena::default_standard()` with the replay's car count, and for each pair of consecutive 30 Hz frames, `set_ball`/`set_car` to the earlier frame's rigid bodies + apply the frame's controls, then `arena.step(N)` (N ≈ `round(120 * dt)`, ~4) to advance to the next frame, emitting each intermediate 120 Hz `Tick`. Snap car/ball state back to the authoritative replay frame at each 30 Hz boundary (avoids drift). Fill each car's `[pitch,yaw,roll]` via `estimate_pyr` (Task 3). Reuse the engine's `set_ball`/`set_car` idioms (see `engine/src/curriculum.rs:102,134`). Drop any tick failing a sanity bound (mirror `engine::episode::state_is_sane` thresholds: pos/vel/ang_vel ≤ 12000/20000/100).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p construct-replay reconstructs_dense_120hz_finite`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add replay/src/reconstruct.rs replay/src/lib.rs replay/tests/reconstruct_test.rs
git commit -m "feat: 120Hz reconstruction via RocketSim tick-stepping"
```

---

### Task 5: Shard writer + `replay-parse` CLI (parallel batch)

**Files:**
- Create: `replay/src/shard.rs`, `replay/src/bin/replay_parse.rs` (replace stub)
- Modify: `replay/src/lib.rs` (add `pub mod shard;`)
- Test: `replay/tests/shard_test.rs`

**Interfaces:**
- Consumes: `Reconstructed` (Task 4), `ReplayMeta` (Task 1).
- Produces: `pub const SHARD_SCHEMA_VERSION: u32 = 1;` and `pub fn write_shard(out_dir: &Path, replay_id: &str, meta: &ReplayMeta, rec: &Reconstructed) -> Result<PathBuf, String>` writing one `.npz` per replay: arrays `ball` `[T,13]`, `cars` `[T,P,17]` (rb 13 + boost + throttle + steer + flags-bitpacked... document exact columns in a header comment), `player_teams` `[P]`, plus a sidecar `<id>.json` with meta + `schema_version` + column names. CLI: `replay-parse --input-dir DIR --output-dir DIR [--fps 30] [--jobs N] [--min-team-size 1]`.

- [ ] **Step 1: Write the failing test** in `replay/tests/shard_test.rs`:
```rust
use construct_replay::{frames::extract_frames, reconstruct::reconstruct_120hz,
    meta::parse_meta, shard::{write_shard, SHARD_SCHEMA_VERSION}};

#[test]
fn writes_loadable_shard_with_schema() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let meta = parse_meta(&bytes).unwrap();
    let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let p = write_shard(dir.path(), "sample", &meta, &rec).unwrap();
    assert!(p.exists());
    let sidecar: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir.path().join("sample.json")).unwrap()).unwrap();
    assert_eq!(sidecar["schema_version"], SHARD_SCHEMA_VERSION);
    assert!(sidecar["num_ticks"].as_u64().unwrap() > 100);
}
```
Add `tempfile = "3"` under `[dev-dependencies]` in `replay/Cargo.toml`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p construct-replay writes_loadable_shard_with_schema`
Expected: FAIL — `shard` module missing.

- [ ] **Step 3: Implement `replay/src/shard.rs`** (build `ndarray::Array2/Array3`, write `.npz` via `ndarray-npy`, write the JSON sidecar) and the `replay-parse` CLI in `replay/src/bin/replay_parse.rs` (clap args; `rayon::par_iter` over `*.replay` in `--input-dir`; skip replays below `--min-team-size` using `parse_meta`; write shards to `--output-dir`; per-replay `catch`-and-log so one bad replay never aborts the batch; print a final `parsed N / skipped M / failed K` summary).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p construct-replay writes_loadable_shard_with_schema`
Expected: PASS. Then smoke the CLI:
```bash
cargo build -p construct-replay --release
mkdir -p /tmp/rp_in /tmp/rp_out && cp replay/tests/fixtures/sample.replay /tmp/rp_in/
./target/release/replay-parse --input-dir /tmp/rp_in --output-dir /tmp/rp_out --jobs 1
ls /tmp/rp_out   # expect sample.npz + sample.json
```

- [ ] **Step 5: Commit**
```bash
git add replay/src/shard.rs replay/src/bin/replay_parse.rs replay/src/lib.rs replay/tests/shard_test.rs replay/Cargo.toml
git commit -m "feat: shard writer + parallel replay-parse CLI"
```

---

### Task 6: ReplayMutator reset-state pool (engine-loadable)

**Files:**
- Create: `replay/src/reset_pool.rs`
- Modify: `replay/src/lib.rs`, `replay/src/bin/replay_parse.rs` (add `--reset-pool-out PATH` + `--reset-samples-per-replay K`)
- Test: `replay/tests/reset_pool_test.rs`
- Reference (do not modify in this task): `engine/src/curriculum.rs` (how the engine sets arena state)

**Interfaces:**
- Consumes: `Reconstructed` (Task 4).
- Produces: `pub fn sample_reset_states(rec: &Reconstructed, k: usize, seed: u64) -> Vec<ResetState>` and a serialized pool format (`.jsonl`, one `ResetState` per line) where `ResetState { ball: RigidFrame, cars: Vec<(CarSpawn)> }` carries exactly the fields the engine needs to `set_ball`/`set_car`. Document the JSON schema in a header comment so a future engine task (curriculum replay-reset mix, spec §4 "replay-state resets 0.7") can deserialize it.

- [ ] **Step 1: Write the failing test** in `replay/tests/reset_pool_test.rs`:
```rust
use construct_replay::{frames::extract_frames, reconstruct::reconstruct_120hz, reset_pool::sample_reset_states};

#[test]
fn samples_k_finite_reset_states_deterministically() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap()).unwrap();
    let a = sample_reset_states(&rec, 8, 42);
    let b = sample_reset_states(&rec, 8, 42);
    assert_eq!(a.len(), 8);
    assert_eq!(a, b, "same seed -> identical sample");
    assert!(a.iter().all(|s| s.ball.pos.iter().all(|x| x.is_finite() && x.abs() < 12_000.0)));
}
```
(Derive `PartialEq` on `ResetState`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p construct-replay samples_k_finite_reset_states_deterministically`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement `replay/src/reset_pool.rs`** — sample `k` ticks uniformly (seeded PCG; skip the first ~2 s of kickoff and any near-terminal goal frames), snapshot ball+cars into `ResetState`, serialize to `.jsonl`. Wire `--reset-pool-out`/`--reset-samples-per-replay` into the CLI to append sampled states across the whole batch into one pool file.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p construct-replay samples_k_finite_reset_states_deterministically`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add replay/src/reset_pool.rs replay/src/lib.rs replay/src/bin/replay_parse.rs replay/tests/reset_pool_test.rs
git commit -m "feat: ReplayMutator reset-state pool sampler + CLI wiring"
```

---

### Task 7: Python acquisition — HF bulk + ballchasing API client

**Files:**
- Create: `python/construct/data/__init__.py`, `python/construct/data/acquire.py`, `python/construct/data/ballchasing.py`
- Test: `tests/python/test_acquire.py`

**Interfaces:**
- Produces:
  - `acquire.pull_hf_subset(dest: Path, allow_patterns: list[str], max_files: int|None) -> int` (returns #files) — wraps `huggingface_hub.snapshot_download` for `chrisrca/rocket-league-replays` (410 GB total; `allow_patterns` like `["grand-champion-3/duels/**"]` scopes it).
  - `ballchasing.Client(token).search(min_rank, playlist, count, after) -> (list[dict], next_cursor)` and `.download(replay_id, dest) -> Path`, rate-limited (token-bucket sized to the tier; default free = 1 download/s, 2 list/s) and resumable (skip existing files, honor `after` cursor). Header `Authorization: <token>` (raw, no `Bearer`). Endpoints: `GET /api/replays` (params `playlist`, `min-rank`, `max-rank`, `count≤200`, `sort-by`, `after`), `GET /api/replays/{id}/file`.

- [ ] **Step 1: Write the failing test** in `tests/python/test_acquire.py` (mock HTTP — no network, no key):
```python
import httpx
from construct.data.ballchasing import Client

def test_search_paginates_and_rate_limits(monkeypatch):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        assert request.headers["Authorization"] == "TESTTOKEN"  # raw, no Bearer
        return httpx.Response(200, json={"count": 1, "list": [{"id": "abc"}],
                                         "next": "https://ballchasing.com/api/replays?after=CUR"})
    client = Client("TESTTOKEN", transport=httpx.MockTransport(handler))
    rows, cursor = client.search(min_rank="grand-champion-3", playlist="ranked-duels", count=150)
    assert rows == [{"id": "abc"}]
    assert cursor == "CUR"
    assert "min-rank=grand-champion-3" in calls[0] and "playlist=ranked-duels" in calls[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/python/test_acquire.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** `ballchasing.py` (`httpx.Client` with injectable `transport` for tests, raw `Authorization` header, token-bucket sleep between calls, `search` returns `(list, after-cursor-parsed-from-next)`, `download` streams to file and skips if present) and `acquire.py` (`pull_hf_subset` wrapping `snapshot_download`). Add `huggingface_hub` and `httpx` via `uv pip install huggingface_hub httpx`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/python/test_acquire.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add python/construct/data tests/python/test_acquire.py
git commit -m "feat: HF + ballchasing acquisition clients"
```

---

### Task 8: Pipeline orchestration (download → parse → index, resumable)

**Files:**
- Create: `scripts/build_replay_dataset.py`, `python/construct/data/index.py`
- Test: `tests/python/test_index.py`

**Interfaces:**
- Consumes: `acquire` (Task 7), the `replay-parse` binary (Task 5).
- Produces: `index.build_index(shard_dir: Path) -> dict` scanning `*.json` sidecars into a single `manifest.json` (`{total_ticks, num_shards, by_team_size, schema_version}`); `scripts/build_replay_dataset.py` CLI: `--source {hf,ballchasing} --dest data/replays --shards data/shards [--max N]` that (1) acquires to `data/replays/`, (2) shells out to `target/release/replay-parse --input-dir data/replays --output-dir data/shards --reset-pool-out data/reset_pool.jsonl`, (3) builds the index; each stage skips already-done work (resumable).

- [ ] **Step 1: Write the failing test** in `tests/python/test_index.py`:
```python
import json
from pathlib import Path
from construct.data.index import build_index

def test_index_aggregates_sidecars(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({"schema_version": 1, "num_ticks": 500, "team_size": 1}))
    (tmp_path / "b.json").write_text(json.dumps({"schema_version": 1, "num_ticks": 300, "team_size": 2}))
    idx = build_index(tmp_path)
    assert idx["total_ticks"] == 800
    assert idx["num_shards"] == 2
    assert idx["by_team_size"] == {"1": 1, "2": 1}
    assert idx["schema_version"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/python/test_index.py -v`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement** `index.py` (`build_index` aggregates sidecars, asserts a single `schema_version`, writes `manifest.json`) and `scripts/build_replay_dataset.py` (argparse; orchestrates acquire → subprocess parse → index; `--source hf` needs no key, `--source ballchasing` reads `BALLCHASING_TOKEN` env).

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/python/test_index.py -v`
Expected: PASS.

- [ ] **Step 5: End-to-end smoke** (fixture-scale, no network):
```bash
mkdir -p data/replays data/shards && cp replay/tests/fixtures/sample.replay data/replays/
./target/release/replay-parse --input-dir data/replays --output-dir data/shards --reset-pool-out data/reset_pool.jsonl
uv run python -c "from construct.data.index import build_index; from pathlib import Path; print(build_index(Path('data/shards')))"
```
Expected: manifest with `num_shards >= 1`, `total_ticks > 100`.

- [ ] **Step 6: Commit**
```bash
git add scripts/build_replay_dataset.py python/construct/data/index.py tests/python/test_index.py
git commit -m "feat: replay dataset orchestration + index (resumable)"
```

---

## Post-plan notes

- **Data-source decision (deferred, non-blocking):** bulk BC data comes from the HF dataset (`chrisrca/rocket-league-replays`, GC1–3, no API key). SSL-grade finetune data (spec §5.4 "winners-side-only, SSL") needs a **ballchasing API key** (Steam-login → ballchasing.com/upload → generate token) — a prerequisite only for `--source ballchasing`, not for standing up the pipeline. Ask the user for the key when the SSL-finetune corpus is actually needed.
- **Feeds forward:** shards (physics + partial actions + events) are the input to the IDM (labels pitch/yaw/roll from the physics window) and BC; the reset pool feeds the engine curriculum's replay-reset mix. Those are separate plans (see backlog tasks "IDM + BC + KL-PPO", "entity transformer + obs v1").
- **Not committed:** raw replays + shards under `data/` (gitignored); only `replay/tests/fixtures/sample.replay` is committed.
