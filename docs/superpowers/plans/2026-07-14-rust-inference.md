# Rust-Side Inference (collect in-engine) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Contract amendment (2026-07-14, during execution):** cross-thread-count exact
> determinism was dropped — candle's gemm rounds batch-size-dependently, flipping
> sampled actions, so no useful cross-config guarantee exists. Binding contract:
> exact determinism for fixed (seed, num_arenas, num_threads). Inference is one
> batched forward per worker per round.

**Goal:** Move rollout collection entirely into the Rust engine — per-worker candle CPU inference, sampling, and buffer accumulation — so sim threads never wait on Python, roughly doubling training throughput.

**Architecture:** Each worker thread owns a candle MLP copy (policy + value heads) rebuilt from raw weight arrays pushed via `Engine.set_weights()` each iteration. `Engine.collect(T)` runs T rounds per worker (forward → sample → step → accumulate) fully in parallel across workers and returns complete rollout buffers as numpy arrays. Python keeps GAE + PPO on GPU unchanged. `Engine.step/reset` stay for the viewer, evals, and parity tests.

**Tech Stack:** candle-core 0.11.0 + candle-nn 0.11.0 (CPU, default features), existing pyo3/numpy stack, PyTorch learner unchanged.

## Global Constraints

- Pin `candle-core = "0.11.0"` and `candle-nn = "0.11.0"` — the two crates are workspace-released and must be the identical version string.
- Linear weight layout is PyTorch's: `[out_features, in_features]`, `y = x@w.t() + b` (candle transposes internally — pass state_dict arrays through untransposed).
- Build tensors with `Tensor::from_vec` (zero-copy move on CPU), never `from_slice` (copies).
- `RAYON_NUM_THREADS=1` must be in effect before the first candle tensor op, or candle's internal gemm parallelism oversubscribes our 14 worker threads. `Engine::new` sets it programmatically iff unset.
- No categorical sampler exists in candle and its CPU RNG is unseedable: sampling is our own softmax→CDF walk with a per-worker PCG32. Determinism contract: same weights + same engine seed + same num_arenas → identical buffers regardless of num_threads.
- Parity tolerances: candle-vs-torch logits/values ≤ 1e-4 absolute (f32 op-order differences); logprob rust-vs-torch-recompute ≤ 1e-4.
- Net architecture fixed by checkpoint format: trunk `Linear(94,512)-ReLU-Linear(512,512)-ReLU`, heads `policy [90,512]`, `value [1,512]`; state_dict keys `trunk.{0,2}.{weight,bias}`, `policy_head.{weight,bias}`, `value_head.{weight,bias}`. Rust must parse trunk layers generically from the `trunk.N.weight` indices (hidden sizes come from array shapes, not hardcoded).
- Trainer contract preserved: `compute_gae`/`ppo_update` signatures untouched; checkpoint format untouched.
- Buffer layout from `collect`: T-major `(T, N, ...)` with agents ordered exactly as `Engine.reset()` orders them (worker-major, arena-major, blue-then-orange).
- Python 3.11 venv, `maturin develop --release` to rebuild, existing tests must stay green (`cargo test`, `pytest tests/python` minus the port-bound render test when a viewer is running).

## File Structure

```
engine/src/policy.rs       # MlpPolicy: candle net from raw layers, forward -> (logits, values)
engine/src/sampler.rs      # Pcg32 + categorical sample + logprob from logits
engine/src/engine.rs       # Cmd::SetWeights, Cmd::Collect, worker collect loop, gather
engine/src/lib.rs          # Engine.set_weights / Engine.collect / debug_policy_forward pyfunctions
engine/Cargo.toml          # candle deps
scripts/gen_policy_fixture.py   # writes tests fixture JSON from a torch net
engine/tests/policy_parity.rs   # golden-fixture parity test
tests/python/test_rust_collect.py  # set_weights/collect integration + determinism + logprob parity
python/construct/learn/train.py    # collect() rewired to engine.collect
scripts/bench_collect.py   # before/after throughput bench
```

---

### Task 1: Candle policy net + golden-fixture parity

**Files:**
- Modify: `engine/Cargo.toml` (add candle deps)
- Create: `engine/src/policy.rs`, `scripts/gen_policy_fixture.py`, `engine/tests/policy_parity.rs`, `engine/tests/fixtures/policy_fixture.json` (generated)
- Modify: `engine/src/lib.rs` (`pub mod policy;`)

**Interfaces:**
- Produces:
  ```rust
  pub struct LayerWeights { pub w: Vec<f32>, pub b: Vec<f32>, pub out_dim: usize, pub in_dim: usize }
  pub struct PolicyWeights { pub trunk: Vec<LayerWeights>, pub policy: LayerWeights, pub value: LayerWeights }
  pub struct MlpPolicy { /* private candle Linears */ }
  impl MlpPolicy {
      pub fn new(w: &PolicyWeights) -> Result<Self, String>;
      /// obs: flat batch*obs_dim f32. Returns (logits batch*actions flat, values batch).
      pub fn forward(&self, obs: &[f32], batch: usize, obs_dim: usize)
          -> Result<(Vec<f32>, Vec<f32>), String>;
  }
  pub fn ensure_single_thread_gemm(); // sets RAYON_NUM_THREADS=1 iff unset, before first op
  ```
- Consumes: nothing from earlier tasks.

- [ ] **Step 1: Add candle deps**

In `engine/Cargo.toml` `[dependencies]`:
```toml
candle-core = "0.11.0"
candle-nn = "0.11.0"
```

- [ ] **Step 2: Write the fixture generator**

```python
# scripts/gen_policy_fixture.py
"""Golden fixture for Rust<->PyTorch policy parity: random small net + inputs
+ expected outputs, all in one JSON."""
import json
from pathlib import Path

import torch

from construct.learn.model import PolicyValueNet

torch.manual_seed(7)
net = PolicyValueNet(obs_size=94, action_count=90, hidden=(32, 32)).eval()
obs = torch.randn(5, 94)
with torch.no_grad():
    logits, values = net(obs)

fx = {
    "obs_size": 94, "action_count": 90, "hidden": [32, 32], "batch": 5,
    "state_dict": {k: v.flatten().tolist() for k, v in net.state_dict().items()},
    "shapes": {k: list(v.shape) for k, v in net.state_dict().items()},
    "obs": obs.flatten().tolist(),
    "expected_logits": logits.flatten().tolist(),
    "expected_values": values.flatten().tolist(),
}
out = Path("engine/tests/fixtures/policy_fixture.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(fx))
print(f"wrote {out} ({out.stat().st_size} bytes)")
```

Run: `source .venv/bin/activate && python scripts/gen_policy_fixture.py`
Expected: `wrote engine/tests/fixtures/policy_fixture.json (...)`. Commit the JSON (it is a test fixture, deliberately in git).

- [ ] **Step 3: Write the failing parity test**

```rust
// engine/tests/policy_parity.rs
use construct_engine::policy::{LayerWeights, MlpPolicy, PolicyWeights};

fn fixture() -> serde_json::Value {
    let text = std::fs::read_to_string(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/policy_fixture.json"),
    )
    .expect("run scripts/gen_policy_fixture.py first");
    serde_json::from_str(&text).unwrap()
}

fn f32s(v: &serde_json::Value) -> Vec<f32> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap() as f32).collect()
}

fn layer(fx: &serde_json::Value, prefix: &str) -> LayerWeights {
    let sd = &fx["state_dict"];
    let shape = fx["shapes"][format!("{prefix}.weight")].as_array().unwrap();
    LayerWeights {
        w: f32s(&sd[format!("{prefix}.weight")]),
        b: f32s(&sd[format!("{prefix}.bias")]),
        out_dim: shape[0].as_u64().unwrap() as usize,
        in_dim: shape[1].as_u64().unwrap() as usize,
    }
}

#[test]
fn candle_forward_matches_pytorch_golden() {
    let fx = fixture();
    let weights = PolicyWeights {
        trunk: vec![layer(&fx, "trunk.0"), layer(&fx, "trunk.2")],
        policy: layer(&fx, "policy_head"),
        value: layer(&fx, "value_head"),
    };
    let net = MlpPolicy::new(&weights).unwrap();
    let obs = f32s(&fx["obs"]);
    let batch = fx["batch"].as_u64().unwrap() as usize;
    let (logits, values) = net.forward(&obs, batch, 94).unwrap();

    let exp_logits = f32s(&fx["expected_logits"]);
    let exp_values = f32s(&fx["expected_values"]);
    assert_eq!(logits.len(), exp_logits.len());
    let max_l = logits.iter().zip(&exp_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    let max_v = values.iter().zip(&exp_values).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
    assert!(max_l < 1e-4, "logits max diff {max_l}");
    assert!(max_v < 1e-4, "values max diff {max_v}");
}
```

- [ ] **Step 4: Run to verify RED**

Run: `cd engine && cargo test --test policy_parity`
Expected: compile FAIL (`policy` module missing). First build downloads/compiles candle — minutes.

- [ ] **Step 5: Implement policy.rs**

```rust
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use std::sync::Once;

static GEMM_GUARD: Once = Once::new();

/// candle's CPU matmul spins a process-wide rayon pool sized by
/// RAYON_NUM_THREADS (default: physical cores). With 14 worker threads each
/// doing their own small matmuls, that pool oversubscribes the box — force
/// single-threaded gemm unless the user overrode it. Must run before the
/// first tensor op (candle reads the var lazily into a OnceLock).
pub fn ensure_single_thread_gemm() {
    GEMM_GUARD.call_once(|| {
        if std::env::var("RAYON_NUM_THREADS").is_err() {
            std::env::set_var("RAYON_NUM_THREADS", "1");
        }
    });
}

pub struct LayerWeights {
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub out_dim: usize,
    pub in_dim: usize,
}

pub struct PolicyWeights {
    pub trunk: Vec<LayerWeights>,
    pub policy: LayerWeights,
    pub value: LayerWeights,
}

pub struct MlpPolicy {
    trunk: Vec<Linear>,
    policy_head: Linear,
    value_head: Linear,
}

fn linear(l: &LayerWeights, dev: &Device) -> Result<Linear, String> {
    if l.w.len() != l.out_dim * l.in_dim || l.b.len() != l.out_dim {
        return Err(format!(
            "layer shape mismatch: w={} b={} expected {}x{}",
            l.w.len(), l.b.len(), l.out_dim, l.in_dim
        ));
    }
    let w = Tensor::from_vec(l.w.clone(), (l.out_dim, l.in_dim), dev).map_err(|e| e.to_string())?;
    let b = Tensor::from_vec(l.b.clone(), l.out_dim, dev).map_err(|e| e.to_string())?;
    Ok(Linear::new(w, Some(b)))
}

impl MlpPolicy {
    pub fn new(weights: &PolicyWeights) -> Result<Self, String> {
        ensure_single_thread_gemm();
        let dev = Device::Cpu;
        Ok(Self {
            trunk: weights.trunk.iter().map(|l| linear(l, &dev)).collect::<Result<_, _>>()?,
            policy_head: linear(&weights.policy, &dev)?,
            value_head: linear(&weights.value, &dev)?,
        })
    }

    /// obs is a flat row-major batch*obs_dim buffer.
    /// Returns (logits flat batch*action_count, values batch).
    pub fn forward(
        &self,
        obs: &[f32],
        batch: usize,
        obs_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let dev = Device::Cpu;
        let mut x = Tensor::from_vec(obs.to_vec(), (batch, obs_dim), &dev)
            .map_err(|e| e.to_string())?;
        for l in &self.trunk {
            x = l.forward(&x).and_then(|t| t.relu()).map_err(|e| e.to_string())?;
        }
        let logits = self.policy_head.forward(&x).map_err(|e| e.to_string())?;
        let values = self.value_head.forward(&x).map_err(|e| e.to_string())?;
        let logits = logits.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?;
        let values = values.flatten_all().and_then(|t| t.to_vec1::<f32>()).map_err(|e| e.to_string())?;
        Ok((logits, values))
    }
}
```
Add `pub mod policy;` to `engine/src/lib.rs`. Note: `flatten_all` — verify it exists on Tensor in 0.11.0 (`cargo doc -p candle-core`); if named differently, use `reshape(batch * n)?` then `to_vec1`.

- [ ] **Step 6: Run to verify GREEN**

Run: `cd engine && cargo test --test policy_parity`
Expected: `candle_forward_matches_pytorch_golden ... ok`

- [ ] **Step 7: Full check + commit**

Run: `cd engine && cargo test` — all pass.
```bash
git add -A && git commit -m "feat: candle MLP policy with PyTorch golden-fixture parity"
```

---

### Task 2: Categorical sampler + logprob

**Files:**
- Create: `engine/src/sampler.rs`
- Modify: `engine/src/lib.rs` (`pub mod sampler;`)

**Interfaces:**
- Produces:
  ```rust
  pub struct Pcg32 { /* state */ }
  impl Pcg32 {
      pub fn new(seed: u64) -> Self;
      pub fn next_f32(&mut self) -> f32;  // uniform [0,1)
  }
  /// Numerically-stable softmax sample from one row of logits.
  /// Returns (action_index, logprob_of_that_action).
  pub fn sample_categorical(logits: &[f32], rng: &mut Pcg32) -> (usize, f32);
  ```
- Consumes: nothing.

- [ ] **Step 1: Write failing tests (in-module)**

```rust
// engine/src/sampler.rs (tests at bottom)
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_given_seed() {
        let logits = vec![0.1f32, 2.0, -1.0, 0.5];
        let (mut a, mut b) = (Pcg32::new(42), Pcg32::new(42));
        for _ in 0..100 {
            assert_eq!(sample_categorical(&logits, &mut a).0,
                       sample_categorical(&logits, &mut b).0);
        }
    }

    #[test]
    fn samples_follow_distribution() {
        // logits [0, ln(3)]: p = [0.25, 0.75]
        let logits = vec![0.0f32, (3.0f32).ln()];
        let mut rng = Pcg32::new(7);
        let n = 100_000;
        let ones = (0..n).filter(|_| sample_categorical(&logits, &mut rng).0 == 1).count();
        let frac = ones as f32 / n as f32;
        assert!((frac - 0.75).abs() < 0.01, "got {frac}");
    }

    #[test]
    fn logprob_matches_log_softmax() {
        let logits = vec![1.0f32, 2.0, 3.0];
        // log_softmax reference computed by hand: subtract max, exp, normalize
        let max = 3.0f32;
        let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
        let z: f32 = exps.iter().sum();
        let mut rng = Pcg32::new(1);
        for _ in 0..50 {
            let (a, lp) = sample_categorical(&logits, &mut rng);
            let expected = (exps[a] / z).ln();
            assert!((lp - expected).abs() < 1e-6, "a={a} lp={lp} expected={expected}");
        }
    }

    #[test]
    fn extreme_logits_do_not_nan() {
        let logits = vec![1000.0f32, -1000.0, 0.0];
        let mut rng = Pcg32::new(3);
        let (a, lp) = sample_categorical(&logits, &mut rng);
        assert_eq!(a, 0);
        assert!(lp.is_finite() && lp <= 0.0);
    }
}
```

- [ ] **Step 2: RED**

Run: `cd engine && cargo test sampler` — compile FAIL.

- [ ] **Step 3: Implement**

```rust
/// Minimal PCG32 (O'Neill) — deterministic, per-worker, no dependencies.
pub struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    pub fn new(seed: u64) -> Self {
        let mut s = Self { state: 0, inc: (seed << 1) | 1 };
        s.next_u32();
        s.state = s.state.wrapping_add(seed);
        s.next_u32();
        s
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    pub fn next_f32(&mut self) -> f32 {
        // 24 high bits -> [0,1)
        (self.next_u32() >> 8) as f32 * (1.0 / (1u32 << 24) as f32)
    }
}

/// Softmax-CDF sample; numerically stable (max-subtracted).
/// Returns (index, ln softmax(logits)[index]).
pub fn sample_categorical(logits: &[f32], rng: &mut Pcg32) -> (usize, f32) {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut z = 0f32;
    // two passes: normalizer, then CDF walk
    for &l in logits {
        z += (l - max).exp();
    }
    let target = rng.next_f32() * z;
    let mut acc = 0f32;
    let mut idx = logits.len() - 1; // fallback: last index (fp round-off)
    for (i, &l) in logits.iter().enumerate() {
        acc += (l - max).exp();
        if acc > target {
            idx = i;
            break;
        }
    }
    let logprob = (logits[idx] - max) - z.ln();
    (idx, logprob)
}
```
Add `pub mod sampler;` to lib.rs.

- [ ] **Step 4: GREEN + full check + commit**

Run: `cd engine && cargo test` — all pass.
```bash
git add -A && git commit -m "feat: deterministic PCG32 categorical sampler with logprobs"
```

---

### Task 3: set_weights through PyO3 + full-path logits parity

**Files:**
- Modify: `engine/src/engine.rs` (Cmd::SetWeights, worker policy storage), `engine/src/lib.rs` (Engine.set_weights, Engine.debug_policy_forward)
- Test: `tests/python/test_rust_collect.py` (first two tests)

**Interfaces:**
- Consumes: `policy::{MlpPolicy, PolicyWeights, LayerWeights}` (Task 1).
- Produces (Python):
  ```python
  eng.set_weights(state_dict_np)   # dict[str, np.float32 ndarray]; keys per Global Constraints
  eng.debug_policy_forward(obs)    # np f32 (B, 94) -> (logits (B, 90), values (B,)) — worker 0's net
  ```
  Rust: `Cmd::SetWeights(Arc<PolicyWeights>)` broadcast; each worker holds `Option<MlpPolicy>`; `parse_state_dict(HashMap<String, (Vec<f32>, Vec<usize>)>) -> Result<PolicyWeights, String>` in engine.rs.

- [ ] **Step 1: Write failing python test**

```python
# tests/python/test_rust_collect.py
import numpy as np
import pytest
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet


def mk(n=2, seed=0, threads=0):
    return Engine(num_arenas=n, blue=1, orange=1, schema_path="schema/v0.toml",
                  reward_config_path="configs/reward_v0.toml", seed=seed, num_threads=threads)


def state_dict_np(net):
    return {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def test_set_weights_accepts_real_state_dict():
    torch.manual_seed(0)
    net = PolicyValueNet(94, 90, (512, 512))
    eng = mk()
    eng.set_weights(state_dict_np(net))  # must not raise


def test_full_path_forward_parity():
    torch.manual_seed(1)
    net = PolicyValueNet(94, 90, (64, 64)).eval()
    eng = mk()
    eng.set_weights(state_dict_np(net))
    obs = np.random.default_rng(2).standard_normal((6, 94)).astype(np.float32)
    logits_r, values_r = eng.debug_policy_forward(obs)
    with torch.no_grad():
        logits_t, values_t = net(torch.from_numpy(obs))
    assert np.abs(logits_r - logits_t.numpy()).max() < 1e-4
    assert np.abs(values_r - values_t.numpy()).max() < 1e-4


def test_set_weights_rejects_garbage():
    eng = mk()
    with pytest.raises(Exception):
        eng.set_weights({"trunk.0.weight": np.zeros((3, 3), dtype=np.float32)})
```

- [ ] **Step 2: RED**

Run: `maturin develop --release && pytest tests/python/test_rust_collect.py -x`
Expected: AttributeError (set_weights missing).

- [ ] **Step 3: Implement Rust side**

engine.rs — extend `Cmd` and worker state:
```rust
// Cmd gains:
SetWeights(std::sync::Arc<crate::policy::PolicyWeights>),
// worker match arm:
Cmd::SetWeights(w) => {
    match crate::policy::MlpPolicy::new(&w) {
        Ok(p) => { policy = Some(p); let _ = otx.send(WorkerOut::ack()); }
        Err(e) => { let _ = otx.send(WorkerOut::err(e)); }
    }
}
```
(Workers get `let mut policy: Option<MlpPolicy> = None;` before their recv loop. Give `WorkerOut` a lightweight variant story: add `error: Option<String>` + `fn ack() -> Self` + `fn err(String) -> Self` with empty buffers, or introduce a response enum — implementer's choice, keep it consistent for Task 4.)

`parse_state_dict` in engine.rs:
```rust
use std::collections::HashMap;

pub fn parse_state_dict(
    arrays: HashMap<String, (Vec<f32>, Vec<usize>)>,
) -> Result<crate::policy::PolicyWeights, String> {
    fn take_layer(
        arrays: &HashMap<String, (Vec<f32>, Vec<usize>)>,
        prefix: &str,
    ) -> Result<crate::policy::LayerWeights, String> {
        let (w, wshape) = arrays.get(&format!("{prefix}.weight"))
            .ok_or_else(|| format!("missing {prefix}.weight"))?;
        let (b, bshape) = arrays.get(&format!("{prefix}.bias"))
            .ok_or_else(|| format!("missing {prefix}.bias"))?;
        if wshape.len() != 2 || bshape.len() != 1 || bshape[0] != wshape[0] {
            return Err(format!("bad shapes for {prefix}: {wshape:?} / {bshape:?}"));
        }
        Ok(crate::policy::LayerWeights {
            w: w.clone(), b: b.clone(), out_dim: wshape[0], in_dim: wshape[1],
        })
    }

    // trunk.N.weight for even N (Sequential interleaves ReLU at odd indices)
    let mut trunk_ids: Vec<usize> = arrays.keys()
        .filter_map(|k| k.strip_prefix("trunk.")?.strip_suffix(".weight")?.parse().ok())
        .collect();
    trunk_ids.sort_unstable();
    if trunk_ids.is_empty() {
        return Err("no trunk layers found".into());
    }
    let trunk = trunk_ids.iter()
        .map(|i| take_layer(&arrays, &format!("trunk.{i}")))
        .collect::<Result<Vec<_>, _>>()?;
    let policy = take_layer(&arrays, "policy_head")?;
    let value = take_layer(&arrays, "value_head")?;
    // chain consistency: obs 94 in, policy/value share trunk output width
    let mut prev = trunk[0].in_dim;
    for l in &trunk[1..] {
        // ok only if widths chain; first layer's in_dim is the obs size
        if l.in_dim != trunk[trunk_ids.len() - trunk_ids.len()].out_dim && l.in_dim != prev {
            return Err("trunk layer widths do not chain".into());
        }
        prev = l.out_dim;
    }
    let _ = prev;
    if policy.in_dim != trunk.last().unwrap().out_dim
        || value.in_dim != trunk.last().unwrap().out_dim
        || value.out_dim != 1
    {
        return Err("head shapes do not match trunk output".into());
    }
    Ok(crate::policy::PolicyWeights { trunk, policy, value })
}
```
(The chaining check above is awkward as written — implementer: simplify to a single loop asserting `trunk[i].in_dim == trunk[i-1].out_dim`; keep the head checks.)

lib.rs pymethods on Engine:
```rust
fn set_weights(&mut self, weights: HashMap<String, PyReadonlyArrayDyn<'_, f32>>) -> PyResult<()> {
    let arrays: HashMap<String, (Vec<f32>, Vec<usize>)> = weights.into_iter()
        .map(|(k, v)| {
            let shape = v.shape().to_vec();
            (k, (v.as_array().iter().copied().collect(), shape))
        })
        .collect();
    let pw = engine::parse_state_dict(arrays).map_err(PyValueError::new_err)?;
    self.inner.set_weights(pw).map_err(PyValueError::new_err)
}
```
`MultiEngine::set_weights` wraps in `Arc`, sends `Cmd::SetWeights(arc.clone())` to every worker, awaits acks, propagates first error. `debug_policy_forward` routes to worker 0 (new `Cmd::DebugForward(Vec<f32>, usize)` returning logits+values) OR — simpler — `MultiEngine` keeps its own `MlpPolicy` copy built at set_weights time and forwards on the calling thread; implementer's choice, document it.

- [ ] **Step 4: GREEN + full check + commit**

Run: `maturin develop --release && pytest tests/python/test_rust_collect.py -v && cd engine && cargo test`
```bash
git add -A && git commit -m "feat: set_weights broadcast + full-path candle/torch forward parity"
```

---

### Task 4: Worker collect loop

**Files:**
- Modify: `engine/src/engine.rs` (Cmd::Collect, worker loop, gather), `engine/src/lib.rs` (Engine.collect)
- Test: extend `tests/python/test_rust_collect.py`

**Interfaces:**
- Consumes: MlpPolicy (T1), sample_categorical/Pcg32 (T2), SetWeights plumbing (T3), EpisodeArena (existing: `write_obs`, `step(action_idx, rewards, flags, final_obs)`).
- Produces (Python — the trainer contract):
  ```python
  out = eng.collect(T)   # requires set_weights called first, else ValueError
  # dict of numpy arrays:
  #  obs (T,N,94) f32 — obs the action was computed FROM
  #  actions (T,N) i64, logprobs (T,N) f32, values (T,N) f32
  #  rewards (T,N) f32, terminated (T,N) bool, truncated (T,N) bool
  #  final_values (T,N) f32 — V(terminal obs) in done rows, 0 elsewhere
  #  last_values (N,) f32 — V(obs after step T-1) for GAE bootstrap
  ```
- Per-worker semantics, per round t: write_obs for all my arenas into a batch → forward once → sample per agent (per-worker Pcg32 seeded `engine_seed × 1e6 + global_base_arena`) → per-arena `step()` → on done rows, forward the final_obs rows to get final_values (batch the done rows in one forward). After T rounds, one more write_obs + forward for last_values.
- Determinism: identical to Engine-level contract — thread-count invariant (seeds keyed on global arena index, same as kickoff seeding).

- [ ] **Step 1: Write failing python tests**

```python
# append to tests/python/test_rust_collect.py

def _weights(hidden=(64, 64), seed=3):
    torch.manual_seed(seed)
    return state_dict_np(PolicyValueNet(94, 90, hidden))


def test_collect_shapes_dtypes():
    eng = mk(n=4)
    eng.set_weights(_weights())
    out = eng.collect(16)
    N = eng.num_agents
    assert out["obs"].shape == (16, N, 94) and out["obs"].dtype == np.float32
    assert out["actions"].shape == (16, N) and out["actions"].dtype == np.int64
    for k in ("logprobs", "values", "rewards", "final_values"):
        assert out[k].shape == (16, N) and out[k].dtype == np.float32, k
    for k in ("terminated", "truncated"):
        assert out[k].shape == (16, N) and out[k].dtype == np.bool_, k
    assert out["last_values"].shape == (N,)
    assert (out["actions"] >= 0).all() and (out["actions"] < 90).all()
    assert np.isfinite(out["logprobs"]).all() and (out["logprobs"] <= 0).all()


def test_collect_requires_weights():
    eng = mk()
    with pytest.raises(Exception):
        eng.collect(4)


def test_collect_deterministic_across_thread_counts():
    w = _weights()
    a, b = mk(n=4, seed=9, threads=1), mk(n=4, seed=9, threads=4)
    a.set_weights(w); b.set_weights(w)
    oa, ob = a.collect(32), b.collect(32)
    for k in oa:
        np.testing.assert_array_equal(oa[k], ob[k], err_msg=k)


def test_collect_logprob_matches_torch_recompute():
    torch.manual_seed(5)
    net = PolicyValueNet(94, 90, (64, 64)).eval()
    eng = mk(n=2, seed=4)
    eng.set_weights(state_dict_np(net))
    out = eng.collect(8)
    obs = torch.from_numpy(out["obs"].reshape(-1, 94))
    acts = torch.from_numpy(out["actions"].reshape(-1))
    with torch.no_grad():
        lp, _, vals = net.evaluate(obs, acts)
    assert np.abs(lp.numpy() - out["logprobs"].reshape(-1)).max() < 1e-4
    assert np.abs(vals.numpy() - out["values"].reshape(-1)).max() < 1e-4
```

- [ ] **Step 2: RED**

Run: `pytest tests/python/test_rust_collect.py -x -k collect`
Expected: AttributeError (collect missing).

- [ ] **Step 3: Implement worker collect**

engine.rs — `Cmd::Collect { steps: usize }`; worker arm outline with real code:
```rust
Cmd::Collect { steps } => {
    let Some(pol) = policy.as_ref() else {
        let _ = otx.send(WorkerOut::err("collect before set_weights".into()));
        continue;
    };
    let agents: usize = arenas.iter().map(|a| a.num_agents()).sum();
    let d = OBS_SIZE;
    let mut out = CollectOut::zeros(steps, agents, d); // plain struct of Vecs
    let mut obs_buf = vec![0f32; agents * d];
    let mut rew_buf = vec![0f32; agents];
    let mut flag_buf = vec![StepFlags::default(); agents];
    let mut fin_buf = vec![0f32; agents * d];

    for t in 0..steps {
        // 1. obs for all my arenas
        let mut off = 0;
        for ar in arenas.iter_mut() {
            let n = ar.num_agents() * d;
            ar.write_obs(&mut obs_buf[off..off + n]);
            off += n;
        }
        out.obs[t * agents * d..(t + 1) * agents * d].copy_from_slice(&obs_buf);
        // 2. one forward for the whole worker batch
        let (logits, values) = match pol.forward(&obs_buf, agents, d) {
            Ok(x) => x,
            Err(e) => { let _ = otx.send(WorkerOut::err(e)); continue; }
        };
        out.values[t * agents..(t + 1) * agents].copy_from_slice(&values);
        // 3. sample per agent
        let mut acts = vec![0i64; agents];
        for a in 0..agents {
            let row = &logits[a * ACTIONS..(a + 1) * ACTIONS];
            let (idx, lp) = crate::sampler::sample_categorical(row, &mut rngs[a_to_arena(a)]);
            acts[a] = idx as i64;
            out.actions[t * agents + a] = idx as i64;
            out.logprobs[t * agents + a] = lp;
        }
        // 4. step arenas, collect rewards/flags/final_obs
        let mut aoff = 0;
        for ar in arenas.iter_mut() {
            let n = ar.num_agents();
            ar.step(&acts[aoff..aoff + n], &mut rew_buf[..n], &mut flag_buf[..n],
                    &mut fin_buf[aoff * d..(aoff + n) * d]);
            for i in 0..n {
                out.rewards[t * agents + aoff + i] = rew_buf[i];
                out.terminated[t * agents + aoff + i] = flag_buf[i].terminated;
                out.truncated[t * agents + aoff + i] = flag_buf[i].truncated;
            }
            aoff += n;
        }
        // 5. final values for done rows (single forward over the done subset)
        let done: Vec<usize> = (0..agents)
            .filter(|&a| out.terminated[t * agents + a] || out.truncated[t * agents + a])
            .collect();
        if !done.is_empty() {
            let mut fobs = vec![0f32; done.len() * d];
            for (j, &a) in done.iter().enumerate() {
                fobs[j * d..(j + 1) * d].copy_from_slice(&fin_buf[a * d..(a + 1) * d]);
            }
            if let Ok((_, fv)) = pol.forward(&fobs, done.len(), d) {
                for (j, &a) in done.iter().enumerate() {
                    out.final_values[t * agents + a] = fv[j];
                }
            }
        }
    }
    // 6. bootstrap values of the post-rollout obs
    let mut off = 0;
    for ar in arenas.iter_mut() {
        let n = ar.num_agents() * d;
        ar.write_obs(&mut obs_buf[off..off + n]);
        off += n;
    }
    if let Ok((_, lv)) = pol.forward(&obs_buf, agents, d) {
        out.last_values.copy_from_slice(&lv);
    }
    let _ = otx.send(WorkerOut::collect(out));
}
```
Notes for the implementer:
- `rngs`: one `Pcg32` per arena, seeded `(engine_seed as u64) * 1_000_000 + global_arena_index as u64` at worker construction; `a_to_arena(a)` maps agent index → its arena's rng (precompute a lookup table). Sampling consumes RNG in agent order within the round → thread-count invariant because agent order is global-arena-major.
- **Wait — RNG-per-arena with agents interleaved is order-sensitive.** Simplest correct scheme: one `Pcg32` per ARENA, and its agents sample in blue-then-orange order (matches iteration). Consumption order within an arena is fixed regardless of thread layout, and arenas never share an rng — invariant holds. Use exactly this.
- `MultiEngine::collect(steps)` fans `Cmd::Collect` to all workers, gathers per-worker `CollectOut`s in worker order, and interleaves into global `(T, N, …)` buffers: for each t, worker w's agent block occupies columns `[w_base .. w_base + w_agents)`. Copy loops are straightforward; keep them in engine.rs.
- lib.rs `Engine.collect` wraps into a `PyDict` of numpy arrays via `into_pyarray` with `py.detach` around the blocking gather; error from any worker → `PyValueError`.

- [ ] **Step 4: GREEN**

Run: `maturin develop --release && pytest tests/python/test_rust_collect.py -v`
Expected: all collect tests pass (determinism test is the hard one — debug seeding first if it fails).

- [ ] **Step 5: Full check + commit**

Run: `cd engine && cargo test && cd .. && pytest tests/python -v --deselect tests/python/test_render_session.py::test_render_session_smoke`
```bash
git add -A && git commit -m "feat: in-engine rollout collection with per-worker candle inference"
```

---

### Task 5: Trainer rewire

**Files:**
- Modify: `python/construct/learn/train.py` (collect() body)
- Test: existing `tests/python/test_train.py` must pass unchanged (contract preserved); extend with one new test.

**Interfaces:**
- Consumes: `eng.set_weights` / `eng.collect` (T3/T4), `compute_gae`, `ppo_update` (unchanged).
- Produces: same Trainer public API; checkpoint format untouched.

- [ ] **Step 1: Write the new failing test**

```python
# append to tests/python/test_train.py

def test_collect_uses_rust_path(tmp_path):
    t = Trainer(small_cfg(tmp_path))
    batch = t.collect(8)
    n = t.engine.num_agents
    assert batch["obs"].shape == (8 * n, t.engine.obs_size)
    assert batch["logprobs"].shape == (8 * n,)
    # rust-collected logprobs must be consistent with the current net
    import torch
    with torch.no_grad():
        lp, _, _ = t.net.evaluate(batch["obs"], batch["actions"])
    assert (lp - batch["logprobs"]).abs().max().item() < 1e-4
```

- [ ] **Step 2: RED**

Run: `pytest tests/python/test_train.py::test_collect_uses_rust_path -x`
Expected: FAIL (old collect returns per-step python loop results but keys/shapes actually match — the logprob-consistency assert is what fails ONLY if weights weren't synced; more robustly it fails because old collect exists... if old path passes all asserts, the test is not RED). **Implementer: the honest RED here is to write the test AFTER deleting the old body — do Step 3's deletion first, run the suite RED, then implement.** Adjust order accordingly and note it in the report.

- [ ] **Step 3: Rewrite Trainer.collect**

```python
    def collect(self, T: int) -> dict:
        N, D = self.engine.num_agents, self.engine.obs_size
        self.engine.set_weights(
            {k: v.detach().cpu().numpy().astype(np.float32)
             for k, v in self.net.state_dict().items()}
        )
        out = self.engine.collect(T)

        values_ext = np.concatenate(
            [out["values"], out["last_values"][None, :]], axis=0
        )
        adv, ret = compute_gae(
            out["rewards"], values_ext, out["final_values"],
            out["terminated"], out["truncated"],
            self.cfg.ppo["gamma"], self.cfg.ppo["lam"],
        )
        dev = self.device
        flat_obs = torch.as_tensor(out["obs"].reshape(T * N, D), device=dev)
        done_frac = (out["terminated"] | out["truncated"]).sum()
        return {
            "obs": flat_obs,
            "actions": torch.as_tensor(out["actions"].reshape(-1), device=dev),
            "logprobs": torch.as_tensor(out["logprobs"].reshape(-1), device=dev),
            "advantages": torch.as_tensor(adv, device=dev).reshape(-1),
            "returns": torch.as_tensor(ret, device=dev).reshape(-1),
            "values": torch.as_tensor(out["values"].reshape(-1), device=dev),
            "ep_reward_mean": float(out["rewards"].sum() / max(1, done_frac)),
        }
```
Remove the old per-step loop and the now-unused `self.obs` init/usage (keep `engine.reset()` call in `__init__` — it is harmless and keeps the engine warm; document or drop it, implementer's call, but be consistent).

- [ ] **Step 4: GREEN — full trainer suite + smoke**

Run: `pytest tests/python/test_train.py -v && python scripts/smoke_test.py`
Expected: all pass; `SMOKE OK: 2048 steps`.

- [ ] **Step 5: Full check + commit**

Run: `cd engine && cargo test && cd .. && pytest tests/python -v --deselect tests/python/test_render_session.py::test_render_session_smoke`
```bash
git add -A && git commit -m "feat: trainer collects via in-engine candle inference"
```

---

### Task 6: Bench, merge-readiness, redeploy both boxes

**Files:**
- Create: `scripts/bench_collect.py`
- No other library code.

**Interfaces:** consumes everything.

- [ ] **Step 1: Write bench script**

```python
# scripts/bench_collect.py
"""Collection throughput: rust in-engine path, by arena count."""
import time

import numpy as np
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet

torch.manual_seed(0)
net = PolicyValueNet(94, 90, (512, 512))
sd = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}

for arenas in (64, 96, 192, 256):
    eng = Engine(num_arenas=arenas, blue=1, orange=1, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=0)
    eng.set_weights(sd)
    eng.collect(16)  # warmup
    t0 = time.perf_counter()
    T = 128
    eng.collect(T)
    dt = time.perf_counter() - t0
    print(f"arenas={arenas:4d} agents={eng.num_agents:4d}: {T * arenas / dt:>10,.0f} env-steps/s")
```

- [ ] **Step 2: Run it, record numbers**

Run: `python scripts/bench_collect.py` (stop the local run B first if CPU-contended, restart after).
Expected: prints table; target ≥ 2x the old ~45k sps at 96 arenas. Put the numbers in the commit message.

- [ ] **Step 3: Training sanity — short real run**

Run: `python -c "from construct.learn.config import TrainConfig; from construct.learn.train import Trainer; cfg = TrainConfig.load('configs/train_v0.toml'); cfg.env.update(num_arenas=96); t = Trainer(cfg); t.run(max_iterations=20)"`
Expected: sps printed >= 2x previous local numbers; ep_rew sane (nonzero, no NaN).

- [ ] **Step 4: Commit + hand back to controller**

```bash
git add -A && git commit -m "feat: collection bench (numbers: <paste>)"
```
Controller (not the implementer) then: final review, merge, wheel rebuild, remote reship, restart both training runs from their latest checkpoints on the new path.

---

## Self-Review Notes

- Spec coverage: candle policy ✓(T1), sampler ✓(T2), weight push ✓(T3), in-engine collect + buffers incl. final/last values ✓(T4), trainer rewire with unchanged GAE/PPO ✓(T5), bench + rollout ✓(T6). RAYON guard ✓(T1). Determinism across thread counts ✓(T4 test).
- Type consistency: `LayerWeights/PolicyWeights/MlpPolicy` names match across T1/T3; collect dict keys match T4↔T5; `values_ext (T+1,N)` matches compute_gae's contract.
- Known rough edges deliberately delegated with instructions: WorkerOut ack/err shape (T3), trunk chaining check simplification (T3), RED-ordering caveat (T5 step 2).
- API risk: `Tensor::flatten_all` (T1 step 5) — verification instruction included; everything else is source-verified against candle 0.11.0.
