# Opponent Pool + TrueSkill League Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train against frozen past checkpoints (not just the current self) and measure real skill with a TrueSkill ladder built from head-to-head matches — including main-vs-run-B fixtures.

**Architecture:** The engine gains opponent-policy slots (frozen candle nets loaded via the existing weights machinery) and a per-arena assignment: self-play arenas train all agents; opponent arenas train only blue while a frozen policy drives orange, and `collect()` returns learner-only buffers. Matches between two frozen checkpoints reuse the same mechanism from Python (policy A as "learner", policy B as opponent, goals counted from the reward stream) — zero extra engine code for match mode. A new `construct/league/` package owns the registry (jsonl + TrueSkill ratings), opponent sampling, and a `league_tick` script that registers new checkpoints, plays rating matches, and updates the ladder.

**Tech Stack:** existing engine crate (no new Rust deps), `trueskill==0.4.5` (verified: `TrueSkill(mu, sigma, beta, tau, draw_probability)`, `create_rating()`, `rate_1vs1(a, b, drawn=, env=)`, `env.expose(r)`), jsonl registry.

## Global Constraints

- **DO NOT DEPLOY**: build/test/commit only; the controller asks the user before any trainer restart or wheel reship (standing instruction).
- Backward compatibility hard gate: `collect(T)` with no assignment (None) must be byte-identical to today's behavior (regression: same seed/weights → `assert_array_equal` old-shape buffers).
- Opponent semantics: assignment vector `arena_opponents: Vec<i32>` of length num_arenas; `-1` = self-play (all agents learn), `k >= 0` = orange team driven by opponent slot k (blue learns, orange rows excluded from ALL returned buffers).
- Learner-agent ordering: worker-major, arena-major, blue-then-orange-if-self-play — i.e., drop orange rows of opponent arenas, preserve relative order of everything else.
- RNG discipline: actions are sampled for EVERY agent every round (learner and opponent) from the same per-arena Pcg32 in the same agent order as today — otherwise rng streams desync and fixed-config determinism breaks. Only the RECORDING of buffers is filtered.
- Opponent forwards batched per (worker, slot): group each worker's opponent-driven agents by slot, one candle forward per slot per round (plus the learner forward). No per-agent forwards.
- Opponent slots: max 8 (`MAX_OPPONENT_SLOTS = 8`); `set_opponents` with more → ValueError; assignment referencing an unset slot → ValueError at collect start.
- TrueSkill env parameters (Seer protocol): `mu=25.0, sigma=25/3, beta=25/6, tau=25/300, draw_probability=0.02`; ladder value = `env.expose(rating)`.
- Match outcome mapping: goals_a > goals_b → A wins; equal → drawn=True.
- Registry file `league/registry.jsonl` (repo-relative, gitignored — add `league/` to .gitignore); one JSON object per line: `{"ck": path, "steps": int, "run": "main"|"b", "reward_config": str, "added_ts": int, "mu": float, "sigma": float, "games": int}`. Rewrites are whole-file atomic (write tmp + rename).
- Python: `trueskill` added to pyproject `[project] dependencies`; venv installs via `uv pip` (NO bare `pip` — uv venv has no pip binary).
- Suites stay green: `cargo test`, `pytest tests/python -q --deselect tests/python/test_render_session.py::test_render_session_smoke`.
- A training process runs from this checkout; builds are safe; never kill python processes; never touch `checkpoints*/` contents.

## File Structure

```
engine/src/engine.rs               # Cmd::SetOpponents, per-arena assignment in Collect, learner-row filtering
engine/src/lib.rs                  # Engine.set_opponents(list of state_dicts), collect(T, arena_opponents=None)
python/construct/league/__init__.py
python/construct/league/registry.py   # jsonl registry + TrueSkill ratings (single file: they change together)
python/construct/league/matches.py    # MatchRunner: two frozen checkpoints -> (goals_a, goals_b)
python/construct/league/sampling.py   # opponent selection for training
scripts/league_tick.py                # register new cks, play rating matches, update ladder, print table
python/construct/learn/{config,train}.py  # league config section + opponent refresh in Trainer
scripts/resume_train.py            # --league flag
tests/python/test_league_engine.py # engine opponent-arena tests
tests/python/test_league.py        # registry/ratings/sampling/match tests
```

---

### Task 1: Engine opponent arenas + learner-only buffers

**Files:**
- Modify: `engine/src/engine.rs`, `engine/src/lib.rs`
- Test: `tests/python/test_league_engine.py` (new), Rust in-module where noted

**Interfaces:**
- Consumes: `MlpPolicy`/`PolicyWeights`/`parse_state_dict` (rust-inference), `Cmd`/`WorkerOut` patterns, per-arena Pcg32 rngs, `EpisodeArena::{step, write_obs, num_agents}` and TC-2's `sizes: Vec<(usize, usize)>`.
- Produces (Python API — Tasks 3/5 depend on):
  ```python
  eng.set_opponents(list_of_state_dicts)      # ≤ 8; rebuilds all slots; [] clears
  out = eng.collect(T, arena_opponents=None)  # None → legacy exact; else list[int] len num_arenas,
                                              # -1 self-play, k ≥ 0 opponent slot; bad slot/len → ValueError
  # out shapes become (T, N_learner, ...); out["learner_agents"] == N_learner as int
  ```
  Rust: `Cmd::SetOpponents(Arc<Vec<PolicyWeights>>)`; `Cmd::Collect { steps, assignment: Arc<Vec<i32>> }` (legacy None becomes a materialized all-`-1` vector — ONE code path); worker holds `opponents: Vec<MlpPolicy>`.

- [ ] **Step 1: Write the failing python tests**

```python
# tests/python/test_league_engine.py
import numpy as np
import pytest
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet


def sd(seed):
    torch.manual_seed(seed)
    return {k: v.detach().numpy().astype(np.float32)
            for k, v in PolicyValueNet(94, 90, (64, 64)).state_dict().items()}


def mk(n=4, seed=0, threads=2):
    return Engine(num_arenas=n, blue=1, orange=1, schema_path="schema/v0.toml",
                  reward_config_path="configs/reward_v0.toml", seed=seed, num_threads=threads)


def test_no_assignment_is_byte_identical_to_legacy():
    w = sd(1)
    a, b = mk(seed=5), mk(seed=5)
    a.set_weights(w); b.set_weights(w)
    oa = a.collect(16)
    ob = b.collect(16, arena_opponents=None)
    for k in oa:
        np.testing.assert_array_equal(np.asarray(oa[k]), np.asarray(ob[k]), err_msg=k)


def test_opponent_arena_shrinks_buffers_and_orders_learners():
    eng = mk(n=4, seed=3)
    eng.set_weights(sd(1))
    eng.set_opponents([sd(2)])
    out = eng.collect(8, arena_opponents=[-1, 0, -1, 0])
    # arenas 0,2 self-play (2 learners each), arenas 1,3 opponent (1 learner each) = 6
    assert out["learner_agents"] == 6
    assert out["obs"].shape == (8, 6, 94)
    assert out["actions"].shape == (8, 6)
    assert np.isfinite(out["logprobs"]).all()


def test_opponent_actually_plays_differently():
    eng1, eng2 = mk(n=2, seed=9), mk(n=2, seed=9)
    for e in (eng1, eng2):
        e.set_weights(sd(1))
    eng1.set_opponents([sd(1)])   # opponent = same weights as learner
    eng2.set_opponents([sd(42)])  # opponent = different net
    o1 = eng1.collect(32, arena_opponents=[0, 0])
    o2 = eng2.collect(32, arena_opponents=[0, 0])
    # same learner weights + same seeds; only the opponent differs. If the opponent
    # net is actually driving orange, trajectories must diverge.
    assert not np.array_equal(o1["obs"], o2["obs"])


def test_bad_assignment_rejected():
    eng = mk()
    eng.set_weights(sd(1))
    eng.set_opponents([sd(2)])
    with pytest.raises(Exception):
        eng.collect(4, arena_opponents=[0, 0])            # wrong length (4 arenas)
    with pytest.raises(Exception):
        eng.collect(4, arena_opponents=[-1, -1, -1, 3])   # unset slot
    with pytest.raises(Exception):
        eng.set_opponents([sd(i) for i in range(9)])      # > 8 slots


def test_determinism_with_opponents():
    w, opp = sd(1), sd(7)
    mk2 = lambda: mk(n=4, seed=11, threads=2)
    a, b = mk2(), mk2()
    for e in (a, b):
        e.set_weights(w); e.set_opponents([opp])
    oa = a.collect(16, arena_opponents=[-1, 0, 0, -1])
    ob = b.collect(16, arena_opponents=[-1, 0, 0, -1])
    for k in oa:
        np.testing.assert_array_equal(np.asarray(oa[k]), np.asarray(ob[k]), err_msg=k)
```

- [ ] **Step 2: RED**

Run: `source .venv/bin/activate && pytest tests/python/test_league_engine.py -x`
Expected: TypeError (`collect() got an unexpected keyword argument`) or AttributeError on set_opponents.

- [ ] **Step 3: Implement Rust side**

engine.rs:
- Worker state: `let mut opponents: Vec<MlpPolicy> = Vec::new();`
- `Cmd::SetOpponents(ws)` arm: rebuild `opponents` from each `PolicyWeights` (`MlpPolicy::new` per slot), reply ack/err exactly like SetWeights (drain-all preserved in `MultiEngine::set_opponents`).
- `Cmd::Collect { steps, assignment }`: worker slices its arena range from the global assignment. Per round:
  1. write obs for ALL agents (existing loop) into `obs_buf`.
  2. Build index lists once per collect (not per round): `learner_idx: Vec<usize>` (all agents of self-play arenas + blue agents of opponent arenas, in agent order) and per-slot `opp_idx[slot]: Vec<usize>` (orange agents of arenas assigned to that slot).
  3. Forwards: learner policy on rows `learner_idx` (gather rows into a scratch batch), each used slot on its `opp_idx[slot]` rows. Scatter logits back to a full-width `logits_all` scratch (agents × 90) and values to `values_all`.
  4. Sample for EVERY agent in agent order from the arena's rng (unchanged order — determinism), collecting `acts` full-width.
  5. Record buffers ONLY for learner rows: maintain `learner_col: Vec<Option<usize>>` mapping agent→output column (None for opponent rows). obs/actions/logprobs/values/rewards/flags/final_values recording all filter through it. `final_values`: done-row forward only for learner rows (opponent rows skipped).
  6. `last_values` after loop: learner rows only.
- `CollectOut` sized by worker learner count. Gather in `MultiEngine::collect` uses per-worker learner counts (new field alongside num_agents).
- Validation at `MultiEngine::collect` entry: assignment len == num_arenas; every k in [-1, set_slot_count); learner count > 0 (all-opponent assignment → Err "no learner agents").
- Legacy: `assignment = vec![-1; num_arenas]` when Python passes None — the SAME code path (learner_idx = all agents) must produce byte-identical output (regression test above pins it).

lib.rs:
```rust
fn set_opponents(&mut self, opponents: Vec<HashMap<String, PyReadonlyArrayDyn<'_, f32>>>) -> PyResult<()> {
    if opponents.len() > 8 {
        return Err(PyValueError::new_err("at most 8 opponent slots"));
    }
    let parsed: Vec<PolicyWeights> = opponents.into_iter()
        .map(|w| { /* same conversion as set_weights */ })
        .collect::<Result<_, _>>().map_err(PyValueError::new_err)?;
    self.inner.set_opponents(parsed).map_err(PyValueError::new_err)
}

#[pyo3(signature = (steps, arena_opponents=None))]
fn collect<'py>(&mut self, py: Python<'py>, steps: usize, arena_opponents: Option<Vec<i32>>)
    -> PyResult<Bound<'py, PyDict>> { /* materialize None -> vec![-1; n], validate, detach, gather */ }
```
Add `out.set_item("learner_agents", n_learner)?` to the returned dict.

- [ ] **Step 4: GREEN + full check**

Run: `maturin develop --release && pytest tests/python/test_league_engine.py -v && cd engine && cargo test && cd .. && pytest tests/python -q --deselect tests/python/test_render_session.py::test_render_session_smoke`

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: opponent-policy arenas with learner-only buffers"
```

---

### Task 2: League registry + ratings

**Files:**
- Create: `python/construct/league/__init__.py` (empty), `python/construct/league/registry.py`
- Modify: `pyproject.toml` (add `"trueskill>=0.4.5"`), `.gitignore` (add `league/`)
- Test: `tests/python/test_league.py` (new)

**Interfaces:**
- Produces:
  ```python
  class Registry:
      def __init__(self, path="league/registry.jsonl"): ...
      def add(self, ck, steps, run, reward_config) -> None            # new member, default rating; no-op if ck known
      def entries(self) -> list[dict]                                  # all, insertion order
      def record_match(self, ck_a, ck_b, goals_a, goals_b) -> None     # trueskill update + games+1, atomic save
      def rating(self, ck) -> tuple[float, float]                      # (mu, sigma)
      def ladder(self) -> list[dict]                                   # sorted by exposed skill desc, adds "skill" key
  TS_ENV  # module-level trueskill.TrueSkill(mu=25.0, sigma=25/3, beta=25/6, tau=25/300, draw_probability=0.02)
  ```

- [ ] **Step 1: Failing tests**

```python
# tests/python/test_league.py
import os

from construct.league.registry import Registry


def test_registry_roundtrip(tmp_path):
    r = Registry(path=str(tmp_path / "reg.jsonl"))
    r.add("ck_a.pt", steps=100, run="main", reward_config="configs/reward_v1.toml")
    r.add("ck_b.pt", steps=200, run="b", reward_config="configs/reward_v2.toml")
    r.add("ck_a.pt", steps=100, run="main", reward_config="configs/reward_v1.toml")  # dup no-op
    assert len(r.entries()) == 2
    r2 = Registry(path=str(tmp_path / "reg.jsonl"))  # reload from disk
    assert [e["ck"] for e in r2.entries()] == ["ck_a.pt", "ck_b.pt"]
    mu, sigma = r2.rating("ck_a.pt")
    assert mu == 25.0 and abs(sigma - 25 / 3) < 1e-9


def test_match_updates_ratings(tmp_path):
    r = Registry(path=str(tmp_path / "reg.jsonl"))
    r.add("A", 1, "main", "x"); r.add("B", 2, "main", "x")
    r.record_match("A", "B", goals_a=3, goals_b=0)
    mu_a, _ = r.rating("A"); mu_b, _ = r.rating("B")
    assert mu_a > 25.0 > mu_b
    r.record_match("A", "B", goals_a=1, goals_b=1)  # draw: ratings converge, order kept
    assert r.rating("A")[0] > r.rating("B")[0]
    ladder = r.ladder()
    assert ladder[0]["ck"] == "A" and "skill" in ladder[0]
    assert ladder[0]["games"] == 2


def test_atomic_persistence(tmp_path):
    p = tmp_path / "reg.jsonl"
    r = Registry(path=str(p))
    r.add("A", 1, "main", "x")
    # file exists and is valid jsonl after every mutation
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    import json
    e = json.loads(lines[0])
    assert e["ck"] == "A" and e["games"] == 0
```

- [ ] **Step 2: RED** — `pytest tests/python/test_league.py -x` → ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# python/construct/league/registry.py
"""Checkpoint registry + TrueSkill ladder (jsonl, atomic rewrites)."""
import json
import os
import time

import trueskill

# Seer protocol parameters
TS_ENV = trueskill.TrueSkill(mu=25.0, sigma=25 / 3, beta=25 / 6, tau=25 / 300,
                             draw_probability=0.02)


class Registry:
    def __init__(self, path="league/registry.jsonl"):
        self.path = path
        self._entries: list[dict] = []
        if os.path.exists(path):
            with open(path) as f:
                self._entries = [json.loads(line) for line in f if line.strip()]

    def _save(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            for e in self._entries:
                f.write(json.dumps(e) + "\n")
        os.replace(tmp, self.path)

    def _find(self, ck):
        for e in self._entries:
            if e["ck"] == ck:
                return e
        raise KeyError(ck)

    def add(self, ck, steps, run, reward_config):
        if any(e["ck"] == ck for e in self._entries):
            return
        self._entries.append({
            "ck": ck, "steps": steps, "run": run, "reward_config": reward_config,
            "added_ts": int(time.time()),
            "mu": TS_ENV.mu, "sigma": TS_ENV.sigma, "games": 0,
        })
        self._save()

    def entries(self):
        return list(self._entries)

    def rating(self, ck):
        e = self._find(ck)
        return e["mu"], e["sigma"]

    def record_match(self, ck_a, ck_b, goals_a, goals_b):
        ea, eb = self._find(ck_a), self._find(ck_b)
        ra = TS_ENV.create_rating(ea["mu"], ea["sigma"])
        rb = TS_ENV.create_rating(eb["mu"], eb["sigma"])
        if goals_a > goals_b:
            ra, rb = trueskill.rate_1vs1(ra, rb, env=TS_ENV)
        elif goals_b > goals_a:
            rb, ra = trueskill.rate_1vs1(rb, ra, env=TS_ENV)
        else:
            ra, rb = trueskill.rate_1vs1(ra, rb, drawn=True, env=TS_ENV)
        ea["mu"], ea["sigma"] = ra.mu, ra.sigma
        eb["mu"], eb["sigma"] = rb.mu, rb.sigma
        ea["games"] += 1
        eb["games"] += 1
        self._save()

    def ladder(self):
        out = []
        for e in self._entries:
            r = TS_ENV.create_rating(e["mu"], e["sigma"])
            out.append({**e, "skill": TS_ENV.expose(r)})
        out.sort(key=lambda e: e["skill"], reverse=True)
        return out
```
pyproject: add `"trueskill>=0.4.5"` to dependencies; `.gitignore`: add `league/`.

- [ ] **Step 4: GREEN + install dep**

Run: `uv pip install trueskill && pytest tests/python/test_league.py -v`

- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: league registry with TrueSkill ladder"`

---

### Task 3: MatchRunner

**Files:**
- Create: `python/construct/league/matches.py`
- Test: append to `tests/python/test_league.py`

**Interfaces:**
- Consumes: `Engine.set_weights/set_opponents/collect(T, arena_opponents)` (Task 1).
- Produces:
  ```python
  class MatchRunner:
      def __init__(self, num_arenas=8, seed=0, reward_config="configs/reward_v0.toml", mode=1): ...
      def play(self, sd_a: dict, sd_b: dict, steps: int = 2700) -> tuple[int, int]
      # A drives blue (as "learner"), B drives orange (opponent slot 0), all arenas
      # opponent-assigned. Goals counted from the learner reward stream:
      # rew >= 9.5 -> A scored; rew <= -9.5 -> B scored (goal weight 10, shaping << 9.5,
      # concede = -10*(1-aggression_bias) with v0 bias 0 -> -10). 2700 steps = 3 game-min/arena.
  def load_sd(ck_path: str) -> dict  # torch checkpoint -> f32 numpy state_dict
  ```
- Requires `reward_config` with |goal| ≥ 9.5 clearance over shaping and bias 0 for symmetric counting — use reward_v0 for matches always (document in code: match REWARDS are only a scoring tape, not training signal).

- [ ] **Step 1: Failing tests**

```python
# append to tests/python/test_league.py
import numpy as np
import torch

from construct.league.matches import MatchRunner, load_sd
from construct.learn.model import PolicyValueNet


def _sd(seed):
    torch.manual_seed(seed)
    return {k: v.detach().numpy().astype(np.float32)
            for k, v in PolicyValueNet(94, 90, (64, 64)).state_dict().items()}


def test_match_runs_and_counts(tmp_path):
    mr = MatchRunner(num_arenas=4, seed=2)
    ga, gb = mr.play(_sd(1), _sd(2), steps=600)
    assert ga >= 0 and gb >= 0  # random-ish nets rarely score in 40s; counting must not crash


def test_match_deterministic(tmp_path):
    a = MatchRunner(num_arenas=4, seed=7).play(_sd(1), _sd(2), steps=400)
    b = MatchRunner(num_arenas=4, seed=7).play(_sd(1), _sd(2), steps=400)
    assert a == b


def test_load_sd_roundtrip(tmp_path):
    net = PolicyValueNet(94, 90, (64, 64))
    ck = {"model": net.state_dict(), "config": {"net": {"hidden": [64, 64]}}, "total_steps": 1,
          "schema_version": 0}
    p = str(tmp_path / "ck.pt")
    torch.save(ck, p)
    sd = load_sd(p)
    assert sd["policy_head.weight"].dtype == np.float32
```

- [ ] **Step 2: RED** — ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# python/construct/league/matches.py
"""Head-to-head matches between two frozen checkpoints, via opponent arenas.

The engine's reward stream is used purely as a scoring tape: with reward_v0
(goal=10, bias 0, shaping << 9.5) a learner-row reward >= 9.5 means A scored,
<= -9.5 means B scored. Matches always use reward_v0 regardless of what the
policies were trained on.
"""
import numpy as np
import torch

from construct._engine import Engine


def load_sd(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    return {k: v.numpy().astype(np.float32) for k, v in ck["model"].items()}


class MatchRunner:
    def __init__(self, num_arenas=8, seed=0, reward_config="configs/reward_v0.toml", mode=1):
        self.eng = Engine(num_arenas=num_arenas, blue=mode, orange=mode,
                          schema_path="schema/v0.toml", reward_config_path=reward_config,
                          seed=seed)
        self.assignment = [0] * num_arenas

    def play(self, sd_a, sd_b, steps=2700):
        self.eng.set_weights(sd_a)
        self.eng.set_opponents([sd_b])
        out = self.eng.collect(steps, arena_opponents=self.assignment)
        rew = np.asarray(out["rewards"])
        goals_a = int((rew >= 9.5).sum())
        goals_b = int((rew <= -9.5).sum())
        return goals_a, goals_b
```
NOTE for implementer: goal events pay every learner agent on the scoring team in that arena — at mode=1 (1v1) one learner row per arena, so the count is exact. Assert `mode == 1` for now with a comment (2v2 matches would multi-count; divide by team size when that arrives — YAGNI today).

- [ ] **Step 4: GREEN + full python suite; Step 5: Commit** — `feat: head-to-head match runner over opponent arenas`

---

### Task 4: Sampling + league_tick

**Files:**
- Create: `python/construct/league/sampling.py`, `scripts/league_tick.py`
- Test: append to `tests/python/test_league.py`

**Interfaces:**
- Consumes: Registry (T2), MatchRunner + load_sd (T3).
- Produces:
  ```python
  # sampling.py
  def choose_opponents(registry, k=4, recent=6, rng=None) -> list[dict]
  # entries: up to k, drawn as: top-2 by skill + uniform from the `recent` newest, dedup,
  # deterministic given rng (random.Random instance)
  ```
  `scripts/league_tick.py [--matches N]`: registers the newest main (`checkpoints/ck_*.pt`) and run-B (`checkpoints_b/ck_*.pt`) checkpoints, plays N rating matches (newest members vs sampled pool), prints the ladder table.

- [ ] **Step 1: Failing test**

```python
# append to tests/python/test_league.py
import random

from construct.league.sampling import choose_opponents


def test_choose_opponents_mix(tmp_path):
    r = Registry(path=str(tmp_path / "reg.jsonl"))
    for i in range(10):
        r.add(f"ck{i}", steps=i, run="main", reward_config="x")
    # give ck0 a huge rating so it must appear as a top pick
    r.record_match("ck0", "ck5", 5, 0)
    r.record_match("ck0", "ck6", 5, 0)
    picks = choose_opponents(r, k=4, recent=5, rng=random.Random(3))
    names = [p["ck"] for p in picks]
    assert "ck0" in names
    assert len(names) == len(set(names)) <= 4
    # deterministic
    again = [p["ck"] for p in choose_opponents(r, k=4, recent=5, rng=random.Random(3))]
    assert names == again
```

- [ ] **Step 2: RED; Step 3: Implement**

```python
# python/construct/league/sampling.py
"""Opponent selection: exploit the ladder top + explore recent additions."""
import random


def choose_opponents(registry, k=4, recent=6, rng=None):
    rng = rng or random.Random()
    ladder = registry.ladder()
    if not ladder:
        return []
    picks = ladder[:2]  # top by exposed skill
    newest = sorted(registry.entries(), key=lambda e: e["added_ts"])[-recent:]
    pool = [e for e in newest if e["ck"] not in {p["ck"] for p in picks}]
    rng.shuffle(pool)
    picks.extend(pool[: max(0, k - len(picks))])
    return picks[:k]
```

```python
# scripts/league_tick.py
"""One league tick: register newest checkpoints, play rating matches, print ladder.
Run manually or on a loop. Cheap: each match is headless engine time only."""
import argparse
import glob
import random

from construct.league.matches import MatchRunner, load_sd
from construct.league.registry import Registry
from construct.league.sampling import choose_opponents


def newest(pattern):
    cks = sorted(glob.glob(pattern))
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--registry", default="league/registry.jsonl")
    args = ap.parse_args()

    reg = Registry(path=args.registry)
    import torch
    for run, pattern in (("main", "checkpoints/ck_*.pt"), ("b", "checkpoints_b/ck_*.pt")):
        ck = newest(pattern)
        if ck:
            meta = torch.load(ck, map_location="cpu", weights_only=False)
            reg.add(ck, steps=meta["total_steps"], run=run,
                    reward_config=meta.get("reward_config_path", "unknown"))

    entries = reg.entries()
    if len(entries) >= 2:
        rng = random.Random()
        fresh = sorted(entries, key=lambda e: e["added_ts"])[-2:]
        mr = MatchRunner(num_arenas=8, seed=rng.randrange(1 << 30))
        played = 0
        for member in fresh:
            for opp in choose_opponents(reg, k=3, rng=rng):
                if opp["ck"] == member["ck"] or played >= args.matches:
                    continue
                ga, gb = mr.play(load_sd(member["ck"]), load_sd(opp["ck"]))
                reg.record_match(member["ck"], opp["ck"], ga, gb)
                print(f"{member['ck']}  {ga}:{gb}  {opp['ck']}")
                played += 1

    print(f"\n{'skill':>7}  {'mu':>6}  {'games':>5}  run   checkpoint")
    for e in reg.ladder()[:15]:
        print(f"{e['skill']:7.2f}  {e['mu']:6.2f}  {e['games']:5d}  {e['run']:<4}  {e['ck']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: GREEN (`pytest tests/python/test_league.py -v`); smoke the script once: `python scripts/league_tick.py --matches 2` (registers live checkpoints, plays 2 short-ish matches — CPU-only, fine alongside training; do NOT run more). Step 5: Commit** — `feat: opponent sampling + league tick script`

---

### Task 5: Trainer integration (NO DEPLOY)

**Files:**
- Modify: `python/construct/learn/config.py`, `python/construct/learn/train.py`, `scripts/resume_train.py`, `configs/train_v0.toml`
- Test: append to `tests/python/test_league.py`

**Interfaces:**
- Consumes: Engine opponent API (T1), Registry/sampling/load_sd (T2-4).
- Produces: `TrainConfig.league: dict` (default `{}`) with keys `enabled` (bool), `opponent_frac` (float, default 0.2), `registry` (path), `refresh_iters` (int, default 200), `slots` (int ≤ 8, default 4). Trainer: when enabled, every `refresh_iters` iterations → `choose_opponents` → `load_sd` each → `set_opponents` → rebuild `self._assignment`: last `round(opponent_frac * num_arenas)` arenas get slots round-robin, rest -1 (arenas are ordered 1s/2s/3s blocks; taking the tail spreads opponents across the largest-team arenas first — document this bias, it is acceptable v1). `collect()` passes `arena_opponents=self._assignment` (or None when disabled). `resume_train.py` gains `--league` (bool flag enabling with defaults).
- Batch bookkeeping: `N = out["learner_agents"]` (no longer `engine.num_agents`); `total_steps` accounting uses N per iteration — document that steps now count LEARNER transitions.

- [ ] **Step 1: Failing test**

```python
# append to tests/python/test_league.py
from construct.learn.config import TrainConfig
from construct.learn.train import Trainer


def test_trainer_league_refresh(tmp_path):
    # seed a registry with two members built from tiny checkpoints
    import torch
    from construct.learn.model import PolicyValueNet
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    for i in (1, 2):
        net = PolicyValueNet(94, 90, (512, 512))
        p = str(tmp_path / f"opp{i}.pt")
        torch.save({"model": net.state_dict(), "total_steps": i,
                    "config": {"net": {"hidden": [512, 512]}}, "schema_version": 0,
                    "reward_config_path": "x"}, p)
        reg.add(p, steps=i, run="main", reward_config="x")

    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=4)
    cfg.ppo.update(rollout_steps=8, minibatch_size=64)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=100)
    cfg.league = {"enabled": True, "opponent_frac": 0.5, "registry": reg_path,
                  "refresh_iters": 1, "slots": 2}
    t = Trainer(cfg)
    t.run(max_iterations=2)
    # 4 arenas, frac 0.5 -> 2 opponent arenas -> learner agents = 4*2 - 2 = 6
    assert t.total_steps == 2 * 8 * 6
```

- [ ] **Step 2: RED; Step 3: Implement**

config.py: add `league: dict = field(default_factory=dict)`.
train.py:
```python
    # in __init__, after engine construction:
    self._assignment = None
    self._league = None
    lg = cfg.league
    if lg.get("enabled"):
        from construct.league.registry import Registry
        from construct.league.sampling import choose_opponents
        from construct.league.matches import load_sd
        self._league = {
            "registry": Registry(path=lg.get("registry", "league/registry.jsonl")),
            "choose": choose_opponents, "load_sd": load_sd,
            "frac": float(lg.get("opponent_frac", 0.2)),
            "refresh": int(lg.get("refresh_iters", 200)),
            "slots": int(lg.get("slots", 4)),
        }

    def _refresh_opponents(self):
        L = self._league
        picks = L["choose"](L["registry"], k=L["slots"])
        if not picks:
            self._assignment = None
            return
        sds = [L["load_sd"](p["ck"]) for p in picks]
        self.engine.set_opponents(sds)
        n = self.engine_num_arenas  # store num_arenas on self in __init__
        n_opp = round(L["frac"] * n)
        a = [-1] * n
        for i in range(n_opp):
            a[n - 1 - i] = i % len(sds)
        self._assignment = a

    # in run(), at loop top:
    #   if self._league and it % self._league["refresh"] == 0: self._refresh_opponents()
    # in collect(): out = self.engine.collect(T, arena_opponents=self._assignment)
    #   N = out["learner_agents"]; all reshapes use N
    # in run(): n = p["rollout_steps"] * <that N> — cache the last N from collect
    #   (simplest: collect returns it in the batch dict as batch["n_agents"])
```
(Implementer: keep it plain — the dict-of-callables sketch above is a hint, a small `_League` dataclass is equally fine. The binding contract is the test + config keys.)
resume_train.py: `p.add_argument("--league", action="store_true")` → `cfg.league = {**cfg.league, "enabled": True}`.
train_v0.toml: commented `[league]` example block.

- [ ] **Step 4: GREEN + full suite + smoke (`python scripts/smoke_test.py` unchanged). Step 5: Commit + STOP (no deploy)** — `feat: trainer opponent-pool integration`

---

## Self-Review Notes

- Coverage vs design: opponent arenas + learner-only buffers ✓(T1), registry/TrueSkill ✓(T2, env params exact), match mode ✓(T3, zero engine additions), sampling + tick ✓(T4), trainer wiring ✓(T5, deploy-gated).
- Determinism: sampling for ALL agents preserves rng streams (T1 constraint + determinism test); match determinism test (T3).
- Type consistency: `collect(T, arena_opponents)` and `learner_agents` key used identically T1/T3/T5; Registry API surface consistent T2/T4/T5; `load_sd` T3/T4/T5.
- Known v1 simplifications, deliberate: opponent arenas taken from the arena-order tail (biases opponents toward 3v3 arenas — documented); match goal-counting asserts mode==1; no PFSP weighting yet (uniform recent + top-2); ladder matches only pit the 2 freshest members per tick.
- The dashboard ladder panel is intentionally OUT of scope (add after data exists).
