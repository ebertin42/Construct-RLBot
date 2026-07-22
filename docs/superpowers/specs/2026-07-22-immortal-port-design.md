# Immortal external training-opponent — in-engine port (design)

Date: 2026-07-22
Status: design — awaiting review
Decision owner: Elliot

## Goal

Make the community bot **Immortal** (RLMarlbot, rlgym-ppo lineage) a first-class
league **training opponent** inside our Rust engine, so PPO trains against a
strong, *externally-derived* policy — not just our own weaker checkpoints.

This is the concrete follow-through on **Option A (opponent diversity)**. As of
2026-07-22 every own-checkpoint arm has failed to beat the champion
`ck_000320471040`:

- reward levers 1/2/3 — all below parity (0.44, backfired, 0.4289)
- **league arm (own-checkpoint diversity), iter-290: 0.4570 FAIL** (null mean
  0.502, threshold 0.55)

Own-checkpoint diversity did not break parity. A genuinely external opponent
is the untried lever. The same path later serves **Nexto** — the project's P3
exit bar ("beats Nexto").

Requirements: **faithful** (byte-exact to the shipped `jit.pt` within cadence),
**fast** (in-engine, no Python-per-step), **reusable** (an `ExternalPolicy`
framework Element/Nexto can later plug into).

## Why in-engine (decided 2026-07-22)

`collect()` runs rollouts autonomously in-engine; opponents are OUR-format nets
forwarded in batched passes (`engine/src/engine.rs` ~388–409). There is **no
Rust→Python per-step hook**. A torch sidecar would need invasive callback
surgery and would be slow in the rollout hot loop; distillation is lossy. So we
bring Immortal's obs + net + action **into** the engine as a new opponent kind.
(Elliot chose "Faithful in-engine port" over "Distill" and "Both, staged".)

## Immortal contract (verified from RLMarlbot @ `9963b3bd`)

Read from the real source — recon's field-count estimates were wrong (it
guessed 101; the real 1v1 obs is 107). Golden tests below are the guard.

- **Obs** — `ExpandAdvancedObs` (standard RLGym `AdvancedObs` + `expand_dims(0)`),
  shape **(1, 107)** for 1v1. Order:
  `ball.pos/2300 (3), ball.linvel/2300 (3), ball.angvel/π (3), prev_action (8),
  boost_pads (34), SELF block (25), OPPONENT block (25) + rel-extras (6)`.
  - Per-player 25 block: `rel_pos/2300 (3), rel_vel/2300 (3), pos/2300 (3),
    forward() (3), up() (3), linvel/2300 (3), angvel/π (3),
    [boost, on_ground, has_flip, is_demoed] (4)`.
  - rel-extras (per *other* player): `(other.pos−self.pos)/2300 (3),
    (other.linvel−self.linvel)/2300 (3)`.
  - Orange side inverts: `inverted_ball / inverted_boost_pads /
    inverted_car_data` (RLGym inversion: negate x,y and flip the basis).
  - `POS_STD = 2300`, `ANG_STD = π`.
- **Net** — TorchScript `jit.pt` (5.74 MB), single head → logits **(1, 126)**.
  Plain MLP (rlgym-ppo). Verified: `torch.jit.load(jit.pt)(zeros(1,107))` →
  `(1,126)`. Deterministic play = `argmax`.
- **Action** — `ImmortalAction` lookup, **126 rows × controls-8**
  (24 ground + 102 aerial), Necto-family. Row =
  `[throttle, steer, pitch, yaw, roll, jump, boost, handbrake]`.
  `update_controls`: `yaw = 0 if jump else steer-slot`; `jump/boost/handbrake = (>0)`.
- **Decision rate** — `tick_skip = 6` (act every 6 physics ticks, 20 Hz),
  controls held between decisions; `prev_action` = the controls-8 chosen last
  decision, fed back into the obs.

## Cadence mismatch — decision needed

Our engine's schema `tick_skip = 8` (15 Hz; `engine/src/schema.rs:77`).
Immortal trained at `tick_skip = 6` (20 Hz). Two options:

- **(v1, recommended) Accept the 8-tick cadence.** The external opponent decides
  on our 8-tick clock like every other agent. Immortal reacts slightly slower
  than trained — acceptable for a *sparring partner* (goal is a strong, distinct
  opponent, not a benchmark-exact replica). Zero cadence machinery.
- **(later) Per-opponent 6-tick sub-clock.** Give external slots their own
  decision cadence. Better fidelity, but a new per-slot clock in the collect
  loop. Defer unless Immortal plays visibly degraded at 8-tick.

## Design

### 1. Opponent slot becomes a kind

Today an opponent slot is an `EntityPolicy`. Introduce:

```
enum OpponentPolicy {
    Native(EntityPolicy),              // today's path — byte-identical
    External(ExternalPolicy),          // new
}
```

`ExternalPolicy` bundles: an obs kind (`AdvancedObs107`), a candle MLP
(extracted weights), an action lookup (`126×8`), and per-car state
(`prev_action`, held controls). Native and self-play paths stay byte-identical
when no external slot is used (non-regression test).

### 2. Obs — AdvancedObs-107 builder (Rust)

New `engine/src/obs_advanced.rs`:
`build_advanced_obs(state, car_idx, prev_action8, out107)` reproducing
`ExpandAdvancedObs` **byte-exact** — RLGym inversion for orange, boost-pad
ordering, `forward()`/`up()` from `rot_mat`, the 6 rel-extras. Distinct from our
94-obs `build_obs`; shares `GameState`/`CarInfo` inputs.

### 3. Net — MLP in candle

- Extract FC params from `jit.pt` in Python (`named_parameters()`), confirm the
  Linear→ReLU stack widths (input 107 → … → 126), export as arrays.
- Reimplement the forward in candle (same widths/activations) → logits(126).

### 4. Action — apply controls-8 directly

`argmax(126) → lookup row → controls-8 → CarControls` on the orange car, via the
existing `actions::to_controls` (already `[f32;8] → CarControls`), **but** apply
Immortal's `update_controls` mapping (yaw/jump/boost/handbrake) rather than ours
where they differ. Bypasses our action-index decode. Refresh `prev_action` each
decision; hold controls between.

### 5. Wiring

- `collect()` opponent path (`engine.rs` ~388–409): when a slot is `External`,
  gather that slot's cars' AdvancedObs → external forward → argmax → controls →
  apply. Learner + native opponents untouched.
- Loading API: `set_external_opponents(...)` (parallel to `set_opponents`)
  taking the extracted MLP arrays + lookup + config; or a kind-tagged
  `set_opponents`. Python side: a loader that reads the weights manifest.
- League/registry: allow an external-opponent id in the pool alongside `ck_*`,
  so `opponent_frac` draws it like any other opponent.

### 6. Determinism / fidelity

- External opponent acts **deterministically** (`argmax`), matching Immortal's
  deploy and preserving gate/rollout determinism.
- 8-tick cadence per the decision above.

## Weights handling / license

Immortal's weights are **unlicensed** (RLMarlbot is a reverse-engineering tool,
no stated license). Therefore:

- **Do NOT commit `jit.pt` (or extracted weights) to the repo.** Store on the
  training box / local cache only; pin SHA256 in a manifest (`deploy/external/`
  or a dedicated `external_weights.manifest`), tamper-evident but not vendored.
- Private-research training use only; not redistributable.
- Source pinned: RLMarlbot @ `9963b3bd`, `rlmarlbot/immortal/{jit.pt,
  agent.py, action/actionparser.py, obs/advanced_obs.py, bot.py}`.

## Testing

- **Golden obs**: Rust `build_advanced_obs` == Python `ExpandAdvancedObs` on N
  random `GameState`s (near-bit, `<1e-5`). *This is the primary guard against
  the silent off-by-one that breaks a ported policy.*
- **Golden net**: candle MLP forward == `torch.jit.load(jit.pt)` on N random
  obs (`max-abs-err < 1e-4`).
- **End-to-end**: short in-engine match — external-opponent controls ==
  Python `Immortal.act` on the same state sequence (argmax indices identical).
- **Non-regression**: native / self-play `collect()` byte-identical with the
  external path compiled in but unused.

## Risks

- **Obs off-by-one silently breaks the policy** (recon's explicit warning) →
  golden obs test is the gate; do not proceed past it on approximate parity.
- **`jit.pt` graph opacity** (widths frozen in the trace) → extract via
  `named_parameters`, verify shapes before reimplementing.
- **Cadence** (8 vs 6 tick) → v1 accepts 8-tick; watch for degraded play.
- **License** → weights out of git; manifest + SHA only.

## Out of scope (this spec)

- **Element** — multi-head (5 categorical + 3 bernoulli), pickled state_dict
  (needs its `Actor` class), 107-dim `CustomObs`, 1v1-hardcoded. Follow-on on
  the same `ExternalPolicy` framework with a different obs + action head.
- **Nexto / Necto** — Perceiver `(q, kv, mask)` obs + 90-action lookup. Bigger
  obs + attention net; reuse `ExternalPolicy` with an attention forward later.
- **RLBot-v5 benchmark wrappers** (real-game, Windows box) — separate, tracked
  under #54; Nexto/Necto already SHA-pinned + fetch-verified in
  `scripts/bench_external.py`.
