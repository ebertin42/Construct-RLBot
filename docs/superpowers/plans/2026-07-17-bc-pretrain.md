# BC Pretrain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Behavior-clone the EntityPolicyNet on the GC2 human-replay corpus, producing a human-prior checkpoint usable as a league anchor and KL-prior.

**Architecture:** Three stages. (1) Shard schema v4: one corpus re-parse enriching shards with the columns BC needs that replays/sim already know — 34 pad states, `has_flip`, 4-horizon ball prediction (all read from the live RocketSim arena during the existing 120 Hz reconstruction), plus the projected 92-table action index (context-gated nearest-row, rlgym-tools `pick_action` style). (2) `bc-export`: a Rust binary (replay crate, now depending on `construct-engine` as a workspace rlib) that converts v4 shards into training-ready obs tensors by rebuilding a minimal `GameState` per row and calling the **same** `obs_v1::build` the live engine uses — train/deploy consistency by construction. Per-player perspectives (each car = one sample from its own mirrored POV). (3) Python BC trainer: inverse-action-frequency-weighted cross-entropy on the projected indices, AdamW, val split, behavior eval of the resulting checkpoint.

**Tech Stack:** Rust (replay crate + construct-engine rlib, ndarray-npy), Python (torch, EntityPolicyNet from model_v1.py).

## Global Constraints
- No deploy; never touch training processes, checkpoints*/ (read-only OK), live registries, remote host. Corpus re-parse and BC training run on the local box, niced.
- `SHARD_SCHEMA_VERSION = 4`; v3 shards remain readable (loader dispatches on sidecar version); BC pipeline requires v4.
- Value-head warm-start (spec §5.4's discounted-return regression) is **deferred** — BC-v1 trains policy head only (value head left at init; the KL-PPO stage treats the BC net as a frozen *prior*, not a critic). Recorded as a follow-up.
- Winners-side-only / SSL filtering deferred to the ballchasing finetune stage (token saved; regular tier).
- Determinism: fixed seed ⇒ identical export sample order and train batches.

## Task B1: shard schema v4 — pads, has_flip, ball-pred columns
**Files:** Modify `replay/Cargo.toml` (add `construct-engine = { path = "../engine" }`), `replay/src/reconstruct.rs`, `replay/src/shard.rs`; Test `replay/tests/shard_test.rs`.
`Tick` gains: `pads: Vec<(f32 /*timer 0..1*/, bool /*active*/); 34]`-equivalent (fixed 34, arena order), per-car `has_flip: bool` (from sim `CarState`), `ball_pred: [[f32;6];4]` (pos+vel at +0.5/1/1.5/2 s via `construct_engine::ballpred::Tracker`, one per stored tick — predict only on STORED (strided) ticks to bound cost). Shard arrays: `pads [T,34,2]`, `ball_pred [T,4,6]`; `cars_state` gains `has_flip` (17 cols). Bump schema to 4; sidecar documents new columns. TDD: extend `writes_loadable_shard_with_schema` (shapes, pad values in 0..1, pred finite and evolving for a moving ball).

## Task B2: action projection — 8-dim controls → 92-table index
**Files:** Create `replay/src/project.rs`; Modify `replay/src/shard.rs` (emit `cars_action_idx [T,P] i64`), `replay/src/reconstruct.rs` if gating inputs needed; Test `replay/tests/project_test.rs`.
`pub fn project_action(action: &[f32;8], on_ground: bool, has_flip: bool, boost_amount: f32, table: &[[f32;8]]) -> usize` — context-gated nearest row over `make_lookup_table_v1()` (92): grounded → match (throttle, steer, handbrake; jump only if pressed & has_flip); aerial → (pitch, yaw, roll; jump only if it matters); boost matched only when `boost_amount > 0` AND boost pressed distinguishes; dodge rows scored by direction dot-product (dodgeDir = (-pitch, yaw+roll)); stall rows reachable (yaw=-roll+jump). Tests: exact tuples map to themselves; grounded steer=yaw equivalence resolves to a ground row; airborne torque rows ignore steer; boost=0 never picks a boost-only-differing row; stall input maps to a stall row.

## Task B3: `bc-export` binary — v4 shards → obs-v1 training tensors
**Files:** Create `replay/src/bin/bc_export.rs`, `replay/src/bc_obs.rs`; Test `replay/tests/bc_export_test.rs`.
Per shard row, per car: rebuild a minimal `rocketsim_rs::GameState` (ball + cars from stored cols, pads applied) — NO stepping, pure struct fill — then call `construct_engine::obs_v1::build` for that car with the stored `ball_pred` snaps and the prev-5 projected indices (from prior rows of the same shard, zeros at episode/gap starts: reset the window when `tick_index` gaps or `is_boundary` after drops indicate discontinuity). Output per input shard: `bc_<id>.npz` with `ents [S,17,26] f32, mask [S,17] u8, query [S,64] f32, prev [S,5] i64, action [S] i64` (S = T×P). CLI: `bc-export --shards DIR --out DIR [--jobs N]`, resumable (skip existing), summary line. Test on the fixture-replay shard: shapes, self one-hot per sample, finite, action in [0,92).

## Task B4 (operational): corpus re-parse to v4 + export
Run (niced, after B1-B3 merge): full re-parse of `data/replays` → `data/shards_v4` (~7-8 h), then `bc-export` → `data/bc` (size estimate: 26+17 floats ≈ 180 B/sample × ~700M samples ≈ manageable at f32; if >400 GB, switch ents/query to f16 in the export — decide by measurement on one batch first). Rebuild manifest. Ledger the numbers.

## Task B5: BC trainer
**Files:** Create `python/construct/learn/bc.py`, `scripts/bc_train.py`, `configs/bc_v1.toml`; Test `tests/python/test_bc.py`.
Dataset streams `bc_*.npz` shards (memory-mapped npz or per-shard load, shuffled shard order + in-shard permutation, fixed seed); split by shard hash (95/5 train/val). Loss: CE over 92 with inverse-action-frequency weights (computed in a first pass over `action` arrays, cached json; clamp weights to [0.1, 10]). AdamW lr 3e-4 cosine, batch 4096, grad-clip 1.0. Metrics per epoch: val top-1/top-3 accuracy, per-class recall for rare classes (jump rows, stalls), loss. Checkpoint format compatible with the existing v1 checkpoint schema (`schema_version: 1`, net dims) so eval_metrics/watch/kickstart-teacher tooling work on BC checkpoints unchanged. Tests: tiny synthetic bc-shards → 2-epoch run improves train loss; checkpoint loads into EntityPolicyNet; weights json cached.
Reference targets (from Seer/Rolv precedent): top-1 ~35-50% is normal; watch per-class recall not just top-1.

## Task B6: BC eval + report
Run eval_metrics + watch.py on the BC checkpoint; behavior read (does it kick off, rotate, aerial?); append findings + go/no-go for the KL-PPO stage to the training journal. Exit criterion for the plan: BC checkpoint plays visibly structured Rocket League (not necessarily strong) and beats a random-init EntityPolicyNet on goals/min and touches.

## Post-plan
- KL-PPO stage (spec §5.5) = separate small plan: config-gate a `kl_prior` (frozen BC net, same obs) into ppo loss next to the kickstart seam.
- SSL finetune corpus via ballchasing (winners-side filter) = separate acquisition task.
- Value warm-start revisit if KL-PPO cold critic hurts.
