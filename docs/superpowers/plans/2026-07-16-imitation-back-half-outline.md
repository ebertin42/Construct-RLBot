# P2 Back Half (IDM → BC → KL-PPO) — Outline & Ordering Decision

**Status: OUTLINE** — full plans get written when their prerequisites land. This
records the load-bearing analysis so it isn't re-derived.

## Key finding: the VPT-style learned IDM stage is (probably) skippable

The spec's §5.3 IDM exists because replays lack pitch/yaw/roll. Our shard
pipeline already solves that **analytically** during reconstruction:
`cars_action` in every shard = replay-native buttons (throttle/steer/jump/
boost/handbrake) + analytic torque-inversion pyr (`replay/src/pyr.rs`, the
rlgym-tools `predict_pyr` approach). This mirrors where the community landed:
rlgym-tools v2 moved from learned IDM (2022-23) to analytic inversion + sim
interpolation (2024-26), and the MSR behavior-cloning paper (arXiv 2502.14998,
2.2M replays) explicitly rejected learned IDMs for variable-rate replay data in
favor of exactly this. A learned IDM remains an *ablation option* if BC
underperforms (train on sim rollouts where true actions are known, refine the
analytic labels), not a prerequisite.

## What replaces it: action projection (small task, Rust)

BC targets our discrete lookup table (90 now, ~110-130 after obs/action v1.1),
not raw 8-dim controls. Needed: context-gated projection of each shard
`cars_action` row onto the nearest table entry, rlgym-tools `pick_action`
style — jump/boost/handbrake matched only when they matter (has_flip, boost>0,
grounded), dodge direction by dot product, indistinguishable-action
equivalence (steer=yaw on ground, pyr irrelevant grounded, etc.). Deliverable:
`cars_action_idx [T]` per shard (or computed at dataloader time — decide in the
full plan; precomputing in Rust is cheaper than Python-side per-epoch).

## Ordering decision (the important one)

**Entity transformer + obs v1 comes BEFORE BC.** Spec §5.4: BC trains the
*exact policy net* via the Rust replay→obs path (train/deploy consistency).
BC-pretraining today's 94-float MLP would be throwaway work once the
transformer lands. So:

1. **Corpus build** — running now (GC2 duels ≈ 130k replays → ~180 GB shards
   @15 Hz, schema v3). GC1 duels + GC3 doubles optional extensions.
2. **Entity transformer + obs v1** (plan: 2026-07-16-entity-transformer-obs-v1)
   — the critical path. Includes the replay→obs-v1 Rust path BC needs.
3. **Action projection** (small task, can ride along with #2's action table).
4. **BC pretrain** — full plan written once #2's obs/net are real. Inverse-
   action-frequency-weighted CE, value head warm-start via discounted-return
   regression on shard rewards (needs reward tape over shards — decide in plan),
   winners-side-only filter for the SSL finetune stage (needs ballchasing pulls;
   token saved, regular tier).
5. **KL-PPO** — PPO + KL-to-frozen-BC-policy decayed over training (Ripple's
   fix for Seer washout), config-gated; ablation groups pure-RL vs BC-init vs
   BC-init+KL per spec §5.6.

## Data status (as of 2026-07-16)

- batch_0000 parsed: 9,989 shards, 56.5M pairs @15 Hz, 14 GB, 160k reset states.
- Full GC2 build running: `scripts/build_gc2_corpus.sh` (12 more batches).
- Reset-state pool feeds engine curriculum replay-resets (spec §4) — separate
  small engine task, independent of BC (curriculum loads `.jsonl`, converts
  quat→rot_mat like `replay/src/reconstruct.rs`).
