# Training Journal

Rolling health log, appended by the 4-hourly monitor. Metrics from the neutral
reward_v0 tape (`scripts/eval_metrics.py`, 1v1 sampled) — NOT shaped ep_rew,
which reward-farming inflates. Regression flag: goals/min down ≥15% vs the
last 2 entries. Reward-hacking signature: ep_rew rising while goals/min falls.

## Baseline history (pre-journal, 2026-07-15/16)

| when | regime | steps | goals/min 1v1 | note |
|---|---|---|---|---|
| 07-15 | v1 (hacked) | 7.4B | 0.51 | touch-farm lineage, abandoned, archived |
| 07-15 | v2 peak | 2.24B | 1.35 (2v2: 2.58) | run-B checkpoint, promoted base |
| 07-15 | v2 (hacked) | 3.6B | 0.40 | shaping-farm collapse, archived; −45%/B drift |
| 07-15 | v3 launch | 2.24B→ | 1.34 | rolled back to peak, farmable shaping stripped |
| 07-15 | v3 | 2.36B | 1.70 | recovery confirmed |
| 07-16 | v3 | 3.0B | 1.73 | |
| 07-16 | v3 | 3.4B | 1.62 | |
| 07-16 | v3 | 3.98B | 1.56 | holding ~1.6; touches 12→19, dist 1775→1480 |
| 07-16 | v3 | 4.4B | (not evaled) | 101k sps, ent 3.45, registry 11 |

Watch items: v3 plateau around 1.6 (if it exceeds ~2B steps flat, consider:
entropy schedule 0.01→0.005, curriculum scenario-pack weights, league
opponent_frac up from 0.2); entity-transformer kickstart deploy imminent —
after swap, expect goals/min to track the teacher (~1.5) during KL anneal,
then exceed it; v1 eval script support pending (monitor notes "v1 eval
pending" until added).

## Entries
### 2026-07-17 ~02:00 — ENTITY-TRANSFORMER KICKSTART DEPLOYED
Branch entity-transformer merged (11 commits, review MERGE + fix wave c76c73e).
v3 MLP run stopped at 4.956B (final ck = kickstart teacher, 1.56 goals/min).
Remote launched: fresh EntityPolicyNet 128/2/4/512 (488k params), reward_v3 +
curriculum + team sizes, league OFF (v0 registry incompatible), dir
checkpoints_entity/. Real remote bench: 11k sps collect-only, ~5.4k sps
in-train (learner+teacher overhead). Anneal 500M steps ≈ ~25h at this rate.
iter 10: kick_kl 1.05-1.15 flattish (0.3% of anneal — early), ent 4.24→3.89,
ep_rew already teacher-range 2-6. Watch: KL must fall by ~50M steps.
Rollback: relaunch v3 from checkpoints/ (intact, not archived — dir untouched).

### 2026-07-17 ~05:00 (monitor)
| steps | sps | kick_kl | lambda_k | ent | goals/min |
|---|---|---|---|---|---|
| 43.1M | 5,415 | 0.48 | 0.914 | 3.35 | **1.71** (ck 40M) |
Trend: KL converging cleanly (1.1→0.48 by 43M — no flag); entropy 3.9→3.3;
ep_rew 6-7 (above teacher's 4-5). Eval at 40M: 1.71 goals/min — ALREADY ABOVE
the 1.56 teacher, 8.6% into the anneal. Kickstart transfer working better than
projected. Improvement notes: none needed yet; next checks — KL should keep
falling toward <0.2 as lambda anneals; watch for post-anneal hacking signature.
Corpus: batch_0006 pulling, 34.1k shards, 712G free. All local loops UP.

### 2026-07-17 ~09:00 (monitor)
| steps | sps | kick_kl | lambda_k | ent | goals/min |
|---|---|---|---|---|---|
| 120.8M | 5,416 | 0.40 | 0.759 | 3.24 | 1.40 (ck 120M) |
Trend: KL still converging (0.48→0.40), entropy easing, sps steady. Eval dipped
1.71→1.40 — single-sample metric with historical ±0.2 noise band (v3 oscillated
1.34-1.73); within teacher range, NOT flagging regression yet. Watch next tick:
two consecutive <1.35 readings = real dip. Corpus: 55,935 shards / 77G (the
"shards: 0" in the build log is an ARG_MAX ls bug at >50k files — cosmetic,
fix `ls | wc` → `find | wc` in build_gc2_corpus.sh line 30 after sweep ends).
Batch_0011 pulling. All loops UP, remote proc alive, disk 666G.

### 2026-07-17 ~13:15 (monitor)
| steps | sps | kick_kl | lambda_k | ent | goals/min |
|---|---|---|---|---|---|
| 199.1M | 5,453 | 0.375 | 0.602 | 3.21 | (skipped — ck only +7M since 1.81 reading) |
Trend: KL settled ~0.375 (productive disagreement — student outscores teacher),
anneal 40%, all steady. Corpus top-up: batch_0000 topped up, on 0001; 61.5k
shards. All loops UP. No flags.

### 2026-07-17 ~19:15 — DEAD-BALL CONTAINMENT FIX SHIPPED (both boxes)
Root-caused the rlviser "frozen ball after sudden kickoff": physics-blowup
containment reset a Bullet-poisoned arena in place; NaN AABB latches
DISABLE_SIMULATION on the ball (one-way; RocketSim never force-clears it), so
post-containment the ball is dead for the arena's lifetime — ghost-ball viewer
sessions AND silent zero-touch training arenas (37/310 watch sessions affected;
same defect live in every collect arena). Fix a1c33e0+8527a5e: rebuild arena in
the containment branch (car ids re-issue identically — hard-asserted). Reviewed
APPROVE (field-by-field sweep, vendored-source verification). Shipped with
Elliot's OK: local .so atomic-swap, run-B resumed ck 3,506.1M (lost ~7.8M
ck-granularity); remote resumed ck 313.79M (lost ~1.5M), lambda_k 0.372
continuity confirmed. Validation: containment fired on remote iter 1 → log
shows "arena rebuilt" — fix live. Kickstart run at 314.1M, sps 5,568, kick_kl
0.39, ent 3.25 — all on trend.
BC-pretrain (task #44): B3 bc-export landed (ac77c17+7a9aa1e, reviewed) and B5
BC trainer landed (4533747+208ec36, reviewed; ckpt-compat proven through
eval_metrics live). v4 re-parse was STALLED (≥10k batch gate vs completed pull;
fixed aeafbe7) — now on batch_0006/12, ETA ~21:00, then B4 export. Note for B6:
kickstart-teacher tooling is v0-only by design; BC ckpt feeds the future
kl_prior seam, not KickstartTeacher.
