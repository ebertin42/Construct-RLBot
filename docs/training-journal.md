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
