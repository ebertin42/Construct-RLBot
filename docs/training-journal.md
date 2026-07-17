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

### 2026-07-18 ~09:40 — ⚠⚠ MECHANISM IDENTIFIED: GOAL-TRADING LOOP
ck 631M → 0.46 goals/min while live ep_rew EXPLODED to 14.3 (2.5x teacher).
Arithmetic gives the exploit: reward_v3 goal +10 / concede -8 → alternating
goals net +2 per agent per exchange. Self-play partners learned COOPERATIVE
GOAL-TRADING (positive-sum loop); vs a non-cooperating eval opponent the
policy looks like camping with 0.46 goals/min. Explains touches ~4 (kickoff
exchanges only), v_loss 0.19 (very predictable income), ent oscillation.
FIXES: (immediate, staged) league opponents break the loop — past selves
don't cooperate; (structural, for Elliot) reward_v3.1 with |concede| ≥ goal
(e.g. -10/-12) makes trading zero/negative-sum — recommend BOTH. Staged fix
unchanged: rollback ck 520M + --league. Awaiting approval.

### 2026-07-18 ~09:00 — ⚠⚠ COLLAPSE ACCELERATING
ck 614M → **0.79 goals/min** (trend 1.90→1.80→1.29→1.21→1.45→0.79), touches
3.2, dist 3123. The 1.45 bounce was noise. ent falling 3.82→3.51 = policy
SHARPENING INTO the camping equilibrium (not recovery); ep_rew still 6.8 —
reward/goal divergence total. Fix attempts still classifier-blocked (both the
full fix and the additive registry-seed alone). Waiting on Elliot: reply "go"
→ I execute journal-f2a89e2 fix (rollback 520M + league). Every hour ≈ 29M
steps deeper into the degenerate equilibrium (discarded on rollback anyway).

### 2026-07-18 ~08:45 — ⚠ REGRESSION CONFIRMED, awaiting approval for fix
Second consecutive sub-1.35: ck 584M → 1.21 goals/min (1.90→1.80→1.29→1.21),
touches ~4.8, ep_rew rising — self-play camping degeneracy post-anneal.
PREPARED FIX (blocked on Elliot approval — permission layer would not let the
overnight session mutate the remote box): (1) seed remote league/registry.jsonl
with v1 entries ck 320M/520M/550M (schema_version=1), (2) restart remote from
ck_000520765440 (the 1.90 peak) with `resume_train.py
checkpoints_entity/ck_000520765440.pt --config configs/train_v1.toml --league`
(v1-native league; opponent_frac 0.2 default; entropy unchanged — one variable
at a time; entropy_coef 0.01→0.005 remains the fallback if league alone
doesn't hold). Drifted steps 520→590M+ get discarded by rollback; cost of
waiting = GPU-hours only. Monitoring continues hourly, documentation-only.

### 2026-07-18 ~07:45 (monitor) — ⚠ POST-ANNEAL DRIFT WARNING
| steps | sps | ent | ep_rew | goals/min |
|---|---|---|---|---|
| 582.7M | 8,095 | 3.77 | **7.26 ↑** | **1.29 ↓** (ck 580M) |
First sub-1.35 reading, WITH the signature: ep_rew rising (5.7→7.3) while
goals fell (1.90→1.80→1.29), touches collapsing monotonically
(15.2→10.5→9.2→4.7), mean ball dist 1452→2715 (agent hanging back). v_loss
0.35 (critic confident in whatever it's doing). One reading below line —
confirmation eval armed on next synced ck. If confirmed: rollback to ck 520M
(1.90 peak) + change needed to break the loop — candidates: entropy_coef
0.01→0.005 (ent climbed 3.26→3.77 post-anneal), league re-enable (v1-native
registry landed in league-v1 merge — self-play degeneracy is the likely
mechanism; both selves camping = no touches, and ep_rew can rise on
vel_to_ball income + own-goal asymmetries without finishing).

### 2026-07-18 ~05:45 (monitor)
Post-anneal reading 1: ck 520M → **1.90 goals/min** (new high; teacher 1.56,
prior peak 1.81). Touches 15.2→10.5/min, dist 1452→1882 — fewer but more
decisive possessions, goals up: efficiency, NOT the hacking signature. Remote
sps 5.5k→8.1k post-anneal (teacher forward gone); log dropped kick_kl/lambda_k
fields as expected. ent 3.67 elevated — watch decline. BC epoch 0 at 32k/177k
batches, loss 1.94. No flags.

### 2026-07-18 ~05:00 (monitor)
| steps | sps | kick_kl | lambda_k | ent | goals/min |
|---|---|---|---|---|---|
| 495.7M | 5,465 | 0.74-0.78 | 0.009 | 3.58-3.63 | 1.44 (ck 494M) |
Anneal effectively done. kick_kl rising into the end (0.40→0.77) + ent up
3.26→3.63 — teacher pull gone, student exploring; expected, KL now vestigial.
Eval 1.44: down from 1.79@320M but inside the ±0.2 noise band and above the
1.35 regression line — single reading, not flagged. Post-anneal watch armed:
two consecutive <1.35 = rollback proposal. BC epoch 0: batch 23.7k/177k,
loss 2.00, 19.2k samples/s. run-B 45k sps. Disk 191.6G host-free. All UP.

### 2026-07-18 ~03:50 — GPU MYSTERY SOLVED: rlviser renders on the training GPU
BC train hit a cliff at 02:41 (20k→2k samples/s; GPU "100%", clocks/temps
normal, zero IO pressure — looked like a loader problem, wasn't). Proven by
kill-test: stopping rlviser.exe instantly restored 21k samples/s. rlviser
renders on the same RTX 4060 WSL CUDA uses; an active-play scene costs ~10x
trainer throughput (cliff = rotation to active segment). Viewer stack paused
overnight (relay stays up); memory updated with the diagnostic signature.
Side finding: scripts/bc_train.py lacks a __main__ guard — an import executed
a second full training run during diagnosis (killed; add guard = todo).
Overnight state: BC 18.9k samples/s (epoch 1 ends ~13:00), run-B restarted
and at 50k sps (best since morning — rlviser was throttling it too), remote
kickstart anneal ends ~04:30. loader prefetch fix 72e38f5 (10x on its own:
0.47→5 batch/s).

### 2026-07-18 ~01:30 — B4 CORPUS EXPORT COMPLETE (+ disk crash postmortem)
v4 re-parse: 67,568 replays, 0 failures, 233G, manifest 381.7M ticks all-1v1.
bc-export: 67,568/67,568 shards → 163G COMPRESSED npz (plan's 180 B/sample
estimate was 11x low — real 2,089 B/sample f32 = 1.6TB; deflate 9x on masked
entity slots saved it, commit 61ea406), 704,498,136 samples, 0 export failures.
OPS: WSL crashed mid-export — vhdx filled the Windows host disk (WSL df lies;
real host free ~180G). Cleanup freed ~105G (v3 shards 85G superseded, uv/HF
caches 20G); host-disk guard monitor added (kills export <100G); run-B lost two
0-byte cks to the crash (quarantined, resumed from 3.5747B intact). Crash also
proved the bc-export fsync gap: truncated-but-renamed bc npz in the pre-crash
cohort — scanned/deleted/re-exported. vhdx compaction recommended to Elliot
(wsl --set-sparse) at next downtime. BC TRAINING (B5 run) starting: 4 epochs,
batch 4096, weighted CE, RTX 4060 (sharing with run-B learner).

### 2026-07-17 ~19:45 (monitor)
| steps | sps | kick_kl | lambda_k | ent | goals/min |
|---|---|---|---|---|---|
| 321.8M | 5,467 | 0.40 | 0.357 | 3.26 | **1.79** (ck 320M) |
Trend: back above teacher (1.56) after the 1.40 dip; KL 0.37→0.40 post-restart
wobble, inside noise. NOTE: this eval logged 4 contained blowups, all "arena
rebuilt" — pre-fix, those arenas ran dead-ball for the rest of the tape,
meaning EARLIER EVAL READINGS (1.71 / 1.40 / 1.81) carried dead-ball
depression noise; the 1.40 "dip" was likely artifact. Expect eval variance to
tighten from here. Run-B resumed on fixed engine, 3.528B (sps depressed by
parse sweep, recovers ~21:00). Re-parse batch_0008/12, ETA ~21:00 → B4.
