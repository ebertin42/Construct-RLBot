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

### 2026-07-18 ~20:30 — CORRECTION: the "terminal" avoidance funk had SELF-RECOVERED
Standard eval of the discarded funk-branch tip (ck 1.38B): **2.14 goals/min,
touches 7.3** — the 0.46-0.79 readings at 608-631M were a trough in an
oscillation that resolved during ~14 unwatched hours. league+v3.1 worked,
noisily. PROCESS LESSON (recorded): single-ck evals on an oscillating run
mislead — two "collapse" calls were cycle troughs; the dashboard's per-ck
auto-eval history (which surfaced this) is the standing fix. COST: K4's
rollback to 562M discarded ~800M steps of a ~2.14 policy (peak 2.25 —
comparable). DECISION (Elliot informed, staying course (a)): anchored run
keeps its ~100M-step window — entropy 1.21 vs funk-era 3.5+ is the
structural difference the anchor was for; ck_001382942720 preserved on both
boxes as fallback (options: un-anchored resume, or re-anchor from 1.38B).
First anchored eval @+11M: 0.62 (expected transition dip; kl_pri 6.9→1.39).

### 2026-07-18 ~19:00 — K4 DEPLOYED: KL-PPO ERA BEGINS (Elliot-approved)
BC-v3 (clean v5 corpus, prev-dropout) B6-v3: probe PASS (copy 0.725≈human
0.70, prev-zeroed top-1 0.441, val top1 62.7) but closed-loop unchanged
(0.29 touches, 0-118 vs 520M) — ball_pred-poisoning hypothesis for closed-
loop failure REFUTED; compounding error is structural per COLT-2025/Seer
precedent (Seer's 820k-replay BC also couldn't play; they lacked the KL
regularizer). DECISION (Elliot): anchor needs conditionals at PPO's states,
not rollout competence — deployed at λ_p 0.05. Remote resumed from
ck_000562083840 (the 2.25 peak) + league + v3.1 + --kl-prior. Verified:
anchored line, kl_pri 6.95@iter1 (large → room to compress), sps warming.
WATCH: kl_pri declining, sps settling ≥6.5k, evals (touches>9, goals≥1.5,
then >2.25); kill-switch = restart without --kl-prior. B1-B6 (#44) and
K1-K4 (#49) COMPLETE — P2 arc closed. Next: reset-pool 0.7 engine task
(levers #1 — aligns PPO states with prior distribution, compounds the
anchor), aux RP head, reward-v4 design pass (zero-sum subtraction).

### 2026-07-19 ~00:45 — B6-v2 preview: FAIL AGAIN — pivot to v5 corpus (BC-v3)
Prev-dropout epoch-1 ck: probe CONFIRMED dropout worked (prev-zeroed acc
0.193→0.451, pred==prev0 0.794→0.755 vs human 0.698, val top1 65.9) — yet
closed-loop STILL dead: 0.31 touches/0.10 goals self-play, and 0-98 vs the
520M peak via league MatchRunner (decisive: not a self-mirror artifact).
Copycat was real but secondary. Prime suspect now: TRAIN/EVAL feature-
distribution mismatch from the 69% poisoned ball_pred corpus — the net
learned pred-slots≈current-ball; live eval feeds HEALTHY predictions (post-
0928e59), so 4 entity slots are off-distribution every tick. AUTONOMY CALL:
killed epoch 2 (same poisoned data, low marginal value); chained: v5 parse
(done soon) → manifest → bc-export --force on v5 corpus (healthy ball_pred +
demoed-skip + kickoff markers) → BC-v3 (prev-dropout, 2 epochs, ~47k/s).
Saves ~4h vs finishing epoch 2 first. K4 NOT deployed (needs a working
prior). Archived: ck_bc_ep00_dropout_poisoned.pt. If BC-v3 also fails
closed-loop → next hypotheses: deeper compounding (needs DAgger-style or
noise-injected BC), or an unfound obs mismatch (recheck B3 parity vs LIVE
engine obs rather than bc-export goldens).

### 2026-07-18 ~17:00 — UNATTENDED EVENING: 3 workstreams landed + v5 re-parse live
Elliot away, full autonomy granted. Landed (all agent-built + reviewed):
(1) shard schema v5 (9b67c67+ce103ca, review APPROVED): is_demoed (col 17,
absent-heuristic; demo-window garbage ang_vel bug found+fixed — pre-v5 those
ticks were silently DROPPED, corrupting live-player coverage too), episode_
marker (dead-ball-run detection), ball_pred divergence assertion restored
(99.95% healthy on fixture post-tracker-fix). In-place v5 re-parse RUNNING
(logs/parse_v5.log, ~7-8h, mixed-schema dir by design). (2) league-tick v1
all-mode (cefaf22, APPROVED) DEPLOYED to remote: first tick rated the v1
pool — avoidance-era frontier ck ranked LAST (-1.96), good-era seeds on top;
PFSP now has real signal. Cutover killed old v0-only loop first. (3) SSL
corpus pull LIVE (0994cc6): ~180/h, 122 replays in, disk-guarded, resumable.
Also: rlviser stopped (nobody watching, was costing BC 12x), BC retrain back
to 17.4k samples/s, loss 2.45@12k batches. Viewer relaunch cmd in memory.

### 2026-07-18 ~16:40 — B6: NO-GO (copycat), prev-dropout retrain launched
Epoch-1 BC ck: val top-1 66.9% / top-3 84.3% (way above Seer 35-50% band —
red flag, not a win) but closed-loop eval near-random: 0.26 touches/min,
0.06 goals/min vs random-init baseline 0.16/0.03. Probe confirmed COPYCAT:
79% of predictions == prev[0] (humans repeat 70%); zeroing the prev ring
collapses accuracy 69.6%→19.3% — ~3/4 of learned "skill" is echoing the
prev-5 input. More epochs can't fix (it's the objective's optimum). FIX
(Elliot-approved): prev-dropout p=0.5 (per-sample zero the prev ring during
training; obs_v1 unchanged), retrain 2 epochs (~20h, prior ready ~mid-day
2026-07-19). Copycat ck kept as ck_bc_ep00_copycat.pt for comparison. K4
deploy waits on B6-v2. Remote left running in avoidance regime (checkpoints
disposable; 562M rollback point intact both boxes).

### 2026-07-18 ~15:00 — ⚠ SECOND DEGENERACY: MUTUAL AVOIDANCE (confirmed)
Post-rollback oscillation resolved downward: 2.25@562 → 2.11@589 → 0.66@608 →
0.61@611 (two consecutive <1.5), touches 2.8, dist 3651, live ep_rew -1.3.
DIFFERENT signature than goal-trading (ep_rew down WITH goals): v3.1 made
goal exchange negative-sum (-2/trade), and the self-play equilibrium of a
negative-sum game is mutual avoidance; league counters in only ~20% of
arenas. DECISION (with Elliot): no third reward surgery — deploy KL-prior
(K4) immediately after B6: the human anchor is the structural anti-avoidance
fix (λ_p tunable upward if 0.05 too gentle). BC at 90%, epoch-1 ck ~15:20.
Day's tally of degeneracy modes on reward_v3.x self-play: trading (+10/-8),
avoidance (+10/-12) — the exploit-free reward likely doesn't exist without a
behavioral prior; that's the whole KL-PPO thesis, now empirically motivated.

### 2026-07-18 ~12:30 (monitor) — RECOVERY CONFIRMED, NEW HIGH
| steps | sps | ent | ep_rew | goals/min |
|---|---|---|---|---|
| 563.3M | 7,479 | 3.55 | 0.61 | **2.25** (ck 562M, all-time high) |
Post-rollback trajectory: 1.19@548M (adaptation dip while critic re-learned
v3.1 values) → 2.25@562M. Touches 4.4→8.3. Turtling refuted; league+v3.1
outperforms the pre-exploit line (peak was 1.90). ep_rew ~0.6 is the honest
scale now (negative-sum goal exchange). BC epoch 0 at 73%, epoch-1 ck ~14:30
→ B6. KL-prior plan written (bccc2e9, task #49) — K1-K3 code next, K4 deploy
after B6 + approval. Exploit-era cks ≥554M quarantined both boxes; viewer
picks by mtime now (91f4472).

### 2026-07-18 ~10:20 — FIX DEPLOYED (Elliot-approved): rollback + league + v3.1
Elliot back, approved "League + reward v3.1". Executed: reward_v3_1.toml
(aggression_bias -0.2 → concede -12 vs goal +10; any trade now net -2 — loop
strictly unprofitable; commit 72b36b5) shipped to remote; league registry
seeded with v1 cks 320M/520M/550M; remote restarted from ck_000520765440
(the 1.90 peak). Verified: "resumed at 520,765,440", league opponents line
lists all 3 seeds, ep_rew instantly 3.0-5.0 (trading income gone), sps ~7.6k
warming. 520→659M exploit-era steps discarded (~4.7h GPU). Watch next: eval
trend recovery toward 1.8-1.9, ep_rew staying in 3-6 band, ent decline;
league_tick on remote still v0-loop — v1 ladder ratings TODO if PFSP weighting
matters. Viewer stays off until BC done (Elliot-approved).

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

### 2026-07-18 ~23:30 — prior upgraded to ep01 (Elliot-approved)
BC-v3 finished 2 epochs: ep01 probe state-only 0.492 (vs ep00 0.441), copy
0.718. Remote eval trend pre-swap: 0.62→0.85→0.78→0.91 (+83M) — climbing, no
λ_p change warranted. Swapped anchor to ep01 via single restart from ck
644.7M: kl_pri 0.47 at resume (no shock), ep_rew 1.96, ent 1.23. Ops: ctl.py
unified CLI landed (ccde06d, 118 tests) — first live status run surfaced the
run-B-prune/league-registry dead refs (cleaned 17→6) and the ep_rew recovery.
Dashboard: multi-run panels + hover datetime tooltips (3ddc53a, 021da44).

### 2026-07-19 ~00:45 (monitor) — anchored run accelerating
| steps | sps | ent | kl_pri | ep_rew | goals/min |
|---|---|---|---|---|---|
| 697M | 4,913 | 1.24 | 0.39 | 1.4 | **1.16** (ck 694M) |
Post-anchor trend: 0.62→0.85→0.78→0.91→1.16; steepest segment is post-ep01-
swap (+0.25/50M). 135M steps degeneracy-free (record for v3.x rewards),
entropy/kl_pri stable. Verdict upgraded: trajectory now competitive, not
just stability. Watch: ~1.5 overnight; 2.14 fallback mark within ~a day at
slope. ctl.py viewer-on bug found+fixed live (missing CONSTRUCT_VISER_ADDR,
869bc23). BC-v3 closed: ep01 val top1 64.2/top3 84.4/jump-recall 51.6.

### 2026-07-19 ~04:45 — kickoff jitter shipped (both boxes)
Anchored policy's textbook symmetric kickoffs manufactured the RocketSim
pinch blowup nearly EVERY kickoff (450 containments remote; viewer stuck in
kickoff-restart loop — Elliot spotted it live). Fix: ±50uu/±0.09rad per-car
kickoff spawn jitter from the existing episode rng (engine commit, bounds+
determinism tests; A/B pinch test documented as physics-chaos-limited).
Shipped: local .so swap + remote resume 741.1M (ep01 anchor intact, kl_pri
0.40). Containment baseline 450 — delta watch. Side-benefit expected:
kickoff-state variety. Run trend pre-ship: 1.16@694M, ep_rew to 5.7@resume.

### 2026-07-19 ~05:40 (monitor) — jitter VERDICT: WIN
Containments frozen at 450 (zero new in 35min/+9.4M steps; pre-fix pace was
hundreds/hr). Viewer kickoff-loop gone. ep_rew ~2.4 sustained, kl_pri 0.37,
ent 1.21. The synthetic A/B test's chaos-limits didn't matter — production
delta was the real experiment, and it's unambiguous.

### 2026-07-19 ~06:30 — jitter v2: REAL win (ball offset was the key)
Elliot caught the confound: ±50uu car jitter "win" was masked by remote's
60%-random curriculum; the viewer (100% kickoffs) still pinched every
episode — the anchored policy course-corrects and re-converges. v2: cars
±150uu/±10°, BALL ±10uu horizontal (contact geometry can't mirror). Shipped
both boxes; VIEWER delta over 12min: 0 containments (pre-fix 20+). Lesson
logged twice tonight: always identify the subpopulation where the bug
actually fires and measure THERE.

### 2026-07-19 ~14:50 — KL-ANCHOR KILL-SWITCH EXECUTED (head-to-head evidence)
Self-play goals/min had the anchored run oscillating 0.62-1.46 and I read it
as "climbing out of the dip". Head-to-head (MatchRunner, 5400 steps, sides
swapped) says otherwise — DECISIVE:
| matchup | result |
|---|---|
| anchored 909M vs unanchored 1.38B | 19-60 / 16-59 (swapped) |
| anchored 909M vs 562M peak (its own start) | 22-77 |
| **1.38B unanchored vs 562M peak** | **36-59 / 30-79 (swapped)** |
So λ_p=0.05 with this prior made the policy ~3.5x weaker over 348M steps —
the "human defense suppresses self-play goals" confound is REFUTED. Executed
the pre-agreed kill-switch: remote resumed from ck_000562083840 (strongest
policy we own), league + v3.1 + jitter-v2, NO anchor; toml [kl_prior]
re-commented so restarts can't silently re-enable. sps 4.9k→6.8k.
BIGGER FINDING: 562M beats the 1.38B unanchored tip too — BOTH post-562M
regimes degraded the policy (anchored severely, unanchored mildly over 800M
steps). 562M is the high-water mark of the whole project.
METRIC LESSON (supersedes the 07-18 20:30 one): self-play goals/min is not a
skill metric — it moves with both sides' defense and hid an 800M-step
regression. Head-to-head vs frozen reference checkpoints is the real ruler;
the league ladder already computes exactly this and should be the primary
signal, with #54 (external bots) as the absolute anchor.
HYPOTHESIS FOR ELLIOT (needs his call): reward_v3.1's asymmetry (concede -12
vs goal +10) rewards risk-aversion; over hundreds of M steps it may breed
passivity — matching v3's trading exploit and v3.1's avoidance equilibrium as
two faces of hand-tuned asymmetry. Research-backed alternative (Seer /
Lucy-SKG, levers doc #3): symmetric goal/concede made zero-sum by subtracting
the opponent's reward, annealed goal weight, multiplicative touch decay. That
is the next regime change I'd propose — NOT another anchor variant.
Anchor post-mortem: it did deliver stability (zero degeneracies in 348M steps,
entropy 1.2 vs 3.5) — the machinery works; λ_p=0.05 with a GC2 prior is just
too strong a leash for a policy already above GC2 level. If retried: λ_p 0.01,
or anneal it in, or anchor to our own best checkpoint instead of humans.

### 2026-07-19 ~15:10 — REWARD V4 DEPLOYED (Elliot-approved): symmetric zero-sum
v4 = goal 10.0 / aggression_bias 0.0 (concede exactly -10) + opp_spirit 1.0
(full opponent subtraction). Engine test pins the point: a trade cycle nets
+2.0 under v3 and EXACTLY 0.0 under v4; a goal event is ±20 after blending.
Deferred (need trainer scheduling that doesn't exist): annealed goal weight,
touch decay (moot at touch=0). Deployed from ck_000562083840 (high-water
mark) + league + jitter-v2 + --reset-optimizer (goal scale doubled, old
value head miscalibrated). First iters: sps 6.5-7.5k (anchor tax gone),
v_loss 2.06->1.36 recalibrating, ent 3.5.
READ ep_rew DIFFERENTLY NOW: under zero-sum it is net advantage vs the
opponent, not shaped income — negative early vs league opponents is expected
and it is structurally unfarmable (trading nets 0). The skill signal is
h2h share vs frozen ck_000562083840, not ep_rew and not self-play goals/min.
Viewer also changed: streams the LIVE run only, cycling 1v1/2v2/3v3 in the
trained 5/3/2 mix (3fd2dfe) — retired lineages dropped.

### 2026-07-19 ~15:40 — v4 -> v4.1 (Elliot-approved): full zero-sum was a mistake
First h2h reading of v4 (the new harness, 33bf03f): ck 564.8M scored 26-49 /
20-51 swapped = 31.5% share vs the frozen 562M peak after only 2.75M steps,
with entropy RISING 3.53->3.83 and ep_rew sinking to -5.
ARITHMETIC FLAW FOUND: in 1v1 the blend reduces to r_i' = r_i - opp_spirit*r_j
(team_spirit is a no-op with one teammate). At opp_spirit=1.0, two cars
chasing the ball symmetrically have their vel_to_ball terms CANCEL EXACTLY —
v4 deleted its own anti-degenerate engagement pull and left a near-sparse
goals-only signal. Rising entropy = policy diffusing without a dense gradient.
KEY INSIGHT: full zero-sum was never needed to kill trading. SYMMETRY does it
— with aggression_bias = 0.0 a trade cycle nets exactly 0.0 at ANY opp_spirit
(v4.1: blue scores +13/-13, orange scores -13/+13, cycle = 0).
v4.1 = symmetric goal (the real fix) + opp_spirit 0.3 (v3-era), keeping ~70%
of the shaping pull and a goal scale (+-13) close to what the resumed value
head knows (v3.1 was +13/-15.6) — so NO --reset-optimizer this time, which
also removes the biggest confound in reading the transient.
First iter: ep_rew +0.79 (v4 was -1.17), v_loss 1.71, ent 3.54, sps 6.7k.
Measurement discipline: auto-h2h fires at ~592M (+30M) vs the 562M peak.
PROCESS NOTE: v4 lived ~30 min. Cheap because the h2h harness caught it in one
reading — under the old self-play metric this would have looked like the usual
"transition dip" and burned a day.

### 2026-07-19 ~16:10 (monitor) — v4.1 holds parity; v4 diagnosis confirmed
h2h vs frozen peak-562M (both side orders, 5400 steps/side):
| regime | ck | share |
|---|---|---|
| v4 (opp_spirit 1.0) | 564.8M (+2.75M) | 31.5% (26-49 / 20-51) |
| **v4.1 (opp_spirit 0.3)** | **575.9M (+13.8M)** | **49.5% (47-46 / 47-50)** |
Parity at 5x the step count where v4 had already collapsed — the
shaping-cancellation diagnosis is confirmed, and v4.1's health markers agree
(ep_rew 0.79->1.58->1.80 rising vs v4's slide to -7; ent stable 3.61 vs v4's
climb; v_loss 0.58). Parity is the CORRECT baseline here (it is the peak +
13.8M steps); the open question is whether it climbs >50% over ~100M, which
is exactly what v3.1 failed to do (it silently lost share over 800M steps).
CONTAINMENTS, honest note: ~275/hr this era vs 43/hr during the anchored era
— jitter-v2 has NOT eliminated pinches for AGGRESSIVE policies (the anchored
policy was passive, which is why the viewer read 0/12min). Distribution is
~96% within 800 ticks of an arena rebuild (cascade-shaped). Rate works out to
~1.4/arena/hr against ~120 episodes/arena/hr = ~1.2% of episodes truncated.
Accepted: containment is safe (rebuild), and the alternatives are worse
(bigger jitter distorts kickoff realism; the real bug is upstream RocketSim).
Escalate only if it exceeds ~5% of episodes.
Also closed: #53 goldens were stale-not-broken — and the check that mattered
came back clean, deploy/bot.py + deploy/obs.py already encode demoed cars the
way obs_v1.rs does, so real-game play was never affected (4334e76).
