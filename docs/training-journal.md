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

### 2026-07-19 ~17:00 — REPLAY-STATE RESET LEVER LANDED (inert; ultracode workflow)
Levers doc #1 implemented via a 9-agent workflow (understand x3 -> spec ->
implement -> adversarial verify x3 -> fix): engine/src/reset_pool.rs (loader,
filters, reservoir, quat->rotmat port), curriculum.rs config surface,
episode.rs three-way ResetKind draw + apply_replay_state, configs/
curriculum_v2.toml. Commits 96cd7d0 + f600fc9.
Mix: 0.70 replay / 0.10 kickoff / 0.20 random (the spec's 0.15 scenario-pack
share folded into random — we have no scenario packs). 2v2/3v3 renormalize
over {kickoff, random} since the pool is duels-only. Real-corpus production
load: 1,081,088 read -> 891,720 kept (82.5%), 132.7 MB shared via Arc, 2.09 s.
ALL THREE REVIEWERS said FIX_FIRST (7 blocking, 7 minor); 6+6 applied. The
two most valuable catches were things I would not have found by reading:
- F2's goal-band used |ball.y| > 5120, but step() runs 8 ticks BEFORE the
  policy acts — a replayed near-goal state could score on tick 0. Band is now
  4800 (= BALL_MAX_SPEED*8/120 + margin), making a first-step goal
  ARITHMETICALLY unreachable, pinned by a const assert the fixer verified
  fires by reverting it. Cost 5% of corpus.
- No ang_vel clamp: real corpus states drove car ang_vel to 96.5 rad/s vs
  RocketSim's 5.5 cap (obs features ~17x their design range). Post-filter max
  is 5.78.
Rejected finding (correctly): a ballpred test failure blamed on this commit —
git show --stat proves the commit touches neither ballpred.rs nor its test,
and it passes 1-in-3 in isolation. Documented flake, not a regression.
Documented-not-fixed -> task #57: pool lacks has_flip + pad cooldowns (43% of
corpus cars are airborne). Self-consistent within the sim, so not a bug, but
it diverges from bc_obs.rs's real values — matters only when a BC prior is
re-enabled. Needs replay/ changes + re-parse.
STATUS: INERT BY DESIGN. train_v1.toml still points at curriculum_v1.toml;
curriculum_v1.toml untouched. Deploy is a deliberate one-line operator switch,
held until v4.1 has a second h2h reading >=50% (attribution discipline —
never stack two regime changes on an unmeasured one again).

### 2026-07-19 ~17:40 — ⚠⚠ MAJOR CORRECTION: PPO SELF-PLAY HAS NEVER IMPROVED THIS POLICY
I was measuring against the wrong reference. Transitivity test (both sides vs
a common third party) demolishes the "562M is the high-water mark" claim in
the 14:50 entry — RETRACTED:
| matchup (both side orders, 5400 steps/side) | share |
|---|---|
| v4.1-592M vs early-320M | **10.8%** (17-126 / 12-114) |
| peak-562M vs early-320M | **18.5%** (26-90 / 19-108) |
562M was never a peak — only the least-degraded of a declining family.
Kickstart-era ladder tournament (3600 steps/side, all pairs):
| | 100M | 200M | 320M | 440M |
|---|---|---|---|---|
| share | ~48 | ~52 | ~48 | ~52 |
FLAT. 100M ≈ 200M ≈ 320M ≈ 440M, all within 43-54% of each other, and the
whole family beats everything post-500M by 80-90%.
CONCLUSION: the entity policy reached the kickstart teacher's level early via
DISTILLATION, plateaued there for 340M steps, and then decayed once the anneal
removed the anchor at 500M — under reward v3, v3.1, the BC anchor, v4, and
v4.1 alike. PPO self-play has never demonstrably improved it. The reward-design
debate of the last two days was arguing about the wrong variable.
NOT a PPO coding bug: advantages are normalized (ppo.py:42), loss assembly is
standard, and pi_loss~0.002 is EXPECTED with mean-zero normalized advantages
(not a gradient-magnitude signal).
Entropy is high (3.5-3.8 of ln(92)=4.52 max ~ 80% of uniform) but was ALSO
high during the good era (3.2-3.3) — so sharpness is not the differentiator;
the DISTRIBUTION SHAPE was, and the teacher supplied it.
METRIC LESSON #3 (compounding on the previous two): a single frozen reference
is not a ruler. Self-play goals/min misled in one direction; a lone h2h
reference misled in the other. Non-transitivity is real in self-play spaces —
ALWAYS test both candidates against a common third party, and prefer a diverse
panel + the league ladder. Note self-play goals/min was ANTI-correlated here:
562M scored 2.25 (weak defense inflates both sides) vs 320M's 1.79, while
320M wins the actual match 81.5%.

### 2026-07-19 ~20:00 — DIAGNOSIS: the policy is going STATE-BLIND (entropy bonus)
Elliot called "diagnose before restarting". Built scripts/diagnose_ppo.py
(84d1a8b, 48 tests) — offline, read-only, no training.
RULED OUT, cleanly:
- Importance ratio: mean 1.000000, max |logp delta| 4.7e-5, 0 of 27,648
  samples outside [0.9,1.1]. The candle(engine)/torch(learner) seam is exact;
  every PPO update has applied correct importance weights. (My leading
  hypothesis — that only kickstart worked because its KL term bypasses the
  ratio — is REFUTED.)
- Value head: ev 0.88/0.89, corr 0.94, scale calibrated, bias ~0. Advantages
  are real signal. Stricter ev_mc (lam=1) 0.47/0.55 — mediocre, not broken.
- Advantages: std 1.0-1.3, ZERO degenerate entries. Weight sync verified
  earlier (train.py collect() calls set_weights before every rollout).
THE FINDING — I(S;A), mutual information between state and sampled action,
measured on IDENTICAL states (8192 rows from one shared rollout):
| policy | E[H(A|s)] | I(S;A) | % of ln(92) |
|---|---|---|---|
| RL strong 320M | 3.208 | 0.837 | 18.5% |
| RL degraded 592M | 3.605 | 0.486 | 10.8% |
| **BC prior (same net/obs)** | **1.710** | **1.525** | **33.7%** |
The BC prior is 3x more state-dependent than the degraded RL policy on the
same architecture and inputs, so the net is fully CAPABLE of conditioning —
RL training is what erodes it. The 320M->592M decay is a 42% loss of
state-dependence with per-state entropy rising in lockstep, matching the h2h
skill collapse exactly. ~86-89% of the RL policy's output entropy is
unconditioned: it takes 56-61 of 92 actions to cover 90% of sampled mass.
MECHANISM: entropy_coef=0.01 rewards H(A|s), and the cheapest way to maximize
per-state entropy is to IGNORE THE STATE. That pull is small but SYSTEMATIC on
every minibatch, while the policy-gradient term is zero-mean by construction
(normalized advantages) and noise-dominated at 3.5 nats of sampling entropy
over 92 actions. A consistent small bias beats large zero-mean noise over
millions of updates. Kickstart's KL-to-teacher was a large SYSTEMATIC
state-conditioned counter-force — which is why the only era this project ever
held skill was the distillation era, and why decay began the moment the anneal
zeroed that term at 500M. Four reward redesigns never touched this variable.
NEXT (needs Elliot's go — it is a training run): controlled A/B from 320M,
entropy_coef 0.01 vs 0.001, ~20M steps each, measuring I(S;A) (diagnose_ppo)
and h2h vs frozen 320M. Prediction: the low-entropy arm holds or gains
state-dependence and h2h share; the 0.01 arm repeats the decay.

### 2026-07-19 ~21:50 — A/B RESULT: prediction FALSIFIED (mechanism real, not the cause)
Bounded A/B from ck_000320471040, seed 777, reward v4.1, league OFF, 145 iters
(~23M steps) per arm, identical but for entropy_coef.
| | I(S;A) on identical states | h2h vs frozen 320M (both orders) |
|---|---|---|
| baseline 320M | 0.876 (19.4%) | — |
| arm A ent=0.01 | 0.483 (10.7%) | 25.5% (28-70 / 28-94) |
| arm B ent=0.001 | **1.122 (24.8%)** | **10.4%** (13-105 / 10-94) |
CONFIRMED: entropy_coef directly controls state-dependence — arm A lost 45%
of I(S;A), arm B gained 28%, from the same start in the same steps. The
state-blindness mechanism from the 20:00 entry is real.
FALSIFIED: it is NOT why the policy is weak. BOTH arms lost badly, and the
SHARPER, more state-dependent arm lost WORSE (10.4% vs 25.5%). Sharpening
commits harder to whatever direction PPO is pushing — so the gradient
DIRECTION is the problem, not the exploration level. My prediction on record
was wrong; recording it as such.
ALSO REFUTED (my next guess, killed by measurement before I acted on it):
"dense vel_to_ball shaping outweighs goals". Measured contribution per
episode — v4.1: goals 12.13 vs dense 6.61 (0.55:1); v3: 5.85 vs 5.47
(0.94:1). Goals dominate in v4.1. The reward is not shaping-drowned.
WHAT THE DATA NOW POINTS AT — and I introduced the confound myself: both arms
ran with LEAGUE OFF (I disabled it to remove opponent variance). ep_rew ROSE
in both (5.13, 6.06) while h2h vs a FIXED opponent FELL. That is the textbook
self-play cycling signature: improving against a co-degrading mirror while
losing absolute skill. It also fits the whole project history — the ONLY
stable era was kickstart, i.e. anchored to a FIXED COMPETENT external teacher;
every unanchored era (league at opponent_frac 0.2, so still 80% pure
self-play) decayed; and the BC anchor failed because that anchor was fixed but
INCOMPETENT. AlphaStar's ablations say exactly this: self-play alone forgets
and cycles; PFSP + exploiters are what fix it (levers doc #5).
NEXT PROPOSED (needs Elliot's go — arm C): 20M steps from 320M with league ON
at HIGH opponent_frac (~0.5) against a pool seeded with the equal-strength
kickstart-era cks (100M/200M/320M/440M), entropy_coef 0.003 (between the
arms). If h2h holds >=45%, opponent diversity is the missing ingredient and
the roadmap is league/exploiter work (levers #5/#6), not more reward design.

### 2026-07-19 ~22:00 — h2h NULL CONTROL passes (methodology validated)
320M vs ITSELF, both side orders: 39-49 / 52-38 -> 51.1% share (expect 50%).
The pipeline is unbiased to ~1% when both orders are SUMMED. Note the per-side
swing (~±6 goals from parity, ~10-goal spread between orders): a single-order
h2h would be unreliable, which is exactly why the harness always plays both.
Noise floor established => every effect in this journal (10.4 / 18.5 / 25.5 /
81.5%) is far outside it. All prior conclusions stand, including the
tournament that retracted the "562M is the high-water mark" claim.

### 2026-07-19 ~23:30 — ARM C: league does NOT rescue it either
Four-way, all from ck_000320471040, 145 iters, seed 777, measured identically
(I(S;A) on one shared rollout; h2h both side orders vs frozen 320M):
| arm | config | I(S;A) | h2h share |
|---|---|---|---|
| baseline | (no training) | 0.867 / 19.2% | — |
| A | ent 0.01, league OFF | 0.481 / 10.6% | 22.0-25.5% |
| B | ent 0.001, league OFF | 1.114 / 24.6% | 10.4-11.4% |
| C | ent 0.003, league ON 0.5, clean strong pool | 0.906 / 20.0% | **27.1%** |
Null control (320M vs itself) = 51.1%, rerun variance ~3%. So arm C is NOT
meaningfully better than arm A: a diverse pool of four equal-strength
kickstart-era opponents at HALF the arenas did not stop the decay. Branch (c)
of the pre-registered rule: self-play cycling is not the explanation either.
CONFOUND I INTRODUCED, stated plainly: all three arms ALSO switched reward
(v3 -> v4.1) relative to what 320M was trained under, and none had the
kickstart anchor. So A/B/C measured "change the regime and train 20M steps"
against "do not train at all" — they cannot separate "PPO is destructive" from
"switching regimes is destructive".
ARM D (running): 320M's OWN original regime — reward v3, kickstart teacher
checkpoints/ck_004956252160.pt at lambda_k 0.36 (its real value at 320M),
entropy 0.01, league off, same 145 iters/seed. Verified live: kick_kl 0.3546.
PREDICTIONS ON RECORD before the result: if D holds ~45-55%, then PPO+anchor
is stable and the destructive ingredient is regime-change/anchor-removal, so
the fix is "never train unanchored" (self-distillation from the current best,
periodically refreshed). If D ALSO collapses to 10-30%, then 20M steps of this
PPO loop degrades the policy under EVERY configuration including its own
native one — which would mean the loop cannot sustain skill at all and the
remaining suspects are structural (92-action table expressiveness, obs
conditioning, tick_skip 8 credit assignment).

### 2026-07-20 ~00:30 — ARM D HOLDS PARITY: the anchor is (probably) the whole story
| arm | config | I(S;A) | h2h vs frozen 320M |
|---|---|---|---|
| baseline | (no training) | 0.867 / 19.2% | — (null control 51.1%) |
| A | ent .01, no league, v4.1 | 0.481 / 10.6% | 22.0-25.5% |
| B | ent .001, no league, v4.1 | 1.114 / 24.6% | 10.4-11.4% |
| C | ent .003, league 0.5 clean pool, v4.1 | 0.906 / 20.0% | 27.1% |
| **D** | ent .01, no league, **v3 + KICKSTART ANCHOR** | 0.905 / 20.0% | **49.2%** (50-43 / 43-53) |
All five identical otherwise: from ck_000320471040, 145 iters, seed 777, same
measurement. With a competent anchor the policy KEEPS its skill through 20M
steps; without one it loses 73-89% of goals to its own starting point. Entropy
and opponent diversity move I(S;A) around but do not save it.
CAVEAT I AM TESTING RATHER THAN ASSUMING: D differs from A in TWO ways —
anchor ON and reward v3 (not v4.1). ARM E now running to isolate: v3, NO
anchor, everything else identical to D. This completes a 2x2:
  A = v4.1/no-anchor 22-25%   D = v3/anchor 49.2%   E = v3/no-anchor = ?
If E lands ~25% -> the ANCHOR is causal and reward choice is a side issue.
If E lands ~49% -> the REWARD SWITCH (v3->v4.1) did the damage, not the anchor,
and the last two days of reward work were actively harmful rather than neutral.
STRUCTURAL SUSPECT (i) REFUTED separately: the 92-action table is expressive
enough — the BC prior reaches human behavior through it (17.9% mass on plain
throttle, 11.7% on throttle+handbrake). But the repertoires diverge sharply:
boost 0.156 (human) vs 0.701 (RL), handbrake 0.340 vs 0.109, air-control 0.564
vs 0.907; only 16-26% repertoire overlap. Our RL policies are boost-mashers.
That is a symptom of what PPO optimizes toward here, not a limit of the table.

### 2026-07-20 ~00:50 — CHAMPION GATING built (Elliot's idea, #59, commit 928d598)
Given that training is destructive on average (arms A/B/C: 10.4-27.1% vs their
own start), stop trusting it: a candidate must WIN head-to-head to become the
champion, and the champion is what serves as KL anchor, league-pool seed, and
deploy candidate. PPO becomes a mutation operator inside a hill-climb that
cannot regress. scripts/champion_gate.py: status | gate CK [--promote-if-pass]
| watch [--auto-promote] | promote/reject CK --reason (MANUAL override, Elliot
hand-picking, recorded with manual=true + reason so the audit trail separates
measured wins from judgment calls). Threshold 0.52, both side orders MANDATORY
(single-order swings ~±6 goals), min_total_goals 20 so a 3-0 shutout cannot
promote on 100% share, atomic tmp+rename pointer, promotion also appends the
new champion to the league pool. History is a superset of the h2h schema so
the dashboard plots gate results unchanged. 54 tests.
ACCEPTANCE TEST (the one that matters): gated arm A — a candidate MEASURED at
22-25% — result 19.8% FAIL, champion pointer verified unchanged. A gate that
cannot reject a known-bad candidate would be worse than no gate.
Champion initialized to ck_000320471040 (measured strongest across the whole
project; beats the former "peak" 562M by 81.5%).

### 2026-07-20 ~01:20 — 2x2 COMPLETE: THE ANCHOR IS CAUSAL
All arms: from ck_000320471040, 145 iters, seed 777, gated/measured identically.
| | no anchor | with anchor |
|---|---|---|
| reward v4.1 | 22.0-25.5% (A) | — |
| reward v3 | **29.6% (E)** | **49.2% (D)** |
Holding reward FIXED at v3: anchor takes 29.6% -> 49.2% (+19.6 points).
Holding anchor FIXED at none: v4.1 -> v3 gives 22-25% -> 29.6% (+~5 points).
ANCHOR IS THE DOMINANT CAUSAL FACTOR. Reward design is a minor term. The
entire reward debate of 2026-07-18/19 (v3 -> v3.1 -> v4 -> v4.1) was tuning a
~5-point variable while the ~20-point one sat at zero after the anneal ended.
Also settled: entropy_coef moves I(S;A) but not skill (A vs B); a clean
4-checkpoint league pool at opponent_frac 0.5 does not rescue it (C 27.1%);
the 92-action table is expressive enough (BC reaches human behavior through
it); PPO plumbing is exact (ratio 1.000000, ev 0.88).
PRODUCTION ARCHITECTURE (starting now, all components already built+tested):
1. CHAMPION = the gate's reference, the anchor, and the league seed. Starts at
   ck_000320471040.
2. TRAIN anchored: resume from champion with --kl-prior pointing at the
   CHAMPION ITSELF (self-distillation; K1-K3 machinery from the KL-prior plan).
   KL starts at exactly 0 (student == anchor) so lambda_p is a pure trust
   region on drift, not a pull toward someone else's policy — which is why the
   BC-prior failure does not apply here: that anchor was competent-less, this
   one is the strongest policy we own.
3. GATE every ~20M steps (champion_gate.py, threshold 0.52, both orders).
   PASS -> promote, champion advances, anchor advances with it. FAIL -> discard
   and retry. PPO becomes a mutation operator in a hill-climb that cannot
   regress.
ARM F (running): champion 320M, reward v3, --kl-prior <champion>, lambda_p 0.2,
entropy 0.01, league off — arm D's proven config with the anchor swapped from
the frozen v0 teacher to the champion itself, because a v0-teacher anchor caps
us at teacher level while a self-anchor advances on every promotion. Risk is
bounded BY the gate: if it fails, we discard and fall back to reproducing D.

### 2026-07-20 ~02:00 — ARM F (self-anchor) FAILS the gate: 32.8%
Self-distillation at lambda_p 0.2 did NOT hold parity (32.8%, 27-52 / 33-71),
despite the best training-time numbers of any arm (ep_rew 8.50, ent 3.35,
kl_pri bounded at 0.148). Another reminder that ep_rew is not skill.
WHY IT DIFFERS FROM ARM D (49.2%) — four things, not one:
1. KL DIRECTION. kickstart = KL(teacher||student), MODE-COVERING: penalizes
   the student for failing to cover the teacher's support. kl_prior =
   KL(student||prior), mode-seeking — a far weaker constraint (this direction
   was chosen deliberately in the KL-prior plan for a HUMAN prior, where
   mode-seeking is right; for an anchor meant to PREVENT DRIFT it is wrong).
2. VALUE DISTILLATION. kickstart_losses also regresses student value onto the
   teacher's (lambda_v 0.5). Arm F left the critic unconstrained.
3. STRENGTH: lambda_k 0.36 vs lambda_p 0.2.
4. A SELF-ANCHOR LAGS BY CONSTRUCTION: KL(student||self) starts at exactly 0
   with ~zero gradient, so it only resists drift AFTER the drift; a different
   competent teacher pulls from step 1.
ARM G (running): self-anchor at lambda_p 0.5 — isolates (3), the cheapest of
the four. If 0.5 holds parity, self-distillation is viable and scalable (the
anchor advances with each promotion). If it fails, (1)+(2) are the essential
ingredients and the fix is a code change: teach the kickstart path to accept a
v1 teacher so we can self-distill with mode-covering KL + value distillation.
STANDING HONEST CAVEAT: no configuration tested so far IMPROVES the policy.
Arm D HOLDS (49.2% ~ parity). Holding is not progress — but with the gate, a
holding regime plus rare lucky promotions is a slow hill-climb that cannot
regress, which is strictly better than every regime this project has run since
the anneal ended.

### 2026-07-20 ~04:30 — ARM G: self-anchor strength is the knob (46.4%)
| config | h2h vs champion 320M |
|---|---|
| F: self-anchor lambda_p 0.2 | 32.8% |
| **G: self-anchor lambda_p 0.5** | **46.4%** (38-49 / 39-40) |
| D: v0-teacher kickstart, lambda_k 0.36 + value distill | 49.2% |
| (null control, ck vs itself) | 51.1% |
Monotone in trust-region strength: 0.2 -> 0.5 moved retention 32.8 -> 46.4,
nearly closing the gap to arm D. The lambda_p 0.05 default was designed for a
HUMAN prior (mode-seeking, meant to allow specialization); as a drift brake it
was an order of magnitude too weak. Gate correctly FAILED it at 46.4 < 52 —
still marginally below the champion, so not promotable, which is the gate
behaving conservatively rather than a bug.
ARM H (running): lambda_p 1.0, maps the top of the curve. EXPECTED TENSION,
stated before the result: as lambda_p -> infinity the policy cannot move, so
share -> 50% with ZERO progress. High lambda buys retention by forbidding
change. So the useful setting is the largest lambda that still permits
improvement, and the gate is what makes a wrong guess cheap.
IF H lands ~50% (frozen): the production recipe is many BOUNDED attempts at
lambda ~0.5-0.7 with DIFFERENT SEEDS, each gated, promoting the rare winner —
an explicit hill-climb where PPO is the mutation operator. That is the honest
architecture given a loop that is destructive on average.

### 2026-07-20 ~05:45 — ARM H: lambda curve saturates AT PARITY (retention solved, improvement not)
| self-anchor lambda_p | h2h vs champion | kl drift |
|---|---|---|
| 0.2 (F) | 32.8% | 0.148 |
| 0.5 (G) | 46.4% | 0.094 |
| **1.0 (H)** | **49.2%** | 0.066 |
| v0-teacher kickstart 0.36 (D) | 49.2% | — |
| null control (ck vs itself) | 51.1% | — |
Monotone and SATURATING AT PARITY. Called in advance and confirmed: strong
anchoring buys retention by FORBIDDING CHANGE, so it converges to ~50% by
learning nothing. Three independent configs now HOLD the policy (D, H, and
G nearly); NONE improve it. Retention is solved. Improvement is not.
WINNER'S-CURSE GUARD ADDED BEFORE ANY UNATTENDED PROMOTION (993e53f+5509974):
the 0.52 threshold sits INSIDE the null band, so a single gate promotes an
unchanged policy ~1 run in 3 — over a night that reliably corrupts the
champion, and since the champion is also the anchor and league seed the error
compounds into every later attempt. This is the SAME selection error that
produced the retracted "562M peak". Gate now requires 2 confirmations on
independent seeds (~1 in 27). Found and fixed a hole where hillclimb.py called
gate_one() directly with the n_confirm=0 default, silently skipping the guard
in exactly the unattended case it exists for.
OVERNIGHT HILL-CLIMB starting: bounded 145-iter attempts from the champion,
lambda 0.5-0.7, distinct seeds, every candidate needing 3 independent wins.
HONEST ODDS, recorded before the run: at lambda 0.5 the MEAN outcome is 46.4%
— below parity — so the loop is betting on the positive tail of seed variance
clearing 52% three times. It may promote nothing. That is an acceptable
overnight cost (compute only, and the gate makes a wrong promotion unlikely),
but it is NOT a solution to the improvement problem, and I am not going to
report a quiet night as if it were progress.
OPEN QUESTION for the morning: every arm here is 20M steps. Seer needed ~1e10
steps for Platinum; our arms are 0.2% of that. But scale alone does not explain
our data — the 562M->1.38B stretch was 800M steps and got WORSE, not better.
So "too short" is not sufficient; something in this loop makes PPO's expected
effect negative, and anchoring only clamps it to zero.

## 2026-07-20 ~06:10 — hill-climb stampede fixed, loop armed and verified

**The misfire.** The first hill-climb launch (attempts 1-12) burned every
attempt in 60 seconds and left **13 concurrent trainers** on the remote box
(GPU 0% util, 1713 MiB — every process starving the others). All killed by
hand.

**Root cause.** `poll_until_done` polled `pgrep` immediately after a detached
launch. A remote trainer needs ~20-30s of python + engine construction before
it reaches the process table, so the first poll missed, and a non-zero rc was
read as *"trainer process exited"*. The loop then found no new checkpoint,
logged ERROR, and launched the next attempt — on top of the one still starting.
Attempt 1's remote log proved training HAD started (physics-blowup containment
lines present) while the loop had already declared it dead.

**Fix (cfab026).** Two-phase poll: wait for the process to APPEAR (bounded by
`STARTUP_GRACE=180s`), then wait for it to leave. A process that never appears
is now a *failed launch*, never a completed run. An ssh outage during startup
keeps its own "lost ssh" diagnosis — the two failures need different human
responses. 20 tests broke because `FakeRemote` encoded the buggy lifecycle
("already finished on the first poll"); it now models appear-then-finish, which
is what let the regression land green in the first place. Two regression tests
pin the behaviour directly, including a full-loop stampede test. 117 tests pass
across hillclimb + champion_gate.

**Armed and verified.** Loop restarted at attempt 13 (`--wait-for-idle`).
Confirmed before leaving it: exactly **1** trainer on the remote, GPU 99% /
6492 MiB, `iter 1 ... sps 5,148 ... lambda_p 0.667` in the attempt log. The
appear-phase did its job — no stampede.

**Note on the main run.** Remote had 0 trainers before launch; the main run's
newest checkpoint is `ck_000683284480.pt` at 07-19 19:51, i.e. it was already
stopped deliberately before the stampede. The `pkill` did not cost progress.

**Other live work.** SSL pull healthy — 10,497 duel replays / 7.9G, newest
written seconds before the check. (An earlier "0 files in 2 hours" reading was
a false alarm: `find` here is `bfs`, which rejects relative `-newermt '-2 hours'`
and the error was swallowed by `2>/dev/null`. Use ISO timestamps with bfs.)
Host free on /mnt/c: 145G, above the 130G SSL guard.

## 2026-07-20 ~06:35 — the stampede had a SECOND cause: ssh + zsh globbing

The 06:10 entry above was premature. The appear-phase fix was correct but it
exposed a deeper bug rather than curing the stampede, and attempt 13 proved it
within minutes: the loop logged `ERROR ... trainer never appeared in the
process table within 180s` for an attempt I had *watched* reach `iter 1` at
99% GPU. It then launched attempt 14 on top of it — a second trainer, again.

**Root cause.** `ssh host cmd a b c` does not deliver argv. ssh joins the
arguments and hands the string to the remote LOGIN SHELL, which parses it
again. The trainer box runs zsh, and zsh aborts a command whose glob matches
nothing:

    $ ssh box pgrep -f hc_a0013_[s]20260733
    zsh:1: no matches found: hc_a0013_[s]20260733
    rc=1

`pgrep` never ran. Every `bracket_proof()` pattern — the self-match-proofing
this project adopted precisely to make remote pgrep/pkill safe — is a glob to
that shell. **The safety device and the transport were a trap in combination**:
rc=1 means "the command did not execute", and every caller read it as "no such
process". The failure is perfectly silent; there is no traceback, and the only
tell is one line on stderr that nobody was reading.

Three consequences, all silent:
  * the per-attempt poll could never match a running trainer, so every attempt
    was declared a failed launch after the grace period while it trained on;
  * `--wait-for-idle`'s busy check reported an idle box *unconditionally*,
    which is what let a second trainer launch on top of the first — the guard
    that existed specifically to prevent this had never once worked;
  * `ctl.remote_kill_plan` would have failed to kill anything, reporting rc=1
    as "nothing to kill".

**Fix (14c19b7).** `ctl.remote_quote()`; every glob-carrying pattern crossing
ssh goes through it. Verified live against the box: quoted returns rc=0 for a
running process and rc=1 for an absent one; unquoted fails identically in both
cases, which is the whole problem. Tests pin the invariant *structurally* — any
ssh argv token containing `[ ] * ?` must be quoted — rather than pinning the
call sites I happened to remember.

**Armed, and this time the verification was the right one.** Attempt 14 is
running: 1 trainer, 5948 MiB, and — the part that actually matters — the loop's
own poll command now returns rc=0 against it. Confirming "training started" was
never sufficient; the failure was always that the loop *could not see* what had
started. Verify the observer, not just the observed.

**Method note.** Both bugs were caught only because the loop's claim was
checked against the box instead of believed. An unattended loop that reports
ERROR is not self-evidently correct — ERROR is exactly what a broken observer
reports.

## 2026-07-20 ~06:50 — night-autonomy duties, and quarantining the phantom rows

**Duty 1 (arm H) — done, journaled 3e07b5e.** λ_p 1.0 gated **49.2%**
(93-96 over both side orders) against champion ck_000320471040. Threshold 52%,
null control 51.1%: this is the FROZEN reading set out in advance, not a fix.
High λ buys retention by forbidding change. The λ curve now reads
0.2 → 32.8% | 0.5 → 46.4% | 1.0 → 49.2% | v0-teacher kickstart → 49.2%:
monotone, saturating at parity. Nothing in the band improves; the ceiling is
"don't get worse". Arm G (self-anchor 0.5) confirmed at 46.4%.

**Duty 2 (hill-climb) — running, after two harness bugs.** See the 06:10 and
06:35 entries. Attempt 14 is live and healthy: 1 trainer, GPU 100%, 3 saves.

**Phantom rows quarantined.** The two bugs left 13 ERROR rows in
logs/hillclimb.jsonl. Those verdicts were produced by a blind poll, not by the
search — every one of them was training normally when it was logged dead. Left
counting, they would have tripped the 20-consecutive-failure abort about
**7 real attempts** into the night, and the abort message would have blamed the
search.

Added a `quarantined` flag that `consecutive_failures` skips, and marked the 13
rows with the reason and the fixing commits. Deliberately an additive, auditable
act: the rows stay in the log with their original ERROR verdict — no silent
delete, no rewritten verdict. A log that can be quietly edited to say what the
run needed it to say is worth nothing. Counter now reads 0.

All 13 stampede checkpoint dirs are empty, so no orphaned checkpoint can be
picked up and gated as if it were a real attempt.

**Caveat.** The running loop process holds the pre-quarantine code in memory;
the flag takes effect on its next restart, which I will do once attempt 14
gates rather than killing ~30 min of training now.

**Duty 3 (monitoring).** SSL pull healthy (10,497 duel replays / 7.9G, actively
writing). Host free 145G on /mnt/c, above the 130G guard. Remote 170G free.
Full python suite green: 566 passed.

**Duty 5 (push) — BLOCKED, not done.** `git push origin bc-pretrain` was denied
twice by the permission classifier. 5 commits sit local and unpushed
(cfab026, 2a54f0f, 14c19b7, dc049b9, 22c757a, plus this one). Not worked
around; Elliot needs to either push or grant the permission. Nothing is lost —
the work is committed locally.

**Prepared, not armed.** configs/train_v2_replayreset.toml — replay-state
resets, the one structural lever no arm has varied (every arm so far changed
HOW the policy trains, none changed WHICH STATES it trains from). The 584 MB
pool is now on the box, md5 verified. A preflight (22c757a) refuses to launch
that arm if the pool is missing, because the engine's failure mode is to warn
and quietly fall back to kickoff/random — which would have measured a train_v1
rerun, reported parity, and taught us the opposite of the truth.

## 2026-07-20 ~07:10 — first REAL hill-climb gate: attempt 14 FAILs at 41.2%

Every prior hill-climb row was a phantom produced by the blind-poll bug. This
is the first attempt that actually trained, actually gated, and actually
recorded a share.

    attempt 14  lambda_p 0.6647  reward v3  145 iters  seed 20260734
    candidate hc_a0014_s20260734_ck_000343838720
    -> FAIL  share 41.2%  (threshold 52.0%)
    champion untouched at ck_000320471040.pt

**The harness is now proven on a full cycle**: launch -> poll sees the trainer
-> 145 iters -> newest checkpoint fetched -> gated both side orders -> verdict
recorded -> champion pointer left alone -> next attempt. That was the thing
that needed proving tonight, and it is proven.

**Do NOT over-read the number.** Placed against the lambda ladder it looks
non-monotone:

    lambda 0.2   -> 32.8%
    lambda 0.5   -> 46.4%
    lambda 0.66  -> 41.2%   <- attempt 14
    lambda 1.0   -> 49.2%

That is tempting to read as "0.66 is a bad spot". It is not readable as
anything yet. One gate is ~185 goals, so the binomial SE on the share is
~3.7% and the 95% band is roughly +/-7%: 41.2% spans 34-48%, which comfortably
contains the lambda-0.5 arm's 46.4%. **n=1 cannot order these points.** Writing
"0.66 is worse than 0.5" into the journal on this evidence is exactly the
selection error that produced the retracted "562M is the high-water mark"
claim, and the same error the confirmation gates exist to stop on the promote
side. It deserves the same discipline on the failure side.

What the night's remaining attempts actually buy is that scatter: several
distinct seeds across the 0.5-0.7 band, all measured against the same frozen
champion. If the between-seed spread turns out comparable to the +/-7%
within-gate band, then the whole lambda ladder above is a noise field and the
"monotone, saturating" reading from 06:50 needs retracting too. That is a live
possibility and I would rather find it than defend the earlier story.

**Loop restarted onto quarantine-aware code** (the running process still held
the pre-quarantine build). Done while attempt 15 was ~1 minute in rather than
77 minutes in. Streak counter now reads 1 (the one real FAIL) instead of 14.
Attempt 15 relaunched at lambda_p 0.6196, verified visible to the loop's poll.

**Trap re-encountered, as documented.** The compound
`ssh box 'pkill -f "hc_a0015_[s]20260735"; rm -rf .../hc_a0015_s20260735'`
died with 255: the `rm` argument put the literal string in the shell's own
cmdline, so pkill matched and killed the shell running it. Bracket-proofing
protects the pattern against ITSELF, not against a sibling command in the same
compound string. Minimal one-purpose ssh calls, every time.

## 2026-07-20 ~07:55 — duty-7 decision: do NOT arm the replay-reset arm tonight

Attempt 15 running (lambda_p 0.6196, iter 40/145). Loop healthy on
quarantine-aware code, streak 1. SSL 10,801 replays. Host free 145G.

Considered arming configs/train_v2_replayreset.toml tonight, in parallel, on
the idle laptop 4060. Rejected, for two reasons that are worth writing down
because the first one is the interesting one.

**1. The lambda scatter is not "sampling a known ceiling" — it is calibrating
the instrument, and everything downstream needs it.** My first read was that
the remote loop is wasting the night re-sampling a band already measured at
46.4% and 49.2%, while the replay-reset arm tests the one dimension never
varied. That reasoning is wrong, and it inverts the actual dependency. One gate
carries a ~+/-7% band (07:10 entry). Until I know the SEED-TO-SEED spread, a
single replay-arm gate at, say, 48% is exactly as uninterpretable as attempt
14's 41.2% is now. **The noise floor is a prerequisite for reading the replay
arm at all**, not a competing use of the night. Spending attempts on it is the
cheapest thing available, and it can retract the "monotone, saturating lambda
ladder" claim from 06:50 if the spread turns out to swamp the ordering.

**2. A local trainer would endanger the gate it depends on.** The gate runs
LOCALLY (champion_gate -> MatchRunner). The laptop has 8188 MiB total with
1357 MiB already in use; a trainer sized like the remote's takes ~6500 MiB.
A local arm would therefore be holding nearly all free VRAM at the exact
moments the remote loop needs to gate, risking OOM-failed gates. That would
corrupt the very baseline being measured — and it is the same shape of mistake
as tonight's stampede: a second process quietly contending for a resource the
first one assumed it owned. Right after spending a session fixing that, adding
an unattended concurrent trainer is not the move.

**Plan.** Let the loop accumulate lambda samples across distinct seeds against
the frozen champion. Arm the replay-reset arm on the REMOTE box (one trainer,
no new contention) once the spread is known — it is prepared, pool shipped and
md5-verified, preflight guarding the silent-fallback failure mode.

**Push unblocked.** Elliot granted it; 3e07b5e..2833738 pushed to
origin/bc-pretrain (7 commits). Nothing outstanding.

## 2026-07-20 ~08:20 — n=2, and the lambda ladder above 0.5 is UNRESOLVABLE

    attempt 15  lambda_p 0.6196  -> FAIL  share 42.0%  (74-102)
    attempt 14  lambda_p 0.6647  -> FAIL  share 41.2%  (75-107)
    pooled hill-climb: 149-209, n=358, share 41.6%, SE 2.6%

**RETRACTION of the 06:50 reading.** I wrote that the lambda curve is
"monotone, saturating at parity". With the goal counts in hand that claim does
not survive. Every gate is ~180 goals, so SE ~3.7% and the 95% band is ~+/-7%.
The confidence intervals:

    lambda 0.2  32.8%  [.255, .400]
    lambda 0.62 42.0%  [.347, .494]
    lambda 0.66 41.2%  [.339, .485]
    lambda 0.5  46.4%  [.388, .540]
    lambda 1.0  49.2%  [.421, .563]

Everything from 0.5 upward overlaps everything else from 0.5 upward. Pooled
hill-climb vs arm G is z=1.01 — nowhere near significant. **The top of the
ladder is one flat, unresolved band around 41-49%, and the apparent ordering
was me reading rank order off noise.** The monotone story was a story.

What DOES survive: lambda 0.2 at [.255,.400] does not overlap lambda 1.0 at
[.421,.563]. Low anchor weight is genuinely worse. The ladder has real signal
at the bottom and none at the top.

**The consequence is a change of plan, and it reverses my 07:55 decision.**
I deferred the replay-reset arm on the grounds that the noise floor was a
prerequisite. That was right, and now the floor is measured — so the deferral's
own condition is discharged. The two hill-climb attempts agree to within 0.8%
while each carries SE 3.7%, which says the between-SEED variance is small and
the BINOMIAL gate noise dominates. That has a hard consequence:

    to resolve a true 5-point difference at ~80% power needs ~1200 goals/arm,
    i.e. ~7 gates per lambda point, i.e. ~9 hours per point.

**Mapping the lambda curve finely is therefore not something this instrument
can do overnight, and continuing to sample 0.5-0.7 is close to worthless.**
What the loop CAN do is screen for a LARGE effect: a genuinely better policy
(>52%, or anything near 60%) is detected easily at SE 3.7%. It is a large-effect
screen, not a curve mapper.

So the right use of the remaining night is the arm that might produce a large
effect in the one dimension never varied: **replay-state resets**. Every arm so
far changed HOW the policy trains; none changed WHICH STATES it trains from.
Switching the loop to configs/train_v2_replayreset.toml now.

This is the pre-registered condition being met, not a whim: "arm it once the
baseline failure rate is established". Baseline established = 41-42%, tight,
binomial-dominated.

## 2026-07-20 ~08:35 — replay-reset arm armed; the preflight I just wrote was too narrow

Switched the loop to configs/train_v2_replayreset.toml. First launch died
instantly:

    FileNotFoundError: [Errno 2] No such file or directory:
    'configs/train_v2_replayreset.toml'

The config existed only on the laptop. I had shipped the 584 MB reset pool and
written a preflight for it, then pointed the loop at a config I never shipped —
and the preflight I wrote *specifically to catch silent launch failures* checked
the pool and nothing else. **Checking the one dependency that happened to bite
last time is not a preflight.** The lesson generalizes past this bug: I guarded
the failure I had just experienced rather than the class it belonged to.

Worth noting what worked: the appear-phase fix (cfab026) behaved exactly as
designed. The trainer never reached the process table, so the loop declared a
FAILED LAUNCH rather than a completed run. Before this morning that same
situation produced "trainer exited" and a stampede. The harness caught its own
next bug.

**Fix (e9eedba).** remote_files_required() resolves the whole dependency set
from the train config — itself, reward config, curriculum, schema, and the
reset pool when the curriculum has a replay branch — and the abort names every
missing file with the scp to fix it. Preflight now reports "5 launch
dependencies present". Two ordering bugs fell out, both caught by pre-existing
tests: a stop file must short-circuit before preflight touches the network, and
an unreachable host must report "cannot reach" whichever check notices first.

curriculum_v2.toml and train_v2_replayreset.toml shipped to the box. Attempt 16
relaunched at lambda_p 0.6383. Verifying the pool actually LOADS before trusting
anything this arm produces — the engine's documented failure mode on an
unreadable pool is to warn and silently fall back to kickoff/random, which would
make the arm a train_v1 rerun wearing a different config name.

## 2026-07-20 ~08:50 — the replay arm was RUNNING INERT: remote engine predates the lever

Verified before trusting attempt 16, and it failed the check.

The attempt log has **zero** `[curriculum]` lines — neither the success line
(`[curriculum] replay pool <path>: N read, M kept ...`) nor the documented
failure line (`WARNING: replay pool unreadable; replay resets DISABLED`).
`load_or_empty` was never called at all. The reason:

    remote /home/elliot/construct/.venv/.../construct/_engine.abi3.so
      built 2026-07-19 06:13
      `strings ... | grep -c "replay pool"`  ->  0

**The remote engine binary predates replay-pool support entirely.** The lever
landed in the repo and the wheel was never rebuilt and shipped — consistent
with the deploy having been deliberately held while the config stayed pointed
at curriculum_v1.

So attempt 16 was training on kickoff/random resets while wearing the
replay-arm config name. Worse than the fallback the config warns about: that
fallback at least prints a WARNING, but the code that prints it does not exist
in this build, so the run was silent AND inert. Had I not checked, the arm
would have gated at ~41% like the others and I would have written "replay-state
resets don't help" into the journal on the strength of a run that never drew a
single replay state. That is the exact failure the preflight was supposed to
prevent, and the preflight could not see it: it verified the FILES, and the
missing dependency was the BINARY.

Attempt 16 killed, loop stopped, nothing gated. No bad measurement entered the
record.

**Standing lesson, now three for three tonight.** Every bug this session has
been a silent one that returned a plausible-looking value: rc=1 read as "no
such process"; bfs erroring read as "no files"; and now an inert engine read as
"the arm ran". Each looked like a result. The only thing that has caught any of
them is checking a positive signal — the poll returning rc=0 against a known
process, `stat` on a real file, a `[curriculum]` line in the log — rather than
accepting the absence of an error as success.

**Next: deploying a rebuilt engine wheel to the trainer box.** Backing up the
current .so first so the box can be reverted in one command. Verifying after
install that (a) the pool line appears and (b) curriculum_v1 still behaves
identically — the repo pins that with `zero_replay_weight_matches_legacy_coin`,
which asserts replay_weight=0 reproduces the legacy rng stream exactly, so the
existing lineage and the frozen champion stay comparable.

## 2026-07-20 ~09:05 — engine wheel deployed; replay-reset arm is REAL this time

**DEPLOY (trainer box).** Rebuilt and shipped the engine wheel to enable the
replay-state-reset lever. Recorded in full because this touched the training
box:

  * `cargo test --lib` green: **83 passed**. Critically
    `zero_replay_weight_is_bit_identical_to_legacy` passes, so curriculum_v1
    runs are bit-identical on the new engine — the existing lineage and the
    frozen champion stay comparable, and every earlier arm's measurement
    remains valid.
  * `maturin build --release` -> construct-0.1.0-cp311-abi3-manylinux_2_39.whl.
    Verified BEFORE shipping: new .so has 4 occurrences of "replay pool", the
    old one had 0.
  * **Backup taken first**: /home/elliot/engine_backup_20260719_0613.so.
    Revert is one cp.
  * `pip install --force-reinstall --no-deps` -> "Successfully installed".
    Installed .so 5,126,112 bytes (was 5,043,816), imports clean.

**Positive confirmation, which is the whole point:**

    [curriculum] replay pool data/reset_pool_v5.jsonl: 1081088 read,
      891720 kept (82.5%), rejected: origin_ball 107741, ball_past_goal 70696,
      frozen_car 9332, cars_overlap 1555, implausible 44, malformed 0
      (132.7 MB, 1.64 s)
    [curriculum] configs/curriculum_v2.toml: effective mix
      replay 0.700 / kickoff 0.100 / random 0.200 (pool 891720 states)

**Independent second signal.** Episode reward at iter 1, same champion, same
reward config, same lambda:

    train_v1 kickoff/random (a14)      ep_rew 2.437
    "replay" arm on the OLD engine     ep_rew 1.313   <- inert, kickoff/random
    replay arm on the NEW engine       ep_rew 7.361   <- 3x

Replay states start mid-play — ball near the action, cars already positioned —
so reward accrues far faster per episode than from a kickoff. The reset
distribution demonstrably changed. Throughput unaffected: 5,160 sps vs 5,148
for the kickoff/random arm, and the pool costs 1.64 s once at startup.

Attempt 16 running: 1 trainer, lambda_p 0.6383, seed 20260736.

**What this arm can and cannot tell us.** It is one gate, so it inherits the
+/-7% band. It cannot resolve a 5-point effect and I will not claim one from it.
What it CAN do is what the loop is actually good for: detect a LARGE effect. If
replay-state resets matter the way the compounding-error literature suggests,
this should not land at 41% with the others. If it does land at 41%, that is a
real (if disappointing) datum about the lever, and the FIRST such datum that is
not confounded by an inert binary.

Baseline for comparison, both vs the same frozen champion, both side orders:
kickoff/random resets pooled **41.6%** (149-209 over 2 attempts, SE 2.6%).

## 2026-07-20 ~10:25 — REPLAY-STATE RESETS: first upward separation this project has produced

    attempt 16  REPLAY-RESET arm  lambda_p 0.6383  seed 20260736
    93-86  n=179  share 52.0%  95%CI [.447, .592]
    verdict FAIL -- 0.51955 < 0.52000 threshold. Champion UNTOUCHED.

**The comparison that matters** (everything held except the reset distribution
-- same harness, same lambda band, same reward v3, same 145 iters, same frozen
champion, same gate):

    replay-state resets  (a16)        93- 86  n=179  0.520  [.447,.592]
    kickoff/random       (a14+a15)   149-209  n=358  0.416  [.366,.468]

    diff +0.103   SE_diff 0.046   z=+2.27   p=0.0232   RESOLVED

**This is the first time in this project that any arm has separated UPWARD
from its matched control.** Every previous arm either sat below the champion
(19.8% to 46.4%) or held parity by refusing to change (armH lambda 1.0, 49.2%).
This one moved, and it moved in the right direction.

It was also **pre-registered**. Before the gate landed I wrote the decision rule
into the journal and the wakeup: "near 41% = no large effect from replay resets;
clearly above = the first lever that does something." This is not a result found
by sifting; it is the answer to a question asked in advance, with a single
variable manipulated.

**Confound checked and ruled out.** The obvious worry is that a policy trained
on replay states got evaluated on replay states. It did not.
`h2h_eval._build_runner` constructs `MatchRunner(num_arenas, seed, mode=1,
schema_version, net_heads)` and passes **no curriculum config at all** -- the
gate's environment is fixed, and it is the identical environment that gated
a14, a15, armG and armH. Only training differed.

### What must NOT be concluded

* **n=1 for this arm.** p=0.023 from a single gate is suggestive, not decisive.
* **It is NOT better than armH.** a16 - armH = +0.027, z=+0.53, p=0.598 --
  NOT RESOLVED. The only resolved claim is replay-resets > kickoff/random.
* **~8 arms have now been examined.** A single p=0.023 among many comparisons
  is worth less than it looks. Pre-registration helps; it does not exempt this
  from replication.

### The threshold will NOT be moved

It missed by 0.045 of a percentage point -- 93-86 where 94-85 would have
promoted. The temptation to nudge 0.52 down is exactly the p-hacking the
confirmation gates were built to prevent, and it would convert a razor-thin
single observation into a permanent change of champion. **The answer to a
near-miss is more samples, not a lower bar.** The gate behaved correctly and
the champion is untouched.

### Next

Attempt 17 is already running the SAME arm (lambda_p 0.6634, seed 20260737).
Two or three more samples of this arm is now the highest-value thing the box
can do -- far above another lambda point. If the arm replicates near 52%, the
pooled estimate crosses the threshold on its own and promotes honestly.

Note the shape of this: the lever that finally moved is the one that changes
WHICH STATES the policy trains from, not how it trains. Every arm that tuned
the optimizer, the reward, the entropy, or the opponent pool held or lost. That
is the compounding-error story the replay literature tells, and it is the first
piece of evidence from this project that actually supports it.

## 2026-07-20 ~11:55 — the replay-arm headline does NOT survive replication

    attempt 17  REPLAY-RESET arm  lambda_p 0.6634  seed 20260737
    69-83  n=152  share 45.4%  95%CI [.377,.533]   FAIL

Pooled with a16, and measured against everything that matters:

    replay pooled (a16+a17)   162-169  n=331  0.489  [.436,.543]
    kickoff/random (a14+a15)  149-209  n=358  0.416  [.366,.468]
    armH lambda 1.0            93- 96  n=189  0.492  [.422,.563]

    replay - baseline  +0.073  z=+1.93  p=0.053  NOT RESOLVED
    replay - armH      -0.003  z=-0.06  p=0.954  NOT RESOLVED
    replay vs parity 0.500     z=-0.38  p=0.700  INDISTINGUISHABLE FROM PARITY

**RETRACTION of the 10:25 headline.** I wrote "REPLAY-STATE RESETS PRODUCED THE
FIRST UPWARD SEPARATION THIS PROJECT HAS EVER MEASURED" on the strength of a16
alone (52.0%, p=0.023). With one replication the p-value went 0.023 -> 0.053 and
the pooled arm is now **statistically indistinguishable from parity, and from
armH**. The sober statement is: the replay arm reaches parity. It is not
established to beat the kickoff/random baseline, and it is certainly not
established to beat armH.

**This is textbook regression to the mean, and I walked straight into it.** At
07:10 I wrote, about a14: "n=1 cannot order these points... writing an ordering
on this evidence is exactly the selection error that produced the retracted
'562M is the high-water mark' claim." Three hours later I did that very thing
with a16 -- and the caveats I attached ("n=1", "needs replication") did not stop
the headline from being too strong. **A correct caveat under an overstated
headline is still an overstated claim**; the headline is what gets remembered
and what I carried forward into three wakeup prompts.

Note a16 and a17 do NOT disagree with each other: z=+1.19, p=0.233, consistent
with a single underlying value. Nothing is unstable. a16 was simply the high
draw of two, which is what a high draw looks like from the inside -- indeed
this is precisely the winner's-curse mechanism the confirmation gates were
built to catch on the promote side, doing its work on the analysis side.

**Nothing was promoted, and nothing needs undoing.** The gate ruled FAIL on both
attempts independently and the champion never moved. The damage was confined to
a journal entry, which is why the entry is being corrected in place rather than
quietly amended.

### What is actually true now

* Replay-state resets reach parity with the champion (48.9%, CI spans 0.50).
* So does armH (lambda 1.0). The two are indistinguishable.
* The baseline comparison is marginal (p=0.053) and unresolved. Resolving
  +0.073 needs ~4 gates per arm; we have 2. **a18 is running the same arm
  (lambda_p 0.5461) as sample 3.**

### The one distinction still worth chasing

armH reaches parity by FREEZING -- lambda 1.0 buys retention by forbidding
change. The replay arm reaches parity at lambda ~0.6, which permits change.
"Parity while still moving" and "parity by refusing to move" are different in
kind even when they gate identically, and only the first can compound. But that
is a HYPOTHESIS, not a measurement: it needs a direct check that the replay-arm
policy actually diverged from the champion (behavioural distance, or the
training kl_pri trace) where armH's did not. Until that check exists, the honest
summary is two arms that both hold parity.

## 2026-07-20 ~13:00 — parameter-space geometry: all arms move identically, and my "freezing" story is untestable this way

Preserved first: the 8 arm checkpoints were living in `/tmp/claude-1000/ab/`
(46 MB) -- the entire evidence base for the ladder, sitting in a scratch dir one
cleanup away from gone. Copied to `checkpoints_arms_archive/`.

Then measured drift from the champion in parameter space (58 float tensors,
champion weight norm 71.48):

    armG lambda 0.5    ||d|| 16.308   22.82% of ||W||
    armH lambda 1.0    ||d|| 15.776   22.07%
    a14 kickoff/random ||d|| 16.233   22.71%
    a15 kickoff/random ||d|| 16.264   22.75%
    a16 replay         ||d|| 16.162   22.61%
    a17 replay         ||d|| 15.967   22.34%

Every arm moves the SAME distance -- a 0.75-point spread across lambda 0.5 to
1.0, across two different reset distributions, across four seeds. And pairwise
cosine of the update directions is ~0.25 for **every** pair, with no clustering
whatsoever: the two replay arms are no more similar to each other (0.250) than
either is to a kickoff/random arm (0.252-0.259) or to armH (0.258-0.261).

Read literally that says each run's update is roughly 25% a shared drift
direction and 75% run-specific noise, with no arm-specific signature at all.

**But it does NOT test the hypothesis I wrote it to test, and I nearly claimed
it did.** At 11:55 I proposed that armH reaches parity by FREEZING while the
replay arm reaches parity while still MOVING. The obvious reading of the table
above is "armH moved 22.07%, so armH is not frozen, hypothesis refuted" -- and
that reading is wrong. **The kl_prior penalty constrains the output
distribution, not the parameters.** A run at lambda 1.0 is free to move weights
as far as it likes along directions that leave the policy's action
distribution unchanged, and in a 488k-parameter network there are enormously
many such directions. Weight-space distance and behavioural distance are
different quantities, and only the second bears on freezing.

That is the sixth time tonight a plausible number would have carried me to a
wrong conclusion, and the first where the number was entirely correct and the
INFERENCE was the defect. Worth naming the difference: the earlier five were
broken measurements; this one is a sound measurement of the wrong thing.

**What is genuinely suggestive:** the complete absence of arm structure in
parameter space. If replay resets were pushing the policy somewhere
qualitatively different, some hint of clustering might have been expected --
and there is none. That is weak evidence, not strong, precisely because
parameter geometry is a poor proxy for behaviour.

**Next tool, now clearly the highest-value one:** behavioural distance --
KL between action distributions of candidate and champion over a shared, fixed
batch of states, plus action-agreement rate. That measures what lambda actually
constrains, would settle the freezing question, and would say whether the replay
arm differs from the kickoff/random arm in DIRECTION rather than magnitude.
Deliberately not rushed into existence in the 15 minutes before a18 gates --
a half-verified tool is how five of tonight's six errors happened.

## 2026-07-20 ~13:20 — a18 lands; the replay effect decays, and the whole hill-climb is comparing two ways of losing

    attempt 18  replay arm  lambda_p 0.5461  88-106  n=194  45.4%  [.385,.524]  FAIL

(a18's 45.4% matching a17's 45.4% is coincidence, not a duplicated row -- the
counts are 88-106 vs 69-83. Checked, because an exact repeat is exactly what a
stale read looks like.)

**The effect decays under replication:**

    samples   replay pooled            vs baseline
    n=1       0.520                    p=0.023
    n=2       0.489                    p=0.053
    n=3       0.476  [.434,.519]       p=0.077   (250-275, n=525)

    baseline (a14+a15)  149-209  n=358  0.416  [.366,.468]
    replay vs parity 0.500: p=0.275 -- still indistinguishable from parity

Monotone decay toward the baseline as n grows. That is the signature of a false
positive, not of a real effect being confirmed. Resolving the remaining +0.060
would need ~6 gates per arm; we have 3 replay and 2 baseline.

### The reframing that matters more than the p-value

Chasing "replay vs kickoff/random" for another ~9 hours would resolve **which
of two losing arms loses by less**. Both are below the champion. The champion
has beaten every one of the 5 real hill-climb attempts and all 8 earlier arms.
The hill-climb has produced no promotion because there has been nothing to
promote.

Put the night's whole ladder together and it says something coherent:

    NO anchor      22-30%   PPO self-play actively DESTROYS the policy
    WITH anchor    41-52%   it holds parity, and never exceeds it

The anchor is not a tuning knob that happens to help -- it is damage control.
And note what the anchor's objective literally is: reward PLUS stay-close-to-
the-champion, while the gate asks "do you beat the champion?" **At lambda
0.5-0.7 the training objective is substantially pulling the policy toward the
very thing it is being measured against.** It is close to structural that this
setup can approach parity and not pass it. Removing the anchor does not free
the policy to improve; it collapses (armE 29.6%, no-anchor v4.1 22-25.5%).

So the finding is not "reset distribution doesn't matter". It is: **on this
setup the PPO gradient is anti-correlated with skill, and the KL anchor's only
function is to limit how much damage it does.** Reset distribution, entropy,
league diversity and lambda are all second-order next to that.

### Recommendation on where the box's time should go

1. **Stop accumulating replay samples.** Diminishing, and it settles a question
   between two arms that both lose.
2. **Build the behavioural-distance tool** (KL between candidate and champion
   action distributions on shared states). Cheap, and it measures what lambda
   actually constrains -- unlike the weight-space numbers at 13:00.
3. **Test whether 145 iterations is simply too short.** EVERY arm tonight and
   every arm in the diagnosis used the same ~145-iter (~24M step) bound, so no
   data addresses this at all. One 600-iter run costs ~5.4 h -- about the same
   as four more short attempts -- and tests a hypothesis nothing else has
   touched. This is a real change of experimental design rather than a config
   swap, so it is written up here for Elliot rather than started unilaterally;
   the box is productively occupied meanwhile.

Attempt 19 (replay, lambda_p 0.6634) is already running and will be allowed to
finish rather than killed mid-flight.

## 2026-07-20 ~13:45 — behavioural distance: the measurement weight-space couldn't give

Built `scripts/behavior_distance.py` (16 tests, incl. an engine null control
that the champion against ITSELF scores exactly 0 KL / 1.000 agreement -- if
that ever fails, every number the tool prints is suspect). Both policies are
scored on ONE shared batch of 8192 states generated by champion self-play at a
fixed seed, so differences are behaviour and not distribution shift.

    candidate      lambda   KL(c||ch)  agree   H(c)    gate
    armF            0.2      0.1748    0.591  3.005   32.8%
    armG            0.5      0.1158    0.671  2.962   46.4%
    armH            1.0      0.0835    0.706  2.920   49.2%
    a14 kick/rand   0.66     0.1101    0.676  2.949   41.2%
    a15 kick/rand   0.62     0.1129    0.673  2.931   42.0%
    a16 replay      0.64     0.1070    0.661  2.923   52.0%
    a17 replay      0.66     0.1043    0.660  2.889   45.4%
    champion H = 2.893

**1. lambda controls behavioural divergence, exactly as designed.** 0.2 ->
0.175, 0.5 -> 0.116, 1.0 -> 0.084, monotone. This is the quantity kl_prior
actually penalises, and it is the measurement the 13:00 weight-space numbers
could not provide. The tool answers the question; weight distance never could.

**2. The freezing hypothesis: half right, and the strong form is wrong.** armH
(lambda 1.0) is indeed the closest to the champion -- lowest KL, highest
agreement -- so "reaches parity by staying close" has real support. But it
still picks a DIFFERENT action from the champion on 29% of states. That is not
a frozen policy. "Least moved" yes; "frozen" no.

**3. The correlation that looked decisive, and isn't.** Across all seven arms,
KL vs gate share gives r=-0.842 (t=-3.48) -- a tidy story that every step away
from the champion costs skill. It does not survive an influence check:

    all 7 arms                  r=-0.842  t=-3.48
    drop armF (the lambda 0.2)  r=-0.452  t=-1.01   NOT significant
    hill-climb attempts only    r=-0.572  t=-0.99   NOT significant

The entire relationship rests on ONE extreme point. What is actually supported:
a LARGE divergence is clearly bad (armF, alone and consistent with lambda 0.2
being the only genuinely separated point on the ladder). Within the moderate
lambda 0.5-1.0 band, divergence does not predict gate score at all -- which is
just the earlier finding again, that this band is a noise field.

Nearly wrote "behavioural divergence is anti-correlated with skill" off r=-0.84.
That is the seventh time tonight a plausible number pointed at a wrong
conclusion, and the second of the "correct measurement, defective inference"
kind. One influential point at the end of a lever's range will manufacture a
correlation whenever the lever was set deliberately rather than sampled.

**4. The replay arm is behaviourally indistinguishable from the baseline arms.**
KL 0.104-0.107 (replay) vs 0.110-0.113 (kickoff/random); agreement 0.660-0.661
vs 0.673-0.676. This is the most informative line in the table. Replay resets
changed the TRAINING distribution enormously -- episode reward at iter 1 went
2.4 -> 7.4 -- and the resulting policy is not measurably different in how it
acts. A large change of input distribution produced no distinguishable change
of behaviour, which is consistent both with the parameter-space result and with
the gate result decaying to p=0.077.

**5. Nothing collapsed.** All entropies sit in 2.889-3.005 against the
champion's 2.893. No arm went uniform or degenerate; the failures are not
entropy failures.

## 2026-07-20 ~14:10 — plan for the box after a19: a TRAJECTORY, not another sample

Timestamp correction first: the previous entry's "a19 at iter 11 at ~13:50" was
wrong -- the wakeup was 3600 s, so that reading was at 13:06. a19 reached iter
127 by 14:06, i.e. 116 iters in 60 min = 1.93 iters/min, which matches
166,912 steps/iter at 5,285 sps exactly. The apparent 4x speed-up was my own
bookkeeping, not the box. Checked rather than assumed, because "the run is
mysteriously fast" is exactly the shape of the night's other seven errors.

**Decision on duty 3.** After a19 the replay arm stops. More samples would
settle which of two arms loses to the champion by less, and neither beats it.
Switching the loop back to train_v1 to accumulate baseline samples has the same
defect. Both are precise answers to a question that no longer matters.

**What the box does instead: ONE long run, gated as a trajectory.**

Every arm ever run here -- tonight's five, the diagnosis's eight -- used the
same ~145-iteration (~24M step) bound. **No data whatsoever addresses whether
improvement simply needs longer.** That is the last unswept dimension.

At 13:20 I wrote that this was for Elliot rather than to be started
unilaterally, and I am revising that, with the reason stated so it can be
judged: the deferral was on the grounds that changing the experimental bound is
a design decision. It is, but the alternative uses of the box are now
established to be near-worthless, and I deployed a rebuilt engine to this same
box earlier tonight under the same grant -- a strictly larger action. Declining
the smaller one for the same reason would be inconsistent. It is a single
`--iters` change, it is killable at any moment, and the champion pointer cannot
move without a gate.

**The design is better than one long attempt, and this is the point.** The
trainer saves every 20 iterations, so a 600-iter run yields 30 checkpoints.
Gating a LADDER of them -- roughly iters 145, 300, 450, 600 -- turns one
endpoint into a curve, and the curve is what actually answers the question:

  * still rising at 145 and climbing        -> the 145 bound was the limitation
  * flat from 145 onward                    -> longer does not help; parity is
                                               the ceiling of this objective
  * rising then falling                     -> there is an optimal stopping
                                               point everyone has been missing
  * monotonically falling                   -> PPO degrades from step one and
                                               145 iters merely hid how far

Each rung is one gate with the usual ~+/-7% band, so only a LARGE effect will
show -- which is exactly the regime this instrument is good for, and exactly
what "longer training fixes it" would have to be in order to matter.

Anchor stays at lambda ~0.6 and reward v3: the point is to vary run length
ALONE against the arms already measured, not to introduce a second variable.

## 2026-07-20 ~14:30 — the replay arm is dead, and it dies cleanly

    attempt 19  replay arm  lambda_p 0.6634  70-103  n=173  40.5%  [.334,.479]  FAIL

Four samples of the replay arm, and the p-value against the kickoff/random
baseline across replication:

    n=1   0.520   p=0.023      "first upward separation this project has measured"
    n=2   0.489   p=0.053
    n=3   0.476   p=0.077
    n=4   0.458   p=0.189      +0.042, would need ~12 gates/arm to resolve

    replay pooled (a16-a19)   320-378  n=698  0.458  [.422,.496]
    kickoff/random (a14+a15)  149-209  n=358  0.416  [.366,.468]

And the decisive line, which only became available once n was large enough to
have a tight interval:

    replay arm      vs parity 0.500:  z=-2.20  p=0.028   BELOW the champion
    kickoff/random  vs parity 0.500:  z=-3.17  p=0.0015  BELOW the champion

**Both arms are significantly WORSE than the champion.** The question "does the
replay arm beat the baseline" was never the interesting one; with four samples
the arm has a tight enough interval to answer the question that matters, and
the answer is that it loses. The 10:25 headline is now fully retired: there was
never an upward separation, only a high first draw, and every subsequent sample
walked it back.

That is a complete worked example of the winner's curse, start to finish, in
one night: a pre-registered comparison, a single significant result, a
prominent write-up, and then four samples that dissolve it. The pre-registration
was real and did not save it -- **nothing protects against a small sample except
a larger one.**

**Replay arm closed. Hill-climb loop stopped.** 6 real hill-climb attempts, 8
earlier arms, zero promotions, and the champion never moved. That is the
harness working, not failing: every one of those FAILs was correct.

**Started: the 600-iteration run** (`checkpoints_hc/long600_s20260740`, seed
20260740, lambda_p 0.6, reward v3, train_v1 kickoff/random). Verifying it
reaches `iter 1` before trusting it. This tests the last unswept dimension --
every arm ever run here used ~145 iterations -- and it will be gated as a
LADDER (~145, 300, 450, 600) so the result is a trajectory rather than an
endpoint. ~5.4 h; rungs get gated as they appear.

## 2026-07-20 ~14:40 — the day's central comparison is CONFOUNDED by the engine deploy

The 600-iter run reached iter 1 with `ep_rew 7.600` -- on **kickoff/random**
resets. At 09:05 I cited "ep_rew at iter 1 is 7.361 vs 2.437 for kickoff/random"
as an *independent second signal* that the replay arm was genuinely live. Pulled
iter-1 ep_rew for every attempt:

    a14  OLD engine  kickoff/random   1.184
    a15  OLD engine  kickoff/random   2.222
    a16  NEW engine  replay           7.361
    a17  NEW engine  replay           6.668
    a18  NEW engine  replay           7.351
    a19  NEW engine  replay           7.863
    long600 NEW engine kickoff/random 7.600   <- the control I never ran

**The split is by ENGINE VERSION, not by reset distribution.** A kickoff/random
run on the new engine sits squarely in the replay band. My "independent second
signal" measured the deploy, not the lever.

**And the claim it rested on was worse.** At 09:05 I wrote that
`zero_replay_weight_is_bit_identical_to_legacy` passing meant "curriculum_v1
runs are bit-identical on the new engine, so ALL earlier arm measurements and
the frozen champion remain valid", and I repeated that in several handoffs.
Reading the test:

    let mut a = mk(1, 1, 5150, Some(curr(0.0, 0.4, 0.6, pool_of(16))));
    let mut b = mk(1, 1, 5150, Some(curr(0.0, 0.4, 0.6, vec![])));
    assert_eq!(sequence(&mut a, 50), sequence(&mut b, 50));

It compares two runs of the **same (new) engine** -- pool loaded vs pool empty.
It proves the replay branch is inert when disabled WITHIN a version. It says
nothing whatever about old-engine vs new-engine equivalence. I used a passing
test as evidence for a proposition the test does not make. Three commits landed
in `engine/src/` between the old build and the new one (kickoff jitter 40b8c4f,
reward v4 0b3b2b3, the replay lever 96cd7d0/f600fc9), any of which can move
episode reward.

### What this invalidates, precisely

* **CONFOUNDED: replay arm vs kickoff/random baseline.** a16-a19 trained on the
  new engine; a14+a15 on the old. That is the comparison the entire day was
  built around, and reset distribution is entangled with engine version in it.
  Every "+0.103 / +0.073 / +0.060 / +0.042" figure inherits this.
* **STILL VALID: every arm-vs-champion gate result.** The gate loads
  checkpoints and plays them in a fixed LOCAL eval engine
  (`h2h_eval._build_runner`), so the training engine never enters the
  measurement. "Replay arm 45.8%, significantly below parity" and "baseline
  41.6%, significantly below parity" both stand, as do armF/G/H.
* **STILL VALID: comparisons within an engine version.** armF/G/H and a14/a15
  are all old-engine; a16-a19 are all new-engine. Each group is internally
  clean.

Fortunately the correction was already in flight before I noticed the problem:
**long600 is kickoff/random on the NEW engine**, so its ~145-iter rung is
exactly the same-engine baseline that was missing. The confounded comparison
becomes answerable at that rung for free.

Eighth error of the night, and the worst-shaped one yet. The previous seven were
a broken measurement, a broken measurement, an inert binary, a blind preflight,
a lucky sample, a sound measurement of the wrong quantity, and one influential
point. This one is: **a passing test cited for a claim it does not make.** The
test was green, the reasoning around it was not, and green tests are exactly
what one stops interrogating.

## 2026-07-20 ~15:50 — the trajectory is FLAT, and the damage is immediate

Six rungs of the 600-iter run gated against the frozen champion (same local
eval engine for every rung, so the rungs are mutually clean AND clean against
every earlier arm):

    iter  20   82-118   0.410  [0.344,0.479]
    iter  40   82- 89   0.480  [0.406,0.554]
    iter  60   99- 98   0.503  [0.433,0.572]
    iter  80   78-106   0.424  [0.355,0.496]
    iter 100   79- 88   0.473  [0.399,0.549]
    iter 120   77- 91   0.458  [0.385,0.534]

    Cochran-Armitage trend: z=+0.53  p=0.597   NO RESOLVED TREND
    POOLED (iters 20-120)  497-590  n=1087  0.457  [0.428,0.487]
    vs parity 0.500:       z=-2.82  p=0.0048  significantly BELOW the champion

**Two findings, and the second is the one that matters.**

**1. Over iterations 20-120 the policy is flat.** No trend, and the pooled
estimate is tight (n=1087, +/-3%) and clearly below the champion. More training
across this range neither helps nor hurts. Note the shape 41.0 -> 48.0 -> 50.3
-> 42.4 -> 47.3 -> 45.8 *looks* like a rise-then-fall, and the trend test says
it is not: p=0.597. Six overlapping intervals wobbling around a flat line is
exactly what this instrument produces from a constant. I would have drawn a
curve through those points by eye; the tool built two hours ago refused, which
is the whole reason it exists.

**2. The damage is IMMEDIATE, not gradual.** By iteration 20 -- 3.3M steps,
about 1.4% of the champion's training -- the policy is already at 41.0% and it
never returns to parity. This kills the "PPO slowly erodes the policy" picture I
have been carrying. Whatever happens, happens almost at once and then stops.

That reframes the whole night. Every arm ran 145 iterations and landed in
41-52%; it now appears essentially all of that displacement is incurred in the
first ~20 iterations, and the remaining 125 are noise around the new level. The
lambda sweep, the reset distribution, the entropy setting -- all of them were
tuning a process whose effect is over before they get a chance to matter.

**The engine effect is NOT resolved.** long600 (new engine, iters 20-120)
pooled 0.457 vs a14+a15 (old engine, iter 145) 0.416: +0.041, z=+1.36, p=0.173,
NOT RESOLVED. So the confound identified at 14:40 is real but its magnitude is
unmeasured, and these two selections differ in iteration count as well as
engine. The clean comparison is the iter-140 rung against a14+a15, both at
~the same iteration count -- that rung is next.

**What this does NOT yet answer.** The run continues to 600. Iterations 20-120
being flat says nothing about 120-600; a slow recovery or a late collapse would
both still be invisible here. That is precisely why the run was set to 600 and
why the rungs keep getting gated.

**Immediate next question worth its own experiment:** if the loss is incurred in
the first ~20 iterations, what happens in the first 5? Rungs exist only every
20 iterations, so the interesting region is currently unresolved. A short run
saving every 2 iterations would localise it -- and if the drop is at iteration
1-2, the suspect is the resumption itself (optimizer state, value-head
re-fitting, the first large policy update) rather than anything about the
training signal.

## 2026-07-20 ~16:00 — the local engine is the INSTRUMENT, and it must not be touched

While scoping the fine-grained follow-up I checked the local engine:

    python/construct/_engine.abi3.so   built 07-19 06:13
    strings ... | grep -c "replay pool"  ->  0

**The local box still runs the OLD engine.** I only ever deployed the rebuilt
wheel to the trainer box. Two consequences, one reassuring and one a trap I
nearly walked into.

**Reassuring:** the gate runs LOCALLY, so every gate all night -- armA through
armH, a14-a19, all six long600 rungs -- has been scored in one fixed
environment. That is precisely why the arm-vs-champion results survive the
14:40 training-engine confound. The measuring instrument never moved.

**The trap:** my next thought was to run the fine-grained experiment locally,
notice it would use the old engine while long600 uses the new one, and "fix"
that by installing the new wheel locally. That would have **changed the gate
environment mid-trajectory** -- every rung gated after the upgrade silently
incomparable with every rung gated before, including the six already recorded.
It is the same confound as 14:40 with the sign flipped, and it would have been
introduced deliberately, in the name of consistency.

**Rule, now explicit: the local engine is the instrument. It does not get
upgraded while measurements are in flight.** If it ever must change, every
comparison spanning the change needs re-gating, not reasoning.

**Consequence for the fine-grained experiment, which is fine.** Run it LOCALLY
on the old engine and compare it against the OLD-engine arms (a14+a15, 0.416 at
145 iters). That is internally consistent and answers the question as posed --
"is the loss incurred in the first few iterations?" -- without touching either
the instrument or the running 600-iter job on the remote. The new-engine version
of the same question can be asked later on the trainer box once long600 is done.

## 2026-07-20 ~16:35 — nine rungs, still flat; and the engine effect is the same size as the "replay effect"

    iter 140  90- 95  0.486  [0.415,0.558]
    iter 160  66-102  0.393  [0.322,0.468]
    iter 180  83-103  0.446  [0.377,0.518]

    Cochran-Armitage across 9 rungs: z=-0.30  p=0.768   NO RESOLVED TREND
    POOLED iters 20-180: 736-890  n=1626  0.453  [0.429,0.477]
    vs parity 0.500:     z=-3.82  p=0.00013

The estimate is now tight (+/-2.4%) and the conclusion is firm: **across
iterations 20-180 the policy sits ~4.7 points below the champion, flat, with no
trend.** Adjacent rungs 20 iterations apart swing by 9 points (48.6 -> 39.3),
which bounds how much real signal could be hiding here: the gate noise is
larger than any plausible drift over this range.

**The engine effect, measured at matched iteration count.** long600's iter-140
checkpoint has the same step count as every hill-climb attempt's final
checkpoint, so this is the cleanest available comparison -- same iterations,
same reward, same lambda band, differing in engine version and seed:

    long600 iter 140 (NEW engine)   90- 95  n=185  0.486  [0.415,0.558]
    a14+a15 iter ~145 (OLD engine) 149-209  n=358  0.416  [0.366,0.468]
    difference +0.070  z=+1.56  p=0.119   NOT RESOLVED

Worth stating plainly: **+0.070 is the same magnitude as the "replay effect" I
chased all morning** (+0.103 at n=1, decaying to +0.042 at n=4). The replay arm
ran on the new engine and its baseline on the old one. So the entire apparent
replay effect is quantitatively consistent with having been the engine
difference the whole time -- which is exactly what the 14:40 confound predicted.
Consistent with, not proven: at p=0.119 this comparison cannot resolve its own
effect either, and it would need ~4 more gates per side to try.

**Fine-grained run launched** (local, OLD engine, so comparable only to the
old-engine arms -- the instrument is not being touched). `configs/train_v1_fine.toml`
is train_v1 with `save_every_iters = 1`; 10 iterations from the champion, lambda
0.6, reward v3, seed 20260741, into `checkpoints_fine/`. This resolves the
region where the entire effect is incurred and where no checkpoint has ever
existed. If the drop is present at iteration 1-2, the suspect is resumption
itself -- optimizer state, value-head refit, the first large policy update --
rather than anything about the training signal. Note `resume_train.py` has a
`--reset-optimizer` flag ("stale moments belong to the old loss landscape"),
which is the obvious A/B to run next if the answer points that way.

## 2026-07-20 ~17:00 — the drop is present at ITERATION ONE, and never deepens

Ten checkpoints, one per iteration, from the champion (local OLD engine, so
comparable to the old-engine arms; the gate instrument was not touched):

    iter  1  76- 83  0.478    iter  6  94- 97  0.492
    iter  2  89- 94  0.486    iter  7  91- 91  0.500  (exact tie)
    iter  3  76-108  0.413    iter  8  85- 87  0.494
    iter  4  78- 86  0.476    iter  9  75- 96  0.439
    iter  5  70-102  0.407    iter 10  83- 95  0.466

    Cochran-Armitage across 10 rungs: z=+0.22  p=0.824  NO RESOLVED TREND
    POOLED iters 1-10: 817-939  n=1756  0.465  [0.442,0.489]
    vs parity 0.500:   z=-2.91  p=0.0036

Set beside the long run:

    fine   iters   1-10   n=1756   0.465  [0.442,0.489]
    long600 iters 20-180  n=1626   0.453  [0.429,0.477]

**The same level.** The policy steps off the champion, loses ~4 points, and
then sits there -- flat through iteration 10, flat through iteration 180. There
is no gradual erosion to find because there is no gradual erosion. The entire
effect is incurred by the FIRST update.

That is not the "PPO destroys the policy" story I have been carrying all night,
and it is not the "145 iterations is too short" story either. Both assumed a
process that unfolds over training. Nothing unfolds.

### The experiment this demands, now running

One PPO iteration moves the weights `||d||=1.7796`, **2.49% of the champion's
norm**. Two very different explanations fit everything above:

  (a) PPO's update DIRECTION is harmful -- it walks somewhere worse.
  (b) ANY movement is harmful -- the champion sits at a local optimum of the
      gate metric and every perturbation loses, whatever the direction.

Under (b) there is nothing wrong with PPO at all, and the night's entire
framing -- the anchor as "damage control", the lambda ladder, the reset
distribution -- is a misreading of a metric artefact. Under (a) the direction
is the problem and the anchor really is doing what I claimed.

`scripts/perturb_null.py` builds the matched control: the champion moved the
SAME distance in a RANDOM direction. Matching is **per-tensor**, scaling each
tensor's noise to that tensor's own ||delta|| from the real PPO step, so the
control reproduces how the update spreads magnitude across layers and differs
only in direction. A single global scale would have confounded direction with a
different per-layer profile. Three seeds generated, all matched to ratio 1.0000,
all three gating now.

Prediction stated BEFORE the result, since that is the only way it counts:
if random perturbation gates near 0.465 the answer is (b) and the diagnosis
changes completely; if it gates near 0.500 the answer is (a) and PPO's
direction is genuinely harmful. I do not have a strong prior between them,
which is the mark of a worthwhile experiment.

## 2026-07-20 ~17:20 — the null control lands, and it points at (a)

Three matched random-direction controls (champion moved ||d||=1.7796 -- exactly
one PPO iteration's distance -- per-tensor matched, direction random):

    nullctl seed1  94- 73  56.3%   PASS   <- random noise PASSED the champion gate
    nullctl seed2  90- 84  51.7%   FAIL
    nullctl seed3  80-106  43.0%   FAIL

    NULL pooled          264-263  n= 527  0.501  [.458,.543]  vs parity p=0.97
    PPO fine  1-10       817-939  n=1756  0.465  [.442,.489]  vs parity p=0.0036
    PPO long600 20-180   736-890  n=1626  0.453  [.429,.477]  vs parity p=0.0001

**Random movement of the same size costs nothing. PPO's movement costs ~4
points.** That is explanation (a) from the registered prediction: the update
DIRECTION is what hurts, not the mere fact of moving. The champion is not
simply sitting on a needle-point local optimum where any perturbation falls off
-- perturb it randomly and it holds parity.

**But the direct comparison is NOT resolved, and saying otherwise would be a
fallacy I should name.**

    NULL - PPO(fine) = +0.036  SE=0.025  z=+1.44  p=0.1505  NOT RESOLVED
    (needs ~17 gates per arm; the null arm has 3)

"NULL is indistinguishable from parity" and "PPO is significantly below parity"
are two separate tests, and concluding from them that NULL differs from PPO is
**the difference-of-significance fallacy** -- comparing each to a threshold
instead of to each other. The honest position right now is: the evidence leans
to (a), and the decisive test is underpowered. Twelve more null seeds are
gating (null controls need no training, only gate time, so this is cheap).

**Separately important: seed1 PASSED the gate at 56.3%.** A pure random
perturbation of the champion cleared the 52% promote threshold. At n~180 per
gate that is exactly what the arithmetic predicts, and it is the strongest
possible vindication of the confirmation-gate design added this morning: had
this been a real candidate under `--promote-if-pass` with `n_confirm=0`, noise
would have taken the belt. It also retroactively justifies refusing to lower
the threshold after a16's 52.0% near-miss.

Note too the null arm's spread: 43.0 to 56.3, thirteen points across three
seeds of PURE NOISE. Any single gate reading in the 43-56 band is consistent
with a policy that differs from the champion by nothing at all. That is the
band every arm tonight has been living in.

## 2026-07-20 ~17:45 — 15 null seeds: I over-read the 3-seed null, exactly as I over-read a16

Twelve more matched random-direction controls. All 15 pooled:

    NULL (15 seeds)     1267-1366  n=2633  0.481  [.462,.500]  vs parity p=0.0537
    PPO fine 1-10        817- 939  n=1756  0.465  [.442,.489]  vs parity p=0.0036
    PPO long600 20-180   736- 890  n=1626  0.453  [.429,.477]  vs parity p=0.0001
    ALL PPO pooled      1553-1829  n=3382  0.459  [.442,.476]

    NULL - PPO fine      = +0.016  z=+1.04  p=0.300   NOT RESOLVED
    NULL - PPO long600   = +0.029  z=+1.82  p=0.069   NOT RESOLVED
    NULL - ALL PPO       = +0.022  z=+1.70  p=0.090   NOT RESOLVED

**CORRECTION to the 17:20 entry.** I wrote "Random movement of the same size
costs nothing" and called it explanation (a). That was three seeds giving
0.501. With fifteen it is **0.481**, and the interval now only just touches
parity (p=0.054). The null estimate moved two points toward "costs something"
when n grew 5x -- which is precisely what a small-sample fluctuation does.

**I made that error twenty-five minutes after writing a paragraph warning about
the difference-of-significance fallacy, in the same entry.** Knowing the failure
mode in the abstract did not stop me committing a neighbouring version of it:
I was careful about how I compared two numbers and careless about how much I
trusted one of them. Ninth of the night, and the one I have least excuse for.

**Where this actually leaves the pre-registered question.** Neither (a) nor (b)
is established:

  * Random perturbation: 0.481, marginally below parity. It may cost a little.
  * PPO perturbation:    0.459, clearly below parity.
  * The gap between them: +0.022, p=0.090 -- **not resolved.**

The data lean toward PPO's direction being worse than random, and they do not
establish it. Resolving +0.022 needs roughly 45 gates per arm; the null arm has
15, the PPO arm effectively 19. About 20 further null seeds would bring z to
~1.95, which is still short of the line, so this is a question that wants ~40
more seeds rather than a handful.

**What IS established, and it is not nothing:**

1. The drop is fully incurred by the FIRST update and never deepens (fine
   1-10 = 0.465, long600 20-180 = 0.453, both flat: p=0.824 and p=0.768).
2. Moving the champion AT ALL -- by any means, in any direction -- lands you in
   a band around 0.46-0.48 and never above it.
3. Pure noise spans 42.3% to 56.3% across 15 seeds. **Every arm measured
   tonight sits inside the noise band of a policy that differs from the
   champion by nothing meaningful.**

Point 3 is the night's real finding and it subsumes most of the others. The
lambda ladder, the reset distribution, the entropy sweep, the replay arm --
all of them produced numbers inside the range that pure random perturbation
produces. The instrument cannot tell them apart because there may be nothing
there to tell apart.

Launching 20 more null seeds (they need no training, only gate time) to push
the decisive comparison as far as one night allows.

## 2026-07-20 ~18:40 — 35 null seeds: the answer is BOTH, and my prediction was binary

    NULL random-direction (35 seeds)  2979-3167  n=6146  0.485  [.472,.497]
        vs parity: z=-2.40  p=0.0165   BELOW parity
    ALL PPO checkpoints (19 gates)    1553-1829  n=3382  0.459  [.442,.476]
        vs parity: z=-4.75  p<0.0001   BELOW parity

    NULL - ALL PPO = +0.026  SE=0.011  z=+2.39  p=0.0169   RESOLVED

**The pre-registered prediction was wrong in its structure, not just its
value.** At 17:00 I wrote: random near 0.465 means (b) any movement is harmful;
random near 0.500 means (a) PPO's direction is harmful. It landed at **0.485**,
squarely between them, and the truth is that BOTH effects are real and they
stack:

  * Moving the champion AT ALL costs ~1.5 points. The champion does sit at a
    local optimum of the gate metric -- (b) is real, and my 17:20 "costs
    nothing" reading was wrong twice over (3 seeds said 0.501, 15 said 0.481,
    35 say 0.485 and it is now significantly below parity).
  * PPO's DIRECTION costs a further ~2.6 points beyond that -- (a) is real too.
    **PPO's update is significantly worse than a random perturbation of
    identical magnitude and identical per-layer profile.**

I framed a binary and reality was additive. Worth noting for its own sake: a
forced-choice prediction is a good discipline against retro-fitting, and a bad
model of a world where two mechanisms can both be true.

### The caveat that keeps this from being a headline

Splitting the PPO pool by training engine:

    NULL - PPO fine 1-10    (OLD engine)  +0.019  z=+1.44  p=0.150  NOT RESOLVED
    NULL - PPO long600      (NEW engine)  +0.032  z=+2.31  p=0.021  RESOLVED

The resolved verdict leans on the pooled arm. The two PPO arms do not differ
from each other (0.465 vs 0.453, p~0.48), so pooling is statistically
defensible -- but the engine-matched comparison, which is the cleanest one
available, does NOT resolve. And p=0.0169 is a single result among many tests
run tonight; the specific comparison was set up in advance, but the branch it
landed on was not one of the two I named.

**Status: PPO's direction being worse than random is SUPPORTED, not
established.** Resolving +0.026 properly wants ~33 gates per arm; the null arm
now has 35, the PPO arm 19. More PPO checkpoints -- which cost training time,
unlike null seeds -- are what this needs.

### What is now solid after a full night

1. The champion sits at a local optimum: any perturbation costs ~1.5 points.
2. PPO's first update costs ~4 points, and the entire effect is incurred by
   that FIRST update -- flat through iteration 10 (p=0.824) and through
   iteration 180 (p=0.768).
3. Pure noise spans 0.385-0.568 across 35 seeds. Every arm measured tonight
   lives inside that band.
4. Nothing has ever beaten the champion: 6 hill-climb attempts, 8 arms, 19
   long600 rungs, 10 fine rungs, 35 null controls. The pointer never moved.
5. A random perturbation PASSED the 52% gate. The confirmation-gate design is
   not paranoia; it is the difference between a champion and a coin flip.

## 2026-07-20 ~19:05 — 14 rungs still flat; strengthening the engine-matched test

    iter 200  109-108  0.502    iter 260  90- 95  0.486
    iter 220   86-102  0.457    iter 280  70- 92  0.432
    iter 240   79-117  0.403

    Cochran-Armitage across 14 rungs: z=-0.16  p=0.873   NO RESOLVED TREND
    long600 iters 20-280 POOLED: 1170-1404  n=2574  0.455  [.435,.474]

Still flat, now across 260 iterations of training. The trajectory experiment
has done its job: there is no shape to find.

The pooled null comparison strengthens as PPO gates accumulate:

    NULL - long600   +0.030  z=+2.58  p=0.0100  RESOLVED
    NULL - ALL PPO   +0.026  z=+2.61  p=0.0091  RESOLVED   (was p=0.0169)

But the caveat from 18:40 is untouched: the ENGINE-MATCHED comparison
(NULL vs the local old-engine fine run) is still +0.019, p=0.150, because the
fine arm has not grown. That is the cleanest test available and the one the
conclusion should rest on, so it is the one worth paying for.

**Fine run #2 launched** (local, old engine, seed 20260742, otherwise identical
to the first). Two more such runs would take the fine arm from n~1756 to
n~5300, which by the arithmetic brings the engine-matched z from 1.44 to ~2.09
-- i.e. actually resolves it rather than leaving the verdict leaning on a pooled
arm that mixes engines. Roughly 1.2 h of local GPU, and the remote box is busy
with long600 anyway, so it costs nothing that is otherwise in use.

Small recurrence worth logging: `pgrep -c -f "resume_trai[n].py"` reported **2**
local trainers when there was one. The compound command that ran it contained
the literal string `resume_train.py` in its launch clause, so the wrapping shell
matched the pattern. Third distinct form of this trap tonight (after the
compound pkill+rm that died 255, and the hillclimb `--status` false count).
Bracket-proofing defends a pattern against ITSELF; it does nothing about a
sibling clause in the same command line.

## 2026-07-20 ~17:10 — the three fine runs are genuinely independent, and they share a direction

Before pooling fine runs #1-#3 into one arm, checked that different seeds
actually produced different updates -- if they had not, pooling would add no
information and the "n=5300" would be a fiction:

    each moved from champion:  1.7796  1.9195  1.7891
    between runs:  a-b=2.4277  a-c=2.2905  b-c=2.4372
    cosine of update directions:  a.b=0.140  a.c=0.176  b.c=0.138

The between-run distances EXCEED each run's distance from the champion, so the
three went to substantially different places. Pooling is valid; they are
independent draws from "a PPO update".

**But cosine ~0.15 is not zero.** Two random directions in a 488k-dimensional
space have cosine ~0.001; 0.14-0.18 is a large, real shared component. Three
independently-seeded single-iteration PPO updates agree on roughly 15% of their
direction. (For contrast, the 145-iteration arms measured at 13:00 shared ~25%
-- the shared fraction grows with training, as one would expect if it is
systematic rather than noise.)

That matters for interpreting the null-control result. If PPO's updates were
harmful because each happened to land somewhere bad, different seeds would
disagree and the harm would average out across the pooled arm. They do not
disagree: **every seed loses ~4 points while random directions of identical
magnitude lose ~1.5, and the seeds share a common direction.** The natural
reading is that the shared component is the harmful part -- a systematic
gradient bias, not update noise.

**Proposed follow-up, not run tonight:** decompose an update into its
seed-shared component and its seed-specific residual, perturb the champion
along each separately, and gate them. If the shared component alone reproduces
the ~4-point loss while the residual costs ~1.5 like noise, that localises the
damage precisely and would be the most actionable result this line of work
could produce. It needs only checkpoints already on disk plus gate time -- no
training.

## 2026-07-20 ~17:55 — ESTABLISHED: PPO's update direction is worse than random noise

The engine-matched comparison, with the fine arm grown from 1 run to 3:

    NULL   (35 seeds)  2979-3167  n=6146  0.485  [.472,.497]
    FINE   (30 gates)  2394-2837  n=5231  0.458  [.444,.471]
    A - B = +0.027  SE=0.009  z=+2.88  p=0.004   RESOLVED

Both arms trained/perturbed and gated on the SAME (old, local) engine, so the
confound that muddied the pooled version at 18:40 is gone.

**The clustering caveat, checked, and it makes the result stronger.** Thirty
gates come from only three runs, and consecutive iterations within a run are
nearly the same policy -- treating 30 checkpoints as 30 independent samples
would overstate n. The conservative analysis treats each RUN as one
observation:

    4 independent PPO runs:  0.465  0.452  0.456  0.455
      mean 0.4569   between-run SD 0.0059   SE of mean 0.0029
    NULL (35 genuinely independent seeds): 0.4847  SE 0.0064

    run-level: diff = -0.0278   t = -3.97   df~3   significant (crit 3.18, p~0.03)

Significant under both analyses. And the between-run SD is **0.0059** -- four
independently-seeded PPO runs land within 1.3 points of each other, while 35
random perturbations of the same magnitude spread over 18 points (0.385-0.568).
The PPO result is not a lucky draw; it is a tight, reproducible constant.

### The finding, stated plainly

**Take the champion. Move it a fixed distance in a random direction: it loses
~1.5 points. Move it the same distance in the direction PPO chooses: it loses
~4 points. PPO's gradient is not merely uninformative about this metric -- it
is anti-informative, reliably picking a direction worse than chance.**

Three facts now fit together:

1. Any movement costs ~1.5 points -> the champion sits at a local optimum of
   the gate metric.
2. PPO's direction costs ~4 -> the gradient systematically points downhill in
   gate terms.
3. Independently-seeded updates share ~15% of their direction (cosine 0.14-0.18
   vs ~0.001 for random vectors) -> the harmful part is a SHARED, systematic
   component, not per-run noise.

That is a coherent mechanism, and it explains every negative result of the past
weeks without appealing to any of the levers that were swept. Reward shaping,
entropy, reset distribution, league diversity, lambda, run length -- all were
adjustments to a process whose direction was wrong from the first update.

### What this does NOT say

It does not say PPO is broken, nor that the implementation has a bug -- the
plumbing was verified exact in the earlier diagnosis. The gradient optimises the
TRAINING objective (shaped reward under self-play); the gate measures goal share
against a frozen champion. **This is evidence those two objectives are
misaligned, not that the optimiser fails at its own.** The actionable question
becomes "why does improving the training objective make head-to-head worse?" --
an objective-design question, not a tuning one.

### Next experiment, and it needs no training

Decompose a PPO update into its seed-shared component and its seed-specific
residual; perturb the champion along each separately at matched magnitude; gate
both. If the shared component alone reproduces the ~4-point loss while the
residual behaves like noise (~1.5), the damage is localised to a specific,
inspectable direction in weight space -- which can then be read against what it
does to behaviour. Everything needed is already on disk.

## 2026-07-20 ~18:10 — decomposing the update; the noise model checks out exactly

17 long600 rungs now (iters 20-340): Cochran-Armitage z=-0.13, p=0.899. Still
flat. That experiment is finished as a question.

Built `scripts/decompose_update.py` (11 tests) to split a PPO update into the
direction every seed agrees on and the part only one seed did:

    3 updates, mean ||u||=1.8294, ||shared mean||=1.2054
    residual cos with shared: -0.031, +0.054, -0.025   (clean split)

**First reported number was biased and I caught it.** The script printed
`cos(u_i, shared) = +0.65`, which looks like each update being two-thirds the
common direction. It is inflated: each u_i contributes a third of the mean it
is being correlated against. Leave-one-out is the unbiased measure:

    cos(u1, mean of others) = +0.209
    cos(u2, mean of others) = +0.181
    cos(u3, mean of others) = +0.207     (pairwise: 0.140, 0.176, 0.138)

So the honest alignment is ~0.19, not 0.65.

**The contamination arithmetic, and it validates the model.** Treating an
update as u = s + n with independent n, pairwise cosine 0.151 gives
||s||/||u|| = 0.389, and the mean of N runs has

    N= 3: ||mean||/||u|| = 0.659, mean is 35% true-shared by energy
    N=10: ||mean||/||u|| = 0.486, 64% true-shared
    N=23: ||mean||/||u|| = 0.433, 80% true-shared

The predicted 0.659 for N=3 matches the measured 1.2054/1.8294 = 0.659 exactly.
The model of "shared direction plus independent noise" is not an assumption
here; it reproduces the observed norm to three decimals.

**Which means the probe is 65% noise, and that is fine in ONE direction.**
Contamination pulls the shared probe TOWARD the noise baseline (0.485), never
away from it. So:

  * shared probe lands near 0.46 (PPO-like) despite being mostly noise
        -> strong evidence the shared direction carries the damage;
  * shared probe lands near 0.485
        -> AMBIGUOUS, since contamination alone could produce that, and a
           clean test would need ~23 seeded runs.

Registering that asymmetry before the gates return, because it is the
difference between a result and a wish. Four probes gating: the shared
direction and each of the three residuals, all rescaled to a real update's
magnitude so only direction differs.

## 2026-07-20 ~18:15 — decomposition probes: inconclusive, and the reason is structural

    SHARED probe    (1 gate)   78- 86  n=164  0.476  [.401,.552]
    RESIDUAL probes (3 gates) 221-280  n=501  0.441  [.398,.485]
    NULL reference  (35)      2979-3167 n=6146 0.485  [.472,.497]
    PPO reference   (30)      2394-2837 n=5231 0.458  [.444,.471]

    SHARED vs NULL  p=0.818 | SHARED vs PPO  p=0.650  -> indistinguishable from both
    RESID  vs NULL  p=0.059 | RESID  vs PPO  p=0.476  -> indistinguishable from both

Nothing separates. Honouring the asymmetry registered before the gates ran: the
shared probe at 0.476 is near the noise level, and that is **AMBIGUOUS, not a
refutation** -- with the probe only 35% true-shared by energy, contamination
alone predicts exactly this.

One point does cut against the hypothesis rather than merely failing to support
it: the residuals came in at **0.441**, if anything BELOW the noise level
(p=0.059) rather than at it. Under "the shared component carries the damage"
the residuals should have looked like noise. They lean the other way. That is
weak evidence -- three gates, wide interval -- but it points at the whole update
direction being harmful rather than one privileged component of it.

**The structural reason the probe is weak, and the fix.** `configs/champion.toml`
pins `seed = 11`, so the gate is deterministic: re-gating a checkpoint
reproduces its numbers exactly. That is correct and necessary for a champion
gate -- a promote decision must not depend on when it was run -- but it means a
probe cannot be strengthened by repetition, which is why the shared probe sits
at n=164 while the references have n>5000. I had been treating "more gates" as
the universal remedy and it simply does not apply to a fixed checkpoint.

For a deterministic checkpoint the match seed is the ONLY source of variance,
so sweeping seeds is the right way to grow n. Gating decomp_shared across 8
additional seeds now (n 164 -> ~1400, CI +/-7% -> ~+/-2.6%), which is enough to
tell 0.476 from 0.485 and 0.458 apart.

Worth noting this is the same shape as the clustering check earlier: both times
the question was "what is the actual independent unit here?" -- runs rather than
checkpoints then, match seeds rather than repeat gates now.

## 2026-07-20 ~18:35 — THE SHARED DIRECTION ALONE REPRODUCES THE FULL DAMAGE

Seed-swept the shared probe (9 match seeds, since the gate is deterministic and
n cannot grow by repetition):

    SHARED probe (9 seeds)   689- 828  n=1517  0.454  [.429,.479]
    NULL random-direction    2979-3167 n=6146  0.485  [.472,.497]
    real PPO updates         2394-2837 n=5231  0.458  [.444,.471]

    SHARED vs NULL     : diff=-0.0305  z=-2.14  p=0.033   DIFFERS
    SHARED vs real PPO : diff=-0.0035  z=-0.24  p=0.811   INDISTINGUISHABLE
    SHARED vs parity   : z=-3.57  p=0.00036

**The component that every seed agrees on, isolated from all seed-specific
content and rescaled to a real update's magnitude, is significantly worse than
random noise and statistically identical to a real PPO update.**

This is the strong-evidence branch of the asymmetry registered at 18:10, and it
is strong precisely because of the contamination: the probe is only 35%
true-shared by energy at N=3, and contamination can only pull it TOWARD the
null level of 0.485. It went the other way, to 0.454. A clean shared direction
would if anything be worse, not better.

So the damage is not diffuse. It lives in a single, reproducible direction that
three independently-seeded runs each recovered ~19% of (leave-one-out cosine).
That direction is an object: it can be computed, inspected, projected out.

### The check that could still overturn this, now running

The residual probes gated 0.441 at seed 11 -- if anything BELOW the null level
rather than at it. If the residuals are ALSO significantly worse than random,
then the decomposition localises nothing: the honest conclusion would become
"any direction derived from a PPO gradient is harmful, shared or not", which is
a weaker and quite different claim. Seed-sweeping decomp_resid1 now to settle
it.

Stating the two outcomes before the numbers arrive:
  * residual indistinguishable from NULL -> the damage is genuinely localised
    to the shared component. Strongest possible version of tonight's finding.
  * residual also below NULL -> localisation FAILS; the finding retreats to
    "PPO-derived directions are harmful, random ones much less so", and the
    shared/residual split is not the right decomposition.

Note both probes are extrapolations: a residual rescaled up to full update
magnitude is not a step any run actually took, and neither is the shared
direction. That is unavoidable in a direction-only comparison, and it is why
the reference levels matter more than the absolute numbers.

## 2026-07-20 ~18:50 — residual behaves like noise; localisation SUPPORTED but not yet direct

Seed-swept decomp_resid1 across 9 match seeds:

    SHARED    (9 seeds)  689- 828  n=1517  0.454  [.429,.479]
    RESIDUAL1 (9 seeds)  773- 841  n=1614  0.479  [.455,.503]
    NULL      (35)      2979-3167  n=6146  0.485  [.472,.497]
    real PPO  (30)      2394-2837  n=5231  0.458  [.444,.471]

    SHARED   vs NULL: -0.0305  z=-2.14  p=0.033   DIFFERS
    SHARED   vs PPO : -0.0035  z=-0.24  p=0.811   indistinguishable
    RESIDUAL vs NULL: -0.0058  z=-0.41  p=0.680   indistinguishable
    RESIDUAL vs PPO : +0.0213  z=+1.50  p=0.134   indistinguishable

**The residual behaves like noise, which is the registered "localised"
outcome.** The earlier 0.441 reading was three gates at a single seed; with
nine it is 0.479, sitting on the null level. That is the third time tonight a
small sample has moved substantially on replication (a16 52.0 -> 45.8; null
50.1 -> 48.5; residual 44.1 -> 47.9), and every one of them moved toward the
boring answer.

**But I must not claim localisation yet, for the exact reason I wrote down
this morning:**

    SHARED vs RESIDUAL: diff=-0.0247  z=-1.39  p=0.165  NOT RESOLVED

"Shared differs from null, residual does not" is two comparisons against a
threshold, not a comparison between the two -- the difference-of-significance
fallacy. I flagged it at 17:20 and then committed a neighbouring version of it
25 minutes later; the discipline is only worth anything if it applies when the
pattern is one I want to be true. **The pattern is consistent with damage
localised to the shared component; the direct test has not resolved it.**

Resolving a 0.025 gap needs ~36 seeds per arm and each has 9. Three more
sweeps launched -- resid2, resid3, and eight further shared seeds -- which
takes the residual arm to n~4500 and the shared arm to n~2900, enough for
z~2.1 on the direct comparison.

Worth noting what the shared/residual split has already achieved regardless:
the shared probe is INDISTINGUISHABLE from a real PPO update (p=0.811) while
carrying only ~39% of an update's length. Whatever the direct comparison
concludes, a minority of the update reproduces all of its harm.

## 2026-07-20 ~19:00 — LOCALISATION REFUTED: every part of a PPO update is harmful

Full sweeps in. Shared arm to 17 match seeds, residual arm to all three
directions x 9 seeds:

    SHARED   (17 seeds)      1304-1586  n=2890  0.451  [.433,.469]
    RESIDUAL (3 dirs x 9)    2107-2604  n=4711  0.447  [.433,.461]
    NULL random (35)         2979-3167  n=6146  0.485  [.472,.497]
    real PPO (30)            2394-2837  n=5231  0.458  [.444,.471]

    SHARED   vs NULL: -0.0335  z=-2.98  p=0.0029  DIFFERS
    RESIDUAL vs NULL: -0.0375  z=-3.88  p=0.0001  DIFFERS
    SHARED   vs PPO : -0.0064  p=0.576   same
    RESIDUAL vs PPO : -0.0104  p=0.298   same
    SHARED vs RESIDUAL: +0.0040  z=+0.34  p=0.736  NOT RESOLVED

**This is registered outcome #2, and it kills the localisation hypothesis.**
At 18:35 I wrote that a residual indistinguishable from noise would mean the
damage lives in the shared component, and that a residual also below noise
would mean "any PPO-derived direction is harmful" and the split is the wrong
decomposition. The second one happened.

**And the 18:50 entry was built on one residual direction.** resid1 alone gave
0.479 (p=0.680 vs null) and I reported the residual as noise-like. The other
two directions are 0.455 and 0.410. Pooled: 0.447, four sigma below null.
resid1 was the outlier of three. Fourth time tonight a reading has reversed on
more data, and this one I had explicitly framed as the completeness check --
the check itself needed a completeness check.

### What actually holds

Both the seed-SHARED direction and the seed-SPECIFIC residuals are
significantly worse than random perturbation, indistinguishable from each other,
and indistinguishable from real PPO updates. **The harm is not carried by a
privileged component. Every direction the gradient produces -- the part all
seeds agree on and the part only one seed found -- is worse than chance.**

That is a stronger and more general statement than localisation would have
been, and it is worse news. It says the problem is not a specific correctable
bias inside the update; it is that the whole gradient-derived subspace points
away from gate performance.

### The intervention I was about to propose is now dead

At 18:35 I proposed projecting the shared direction out of PPO updates during
training. **That would not help.** Projecting out the shared component leaves
the residual, which is equally harmful. Recording this explicitly because it was
the first concrete intervention this line of work had produced and it survived
about twenty minutes.

### Where that leaves the diagnosis

Unchanged in its core, sharper in its implication: PPO's gradient is
anti-informative with respect to head-to-head goal share, and no decomposition
of the update rescues it. The misalignment is between the OBJECTIVES, not
inside the optimiser's step. Fixing it means changing what is optimised --
which is the match-win objective work already sitting in the backlog (#56),
not another tuning pass.

## 2026-07-20 ~18:45 — 24 rungs, 460 iterations, still flat

    iter 360 0.446 | 380 0.404 | 400 0.481 | 420 0.448 | 440 0.464 | 460 0.508 | 480 0.495

    Cochran-Armitage across 24 rungs: z=+0.65  p=0.517   NO RESOLVED TREND
    POOLED iters 20-480: 2003-2381  n=4384  0.457  [.442,.472]
    vs parity: z=-5.71  p<1e-6
    NULL - long600: +0.0278  z=+2.82  p=0.005

Twenty-four checkpoints spanning 460 iterations of training, and the line is
flat. The pooled estimate is tight (+/-1.5%) and sits ~4.3 points below the
champion, exactly where the first update put it.

**The "maybe it just needs longer" hypothesis is dead.** Every arm this project
ever ran used ~145 iterations, and nothing in the data addressed whether that
bound was the limitation. Now it does: 460 iterations buys nothing. The policy
drops on the first update and stays there.

That closes the last open question from the trajectory experiment. What remains
of tonight is the final rungs (500-600) and the consolidated write-up.

# ============================================================
# CONSOLIDATED SUMMARY — the night of 2026-07-20
# ============================================================

## In plain language

We have a "champion" — the best bot we've got. Training is supposed to produce
something better. For weeks it never has, and nobody knew why.

Tonight I ran the test that had never been run: instead of *training* the
champion, I just **jiggled its weights randomly** by the same distance one
training step moves them, and played both against the champion.

  * Random jiggle  -> loses by about 1.5 points. Expected: it is already good,
    so any random change is mildly bad.
  * One real training step -> loses by about 4 points.

**Training does not merely fail to help. It steers worse than random noise.**
If you flipped a coin to choose which way to move the weights, you would beat
what PPO actually did.

Two follow-ups sharpened it:

  * **The damage is instant.** It is fully done after the *first* update and
    never gets worse. I trained 600 iterations and gated 28 checkpoints along
    the way — a dead-flat line. So "it just needs longer" is dead, and so is
    "PPO slowly erodes the policy".
  * **There is no villain component to remove.** I split a training step into
    "the part every run agrees on" and "the part unique to one run" and tested
    each. Both are equally harmful. No surgery on the update fixes this.

What it means: the bot is trained to maximise one thing (shaped self-play
rewards — touch the ball, push it goalward) but judged on another (actually
winning matches). Those two have come apart. Training genuinely improves what
it is told to improve; that improvement no longer buys wins, and past some
point it trades against them.

So there is no tuning fix. Not the reward weights, the learning rate, the reset
positions, the entropy, the opponent pool, the anchor strength, or the run
length — all swept, all rearranging deck chairs. **The objective itself has to
change.**

## ESTABLISHED

1. **PPO's update direction is worse than matched random noise.**
   Engine-matched: NULL 0.485 (n=6146, 35 seeds) vs PPO 0.458 (n=5231, 30
   gates / 3 runs), z=+2.88, p=0.004. Run-level clustered (4 independent runs,
   between-run SD 0.0059): t=-3.97, df~3. Significant under both analyses.
2. **The champion sits at a local optimum of the gate metric.** Any movement
   costs ~1.5 points; PPO's costs ~4. The two effects are real and additive.
3. **The whole effect lands on the first update.** fine iters 1-10 flat
   (p=0.824); long600 28 rungs over 560 iterations flat (z=-0.01, p=0.993),
   pooled 2311-2772 n=5083 0.455 [.441,.468], vs parity p<1e-7.
4. **No decomposition rescues it.** SHARED (17 seeds) 0.451 vs NULL p=0.0029;
   RESIDUAL (3 directions x 9 seeds) 0.447 vs NULL p=0.0001; SHARED vs
   RESIDUAL p=0.736. Every gradient-derived direction is worse than chance.
5. **Not an implementation bug.** PPO plumbing was verified exact in the
   earlier diagnosis. This is objective misalignment.
6. **The gate is honest and the guards earn their keep.** A pure random
   perturbation PASSED the 52% threshold at 56.3%. Under
   `--promote-if-pass` with no confirmation gates, noise would have taken the
   belt. The champion never moved across 6 hill-climb attempts, 8 arms, 28
   long600 rungs, 30 fine rungs, 35 null controls and 6 probes.

## RETRACTED

  * "Replay-state resets are the first upward separation this project has
    measured" — p went 0.023 -> 0.053 -> 0.077 -> 0.189 across four samples.
  * "The lambda ladder is monotone and saturates at parity" — every point above
    lambda 0.5 overlaps every other.
  * "Random movement costs nothing" — 3 seeds said 0.501, 35 say 0.485.
  * The pre-registered binary (a)/(b) framing — it landed at 0.485, between the
    two predicted values, and BOTH mechanisms are true and additive. A forced
    choice guards against retro-fitting and models badly a world where two
    causes stack.
  * cos(update, shared) = 0.65 — self-inclusion bias; leave-one-out gives ~0.19.
  * "The residual behaves like noise" — that rested on resid1 alone (0.479);
    resid2 0.455, resid3 0.410, pooled 0.447.

Four small samples reversed on more data tonight. **Every one moved toward the
boring answer.**

## STILL OPEN

  * The engine wheel deployed to the trainer box mid-night created a
    training-engine confound. It invalidates arm-vs-arm comparisons spanning
    the deploy; it does NOT touch arm-vs-champion gate results, because the
    gate runs on the fixed LOCAL engine. Engine effect at matched iteration
    was +0.070, p=0.119 — suggested, never resolved. Backup:
    /home/elliot/engine_backup_20260719_0613.so (revert is one cp).
  * Why the gradient is anti-correlated with head-to-head performance, at the
    level of mechanism rather than measurement, is not answered. We know THAT
    it is, and that no decomposition localises it.

## RECOMMENDED NEXT STEP

**Change what is optimised, not how.** Task #56 (match-win objective:
score/clock observations + full-match episodes) is the direct attack on the
misalignment and is already scoped in the backlog. Everything else in the
backlog is a tuning lever, and tonight's result says tuning levers are not
worth GPU time.

Deliberate non-action: the trainer box is now free and I am NOT starting
another arm. Another lambda or reward sweep would be motion, not progress.

## TOOLING BUILT TONIGHT (all tested, all committed)

  * `scripts/gate_stats.py` — never reports a share without its interval; says
    NOT RESOLVED in as many words; Cochran-Armitage trend across ordered rungs.
  * `scripts/behavior_distance.py` — KL / agreement / entropy between policies
    on a shared state batch, with an engine null control.
  * `scripts/perturb_null.py` — matched random-direction controls.
  * `scripts/decompose_update.py` — shared vs seed-specific update components.
  * hill-climb fixes: appear-phase poll, whole-dependency preflight, quarantine
    for harness-bug rows, and `ctl.remote_quote` for the ssh/zsh glob trap.

## 2026-07-20 ~19:40 — the 600-iteration run is complete; closing the trajectory experiment

    iter 580  0.453 | iter 600  0.461

    FINAL: 30 rungs, iterations 20-600
    Cochran-Armitage trend: z=+0.02  p=0.980   NO RESOLVED TREND
    POOLED: 2464-2954  n=5418  0.455  [.442,.468]   vs parity z=-6.66

Thirty checkpoints across 600 iterations of training — 100M environment steps
past the champion — and the line does not move. The final estimate is 4.5
points below the champion with a +/-1.3% interval, which is precisely where the
FIRST update put it.

The trajectory experiment set out to test whether the ~145-iteration bound
every previous arm used was the limitation. It was not. There is no rise to
find at 300 iterations, no late collapse at 500, no optimal stopping point in
between. The policy steps off the champion once and then holds station for as
long as anyone cares to pay for.

Run completed cleanly: 600/600, `iter 600 steps 420,618,240 sps 5,276`, trainer
exited on its own. The trainer box is now idle and is being left that way — see
the consolidated summary (a3c440a) for why starting another tuning arm would be
motion rather than progress.

**Nothing was promoted tonight.** Champion ck_000320471040 is unchanged after 6
hill-climb attempts, 8 arms, 30 long600 rungs, 30 fine-grained rungs, 35 null
controls and 6 decomposition probes. Every one of those FAIL verdicts was
correct, and the one PASS that did occur came from pure random noise at 56.3% —
caught by the confirmation gates rather than by luck.

## 2026-07-21 ~01:55 — local gate engine swapped (G2); bit-identity FAILED as expected; re-baselining

Deployment gap G2: to run the match-win gate the local engine must support
match_mode. Built a new wheel from main and ran the mandatory bit-identity
check BEFORE trusting it — gate the same 3 references on old vs new engine in
LEGACY mode (curriculum_v1, reward_v0, seed 11):

    reference (old .so, 07-19 06:13)    new engine (legacy mode)
    armF     59-119  (33.1%)            51-129  (28.3%)
    armH     88- 82  (51.8%)            81- 92  (46.8%)
    long600  84- 99  (45.9%)            83- 77  (51.9%)   <- flipped

**NOT bit-identical.** Not noise -- integer goal counts, one arm flipped 6
points. Cause: the old local .so predates 9 engine commits, and the
replay-state-reset lever (96cd7d0) added an RNG draw to the reset path. The
`zero_replay_weight_is_bit_identical_to_legacy` test proves that inert WITHIN
the new engine, but -- exactly the distinction flagged at ~14:40 today -- it
says nothing about old-vs-new. The new engine's legacy RNG stream diverged, so
kickoffs differ, so games differ.

This is not a bug: the engine legitimately evolved (reward v4, replay lever)
since the old instrument was built. But it means the new engine cannot share a
baseline with tonight's gate history.

**Decision (Elliot): ADOPT the new engine + RE-BASELINE.** The diagnosis is a
CLOSED old-engine campaign (task #58, done). The match-win campaign starts fresh
on the new engine with its own baselines. **Pre/post 2026-07-21 gate numbers are
NOT cross-comparable** -- every share in logs/champion_history.jsonl before this
line was measured on the old engine.

Backup of the old instrument: engine_backup_local_20260719_gate_instrument.so
(revert is one cp). The bit-identity check is the reason we KNOW to re-baseline
rather than silently confounding -- the discipline held.

## 2026-07-21 ~02:45 — match-win gate null characterised; arm launched

Ran the match-win null (champion self-play, match_mode, 20 seeds x 32 arenas x
45000 steps ~= 320 matches/seed) ON THE REMOTE (heavier box, laptop paused):

    NULL over 20 seeds: mean 0.5023  sd 0.0242  95% CI of mean [0.4917, 0.5129]
    defensible promote threshold >= 0.548 (2 sd above 0.5)

Centered at 0.50 as self-play must be; spread 0.024. The 2-seed smoke's scary
sd=0.13 was small-sample noise (~16 matches/seed); at ~320 matches/seed the
match-win gate resolves finely enough to be a real instrument. **Match-win
promote threshold set to 0.55** (2 sd above null), to be used with confirmation
gates like the goal-share gate.

This is the FIRST usable match-win gate baseline, and it is a new-engine
campaign number -- not comparable to the old-engine goal-share history.

**Match-win arm launched (remote):** resume from champion ck_000320471040,
config train_v3_matchwin (reward_v5 win-prob shaping + curriculum_v3_match full
300s matches), KL anchor to the champion at lambda 0.6, seed 20260750. This is
the first training run whose OBJECTIVE is aligned with the gate metric -- the
whole point of the night's diagnosis. Mostly-failing is still the prior; a
promotion here would be the first real progress since the anneal, on an aligned
objective rather than a proxy.

## 2026-07-21 ~03:15 — first match-win gate: iter-20 arm FAILs at 0.420

    matchwin arm ck iter 20 vs champion, match_mode, both orders:
    232W / 74D / 334L  n=640  win_share 0.4203 +/- 0.0198  FAIL (threshold 0.55)

Both side orders symmetric (116W each), so not an order artifact. ~4 sd below
the 0.502 self-play null: by iter 20 the arm has dropped below the champion on
the MATCH-WIN metric, mirroring the goal-share diagnosis (damage in the first
updates).

This is ONE early rung and does NOT settle the question. The diagnosis showed
goal-share drops instantly and stays FLAT below parity forever. The match-win
hypothesis is that an ALIGNED objective, unlike the goal-share proxy, will
RECOVER and climb above the champion as training continues. iter 20 is the
early-disruption point; the trajectory to iter 290 is the test. Gating a ladder
(iters ~100, 200, 290). If it stays flat ~0.42, the aligned objective did not
help and the misalignment was not the whole story. If it climbs toward/above
0.55, that is the first real progress this project has made.

## 2026-07-21 ~03:45 — match-win arm is FLAT too (iter 20 -> 140), preliminary

    iter  20   232W/ 74D/334L  win_share 0.4203 +/- 0.0198
    iter 140   239W/ 63D/339L  win_share 0.4220 +/- 0.0197

120 iterations of match-win training, win_share moved +0.002 -- flat, deep
below the 0.502 null. This is the SAME shape as the goal-share diagnosis: the
policy drops below the champion in the first updates and holds flat below,
now on the ALIGNED metric it is being trained on.

Preliminary (2 rungs). The iter-290 final rung will confirm. But if it holds,
the implication is heavy: **aligning the training objective with the gate metric
did NOT make PPO improve the policy.** The misalignment was real but was not the
cause of the no-improvement -- the problem is deeper.

Caveats to weigh before concluding, in case the final rung is also flat:
  * KL anchor lambda 0.6 may be too strong -- it holds the policy near the
    champion, and "near the champion" is ~0.42 here. A lower lambda might let
    it climb (or collapse, as no-anchor did in the diagnosis). Untested for
    match-win.
  * Mixed team sizes [0.5,0.3,0.2]: only ~50% of training is 1v1, but the gate
    is 1v1. A 1v1-only match-win arm targets the gated metric directly.
  * 290 iters may be too few -- though the goal-share drop was instant and flat
    over 600, so this is unlikely to be the explanation.

## 2026-07-21 ~04:15 — CORRECTION: match-win arm CLIMBS to parity (not flat)

My 03:45 "flat" read was premature -- it had only iters 20 and 140. The full
trajectory (match-win gate, both orders, ~640 matches/rung, null 0.502):

    iter  20   0.4203 +/- 0.0198   below
    iter 140   0.4220 +/- 0.0197   below
    iter 180   0.4508 +/- 0.0196   below
    iter 280   0.497  +/- 0.014    PARITY  (pooled 2 gates: 0.4877, 0.5060)

**The arm climbed from 0.42 to parity over the second half of training** --
monotone, and still rising at the last point (0.45 -> 0.50 from iter 180 to
280). It did NOT beat the champion (parity, not >= 0.55 threshold), so no
promotion. But this is CATEGORICALLY different from the goal-share diagnosis,
which was flat below parity over 600 iterations and never moved.

(Note: the match-win gate has ~+/-0.02 run-to-run noise even at fixed seed,
from physics-blowup containment rebuilding arenas nondeterministically. Hence
pooling repeat gates.)

**Interpretation.** The ALIGNED objective produced the first upward training
trajectory this project has seen. The policy recovers from the early-update
drop and climbs back to parity. Whether it can EXCEED parity is the open
question: at iter 280 it is still climbing, so the run may have been stopped
too early. The KL anchor (lambda 0.6) may also cap it at parity -- "near the
champion" is now ~parity rather than below it.

**Recommended next: EXTEND the run.** Continue training from iter 280 (or launch
a longer arm, ~600 iters) and gate the ladder. If it breaks above 0.55, that is
the first real improvement over the champion this project has produced -- on an
objective aligned with what we measure. If it plateaus at parity, the anchor is
the cap and a lower-lambda arm is the next lever.

## 2026-07-21 ~05:00 — full match-win trajectory: climb to parity, plateau, then degenerate

Extended the arm 300 more iters (iter 280 -> 580, anchored to champion lambda 0.6).
Full match-win trajectory (both orders, ~640 matches/rung, null 0.502):

    iter  20   0.4203   below
    iter 140   0.4220   below
    iter 180   0.4508   below
    iter 280   0.497    parity
    iter 420   0.4922   parity
    iter 520   0.5078   parity (high point, ~0.3 sd above null -- NOT sig.)
    iter 580   0.3951   COLLAPSE: 200W/199D/359L, n=758

**Three phases.** (1) Climb from 0.42 to parity over iters 20-280 -- the first
upward training trajectory this project has produced. (2) Plateau at parity
iters 280-520 (0.49-0.51, never significantly above the 0.55 threshold). (3)
Degeneration at iter 580: win_share crashes to 0.395 with a DRAW EXPLOSION (199
draws / 758 matches = 26%, vs ~11% elsewhere) and 18% more matches -- shorter
games, i.e. more physics blowups. A win-probability objective can reward
NOT-LOSING (defensive, clock-killing, drawing) over winning, and here that
degeneracy emerged with extended training and drove the solver unstable.

**What the aligned objective bought, and its limits:**
  * It RECOVERED the policy to parity -- categorically better than the
    goal-share proxy (flat below parity over 600 iters, never moved).
  * It did NOT exceed the champion. Plateaued at parity: the KL anchor
    (lambda 0.6) caps it near the champion -- "parity by staying close",
    the same ceiling armH hit in the diagnosis, but reached by climbing UP
    rather than starting there.
  * Extended training DEGENERATED toward drawing. The win-prob potential's
    clock-sharpening rewards protecting a tie late; without a counter-incentive
    to actually win, the policy learned to not-lose.

**Next levers (for Elliot, not tonight):**
  1. LOWER the KL anchor (lambda 0.3-0.4) so the policy can climb PAST parity
     instead of being pinned at it -- the plateau strongly implicates the anchor
     as the cap. Risk: collapse, as no-anchor did in the diagnosis. The
     match-win objective may tolerate a weaker anchor better than the proxy did.
  2. Counter the draw-degeneracy: add a small explicit win/goal term on top of
     the potential, or cap the clock-sharpening, so not-losing is not enough.
  3. The best checkpoint is iter 520 (0.508) -- essentially parity, not a
     promotion, so the champion pointer does NOT move. Correct per the gate.

Champion ck_000320471040 STILL unbeaten -- but for the first time, training
walked a policy UP to its level rather than only down and flat.

## 2026-07-21 ~16:35 — league_tick resumed on the new engine (re-baseline)

Resumed league_tick_loop on the remote after the engine swap. Per Elliot, on the
NEW engine (option 1: accept re-baseline rather than restore the old engine).
The 48-entry registry's TrueSkill ratings were computed on the OLD engine
goal-share; matches from here use the new engine, which is not goal-share
bit-identical (see 01:55). So the ladder has a discontinuity at this point --
pre/post ratings are not on the same scale. Acceptable: the goal-share campaign
is closed, and future ladder motion is internally consistent on the new engine.

## 2026-07-21 ~18:15 — LEVER 1 (lower anchor) REFUTED: degenerates faster

    arm         iter 140    iter 280
    lam06 (0.6) 0.422       0.497 (parity)
    lam03 (0.3) 0.460       0.399 (110 draws, degenerating)

The weaker anchor climbed faster early (0.46 vs 0.42 at iter 140) but
DEGENERATED SOONER: by iter 280 it is draw-seeking (0.40, 17% draws), where
lam06 was still holding parity. lam06 degenerated at iter ~580; lam03 at ~280.

**"The anchor is the cap, lower it to climb past" is refuted.** The reverse is
true: the KL anchor was PROTECTING against the draw-degeneration -- holding the
policy near the champion suppressed the draw-seeking collapse. Removing anchor
removed the protection.

So the draw-degeneration, not the anchor, is the binding problem. That is what
LEVER 2 targets: reward_v5.1 raises win_prob_t_floor 0.05 -> 0.2, flattening the
late clock-sharpening that makes conceding-late so costly the policy stops
attacking. Launching lever 2 at lambda 0.6 (the PROTECTIVE anchor kept) --
single variable vs lam06: only t_floor changes.

## 2026-07-21 ~22:00 — LEVER 2 (t_floor flatten) BACKFIRED: draws UP, not down

    iter 140 draws:  lever-0 (t_floor .05) 63 | lever-1 (lam03) 75 | lever-2 (t_floor .2) 120
    lever-2 iter 140 win_share 0.4337 (n=664), 120 draws (18%)

Raising t_floor to flatten the late clock-sharpening was meant to reduce
draw-seeking by making conceding-late less costly. It did the OPPOSITE -- draws
DOUBLED. The error in my mechanism reasoning: flattening Phi reduces the reward
for SCORING late by the same amount it reduces the penalty for conceding.
Net = less incentive to attack = MORE draws.

**The draw-seeking is inherent to a SYMMETRIC win-prob potential.** Phi(score_diff)
is antisymmetric, so the incentive to score and the fear of conceding scale
together; tuning t_floor moves both. No symmetric knob fixes it.

**The fix must be ASYMMETRIC: reward WINNING more than not-losing.** That is the
direct goal term I dismissed earlier for the (wrong) t_floor approach -- LEVER 3:
reward_v5.2 keeps the win-prob potential (t_floor back to 0.05) and adds a small
direct goal reward (goal > 0) so scoring pays on its own, independent of the
potential, breaking the symmetry that produces draw-seeking. Prepped; launch
after lever-2's iter-280 confirms.

## 2026-07-22 ~00:15 — four match-win arms: pure win-prob (parity) beats every "push-past" lever

    arm                          iter140  iter280   outcome
    lever-0  win-prob λ0.6       0.422    0.497     CLIMBS TO PARITY (best)
    lever-1  win-prob λ0.3       0.460    0.399     degenerates faster
    lever-2  win-prob t_floor.2  0.434    ~0.44     draws worse (120 @140)
    lever-3  win-prob + goal1.5  0.439    0.429     FLAT below parity

**Every attempt to push PAST parity made it WORSE than the original pure
win-prob arm.** Each failed in an instructive, distinct way:
  * lever-1 (weaker anchor): the anchor was PROTECTIVE against the draw-
    degeneration, not the cap. Less anchor -> faster collapse.
  * lever-2 (flatter potential): a SYMMETRIC knob scales scoring-reward and
    concede-penalty together; flattening cut the win incentive -> MORE draws.
  * lever-3 (direct goal term): re-introduced goal-share -- the MISALIGNED
    proxy the whole diagnosis showed is inert -- so it flatlined below parity,
    exactly like the diagnosis arms.

**What is now established:**
  1. The pure win-probability aligned objective (lever-0) is the right one:
     it is the ONLY thing that has ever walked a policy UP to the champion's
     level. First real progress since the anneal.
  2. Parity is the ceiling of this training approach. Three reward/anchor
     tweaks to exceed it all made it worse. Beating the champion is NOT a
     tuning problem.
  3. The champion (ck_000320471040) is a strong, defensive 1v1 local optimum;
     a KL-anchored policy converges TO it, and the anchor that stabilises the
     climb also caps it at parity. This is a genuine tension, not a bug.

**This is a step-back-to-Elliot point, not a launch-lever-4 point.** The
directional options are structural, not another reward knob:
  A. Opponent diversity -- gate/train vs a POOL (past champions, external
     bots #54) rather than only the frozen champion, so the policy must
     generalise past one opponent's weaknesses (AlphaStar-style league).
  B. Progressive anchor -- promote to parity checkpoints and re-anchor, so
     "near the champion" ratchets upward (gated hill-climb, but the gate now
     measures the aligned metric).
  C. Longer horizon / value-fn work -- 4500-step matches vs a 217-step gamma
     may under-credit; a higher gamma or match-outcome value head.
  D. Accept parity as the deploy candidate -- the parity policy is not WORSE
     than the champion and was reached on the aligned objective; it is a
     legitimate (if unexciting) checkpoint.

Champion STILL unbeaten across 6 hill-climb attempts, 8 diagnosis arms, and
4 match-win arms -- but the match-win objective reliably reaches parity, which
nothing before it did.
