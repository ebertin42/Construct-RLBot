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
