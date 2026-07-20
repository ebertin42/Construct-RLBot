# Match-win objective (task #56) — design

**Date:** 2026-07-20
**Status:** approved, ready for implementation planning
**Supersedes nothing. Blocks nothing.**

## Why

Tonight's diagnosis (docs/training-journal.md, consolidated summary at commit
a3c440a; task #58) established, with a matched random-direction null control:

* Perturbing the champion a fixed distance in a **random** direction costs
  ~1.5 goal-share points. It sits at a local optimum of the gate metric.
* Perturbing it the same distance in the direction **PPO chooses** costs ~4
  points. Engine-matched: NULL 0.485 (n=6146, 35 seeds) vs PPO 0.458
  (n=5231, 30 gates / 3 runs), z=+2.88, p=0.004. Run-level clustered across 4
  independent runs (between-run SD 0.0059): t=-3.97, df~3.
* The entire effect lands on the **first** update and never deepens — flat
  through iteration 10 (p=0.824) and across 30 rungs of a 600-iteration run
  (z=+0.02, p=0.980).
* **No decomposition of the update rescues it.** The seed-shared direction
  (0.451, p=0.0029 vs null) and the seed-specific residuals (0.447, p=0.0001)
  are each worse than random and indistinguishable from each other (p=0.736).

PPO's gradient is *anti-informative* with respect to head-to-head performance,
not merely uninformative. The plumbing was verified exact in the earlier
diagnosis, so this is not a bug: **the training objective and the evaluation
metric are measuring different things.** Training maximises shaped self-play
reward (touch the ball, carry it goalward); the gate measures goal share
against a frozen champion. Those have come apart, and every tuning lever —
reward weights, entropy, lambda, reset distribution, opponent pool, run length
— was swept without effect, because each adjusts *how* the optimiser steps
rather than *what* it optimises.

This spec changes what is optimised.

## Decisions taken

1. **Objective.** Training and evaluation both target match wins.
2. **Mechanism.** Potential-based reward shaping with a win-probability
   potential — NOT a terminal match reward (see "Why not a terminal reward").
3. **Gate.** Two-stage: cheap goal-share screen, then a match-win gate that
   decides promotion.
4. **Phasing.** Ship the reward change first with no observation change; add
   score/clock observations only afterwards, as a separate phase.

## Why not a terminal reward

The obvious design — pay +1 for winning the match, -1 for losing — is dead on
arithmetic. `tick_skip = 8` at 120 Hz gives 15 Hz decisions, so a 5-minute
match is **4500 decision steps**. At the live `gamma = 0.9954`
(configs/train_v1.toml:26) the effective horizon is ~217 steps (~14.5 s), and a
reward 4500 steps away arrives discounted by 0.9954^4500 ~= **1e-9**. Not weak
— numerically absent. `rollout_steps = 256` compounds it: one rollout covers
17 seconds of a five-minute match.

Raising gamma to ~0.9995 would reach a ~2000-step horizon but destabilises
value learning and invalidates the tuning of everything downstream of the
current horizon. Rejected.

**Potential-based shaping avoids the problem entirely.** With
`F(s,s') = gamma * PHI(s') - PHI(s)`, the shaped rewards telescope to
`PHI(end) - PHI(start)` — the match outcome — while each individual reward is
delivered *at the step where the game state changes*, well inside the current
217-step horizon. Ng, Harada & Russell (1999) proves potential-based shaping
leaves the optimal policy unchanged for **any** PHI.

That guarantee is doing real work here: PHI's accuracy affects only sample
efficiency, never what the policy converges to. So Phase 1 can ship a simple
analytic PHI and refine it later from the 13k-replay SSL corpus without
changing the target. Getting PHI wrong costs learning speed, not correctness.

## Phase 1 — reward and match layer (this spec)

### The match layer

Today a goal *terminates* the RL episode and the arena resets
(engine/src/episode.rs:748-751, reset at :803). A goal is detected at :725 and
immediately discarded — nothing accumulates it. There is no score, no match
clock, no match concept anywhere in the engine, and rocketsim_rs supplies none
(documented at engine/src/obs_v1.rs:26-33).

The change decouples **match** from **RL trajectory**:

| event | today | Phase 1 |
|---|---|---|
| goal scored | `terminated` -> reset | score updated, kickoff reset, match and trajectory both CONTINUE |
| clock expires (300 s) | truncated | `terminated` — the real trajectory boundary |
| 30 s no touch | truncated | unchanged (stall guard) |
| physics blowup | terminated + arena rebuild | unchanged |

New state on `EpisodeArena` (engine/src/episode.rs:190-217): `score_blue`,
`score_orange`, `match_start_tick`. Three fields.

`MAX_TICKS` is already `300 * TICKS_PER_SEC` (episode.rs:21), so the clock cap
needs no change — today it simply never fires, because a goal always ends the
episode first.

**Why this matters for credit assignment:** `gae.py:22` bootstraps 0 on
`terminated`. Today that means every goal severs credit. After the change a
goal is a scoring event *inside* a trajectory and credit flows across it —
which is precisely the judgement we want learned ("was that goal worth
conceding the counter-attack?").

Trajectories now span ~4500 steps against `rollout_steps = 256`, i.e. ~18
rollouts. This is already how the engine behaves: arenas are not reset between
`collect()` calls, and GAE bootstraps across rollout boundaries via
`values[t+1]`. No change required.

### The reward

New `configs/reward_v5_winprob.toml`. Adds one term:

* `win_prob_weight` — coefficient on `gamma * PHI(s') - PHI(s)`.

PHI is a win probability in [0,1] computed from `(score_diff, time_remaining)`
from the agent's team perspective. Phase 1 pins the analytic form explicitly so
there is nothing to interpret:

```
t_frac = time_remaining / MATCH_SECONDS        # 1.0 at kickoff -> 0.0 at final whistle
k      = K_BASE / max(t_frac, T_FLOOR)         # slope sharpens as the clock runs down
PHI    = sigmoid(k * score_diff)
```

with `K_BASE` and `T_FLOOR` as config constants (`win_prob_k_base`,
`win_prob_t_floor`). `T_FLOOR` prevents division blow-up in the final seconds
and caps how sharp PHI can get.

Properties this form guarantees, each of which is a test below:
* `PHI = 0.5` at 0-0 for any clock, so kickoff carries no bias
* `PHI(score_diff, t) = 1 - PHI(-score_diff, t)`, so the two teams' potentials
  are exact complements and the shaped game stays zero-sum
* a fixed lead is worth strictly more as `t_frac` falls

The gamma in `gamma * PHI(s') - PHI(s)` is the TRAINING gamma
(`cfg.ppo["gamma"]`), not a separate constant. Using any other value breaks the
Ng et al. guarantee.

Existing shaping terms (`goal`, `touch`, `vel_to_ball`, `vel_ball_to_goal`,
`offensive_potential`, `touch_accel`, `aggression_bias`) remain in
`RewardConfig` (engine/src/reward.rs:5-24) and default to 0.0 in v5. This keeps
every historical config loadable and makes the A/B against reward_v4_1 a config
swap rather than a code change.

`team_spirit` / `opp_spirit` blending (episode.rs:135-143) applies unchanged.

**Reward is currently a pure function of raw game state** (reward.rs:56) with no
episode or match state threaded in. Phase 1 must thread score and match clock
into that call. This is the largest single change to an existing signature and
should be done explicitly rather than via a global.

### The gate

Two stages, reusing the confirmation-gate architecture already built:

1. **Screen** — the existing goal-share gate, unchanged. Cheap, high power,
   rejects candidates that are clearly worse. Most candidates stop here.
2. **Confirm** — full match-win gate. Only survivors pay for it. This decides
   promotion.

**The match layer must be config-gated.** Turning it on unconditionally would
also change the goal-share SCREEN: goals would stop terminating episodes there
too, altering reset dynamics and making the screen's numbers incomparable with
every historical gate in `logs/champion_history.jsonl`. That would destroy the
baseline this whole spec is measured against. So the match layer is selected by
the curriculum/episode config, the screen keeps running in legacy episode mode,
and only the confirm stage runs full matches. The two stages measure different
things deliberately.

**No new engine API is needed.** `collect()` already returns `terminated`
(engine/src/lib.rs:351-364), which now means "match ended", and
`python/construct/league/matches.py:38` already forces
`reward_config="configs/reward_v0.toml"` as a neutral scoring tape independent
of the training reward — so the `GOAL_THRESHOLD = 9.4` spike detection
(matches.py:23, :64-78) keeps working regardless of what the policy trained on.
Grouping that tape by `terminated` boundaries per arena yields per-match scores,
hence W/D/L.

**Cost, honestly.** A match-win gate is far less sample-efficient than goal
share: today's ~96 arena-minutes yields ~180 goals (SE 3.7%) but only ~19
matches (SE 11.5%, a +/-22% band). Matching today's precision needs ~183
matches, ~10x the compute.

The mitigation is parallelism, not cleverness: the gate runs **8 arenas**
(configs/champion.toml) while training runs 192. Raising gate arenas to 32-64
recovers most of the wall-clock, because the work is parallel. **This must be
measured, not assumed** — engine throughput does not scale linearly in arenas
forever.

### Why no observation change in Phase 1

`query[60..62]` are reserved scoreboard slots, hardcoded to 0.0 in three
synchronised places (engine/src/obs_v1.rs:342-344, deploy/obs.py:260-263, and
transitively replay/src/bc_obs.rs:433). Filling them is cheap — `Q_FEAT` stays
64, so no checkpoint is invalidated.

But it makes the champion **uncomparable**. The champion is a v1 checkpoint
trained with those slots always zero; an engine that fills them with real
values feeds it out-of-distribution input. Bumping to schema v2 makes
`require_compatible` (scripts/h2h_eval.py:79-109) refuse the pairing outright,
leaving no way to ask "is this better than the champion?" — the only question
that matters.

Phase 1 therefore ships the reward alone. The policy learns lead-protection
implicitly through the reward without being told the score. This is also the
right call on evidence hygiene: on 2026-07-20 an engine deploy silently
confounded a full day of arm-vs-arm comparisons, and the fix was to vary one
thing at a time.

## Phase 2 — score/clock observations (deferred, sketched only)

Only if Phase 1 shows life. Fill `query[60] = clamp(score_diff/5, -1, 1)`
(signed from the agent's team perspective), `query[61] = time_frac`,
`query[62] = overtime`, per the original plan
(docs/superpowers/plans/2026-07-16-entity-transformer-obs-v1.md:42).

Requires per-policy obs-variant plumbing so old and new stay comparable: zero
the slots for v1-schema policies, real values for v2, both playing the same
match. This works precisely because `Q_FEAT` is unchanged — each policy is
evaluated on the observation distribution it was trained on.

`deploy/obs.py:260-268` already has the real values available from RLBot and
deliberately discards them; Phase 2 stops discarding them.

**Test that must be rewritten, not deleted:** `obs_v1.rs:425-444`
(`orange_obs_mirrors_blue_at_kickoff`) and `engine/tests/obs_v1_test.rs:78-86`
assert blue and orange query rows match. `score_diff` is sign-flipped per team,
so it is 0 at kickoff and the assertion still passes — but its *premise* (every
query float is team-symmetric) becomes false. Rewrite to assert
mirror-ANTIsymmetry for the score slot specifically.

## Testing

**Load-bearing test:** summed shaped reward across a complete match must equal
`PHI(end) - PHI(start)` to floating-point tolerance. This is what proves the
shaping is genuinely potential-based and has not quietly become an exploitable
dense reward. If it fails, the Ng et al. guarantee does not apply and the
policy is optimising something other than match wins.

Rust (engine):
* a goal does NOT set `terminated`; score increments; kickoff reset occurs
* clock expiry DOES set `terminated`
* score resets to 0-0 at match start, not at kickoff
* PHI is complementary: `PHI(d, t) == 1 - PHI(-d, t)` for all d, t
* PHI at 0-0 is exactly 0.5 for EVERY clock value, not just kickoff
* PHI sharpens monotonically: a +1 lead is worth strictly more at 30 s
  remaining than at 270 s
* `T_FLOOR` holds: PHI stays finite and bounded at t_remaining = 0
* the shaped game is zero-sum: blue's shaped reward equals -orange's on every
  transition (follows from complementarity, but assert it directly)
* v5 config parses; historical reward_v0/v3/v4_1 configs still parse

Python:
* reward tape + `terminated` boundaries group into the expected match records
* W/D/L aggregation, including a 0-0 draw
* win-share statistics and their intervals (reuse scripts/gate_stats.py)

**Null control (mandatory).** A policy playing itself must gate at 50% win
share. Tonight a random perturbation PASSED the 52% goal-share threshold at
56.3%, which is the entire justification for the confirmation gates. The
match-win gate is noisier per unit compute and needs its own null distribution
characterised BEFORE any promotion decision is trusted.

## Error handling

Every failure in this path must be loud. On 2026-07-20 a replay-reset arm ran a
full attempt inertly because a missing capability degraded to a silent no-op,
and the run looked completely normal. Accordingly:

* a missing or malformed PHI config is a hard error, never a default
* the engine logs a positive confirmation line when the win-prob term is active
  (mirroring `[curriculum] replay pool ...`), so "is it live?" is answerable
  from the log rather than inferred from absence of error
* match-layer state is asserted consistent at match start (score 0-0, clock
  reset), not assumed

## Out of scope

* Raising gamma or changing `rollout_steps`
* Overtime / golden goal (Phase 2 reserves `query[62]`; matches draw for now)
* Fitting PHI from the SSL corpus (a refinement; the Ng guarantee means it
  cannot change the optimum)
* 2v2 / 3v3 match gating — `matches.py:45` hard-asserts `mode == 1` because
  2v2+ multi-counts goals across teammates. Unchanged.
* Any change to `python/construct/_engine.abi3.so` deployment cadence. The
  LOCAL engine is the gate instrument and must not move mid-measurement
  (see the gate-instrument memory and the 2026-07-20 ~16:00 journal entry).

## Risks

1. **PHI shape is a free parameter.** The Ng guarantee protects the optimum,
   not the learning dynamics. A badly-scaled PHI could make the signal too weak
   to learn from within a reasonable budget. Mitigation: the telescoping test
   plus an early check that per-step shaped reward magnitudes are comparable to
   the v4.1 goal event (~±13 after spirit blending).
2. **Longer trajectories change the data distribution per rollout.** 18
   rollouts per trajectory instead of ~1 means the value function sees far more
   mid-match states and far fewer kickoffs. This is intended, but it is a real
   distribution shift and could interact with the KL anchor.
3. **The match-win gate's null distribution is uncharacterised.** Must be
   measured before trusting any promotion (see Testing).
4. **Phase 1 may show nothing.** Tonight's result says the objective is
   misaligned; it does not prove that THIS realignment is sufficient. The
   honest prior is that this is the best-motivated single change available, not
   that it is guaranteed to work.
