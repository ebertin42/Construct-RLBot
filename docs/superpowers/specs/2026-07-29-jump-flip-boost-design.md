# Jump, flip and boost-pad behaviour — reward fix (run v10 fork)

Date: 2026-07-29
Status: approved, ready for implementation plan

## Problem

Three behaviours are absent from the v9 policy at 2.02B steps, reported from
watching the viewer and confirmed by telemetry:

1. **The bot never jumps.** `p_jump` has fallen to 2.5e-3 — about 123x below
   the 32/104 = 30.8% a uniform policy over the action table would give. It has
   fallen monotonically all run (1.20e-2 at 400-600M).
2. **The bot never jump-flips.** A dodge needs a jump rising edge while
   airborne with a flip available, i.e. jump -> release -> jump. At
   p_jump = 2.5e-3 that sequence is sampled at roughly the square of an already
   tiny number.
3. **The bot ignores boost pads**, including free ones already on its path.

### Root cause

All three share one cause: **the reward function only pays for ball-directed
behaviour.** The nine reward terms are `goal, touch, vel_to_ball, touch_accel,
vel_ball_to_goal, offensive_potential, aerial_touch, air_setup, win_prob`. Two
of those are aerial terms that have never fired.

- `aerial_touch` is gated by `height_ramp`, a clamped linear ramp on
  `cur.ball.pos.z` from `aerial_z_lo = 400` to `aerial_z_hi = 1400`. Measured
  over 122 telemetry windows / 25,016 airborne touches, the ball-z distribution
  at touch is **69% below 150uu, 29% in 150-300uu, 1.67% above 300uu**, and the
  >300uu fraction is FLAT across the whole run (t = -1.24). The payout ramp
  begins at 400uu, so **the aerial payout has never paid out.**
  `air_gate_frac` (learner) averages 0.00021 and is nonzero in only 1,937 of
  17,606 iterations.
- `air_setup` is negative in 99.0% of iterations (mean -0.718). The magnitude is
  small — about 0.3% of `r_touch` (~250) — so this is not punishment; it is the
  absence of any positive counterweight.
- There is **no boost or pad reward term at all**. Pads ARE observable (34 pad
  entities with positions and cooldown timers, correctly mirrored) and own-boost
  is in the observation as `f20 = s.boost / 100.0`. `vel_to_ball` pays for
  moving at the ball, so a detour is a small loss and a free pad is worth
  exactly zero. The behaviour is indifference, not blindness.

Neither the action space nor the observation is the blocker. 22 of 32 jump rows
carry nonzero pitch or yaw (dodge-capable share is 25% of the table by
`actions.rs`'s own accounting), and `has_flip_or_jump()` is in the observation
at `obs_v1.rs:165`. `actions.rs` sized the v1air table's +12 rows against a
UNIFORM policy, quoting "one takeoff per car per 27 seconds". At the trained
policy's actual jump rate that becomes roughly one per 55 minutes. **The action
table work was undone by a reward term that never paid.**

## Approach

Fork a second run rather than modifying the live one. Run A (v9) is at its
fastest measured improvement rate of the experiment (1.3e-3 win-share per
Mstep, ~2x its earlier rate) and is not to be risked. Run B is a controlled
comparison on the same box, same engine, same wall-clock.

Move reward levers ONLY. Entropy stays at 0.006. Two variables moving at once
would make a null result unattributable — the failure mode this session has
repeatedly paid for.

## Changes

### Lever 1 — aerial payout ramp floor (config only)

```toml
aerial_z_lo       400.0 -> 150.0     # payout ramp floor
aerial_meas_z_lo  400.0  UNCHANGED   # instrument stays fixed
aerial_z_hi      1400.0  UNCHANGED
aerial_touch        1.5  UNCHANGED
```

`height_ramp` is `((ball_z - z_lo) / (z_hi - z_lo)).clamp(0,1)` — a graded ramp,
not a step. Lowering the floor strictly RAISES payout for every touch above
150uu and introduces no cliff:

| ball z at touch | payout now | after |
|---|---|---|
| 300uu | 0 | 12% |
| 800uu | 40% | 52% |
| 1400uu | 100% | 100% |

31% of current airborne touches land above 150uu, so a term that has paid zero
for 2B steps begins paying on roughly a third of them, with the gradient
pointing up.

`aerial_meas_z_lo` is deliberately NOT moved. `reward.rs` separates the payout
floor from the measurement floor precisely so "the payout ramp can move without
moving the instrument that scores it". Keeping it at 400 leaves
`T_AERIAL_TOUCH_EVENTS` / `air_gate_frac` directly comparable between run A and
run B.

### Lever 2 — new `boost_pickup` event term (engine)

```
T_BOOST_PICKUP = 20   reward      cfg.boost_pickup * (gained / 100.0)
T_BOOST_GAINED = 21   instrument  raw boost units gained (coefficient-free)
T_FLIP_EVENTS  = 22   instrument  dodge fires (has_flipped false -> true)
N_TERMS 20 -> 23
cfg.boost_pickup: f32  with #[serde(default)] = 0.0
```

where `gained = cur.cars[i].state.boost - prev.cars[i].state.boost`, counted
only when positive.

**Event, not potential — deliberately.** A potential on boost AMOUNT
(`F = γΦ(s') - Φ(s)` with `Φ = boost`) penalises SPENDING boost and yields a
hoarder. That is the identical γ-drag trap that left `air_setup` net negative
for 2B steps. Scaling by amount gained makes a big pad (100) worth ~8x a small
pad (12), matching their real value.

**Coefficient: `boost_pickup = 0.2`**, derived rather than guessed. Against the
live scale (`goal 10.0, touch 0.5, touch_accel 0.5, aerial_touch 1.5,
air_setup 0.3, vel_to_ball 0.05`):

- a big pad (gained 100) pays `0.2 * 1.0 = 0.2` — 40% of a `touch`
- a small pad (gained 12) pays `0.2 * 0.12 = 0.024`
- `vel_to_ball` pays up to 0.05 per decision, so a big pad is worth about
  `0.2 / 0.05 = 4` decisions of forgone ball approach, i.e. roughly a 0.27s
  detour at decision period 8

That is the intended shape: a pad already on the path is always worth taking
(any positive value beats zero when the detour cost is zero), while a detour of
more than a fraction of a second never is. A coefficient of 0.5 would buy a
~0.67s detour and would start distorting play; 0.2 does not.

`#[serde(default)]` means any tape that does not set `boost_pickup` is
reward-identical to today, so run A is unaffected if it ever restarts onto the
new engine.

### Instrumentation rationale

`T_FLIP_EVENTS` exists because flips are currently **unmeasurable** — the
question "does the bot flip?" cannot be answered except by watching. Adding
`air_z` telemetry is what made the aerial question decisive earlier in this
session; this is the same move. `T_BOOST_GAINED` is coefficient-independent so
the instrument does not move when the payout is tuned.

## Rollout

1. Build the wheel with `maturin build -o <path outside the repo>`.
   **NEVER `maturin develop`** — it would overwrite
   `python/construct/_engine.abi3.so` (sha `deab5d3b988d62d3...`), the local
   gate instrument every gate and bench result in this session was scored on.
2. Ship the wheel to the remote box and `pip install` into its venv. Run A is
   unaffected: it holds its old `.so` in memory and would only pick up the new
   engine on its own next restart.
3. Start run B: `checkpoints_v10/`, `configs/reward_v10_jumpboost.toml`, forked
   from run A's newest checkpoint. A common ancestor makes divergence
   attributable.
4. **Run A is never restarted by this work.**

Curriculum state restores from the forked checkpoint, so run B begins at rungs
currently producing in-band win rates. This satisfies
`regime-change-needs-contested-start`, which records a new reward regime
collapsing when started against an easy opponent with a stale critic.

Remote box has 16 cores at load 5.32 and 22 GB free, so a second trainer fits
with headroom. Expect a modest sps drop, not a halving — to be measured, not
assumed.

## Success criteria

Read at ~200M steps on run B, with run A as the control.

| signal | now | fix worked if |
|---|---|---|
| `p_jump` | 2.5e-3 | rises toward 1e-2+ |
| `air_gate_frac` (fixed 400uu instrument) | 0.0002 | clearly nonzero |
| `air_z` >300uu share | 1.67%, flat over 2B steps | rises, \|t\| > 3 |
| `T_FLIP_EVENTS` | unmeasured | nonzero at all |
| `T_BOOST_GAINED` | unmeasured | rises vs run A |
| p5 bench cluster | 0.3524 | **must not regress** |

The last row is the guardrail. If jumping appears but the bench drops, the
trade was not worth making and run B should be abandoned rather than tuned.

Bench comparisons use CLUSTER means (>=3 checkpoints x 2 seeds per window),
never single checkpoints: between-checkpoint sd is ~0.09 against ~0.017 across
seeds within a checkpoint.

## Risks

- **The fix may simply not bite.** p_jump = 2.5e-3 may be too thin to climb even
  a reachable gradient. Accepted knowingly in exchange for an attributable
  result; the fallback (raise entropy, or a small direct jump bonus) stays
  available as a second, separately-measured experiment.
- **Two trainers share the box.** Both runs slow. Any bench or gate measured
  during this period is on a loaded box — `gates-need-idle-box` measured CPU
  load biasing win_share 2-3 points low. Bench run B and run A under the SAME
  load, or not at all.
- **Reward farming.** A pickup reward is farmable in principle. Mitigated by a
  small coefficient, pad cooldowns, and `T_BOOST_GAINED` being watched directly
  so farming is visible rather than inferred.

## Out of scope

- Any change to `air_setup` — it is the term already suspected of being
  quietly negative, and moving it alongside the payout gate would make a null
  result unattributable.
- Any entropy change.
- Side-dodge action rows. `actions.rs` excluded them deliberately; front/back
  flips are already expressible and are what is being tested.
