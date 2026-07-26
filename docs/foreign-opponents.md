# Foreign opponents (ported community bots)

A **foreign opponent** is an externally-trained community bot ported into our Rust
engine so PPO can train against it. Unlike a league opponent (`set_opponents`,
which loads one of OUR `EntityPolicyNet` checkpoints), a foreign bot brings its
own observation builder, network and action table — it only shares the arena.

This exists because every own-checkpoint arm failed to beat the champion
`ck_000320471040`, including the Option-A league arm (iter-290 win_share **0.4570
FAIL**, null 0.502, threshold 0.55). Own-checkpoint "diversity" is just beating
versions of yourself. A foreign bot is genuinely new blood. See
`docs/superpowers/specs/2026-07-22-immortal-port-design.md`.

Currently ported: **Immortal**, **Element** (RLMarlbot, rlgym-ppo/MLP, unlicensed),
**Nexto** (~GC1) and **Necto** (~Diamond) (Rolv-Arild/Necto, CC BY-NC-SA 4.0).

Measured strength of our champion `ck_000320471040` against each (`bench_foreign.py`):

| opponent | our win_share |
|---|---|
| Immortal | 0.0953 (25W/11D/284L) |
| Nexto | 0.0000 (0W/0D/96L) |
| Necto | NOT MEASURABLE YET (obs builder unimplemented — see below) |

Nexto is far too strong to be a useful teacher today; Immortal is hard but
playable. Pick the rung you can actually contest.

**Necto is deliberately refused at construction.** Its net is ported and
golden-tested, but its OBSERVATION is not Nexto's despite identical tensor
widths (both q32/kv24): entity order is ball-first, the relative transform
subtracts position AND velocity with no heading rotation, team flags are keyed
to blue then column-swapped, and pad/car column 21 carries stateful respawn and
demo TIMERS rather than binary flags. Running it on Nexto's obs yields a bot
driving on nonsense while every net-level golden test still passes — those feed
torch and candle the same q/kv, so they check the net, not the obs. A 96-0
benchmark was produced exactly that way and discarded.

## Getting the weights

Immortal's weights are **not in this repo and must never be committed** —
RLMarlbot publishes no license, so they are private-research-use only.

```bash
git clone https://github.com/LoveSexDrugz/RLMarlbot /tmp/RLMarlbot
git -C /tmp/RLMarlbot checkout 9963b3bd328267d424ebac4a63429422df053e9c
.venv/bin/python scripts/extract_immortal_weights.py \
    /tmp/RLMarlbot/rlmarlbot/immortal/jit.pt
# -> ~/.cache/construct/immortal_weights.npz  (prints sha256)
```

Check the printed sha256 against `external_weights.manifest`
(`d72949bf…f453a4f`). Tests that need the weights **skip** when they're absent,
so CI stays green without them.

## Using one

Foreign opponents live in their **own slot space**, separate from
`set_opponents`, and are addressed from a collect assignment as `-(slot) - 2`:

```python
eng.set_foreign_opponents([immortal_sd], ["immortal"])   # or "nexto" / "necto"
# arena 0 -> foreign slot 0; arena 1 -> self-play; arena 2 -> native slot 0
out = eng.collect(steps, arena_opponents=[-2, -1, 0])
```

In a foreign arena the **orange** cars are driven by the ported bot and the
**blue** cars remain learner rows. `-1` (self-play) and `k >= 0` (native
opponent) behave exactly as before.

## How the port works

| piece | where | verified against |
|---|---|---|
| AdvancedObs-107 | `engine/src/obs_advanced.rs` | `immortal_obs_test.rs`, 32 cases blue+orange, <1e-5 |
| 126-row action table | `engine/src/actions.rs::make_immortal_table` | `immortal_action_test.rs`, row-for-row |
| MLP (LeakyReLU 0.01) | `engine/src/foreign.rs::ForeignMlp` | `foreign_mlp_test.rs` vs torch, <1e-5 |
| assembly + prev-action | `engine/src/foreign.rs::ForeignPolicy` | `foreign_policy_test.rs` |
| engine wiring | `engine.rs`, `episode.rs` | `tests/python/test_foreign_opponent.py` |

Two things that would have silently corrupted Immortal's input and are worth
remembering:

- **Boost-pad order.** The engine stores pads in RocketSim canonical order (6 big
  then 28 small); Immortal's obs expects rlgym `BOOST_LOCATIONS` order. We
  reorder by nearest-xy position match, and orange uses `inverted_boost_pads`
  (the rlgym array reversed — that order is reverse-antisymmetric within ~2 uu).
- **`get_game_state()` does not preserve car add-order** (it comes back
  id-descending). `EpisodeArena::foreign_controls` maps agent → car BY CAR ID.

## Element and Immortal above 1v1: a degraded LOCAL variant

**Read this before quoting any element/immortal result at 2v2 or 3v3.**

Element and Immortal consume `AdvancedObs-107`, and 107 is a *hard* width:
their extracted weights start with `net.0.weight [512,107]` (immortal) and
`fc1.weight [256,107]` (element). The layout is
`ball 9 + prev_action 8 + pads 34 + self 25 + 31 * (other cars)`, i.e. exactly
**one** other car. Only the last block depends on car count.

Until 2026-07-26 the engine **refused** these two above 1v1. That was wrong in
its reasoning: the limit belongs to *the extracted artifact*, not to the bot.
Upstream RLMarlbot shares it — `rlmarlbot/immortal/obs/advanced_obs.py` loops
over all other cars with no padding and no truncation, so upstream would emit
169 floats into a 107-wide net at 2v2 and crash. There is no "correct" 2v2
behaviour to be faithful to.

So we truncate. Above 1v1 these two see **the opponent nearest the ball and
nothing else** — not their teammates, not the other opponents
(`obs_advanced::build_advanced_obs_one_other`, selector
`nearest_opponent_to_ball`).

**At 1v1 nothing changed.** There is one other car and it is the enemy, so the
truncated builder is byte-identical to the old one — asserted over the real
fixture states in `engine/tests/immortal_obs_test.rs`
(`truncated_obs_is_byte_identical_to_full_obs_at_1v1`). Unlike the necto reset
fix, this change invalidates **no** existing 1v1 measurement.

**Why `near-ball`.** The shown car lands at offset 76..107, the slot the net
was trained to read as the *enemy*, so it must be an opponent — showing a
teammate writes an ally into an enemy-typed slot (measured as consistently the
worst selector). Among opponent selectors, `near-ball` is self-correcting
against the obvious exploit: with `near-self` our net could drive car B at the
ball while car A parks next to the bot and hogs the visible slot, whereas with
`near-ball` whichever of our cars is contesting the ball *is* the visible one.
`lowest-id` is additionally degenerate when the bot drives several cars — they
would all watch the same opponent.

**Measured** (4000 steps/row x 3 seeds vs a random opponent). A truncated bot
in a team arena is indistinguishable from the same bot at 1v1:

| bot | regime | goals/600s | ball-facing | approach | touches/min |
|---|---|---|---|---|---|
| element | 1v1 (reference) | 61 ± 5 | 0.555 | 0.531 | 39.4 |
| element | 3v3 truncated | 58 ± 5 | 0.570 | 0.579 | 44.7 |
| immortal | 1v1 (reference) | 43 ± 12 | 0.558 | 0.546 | 37.4 |
| immortal | 3v3 truncated | 41 ± 2 | 0.577 | 0.603 | 39.3 |

No spin (ground `|ang_vel.z|` 0.8–1.1 rad/s against a 5.5 cap; a spinner would
sit at the cap), no idling, zero insane physics states in 42 rows. Against
nexto they track a **non-truncated** necto control cell for cell as team size
grows, so the collapse with team size is the cost of playing nexto, not the
cost of truncation.

**What this is NOT.** Above 1v1 this is a *degraded local variant*, not the
ported bot. A claim of the form "we beat Element at 3v3" is meaningless outside
this repo. Keep `bench_foreign.py` / Immortal-at-1v1 as the absolute ruler.

**Exploitability tripwire.** The bot cannot see 1 (2v2) or 3 (3v3) of our cars,
so "occupy the visible slot with car A, walk car B in" is a free goal that
exists against no real opponent — the `replay-bc-cannot-play` failure mode of
training against an arm with a structural hole. Three things bound it:
`near-ball` makes the exploit require the exploiting car to approach the ball,
at which point it *becomes* the visible car; the auto-curriculum's [0.35, 0.65]
band caps how much experience can come from a slot we are farming; and in v9
these are 4 of 12 slots. **If win share against an element/immortal team slot
ratchets to the easy pin and stays there while the nexto/necto slots do not,
that is the exploit — pull the slots.**

**Difficulty does not carry over from 1v1.** Measured with a fresh random net
at 1 car / p24 (`_measure_foreign_winrates`, ~18 matches/slot): at 2v2 element
is the *hardest* of the four bots for us (0.333) even though at 1v1 it is one
of the two easiest (0.677 at a shared p4, against nexto's 0.146). Blinding a
bot does not make it easy — it keeps chasing the ball. Seed team rungs from a
team measurement, never from the 1v1 ordering.

## Fidelity caveats

- **Cadence.** Our engine's `tick_skip = 8` (15 Hz); Immortal trained at 6
  (20 Hz). It decides on our clock, so it reacts slightly slower than trained.
  Acceptable for a sparring partner; revisit if it plays visibly degraded.
- **Deterministic play.** The bot argmaxes (matching its deploy). Its cars still
  consume their per-agent rng draw so the arena's random stream is unperturbed;
  the sampled index is simply overridden.

## Protecting the gate instrument

Any engine rebuild risks silently changing the instrument every gate baseline was
scored on. Before and after a rebuild:

```bash
.venv/bin/python scripts/instrument_fingerprint.py --save before.json
# ... rebuild ...
.venv/bin/python scripts/instrument_fingerprint.py --check before.json
```

It replays the gate's exact MatchRunner construction champion-vs-champion at a
fixed seed and hashes every output array. The foreign-opponent port was verified
this way: `INSTRUMENT UNCHANGED`.

## Handicapping a bot (the difficulty knob)

Every ported bot beats the champion 96-0 at full strength, so none is usable as a
TEACHER as-is: at a 0% win rate the win-probability potential saturates near -0.5
and those arenas contribute almost no gradient. `decision_periods` fixes this -- a
bot recomputes a decision every Nth call and holds its controls in between, which
degrades skill smoothly without touching its policy or observation:

```python
eng.set_foreign_opponents([element_sd], ["element"], decision_periods=[4])
```

Measured champion win_share vs Element by period: 1 -> 0.00, 2 -> 0.00,
4 -> 0.375, 8 -> 0.984. One bot spans the whole range. The honest absolute
progress metric is "the period at which win_share crosses 0.5" -- measure it with
`bench_foreign.py --period N`.

In a training config this is `[foreign].decision_periods` (same length as
`kinds`); see `configs/train_v5_element_handicap.toml`.

## Adding another bot

`ForeignKind` in `engine/src/foreign.rs` is the extension point. Element
(multi-head, pickled state_dict, 107-dim `CustomObs`) and Nexto/Necto (Perceiver
`(q, kv, mask)` obs, 90-action table) would each add a variant plus their own obs
builder — the slot plumbing and controls override are already generic.
