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

Currently ported: **Immortal** (RLMarlbot, rlgym-ppo), **Nexto** (~GC1) and
**Necto** (~Diamond) (Rolv-Arild/Necto, CC BY-NC-SA 4.0).

Measured strength of our champion `ck_000320471040` against each (`bench_foreign.py`):

| opponent | our win_share |
|---|---|
| Necto  | see docs/training-journal.md |
| Immortal | 0.0953 |
| Nexto | 0.0000 (0W/0D/96L) |

Nexto is far too strong to be a useful teacher today; Immortal is hard but
playable. Pick the rung you can actually contest.

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

## Adding another bot

`ForeignKind` in `engine/src/foreign.rs` is the extension point. Element
(multi-head, pickled state_dict, 107-dim `CustomObs`) and Nexto/Necto (Perceiver
`(q, kv, mask)` obs, 90-action table) would each add a variant plus their own obs
builder — the slot plumbing and controls override are already generic.
