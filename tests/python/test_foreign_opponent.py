"""Foreign (ported community bot) opponents.

Two things matter here:
  1. NON-REGRESSION: with no foreign slot assigned, collect() output must be
     byte-identical to before the foreign machinery existed. Every gate baseline
     (null mean 0.502 sd 0.024, threshold 0.55) was scored on that behaviour.
  2. The foreign path actually runs: an arena assigned `-2` drives its ORANGE
     cars with the ported bot and still yields learner rows for its BLUE cars.

Tests needing Immortal's weights SKIP when the (unlicensed, never-committed)
npz is absent -- see docs/foreign-opponents.md.
"""
import pathlib

import numpy as np
import pytest

from construct._engine import Engine

WEIGHTS = pathlib.Path.home() / ".cache/construct/immortal_weights.npz"
CHAMPION = pathlib.Path("checkpoints_entity/ck_000320471040.pt")


def _v1_engine(num_arenas=2, seed=7):
    return Engine(num_arenas=num_arenas, blue=1, orange=1,
                  schema_path="schema/v1.toml",
                  reward_config_path="configs/reward_v0.toml",
                  seed=seed, num_threads=1, net_heads=4)


def _champion_sd():
    from construct.league.matches import load_sd
    return load_sd(str(CHAMPION))


def _immortal_sd():
    return {k: v.astype(np.float32) for k, v in np.load(WEIGHTS).items()}


needs_champion = pytest.mark.skipif(
    not CHAMPION.exists(), reason="champion checkpoint not present")
needs_weights = pytest.mark.skipif(
    not WEIGHTS.exists(), reason="Immortal weights npz not present (unlicensed)")


_DETERMINISM_SNIPPET = """
import numpy as np
from construct._engine import Engine
from construct.league.matches import load_sd

sd = load_sd("checkpoints_entity/ck_000320471040.pt")
outs = []
for _ in range(2):
    eng = Engine(num_arenas=2, blue=1, orange=1, schema_path="schema/v1.toml",
                 reward_config_path="configs/reward_v0.toml", seed=7,
                 num_threads=1, net_heads=4)
    eng.set_weights(sd)
    outs.append(eng.collect(64))
a, b = outs
for key in a:
    assert np.array_equal(np.asarray(a[key]), np.asarray(b[key])), key
print("OK")
"""


@needs_champion
def test_selfplay_collect_is_deterministic_in_a_fresh_process():
    """Two identically-seeded engines produce identical self-play rollouts. This is
    the property the gate depends on, and the foreign code path must not perturb it.

    RUN IN A SUBPROCESS deliberately. There is a PRE-EXISTING engine limitation
    (verified 2026-07-22 against the build BEFORE foreign opponents existed): once
    other arenas have been created in the same process, two freshly-constructed
    same-seed engines can diverge by ~1e-5 in rewards. Gates always run in a fresh
    process, so the guarantee they rely on is the fresh-process one asserted here.
    See docs/foreign-opponents.md and scripts/instrument_fingerprint.py.
    """
    import subprocess, sys
    r = subprocess.run([sys.executable, "-c", _DETERMINISM_SNIPPET],
                       capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, f"determinism check failed:\n{r.stdout}\n{r.stderr}"
    assert "OK" in r.stdout


@needs_champion
@needs_weights
def test_foreign_arena_runs_and_yields_blue_learner_rows():
    sd = _champion_sd()
    eng = _v1_engine(num_arenas=2)
    eng.set_weights(sd)
    eng.set_foreign_opponents([_immortal_sd()], ["immortal"])
    # arena 0 -> foreign slot 0 (encoded -2); arena 1 -> self-play
    out = eng.collect(64, arena_opponents=[-2, -1])
    # arena 0 contributes blue only (1 row), arena 1 contributes both (2 rows)
    assert np.asarray(out["actions"]).shape[1] == 3
    assert np.isfinite(np.asarray(out["rewards"])).all()


@needs_champion
@needs_weights
def test_foreign_opponent_changes_the_rollout():
    """A foreign-driven arena must differ from the same arena self-playing --
    otherwise the override silently isn't being applied."""
    sd = _champion_sd()

    eng_a = _v1_engine(num_arenas=1)
    eng_a.set_weights(sd)
    self_play = np.asarray(eng_a.collect(64)["rewards"])

    eng_b = _v1_engine(num_arenas=1)
    eng_b.set_weights(sd)
    eng_b.set_foreign_opponents([_immortal_sd()], ["immortal"])
    foreign = np.asarray(eng_b.collect(64, arena_opponents=[-2])["rewards"])

    # different shapes (self-play has 2 learner rows, foreign has 1) OR different
    # values -- either way the foreign path demonstrably changed the rollout.
    assert self_play.shape != foreign.shape or not np.array_equal(self_play, foreign)


@needs_weights
def test_rejects_unknown_kind():
    eng = _v1_engine()
    with pytest.raises(Exception):
        eng.set_foreign_opponents([_immortal_sd()], ["not-a-bot"])


@needs_weights
def test_rejects_length_mismatch():
    eng = _v1_engine()
    with pytest.raises(Exception):
        eng.set_foreign_opponents([_immortal_sd()], [])


@needs_champion
@needs_weights
def test_rejects_non_1v1_arena():
    """Immortal's AdvancedObs is 107 floats == exactly 1v1. A 2v2 arena would need
    169 and the net cannot consume it, so the engine must refuse rather than write
    out of bounds."""
    sd = _champion_sd()
    eng = Engine(num_arenas=1, blue=2, orange=2, schema_path="schema/v1.toml",
                 reward_config_path="configs/reward_v0.toml", seed=7,
                 num_threads=1, net_heads=4)
    eng.set_weights(sd)
    eng.set_foreign_opponents([_immortal_sd()], ["immortal"])
    with pytest.raises(Exception, match="1v1"):
        eng.collect(8, arena_opponents=[-2])


@needs_champion
def test_rejects_unset_foreign_slot():
    eng = _v1_engine(num_arenas=1)
    eng.set_weights(_champion_sd())
    with pytest.raises(Exception):
        eng.collect(8, arena_opponents=[-2])   # no foreign slots loaded
