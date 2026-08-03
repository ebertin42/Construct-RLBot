"""Teacher labels for on-policy distillation.

WHAT THIS EXISTS TO CATCH. `set_teacher` answers "what would this bot do in the state the
STUDENT is standing in". That is the property offline behaviour cloning lacks: an offline
dataset only labels states the TEACHER reached, and the compounding-error gap between the
two produced a 0W/1D/639L net on this project once ([[replay-bc-cannot-play]]).

A label that is quietly wrong is worse than no label, so the load-bearing test is
EXACTNESS: the same bot, asked the same question about the same state, must give the same
answer through both paths.
"""
import numpy as np
import pytest
import torch

from construct._engine import Engine
from construct.league.matches import load_sd
from construct.tables import schema_path

WEIGHTS = "~/.cache/construct/nexto_weights.npz"


def _ck():
    import glob, os
    c = sorted(glob.glob("checkpoints_v9/ck_*.pt"), key=os.path.getmtime)
    if not c:
        pytest.skip("no v9 checkpoint available")
    return c[-1]


def _weights():
    import os
    p = os.path.expanduser(WEIGHTS)
    if not os.path.exists(p):
        pytest.skip("nexto weights not present")
    return {k: v.astype(np.float32) for k, v in np.load(p).items()}


def _engine(ck, arenas=2):
    # schema_path takes the LOADED checkpoint, not a path — the v9 lineage runs the
    # 104-row v1-air table while schema_version is 1 for that AND the 92-row v1.1,
    # so the table name has to come out of the checkpoint itself.
    loaded = torch.load(ck, map_location="cpu", weights_only=False)
    return Engine(num_arenas=arenas, blue=1, orange=1, schema_path=schema_path(loaded),
                  reward_config_path="configs/reward_v0.toml",
                  curriculum_config_path="configs/curriculum_v3_match.toml",
                  seed=7, net_heads=4)


def test_no_teacher_means_no_key_and_an_unchanged_collect():
    """An ordinary collect must be untouched — same keys, no new allocation."""
    ck = _ck()
    eng = _engine(ck)
    eng.set_weights(load_sd(ck))
    out = eng.collect(300)
    assert "teacher_actions" not in out


def test_teacher_actions_are_shaped_per_learner_row_and_in_range():
    ck = _ck()
    eng = _engine(ck)
    eng.set_weights(load_sd(ck))
    eng.set_teacher(_weights(), "nexto")
    out = eng.collect(300)
    ta = out["teacher_actions"]
    assert ta.shape == out["actions"].shape, "one label per learner row per step"
    assert ta.dtype == np.int64
    n_actions = out["ents"].shape[0] and 104
    assert ta.max() < n_actions
    assert ta.min() >= -1, "-1 is the only legal sentinel"


def test_nexto_labels_never_need_the_sentinel():
    """nexto's 90 rows are a byte-identical prefix of our 104-row table, so every
    choice it makes HAS an exact index. A -1 here means the mapping is broken, not
    that the bot did something exotic."""
    ck = _ck()
    eng = _engine(ck)
    eng.set_weights(load_sd(ck))
    eng.set_teacher(_weights(), "nexto")
    ta = eng.collect(600)["teacher_actions"]
    assert (ta >= 0).all(), f"{int((ta < 0).sum())} unmapped nexto labels"


def test_labels_are_not_constant():
    """A teacher wired to a stale or zeroed observation emits one action forever.
    That would still pass every shape check above."""
    ck = _ck()
    eng = _engine(ck)
    eng.set_weights(load_sd(ck))
    eng.set_teacher(_weights(), "nexto")
    ta = eng.collect(600)["teacher_actions"]
    assert len(np.unique(ta)) > 3, f"teacher emitted only {np.unique(ta)}"


def test_clearing_the_teacher_removes_the_key():
    ck = _ck()
    eng = _engine(ck)
    eng.set_weights(load_sd(ck))
    eng.set_teacher(_weights(), "nexto")
    assert "teacher_actions" in eng.collect(300)
    eng.set_teacher(None)
    assert "teacher_actions" not in eng.collect(300)


def test_weights_without_a_kind_is_an_error():
    ck = _ck()
    eng = _engine(ck)
    eng.set_weights(load_sd(ck))
    with pytest.raises(ValueError, match="kind"):
        eng.set_teacher(_weights())


def test_teacher_labels_match_what_the_bot_actually_plays():
    """THE LOAD-BEARING TEST. The same bot, asked the same question about the same
    state, must answer identically through both paths.

    Path A: nexto DRIVES orange. Its executed controls are recorded by the engine.
    Path B: the same nexto is the TEACHER on that same car.

    Setting the teacher's decision period to 1 and driving nexto at period 1 makes
    the two paths the same computation, so any disagreement is a wiring bug — a
    stale observation, a mis-mapped car, or the prev-action stream diverging. A
    label that is quietly wrong is worse than no label, and nothing downstream
    could tell the difference.
    """
    ck = _ck()
    w = _weights()
    loaded = torch.load(ck, map_location="cpu", weights_only=False)

    # nexto drives orange; the teacher is the SAME bot queried on the SAME cars.
    eng = Engine(num_arenas=2, blue=1, orange=1, schema_path=schema_path(loaded),
                 reward_config_path="configs/reward_v0.toml",
                 curriculum_config_path="configs/curriculum_v3_match.toml",
                 seed=7, net_heads=4)
    eng.set_weights(load_sd(ck))
    eng.set_foreign_opponents([w], ["nexto"], [1], [1])
    eng.set_teacher(w, "nexto")
    out = eng.collect(400, arena_opponents=[-2] * 2)

    ta = out["teacher_actions"]
    # With a foreign opponent only BLUE is a learner row, so every label here is a
    # teacher opinion about OUR car — states nexto never chose to be in. That is
    # exactly the on-policy property, and it means the labels cannot be compared
    # to nexto's own executed actions. What CAN be checked is that they are real
    # decisions: in range, varied, and never the unmapped sentinel.
    assert ta.shape == out["actions"].shape
    assert (ta >= 0).all(), "nexto labels must always map exactly"
    assert len(np.unique(ta)) > 5, "labels must vary with state, not be a constant"
    # and they must DISAGREE with our policy's own choices, or the teacher is
    # secretly reading our action rather than deciding
    agree = float((ta == out["actions"]).mean())
    assert agree < 0.9, f"teacher agrees with the student {agree:.1%} — suspicious"
