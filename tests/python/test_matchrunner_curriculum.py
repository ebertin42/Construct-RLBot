"""Tests for MatchRunner's curriculum_config plumbing (deployment-gap G1).

MatchRunner never passed curriculum_config_path to the engine, so it could
never run full-match ("match_mode") games -- the match-win gate would see a
boundary per goal (single-goal "matches") instead of real 300s matches.

These tests target the pure engine_kwargs-assembly logic, NOT a live engine:
constructing a real Engine is expensive and match_mode support in the local
engine build is a separate step (G2, tracked elsewhere). `_engine_kwargs` is
extracted out of MatchRunner.__init__ specifically so this can be verified
without ever touching construct._engine.Engine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from construct.league.matches import _engine_kwargs  # noqa: E402


def test_engine_kwargs_omits_curriculum_path_by_default():
    # Legacy construction (no curriculum_config) must be byte-for-byte
    # unchanged: the key must not appear at all, not even as None -- some
    # engine bindings treat an explicit None differently from an absent kwarg.
    kw = _engine_kwargs(
        num_arenas=8, seed=0, reward_config="configs/reward_v0.toml",
        mode=1, schema_version=0, net_heads=4,
    )
    assert "curriculum_config_path" not in kw


def test_engine_kwargs_omits_curriculum_path_when_explicitly_none():
    kw = _engine_kwargs(
        num_arenas=8, seed=0, reward_config="configs/reward_v0.toml",
        mode=1, schema_version=1, net_heads=4, curriculum_config=None,
    )
    assert "curriculum_config_path" not in kw


def test_engine_kwargs_threads_curriculum_path_when_given():
    kw = _engine_kwargs(
        num_arenas=8, seed=0, reward_config="configs/reward_v0.toml",
        mode=1, schema_version=1, net_heads=4,
        curriculum_config="configs/curriculum_v3_match.toml",
    )
    assert kw["curriculum_config_path"] == "configs/curriculum_v3_match.toml"


def test_engine_kwargs_unchanged_fields_v0_no_net_heads():
    # schema_version=0 must never carry net_heads, curriculum threading must
    # not disturb that existing v0/v1 branch.
    kw = _engine_kwargs(
        num_arenas=4, seed=7, reward_config="configs/reward_v0.toml",
        mode=1, schema_version=0, net_heads=4,
        curriculum_config="configs/curriculum_v3_match.toml",
    )
    assert "net_heads" not in kw
    assert kw == {
        "num_arenas": 4, "blue": 1, "orange": 1,
        "schema_path": "schema/v0.toml", "reward_config_path": "configs/reward_v0.toml",
        "seed": 7, "curriculum_config_path": "configs/curriculum_v3_match.toml",
    }


def test_matchrunner_init_accepts_curriculum_config_kwarg():
    # MatchRunner.__init__'s signature must accept curriculum_config without
    # raising a TypeError before it ever reaches Engine construction -- checked
    # by inspecting the signature rather than constructing a real engine.
    import inspect

    from construct.league.matches import MatchRunner
    sig = inspect.signature(MatchRunner.__init__)
    assert "curriculum_config" in sig.parameters
    assert sig.parameters["curriculum_config"].default is None
