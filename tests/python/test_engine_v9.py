"""Python-facing v9 engine API (E2 / E5 / E6 / E9 / E10).

Each of these exists because the trainer, the viewer or a pre-launch measurement
needs something the engine did not previously expose -- and in E2's case because
the missing check was a PANIC, not an error.

Tests needing the (unlicensed, never-committed) bot weights SKIP when absent.

THE WHOLE MODULE SKIPS when the INSTALLED engine predates v9. That is not
cowardice about a failing test -- it is the same `needs_weights` idiom
test_foreign_opponent.py already uses, and it exists because the v9 wheel is
deliberately NOT installed over python/construct/_engine.abi3.so until the null
re-measurement is done: that .so is the gate instrument, and every gate and
bench number in the repo was scored on it (see the `gate-instrument-is-local-
engine` note). Without this guard the repo's documented test command
(`pytest tests/python -q --ignore=tests/python/test_deploy_v1.py`) is RED in a
clean tree -- 9 failures that are pure environment -- and a red suite is how a
real regression gets waved through. `maturin develop --release -m
engine/Cargo.toml` turns all of these back on.
"""
import pathlib
import socket

import numpy as np
import pytest
import torch

from construct._engine import Engine, RenderSession, BotMatch, action_table_v1
from construct.learn.model_v1 import EntityPolicyNet

CACHE = pathlib.Path.home() / ".cache/construct"
_VISER_PORT = 34254

# `team_sizes` is E6 and is the first thing v9 added to the Engine surface, so
# it is a sufficient probe for "is the installed .so the v9 build?". Checked by
# attribute rather than by version string because the wheel carries no version.
pytestmark = pytest.mark.skipif(
    not hasattr(Engine, "team_sizes"),
    reason="installed engine predates the v9 wheel; run "
           "`.venv/bin/maturin develop --release -m engine/Cargo.toml`",
)


def _weights(name):
    p = CACHE / f"{name}_weights.npz"
    if not p.exists():
        pytest.skip(f"{p} absent (unlicensed, never committed)")
    return {k: v.astype(np.float32) for k, v in np.load(p).items()}


def _net_sd(seed=0):
    torch.manual_seed(seed)
    net = EntityPolicyNet(d_model=32, layers=1, heads=2, ff=64,
                          action_table=action_table_v1())
    return {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def _engine(num_arenas, weights=None, **kw):
    kw.setdefault("blue", 1)
    kw.setdefault("orange", 1)
    e = Engine(num_arenas=num_arenas, schema_path="schema/v1.toml",
               reward_config_path="configs/reward_v0.toml", seed=5,
               num_threads=1, net_heads=2, **kw)
    return e


def _viser_port_busy() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("0.0.0.0", _VISER_PORT))
    except OSError:
        return True
    finally:
        probe.close()
    return False


# ---------------------------------------------------------------- E6 --------

def test_team_sizes_getter_lists_every_arena_in_ascending_blocks():
    """The trainer places opponents PER BLOCK and could not previously tell
    which arenas were which -- `allocate_team_sizes` was Rust-only, so v8 had to
    stamp the front and the tail and ended up with zero 3v3 self-play arenas."""
    e = _engine(12, team_size_weights=[6, 3, 2])
    ts = list(e.team_sizes)
    assert len(ts) == 12
    assert ts == sorted(ts), "blocks must be ordered 1s, 2s, 3s"
    assert ts == [1] * 7 + [2] * 3 + [3] * 2
    # and it agrees with the agent count the engine actually built
    assert e.num_agents == sum(2 * s for s in ts)


def test_team_sizes_without_weights_is_all_ones():
    assert list(_engine(3).team_sizes) == [1, 1, 1]


# ---------------------------------------------------------------- E5 --------

def test_learner_team_size_labels_every_row():
    e = _engine(12, team_size_weights=[6, 3, 2])
    e.set_weights(_net_sd())
    out = e.collect(4)
    lab = np.asarray(out["learner_team_size"])
    assert lab.shape == (out["learner_agents"],)
    ts = list(e.team_sizes)
    # self-play: every arena contributes 2m rows
    for m in (1, 2, 3):
        assert int((lab == m).sum()) == 2 * m * ts.count(m)
    # rows are grouped arena-major, so the label is non-decreasing here
    assert list(lab) == sorted(lab)


def test_learner_team_size_follows_rows_not_arenas_with_opponents():
    """An opponent arena contributes only its BLUE cars. The label has to follow
    the ROW layout -- a label indexed by arena would silently mis-scale every
    gradient in the per-group advantage standardisation."""
    e = _engine(12, team_size_weights=[6, 3, 2])
    sd = _net_sd()
    e.set_weights(sd)
    e.set_opponents([sd])
    ts = list(e.team_sizes)
    # give the first arena of each block a native opponent
    assign = [-1] * 12
    for m in (1, 2, 3):
        assign[ts.index(m)] = 0
    out = e.collect(4, arena_opponents=assign)
    lab = np.asarray(out["learner_team_size"])
    for m in (1, 2, 3):
        want = sum((m if assign[i] == 0 else 2 * m)
                   for i, s in enumerate(ts) if s == m)
        assert int((lab == m).sum()) == want, f"team size {m}"


# ---------------------------------------------------------------- E4 --------

def test_partial_foreign_team_forwards_the_mirror_cars():
    """`forward_rows` is the direct evidence that the cars the bot does NOT
    drive are played by our policy rather than sampling from an all-zero logits
    row (i.e. uniformly at random)."""
    w = _weights("necto")
    e = _engine(1, blue=3, orange=3)
    e.set_weights(_net_sd())
    e.set_foreign_opponents([w], ["necto"], [12], [1])
    out = e.collect(4, arena_opponents=[-2])
    assert out["learner_agents"] == 3, "only BLUE learns, whatever the bot drives"
    assert out["forward_rows"] == 5, "3 blue + 2 mirror cars through our policy"

    e2 = _engine(1, blue=3, orange=3)
    e2.set_weights(_net_sd())
    e2.set_foreign_opponents([w], ["necto"], [12], [3])   # full team
    out2 = e2.collect(4, arena_opponents=[-2])
    assert out2["learner_agents"] == 3
    assert out2["forward_rows"] == 3, "a full foreign team leaves no mirrors"


def test_foreign_cars_length_mismatch_is_rejected():
    w = _weights("necto")
    e = _engine(1)
    with pytest.raises(ValueError, match="foreign_cars"):
        e.set_foreign_opponents([w], ["necto"], [1], [1, 1])


# ---------------------------------------------------------------- E9 --------

def test_reward_terms_are_a_decomposition_and_reset_on_read():
    e = _engine(2)
    e.set_weights(_net_sd())
    e.reward_terms()                       # drain construction-time counters
    out = e.collect(32)
    t = e.reward_terms()
    assert t["agent_steps"] == 32 * 4      # 2 arenas x 2 cars x 32 steps
    reward_keys = ("goal", "touch", "vel_to_ball", "touch_accel", "vel_ball_to_goal",
                   "offensive_potential", "aerial_touch", "air_setup", "win_prob")
    total = float(np.asarray(out["rewards"]).sum())
    assert abs(sum(t[k] for k in reward_keys) - total) < 1e-3
    # reward_v0 has none of the v9 fields set, so they must all be exactly zero
    for k in ("aerial_touch", "air_setup", "touch_accel", "win_prob"):
        assert t[k] == 0.0, k
    assert all(v == 0.0 for v in e.reward_terms().values()), "must reset on read"


# ---------------------------------------------------------------- E10 -------

def test_botmatch_runs_at_mode_2_for_every_kind():
    """Measuring the (bot x team size) rung ladder before launch needs team
    bot-vs-bot matches; `BotMatch` was hardcoded to 1v1.

    INVERTED on 2026-07-26: this used to assert that element/immortal were
    REFUSED at mode > 1 ("1v1-only"). They now run there on a TRUNCATED
    AdvancedObs-107 showing only the opponent nearest the ball, so a team
    match with a fixed-width bot must produce finite rewards rather than
    raise. Above 1v1 those two are a degraded local variant of the bot, not
    the ported bot -- see docs/foreign-opponents.md.
    """
    necto, nexto = _weights("necto"), _weights("nexto")
    bm = BotMatch("necto", necto, 12, "nexto", nexto, 12, arenas=1, mode=2)
    rew, term = bm.run(8)
    assert rew.shape == (8, 1), "one column per ARENA at every mode"
    assert term.shape == (8, 1)
    assert np.isfinite(rew).all()
    # element (AdvancedObs-107, fixed width) now drives a 2v2 side too.
    bm2 = BotMatch("element", _weights("element"), 1, "nexto", nexto, 1,
                   arenas=1, mode=2)
    rew2, term2 = bm2.run(8)
    assert rew2.shape == (8, 1)
    assert term2.shape == (8, 1)
    assert np.isfinite(rew2).all(), "truncated obs must not produce NaN rewards"
    # mode is still validated, and an unknown kind still fails fast.
    with pytest.raises(ValueError, match="mode"):
        BotMatch("necto", necto, 1, "nexto", nexto, 1, arenas=1, mode=4)
    with pytest.raises(ValueError, match="unknown foreign kind"):
        BotMatch("nosuchbot", necto, 1, "nexto", nexto, 1, arenas=1, mode=2)


# ---------------------------------------------------------------- E2 --------

def test_render_session_accepts_a_fixed_width_bot_in_a_team_arena():
    """`RenderSession(blue=2, orange=2)` + element originally PANICKED inside
    build_advanced_obs (index 107 out of bounds) and took the viewer process
    down; a ValueError refusal replaced the panic on 2026-07-26.

    INVERTED later the same day: element/immortal now run above 1v1 on the
    TRUNCATED AdvancedObs-107, so the viewer must ACCEPT them and render the
    matchup. The panic is still gone -- that is what this asserts now, by
    actually stepping the session.
    """
    if _viser_port_busy():
        pytest.skip(f"UDP port {_VISER_PORT} is in use (viewer/relay likely running)")
    w = _weights("element")
    sess = RenderSession(blue=2, orange=2, schema_path="schema/v1.toml",
                         reward_config_path="configs/reward_v0.toml", seed=3)
    try:
        sess.set_foreign_opponent(w, "element")
        assert sess.foreign_kind == "element"
        # ...and a team-capable EARL bot is still accepted at 2v2
        sess.set_foreign_opponent(_weights("necto"), "necto", 12)
        assert sess.foreign_kind == "necto"
    finally:
        sess.close()


def test_render_session_still_accepts_a_fixed_width_bot_at_1v1():
    if _viser_port_busy():
        pytest.skip(f"UDP port {_VISER_PORT} is in use (viewer/relay likely running)")
    w = _weights("element")
    sess = RenderSession(blue=1, orange=1, schema_path="schema/v1.toml",
                         reward_config_path="configs/reward_v0.toml", seed=3)
    try:
        sess.set_foreign_opponent(w, "element", 4)
        assert sess.foreign_kind == "element"
    finally:
        sess.close()
