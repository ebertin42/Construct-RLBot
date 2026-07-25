"""Per-bot auto-curriculum control (train.py `_auto_curriculum_step`).

Guards the 2026-07-25 change from ONE shared decision_period to one period PER
foreign slot. The motivating measurement (ck_001171502080, all four bots at a
shared p4): immortal 0.896, element 0.677, necto 0.615, nexto 0.146 -- a single
knob cannot put those in the same band, so it parks two bots as punching bags
and one as an unwinnable wall.

`_auto_curriculum_step` is exercised unbound against a minimal stub so the test
needs no Engine, no CUDA and no checkpoint: the arithmetic (which way each
period moves, and that slots move INDEPENDENTLY) is the whole contract.
"""
import types

from construct.learn.train import Trainer


def _stub(win_rates, periods, kinds=None, alpha=1.0, dwell=0):
    """Minimal object carrying only what `_auto_curriculum_step` touches.

    alpha=1.0 makes the per-slot EMA equal the raw win rate, so a single call
    is enough to assert the direction of travel. dwell=0 keeps most tests
    focused on the band arithmetic; the dwell behaviour has its own test.
    """
    kinds = kinds or [f"bot{i}" for i in range(len(periods))]
    s = types.SimpleNamespace(
        _ac_on=True,
        _foreign_slots=len(periods),
        _foreign_frac=0.4,
        num_arenas=100,
        _ac_ema=None,
        _ac_alpha=0.2,
        _ac_every=1,
        _ac_win_lo=0.45,
        _ac_win_hi=0.55,
        _ac_band=5.0,
        _ac_period_min=1,
        _ac_period_max=12,
        _ac_wr_ema=[None] * len(periods),
        _ac_wr_alpha=alpha,
        _ac_dwell=dwell,
        _ac_since=[10 ** 6] * len(periods),      # free to move on the first eval
        _foreign_periods=list(periods),
        _foreign_kinds=list(kinds),
        _foreign_sds=[{} for _ in periods],
        _ac_eval=object(),          # non-None so the eval branch is taken
        engine=types.SimpleNamespace(set_foreign_opponents=lambda *a, **k: None),
    )
    s._measure_foreign_winrates = lambda: list(win_rates)
    return s


def _step(stub, it=1):
    Trainer._auto_curriculum_step(stub, it, ep_reward_mean=0.0)


def test_punching_bag_hardens_and_wall_eases_in_the_same_pass():
    """The whole point of per-slot control: opposite moves, one pass."""
    # slot0 far too easy (0.90), slot1 far too hard (0.15) -- the real p4 spread.
    s = _stub([0.90, 0.15], [4, 4], kinds=["immortal", "nexto"])
    _step(s)
    assert s._foreign_periods[0] == 3, "a bot we beat 0.90 must get HARDER (lower period)"
    assert s._foreign_periods[1] == 5, "a bot we lose 0.15 to must get EASIER (higher period)"


def test_in_band_bot_is_left_alone():
    s = _stub([0.50, 0.90], [4, 4])
    _step(s)
    assert s._foreign_periods[0] == 4, "0.50 is inside [0.45,0.55] -- hold"
    assert s._foreign_periods[1] == 3, "the other slot still moves independently"


def test_periods_clamp_to_configured_bounds():
    s = _stub([0.99, 0.01], [1, 12])          # already at min / max
    _step(s)
    assert s._foreign_periods == [1, 12], "must not walk past period_min/period_max"


def test_ema_smooths_noise_while_the_period_is_stable():
    """An in-band bot stays put, and its EMA accumulates across evals."""
    s = _stub([0.52], [4], alpha=0.5)
    _step(s)
    assert s._foreign_periods[0] == 4 and s._ac_wr_ema[0] == 0.52
    s._measure_foreign_winrates = lambda: [0.40]      # noisy dip, still smoothed
    _step(s)
    assert s._ac_wr_ema[0] == 0.46, "ema must blend, not jump to the raw value"
    assert s._foreign_periods[0] == 4, "0.46 is still in band -> hold, no thrash"


def test_period_change_resets_that_slots_ema():
    """Regression: stale EMA must not ratchet a second time.

    Seeding at 0.90 hardens p4->p3. The EMA still describes p4, so carrying it
    over would harden again to p2 without ever measuring p3.
    """
    s = _stub([0.90], [4], alpha=0.3)
    _step(s)
    assert s._foreign_periods[0] == 3, "first pass hardens on the 0.90 evidence"
    assert s._ac_wr_ema[0] is None, "changing the period must clear its stale history"
    s._measure_foreign_winrates = lambda: [0.50]      # p3 turns out to be fair
    _step(s)
    assert s._foreign_periods[0] == 3, "a fresh in-band measurement holds p3"
    assert s._ac_wr_ema[0] == 0.50


def test_slot_with_no_completed_match_is_skipped_not_crashed():
    s = _stub([None, 0.90], [4, 4])
    _step(s)
    assert s._foreign_periods[0] == 4, "no signal -> leave that slot's period alone"
    assert s._foreign_periods[1] == 3
    assert s._ac_wr_ema[0] is None


def test_dwell_blocks_a_second_move_until_the_ema_re_accumulates():
    """Without a dwell the controller moves on ~85-90% of evals (sd 0.118 vs a
    0.05 half-width) and wanders 4 rungs. Dwell holds it still while the reset
    EMA refills."""
    s = _stub([0.90], [4], alpha=1.0, dwell=2)
    _step(s)
    assert s._foreign_periods[0] == 3, "first eval is free to move"
    assert s._ac_since[0] == 0, "the move starts the dwell"
    for expected in (1, 2):                       # two evals held
        _step(s)
        assert s._foreign_periods[0] == 3, "dwell must block further moves"
        assert s._ac_since[0] == expected
    _step(s)                                      # dwell served -> may move
    assert s._foreign_periods[0] == 2, "after the dwell a still-0.90 bot hardens"


def test_dwell_still_updates_the_ema_while_holding():
    """Held evals are not wasted: they feed the EMA so the next decision is
    based on several matches, not one."""
    s = _stub([0.90], [4], alpha=0.5, dwell=2)
    _step(s)                                      # moves, resets ema to None
    assert s._ac_wr_ema[0] is None
    s._measure_foreign_winrates = lambda: [0.50]
    _step(s)
    assert s._ac_wr_ema[0] == 0.50, "first held eval seeds the ema"
    _step(s)
    assert s._ac_wr_ema[0] == 0.50, "second held eval keeps folding it in"
    assert s._foreign_periods[0] == 3, "and the period stayed put throughout"


def test_falls_back_to_shared_knob_without_an_eval_engine():
    s = _stub([0.90, 0.90], [4, 4])
    s._ac_eval = None
    s._measure_foreign_winrates = lambda: None
    s._ac_ema = 99.0                      # way above band -> harden
    _step(s)
    assert s._foreign_periods == [3, 3], "ep_rew fallback still drives one shared knob"
