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


def _stub(win_rates, periods, kinds=None, alpha=1.0, dwell=0,
          ladder=None, cars=None, modes=None, big=1.0):
    """Minimal object carrying only what `_auto_curriculum_step` touches.

    alpha=1.0 makes the per-slot EMA equal the raw win rate, so a single call
    is enough to assert the direction of travel. dwell=0 keeps most tests
    focused on the band arithmetic; the dwell behaviour has its own test.

    `ladder` (v9) switches the controller from the legacy +/-1 integer period to
    the LEXICOGRAPHIC (foreign_cars, period-ladder index) rung. Left None it is
    the pre-v9 control law, which is what every test above the v9 section
    exercises -- those assertions are unchanged on purpose: adding a ladder must
    not silently change how a run WITHOUT one behaves.

    `big` defaults to 1.0 so |ema - 0.5| can never exceed it and the 2-rung jump
    stays out of the legacy tests' way.
    """
    kinds = kinds or [f"bot{i}" for i in range(len(periods))]
    n = len(periods)
    pidx = ([min(range(len(ladder)), key=lambda i, p=p: abs(ladder[i] - p))
             for p in periods] if ladder else [0] * n)
    s = types.SimpleNamespace(
        _ac_on=True,
        _foreign_slots=n,
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
        _ac_wr_ema=[None] * n,
        _ac_wr_alpha=alpha,
        _ac_dwell=dwell,
        _ac_since=[10 ** 6] * n,      # free to move on the first eval
        _ac_ladder=list(ladder) if ladder else None,
        _ac_pidx=pidx,
        _ac_big=big,
        _ac_cycle=0,
        _ac_stagger=False,
        _foreign_periods=list(periods),
        _foreign_kinds=list(kinds),
        _foreign_cars=list(cars) if cars else [1] * n,
        _foreign_modes=list(modes) if modes else [1] * n,
        _foreign_sds=[{} for _ in periods],
        _ac_eval=object(),          # non-None so the eval branch is taken
        engine=types.SimpleNamespace(set_foreign_opponents=lambda *a, **k: None),
    )
    s._measure_foreign_winrates = lambda modes=None: list(win_rates)
    s._rung_harder = lambda i: Trainer._rung_harder(s, i)
    s._rung_easier = lambda i: Trainer._rung_easier(s, i)
    s._rung_str = lambda i: Trainer._rung_str(s, i)
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
    s._measure_foreign_winrates = lambda modes=None: [0.40]      # noisy dip, still smoothed
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
    s._measure_foreign_winrates = lambda modes=None: [0.50]      # p3 turns out to be fair
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
    s._measure_foreign_winrates = lambda modes=None: [0.50]
    _step(s)
    assert s._ac_wr_ema[0] == 0.50, "first held eval seeds the ema"
    _step(s)
    assert s._ac_wr_ema[0] == 0.50, "second held eval keeps folding it in"
    assert s._foreign_periods[0] == 3, "and the period stayed put throughout"


def test_falls_back_to_shared_knob_without_an_eval_engine():
    s = _stub([0.90, 0.90], [4, 4])
    s._ac_eval = None
    s._measure_foreign_winrates = lambda modes=None: None
    s._ac_ema = 99.0                      # way above band -> harden
    _step(s)
    assert s._foreign_periods == [3, 3], "ep_rew fallback still drives one shared knob"


def test_an_eval_with_no_completed_match_holds_instead_of_using_ep_rew():
    """With an eval engine present, "no match completed" must HOLD.

    Falling through to the ep_rew fallback would apply ONE shared knob to every
    slot, driven by a shaped reward the whole auto-curriculum exists to avoid
    trusting (L8: win-prob shaping once drove ep_rew +510 while real skill fell).
    Under the staggered one-mode-per-cycle eval this case also looks routine --
    a cycle measuring only mode 3 leaves every mode-1 slot at None.
    """
    s = _stub([None, None], [4, 4])
    s._ac_ema = 99.0                      # would harden, if it were consulted
    _step(s)
    assert s._foreign_periods == [4, 4], "hold; do not move on the ep_rew margin"
    assert s._ac_wr_ema == [None, None]


# =====================================================================
# v9: LEXICOGRAPHIC rung control (foreign_cars coarse, period-ladder fine)
#
# Why it exists: `decision_period` has almost no authority above 1v1. Same net,
# reward_v0 tape, 3v3 goal share for us against a FULL nexto team at
# p12/p24/p48/p96 = 0.092/0.256/0.314/0.371 -- at one decision every 6.4 SECONDS
# it still out-scores us. At 1v1 the same knob sweeps 0.02 -> 0.95 over p1->p12.
# Car count is the knob that works at m>1.
#
# The control rule is monotone BY CONSTRUCTION, which is the point: we never
# have to know a priori whether (2 cars, p12) is harder than (1 car, p1).
# =====================================================================

LADDER = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]


def test_ladder_moves_one_rung_along_the_ladder_not_by_one_period():
    # p12 is index 8; one rung harder is index 7 = p10, NOT p11 (which is not on
    # the ladder at all). The ladder is coarse on purpose at the easy end.
    s = _stub([0.90], [12], ladder=LADDER, modes=[3])
    _step(s)
    assert s._foreign_periods[0] == 10
    assert s._foreign_cars[0] == 1, "period headroom remains, so cars must not move"


def test_running_out_of_period_headroom_adds_a_car_and_resets_the_period():
    # At the HARDEST period (index 0 = p1) with cars < mode, "harder" must spend
    # the coarse knob and drop back to the easiest period of the new car count.
    s = _stub([0.90], [1], ladder=LADDER, cars=[1], modes=[3])
    _step(s)
    assert s._foreign_cars[0] == 2, "out of period headroom -> add a foreign car"
    assert s._foreign_periods[0] == LADDER[-1], (
        "the new car count starts at its EASIEST period; the controller does not "
        "need to know whether that is a net step up or down"
    )


def test_running_out_of_the_other_way_removes_a_car():
    s = _stub([0.10], [24], ladder=LADDER, cars=[2], modes=[3])
    _step(s)
    assert s._foreign_cars[0] == 1
    assert s._foreign_periods[0] == LADDER[0], "fewer cars start at the hardest period"


def test_rung_saturates_at_one_car_and_the_easiest_period():
    # Nothing left to give: a slot pinned at (1 car, easiest period) that is
    # still unwinnable must simply hold. R3's "the 3v3 wall" is detected by
    # watching for exactly this state, not by the controller crashing.
    s = _stub([0.05], [24], ladder=LADDER, cars=[1], modes=[3])
    _step(s)
    assert (s._foreign_cars[0], s._foreign_periods[0]) == (1, 24)
    # ...and symmetrically at the hard end of a 1v1 slot (mode 1 -> max 1 car).
    s = _stub([0.99], [1], ladder=LADDER, cars=[1], modes=[1])
    _step(s)
    assert (s._foreign_cars[0], s._foreign_periods[0]) == (1, 1)


def test_a_car_count_change_resets_that_slots_ema():
    """The tested invariant extends to the COARSE knob.

    Stale evidence ratchets twice: the EMA that justified adding a car describes
    the OLD car count, and keeping it would immediately justify adding another
    without ever measuring the first.
    """
    s = _stub([0.90], [1], ladder=LADDER, cars=[1], modes=[3], alpha=0.3)
    _step(s)
    assert s._foreign_cars[0] == 2, "precondition: the coarse knob moved"
    assert s._ac_wr_ema[0] is None, "a foreign_cars change must clear the EMA too"


def test_two_rung_jump_only_outside_three_sigma():
    # |ema - 0.5| > big_move_margin (0.35) -> two rungs. Single-eval sd is
    # 0.118, so 0.85 is ~3 sd outside the band: not a noise-driven move.
    s = _stub([0.90], [12], ladder=LADDER, modes=[3], big=0.35)
    _step(s)
    assert s._foreign_periods[0] == 8, "0.90 is 3 sd out -> index 8 -> 6, i.e. p8"
    # ...and a merely out-of-band measurement still moves exactly one rung.
    s = _stub([0.70], [12], ladder=LADDER, modes=[3], big=0.35)
    _step(s)
    assert s._foreign_periods[0] == 10


def test_dwell_and_independence_still_hold_on_the_ladder():
    # The two properties the v8 controller was fixed for must survive the rung
    # rewrite: slots move independently, and a slot that just moved dwells.
    s = _stub([0.90, 0.10], [12, 12], kinds=["nexto", "necto"],
              ladder=LADDER, cars=[1, 2], modes=[3, 3], dwell=2)
    _step(s)
    assert s._foreign_periods[0] == 10, "the punching bag hardens"
    assert s._foreign_periods[1] == 16, "the wall eases, in the same pass"
    assert s._ac_since == [0, 0]
    s._measure_foreign_winrates = lambda modes=None: [0.90, 0.10]
    for _ in range(2):
        _step(s)
        assert s._foreign_periods == [10, 16], "dwell must block further moves"
    _step(s)
    assert s._foreign_periods == [8, 20], "after the dwell both may move again"
