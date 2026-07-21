"""Tests for scripts/matchwin_gate.py — the match-win promotion gate.

This gate DECIDES PROMOTIONS, so the two error-prone pure pieces are pinned
here: the side-order flip (order 2 plays the champion, so its records must be
inverted to the candidate's perspective before summing) and the win-share /
threshold arithmetic. A bug in either silently promotes or rejects.
"""
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import matchwin_gate as mg  # noqa: E402


# --- the side-order flip ----------------------------------------------------

def test_flip_swaps_each_pair_to_candidate_perspective():
    # order 2 records are (champion_goals, candidate_goals); flipped they must
    # become (candidate_goals, champion_goals).
    assert mg.flip_to_candidate([(3, 1), (0, 2)]) == [(1, 3), (2, 0)]


def test_flip_of_empty_is_empty():
    assert mg.flip_to_candidate([]) == []


def test_flip_is_its_own_inverse():
    m = [(5, 2), (0, 0), (1, 4)]
    assert mg.flip_to_candidate(mg.flip_to_candidate(m)) == m


def test_a_champion_win_in_order2_becomes_a_candidate_loss():
    # (champion 2, candidate 0) -> candidate perspective (0, 2): a loss.
    flipped = mg.flip_to_candidate([(2, 0)])
    a, b = flipped[0]
    assert a < b, "a champion win must read as a candidate loss after the flip"


# --- the aggregation arithmetic ---------------------------------------------

def test_aggregate_sums_both_orders():
    r = mg.aggregate((10, 2, 8), (12, 4, 6), 0.55)
    assert r["wins"] == 22 and r["draws"] == 6 and r["losses"] == 14
    assert r["n"] == 42


def test_draws_count_half():
    # 0W, 4D, 0L over both orders -> exactly 0.5.
    assert mg.aggregate((0, 2, 0), (0, 2, 0), 0.55)["share"] == pytest.approx(0.5)


def test_pass_requires_share_at_or_above_threshold():
    # 60% win share clears 0.55.
    r = mg.aggregate((30, 0, 20), (30, 0, 20), 0.55)
    assert r["share"] == pytest.approx(0.6) and r["verdict"] == "PASS"


def test_parity_is_a_fail_not_a_promotion():
    # Exactly the match-win arm's outcome: ~parity must NOT promote.
    r = mg.aggregate((25, 0, 25), (25, 0, 25), 0.55)
    assert r["share"] == pytest.approx(0.5) and r["verdict"] == "FAIL"


def test_threshold_boundary_is_inclusive():
    # share exactly == threshold passes (>=).
    r = mg.aggregate((55, 0, 45), (55, 0, 45), 0.55)
    assert r["share"] == pytest.approx(0.55) and r["verdict"] == "PASS"


def test_just_below_threshold_fails():
    r = mg.aggregate((54, 0, 46), (55, 0, 45), 0.55)
    assert r["share"] < 0.55 and r["verdict"] == "FAIL"


def test_no_completed_matches_is_none_share_not_zero():
    # 0 of 0 is not a total loss; share must be None, never 0.0, so the gate
    # never rejects on zero evidence.
    r = mg.aggregate((0, 0, 0), (0, 0, 0), 0.55)
    assert r["share"] is None and r["verdict"] == "FAIL"
    assert r["n"] == 0


def test_se_matches_binomial_at_half():
    r = mg.aggregate((160, 0, 160), (160, 0, 160), 0.55)
    assert r["se"] == pytest.approx(math.sqrt(0.25 / 640))


def test_draws_do_not_move_share_off_parity():
    # A draw-heavy split with equal W and L stays at 0.5 -- draws are neutral,
    # exactly why the iter-580 draw-degeneration read as ~parity-to-below, not
    # a win.
    r = mg.aggregate((100, 199, 359), (0, 0, 0), 0.55)
    # wins 100, draws 199, losses 359, n=658
    expected = (100 + 0.5 * 199) / 658
    assert r["share"] == pytest.approx(expected)
    assert r["verdict"] == "FAIL"
