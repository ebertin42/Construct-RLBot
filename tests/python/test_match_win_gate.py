import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from construct.league.matches import match_record, split_matches  # noqa: E402

TH = 9.4


def tape(*rows):
    """(T,1) reward tape for a single arena."""
    return np.array(rows, dtype=np.float32).reshape(-1, 1)


def flags(*rows):
    return np.array(rows, dtype=bool).reshape(-1, 1)


def test_splits_on_terminated_boundaries():
    r = tape(10.0, 0.1, -10.0, 0.0, 10.0, 0.0)
    t = flags(False, False, False, True, False, True)
    assert split_matches(r, t, TH) == [(1, 1), (1, 0)]


def test_sub_threshold_rows_are_not_goals():
    """Shaping noise must never be counted as a goal."""
    r = tape(0.55, -0.55, 9.39, -9.39, 0.0)
    t = flags(False, False, False, False, True)
    assert split_matches(r, t, TH) == [(0, 0)]


def test_trailing_incomplete_match_is_discarded():
    """A match still in progress at the end of the tape has no outcome and must
    not be scored as a draw -- that would bias every gate toward 0.5."""
    r = tape(10.0, 0.0, 10.0)
    t = flags(False, True, False)
    assert split_matches(r, t, TH) == [(1, 0)]


def test_match_record_counts_wins_draws_losses():
    rec = match_record([(2, 1), (0, 0), (1, 3), (1, 0)])
    assert rec["wins"] == 2 and rec["draws"] == 1 and rec["losses"] == 1
    assert rec["win_share"] == pytest.approx((2 + 0.5) / 4)


def test_a_draw_counts_as_half():
    """Standard convention, and it keeps a self-play null control at exactly
    0.5 rather than pushing it around by the draw rate."""
    assert match_record([(0, 0), (0, 0)])["win_share"] == pytest.approx(0.5)


def test_no_completed_matches_is_none_not_zero():
    """0 wins of 0 matches is not 0% -- returning 0.0 would read as a total
    loss and could promote or reject on nothing."""
    assert match_record([])["win_share"] is None
