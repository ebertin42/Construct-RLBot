import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import match_gate_null as m  # noqa: E402


def test_null_summary_reports_spread_not_just_a_mean():
    """A mean alone hides the spread, and the spread is the entire point: it
    tells us how large a win-share difference the gate can even resolve."""
    s = m.null_summary([0.5, 0.6, 0.4, 0.55, 0.45])
    assert s["n"] == 5
    assert s["mean"] == pytest.approx(0.5)
    assert s["sd"] > 0
    assert s["lo"] < s["mean"] < s["hi"]


def test_single_sample_has_no_defined_spread():
    s = m.null_summary([0.5])
    assert s["sd"] is None and s["lo"] is None and s["hi"] is None


def test_empty_input_is_not_a_crash():
    assert m.null_summary([])["n"] == 0


def test_empty_input_mean_is_none_so_main_must_guard_before_formatting():
    """main() formats s['mean'] with :.4f; f"{None:.4f}" raises TypeError.
    When every seed is dropped (win_share is None -- the realistic case until
    match_mode is wired into MatchRunner), shares is [] and mean must be None
    here so main()'s `if s["mean"] is None: ... return 0` guard fires instead
    of crashing on the format spec. main() itself isn't unit-testable without
    the engine (it constructs a MatchRunner), so this pins the precondition
    the guard depends on; the guard body is verified by inspection."""
    s = m.null_summary([])
    assert s["mean"] is None
