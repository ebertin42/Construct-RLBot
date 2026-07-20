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
