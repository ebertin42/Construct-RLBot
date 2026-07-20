"""Tests for scripts/gate_stats.py.

The point of this tool is to stop a share from being read without its error
bar, so the tests are mostly about the tool REFUSING to imply an ordering it
cannot support.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import gate_stats  # noqa: E402


def write_history(tmp_path, rows):
    p = tmp_path / "champion_history.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def row(name, gc, gh):
    return {"candidate": f"checkpoints/{name}", "goals_c": gc, "goals_champ": gh,
            "share": gc / (gc + gh), "verdict": "FAIL"}


# --- intervals --------------------------------------------------------------

def test_wilson_interval_brackets_the_point_estimate():
    lo, hi = gate_stats.wilson(75, 182)
    assert lo < 75 / 182 < hi


def test_wilson_stays_inside_unit_interval_at_the_extremes():
    for k, n in [(0, 50), (50, 50), (1, 200)]:
        lo, hi = gate_stats.wilson(k, n)
        assert 0.0 <= lo <= hi <= 1.0


def test_a_real_gate_carries_a_band_around_seven_points():
    """~180 goals -> the 95% band is wide enough to swallow most differences
    anyone is tempted to interpret. If this ever tightens a lot, the sample
    size changed and the journal's epistemics need revisiting."""
    s = gate_stats.summarize(75, 107)
    assert 0.05 < (s["hi"] - s["lo"]) < 0.16


# --- pooling ----------------------------------------------------------------

def test_pooling_adds_goals_not_averages_shares():
    """Averaging shares would weight a short gate equally with a long one."""
    rows = [row("hc_a", 75, 107), row("hc_b", 74, 102)]
    s = gate_stats.pool(rows)
    assert s["goals_c"] == 149 and s["goals_champ"] == 209
    assert s["n"] == 358
    assert s["se"] < gate_stats.summarize(75, 107)["se"], "pooling must tighten the SE"


def test_pool_ignores_rows_without_raw_counts():
    rows = [row("hc_a", 75, 107), {"candidate": "old.pt", "share": 0.5}]
    s = gate_stats.pool(rows)
    assert s["gates"] == 1


# --- the ordering guard -----------------------------------------------------

def test_compare_refuses_to_order_the_2026_07_20_lambda_ladder(tmp_path, capsys):
    """THE REGRESSION THIS TOOL EXISTS FOR: armG 46.4% vs pooled hill-climb
    41.6% looks like an ordering and is not one (z=1.01)."""
    hist = write_history(tmp_path, [
        row("armG_selfanchor05.pt", 77, 89),
        row("hc_a0014_x.pt", 75, 107),
        row("hc_a0015_x.pt", 74, 102),
    ])
    rc = gate_stats.main(["--history", hist, "compare", "armG", "hc_a"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "NOT RESOLVED" in out
    assert "Do not report an ordering" in out


def test_compare_does_resolve_a_genuinely_large_gap(tmp_path, capsys):
    """Low lambda really is worse -- the tool must not cry noise at everything."""
    hist = write_history(tmp_path, [
        row("armF_lambda02.pt", 60, 123),
        row("armH_lambda10.pt", 93, 96),
    ])
    gate_stats.main(["--history", hist, "compare", "armH", "armF"])
    out = capsys.readouterr().out
    assert "RESOLVED" in out and "NOT RESOLVED" not in out
    assert "better" in out


def test_list_never_prints_a_share_without_its_interval(tmp_path, capsys):
    hist = write_history(tmp_path, [row("hc_a0014_x.pt", 75, 107)])
    gate_stats.main(["--history", hist, "list"])
    out = capsys.readouterr().out
    assert "share=" in out and "95%CI=[" in out and "SE=" in out


def test_list_flags_rows_it_cannot_put_a_band_on(tmp_path, capsys):
    hist = write_history(tmp_path, [{"candidate": "legacy.pt", "share": 0.5}])
    gate_stats.main(["--history", hist, "list"])
    out = capsys.readouterr().out
    assert "skipped" in out and "not reportable" in out


# --- power ------------------------------------------------------------------

def test_needed_gates_matches_the_journals_seven_gates_per_point():
    """Journal f9188d5 claims ~7 gates per arm to resolve 5 points. If the
    arithmetic behind that claim drifts, this test should catch it."""
    assert 4 <= gate_stats.needed_gates(0.05) <= 10


def test_needed_gates_grows_as_the_effect_shrinks():
    assert gate_stats.needed_gates(0.02) > gate_stats.needed_gates(0.10)


def test_z_test_is_symmetric_under_argument_order():
    a = gate_stats.summarize(93, 96)
    b = gate_stats.summarize(60, 123)
    ab = gate_stats.two_proportion_z(a, b)
    ba = gate_stats.two_proportion_z(b, a)
    assert ab["z"] == pytest.approx(-ba["z"])
    assert ab["p"] == pytest.approx(ba["p"])


def test_p_value_agrees_with_a_known_normal_tail():
    """z=1.96 two-sided should be ~0.05."""
    a = {"share": 0.5 + 1.96 * 0.01, "se": 0.01}
    b = {"share": 0.5, "se": 0.0}
    assert gate_stats.two_proportion_z(a, b)["p"] == pytest.approx(0.05, abs=0.002)


def test_missing_history_file_is_not_a_crash(tmp_path):
    assert gate_stats.load_rows(tmp_path / "nope.jsonl") == []


# --- multi-pattern selection ------------------------------------------------

def test_grep_ors_over_a_comma_separated_list(tmp_path, capsys):
    """Needed to pool a chosen set of attempts. Comma rather than regex because
    the natural `hc_a001[45]` dies in zsh before the tool sees it."""
    hist = write_history(tmp_path, [
        row("hc_a0014_x.pt", 75, 107),
        row("hc_a0015_x.pt", 74, 102),
        row("hc_a0016_x.pt", 93, 86),
    ])
    gate_stats.main(["--history", hist, "pool", "--grep", "hc_a0014,hc_a0015"])
    out = capsys.readouterr().out
    assert "2 gates" in out
    assert "149- 209" in out.replace("  ", " ").replace("  ", " ") or "149" in out


def test_compare_selectors_accept_comma_lists(tmp_path, capsys):
    hist = write_history(tmp_path, [
        row("hc_a0014_x.pt", 75, 107),
        row("hc_a0015_x.pt", 74, 102),
        row("hc_a0016_x.pt", 93, 86),
    ])
    gate_stats.main(["--history", hist, "compare", "hc_a0016", "hc_a0014,hc_a0015"])
    out = capsys.readouterr().out
    assert "1 gates" in out and "2 gates" in out


def test_whitespace_in_the_comma_list_is_tolerated(tmp_path):
    rows = [row("armG_x.pt", 77, 89), row("armH_x.pt", 93, 96)]
    assert len(gate_stats.select(rows, " armG , armH ")) == 2
