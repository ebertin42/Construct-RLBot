import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
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


def test_expected_sd_halves_the_variance_when_both_orders_double_n():
    """The reason this is computed and not remembered. The published ~0.025 is a
    SINGLE-ORDER number (n~320); --both-orders is mandatory at m>1 and puts the
    same run at n~640, where a correct sd is smaller by sqrt(2). Quoting the
    single-order constant there would flag a clean run as broken."""
    one = m.expected_sd(320, 0.205)
    both = m.expected_sd(640, 0.205)
    assert one == pytest.approx(0.0249, abs=5e-4)
    assert both == pytest.approx(0.0176, abs=5e-4)
    assert both == pytest.approx(one / (2 ** 0.5), rel=1e-9)


def test_expected_sd_tracks_the_draw_rate():
    """Draws pull an outcome to the mean, so a drawier regime is LESS variable.
    m=2's higher draw rate (0.205 vs 0.155) is why its sd is not the 1v1 one."""
    assert m.expected_sd(320, 0.205) < m.expected_sd(320, 0.155)
    assert m.expected_sd(320, 0.155) == pytest.approx(0.0257, abs=5e-4)


def test_expected_sd_refuses_an_empty_run():
    assert m.expected_sd(0, 0.2) is None


def test_sd_band_reproduces_the_published_1v1_band():
    """Evidence that the Wilson-Hilferty generalisation is the same instrument
    that produced the historical hardcoded band, not a new invention: the 1v1
    protocol (sigma ~0.0257, 20 seeds) comes back as [0.0176, 0.0338], which is
    the published [0.018, 0.035] at the precision that band was quoted to."""
    lo, hi = m.sd_band(0.0257, 20)
    assert lo == pytest.approx(0.018, abs=1e-3)
    assert hi == pytest.approx(0.035, abs=2e-3)


def test_sd_band_on_the_both_orders_team_scale_accepts_a_correct_run():
    """The bug this pins: the historical band [0.018, 0.035] REJECTS a correct
    both-orders 2v2 null, whose sd lands at ~0.0176 -- below the old lower edge.
    On its own scale the same correct run sits comfortably inside."""
    exp = m.expected_sd(640, 0.205)
    lo, hi = m.sd_band(exp, 20)
    assert lo < exp < hi
    assert lo == pytest.approx(0.0121, abs=1e-3) and hi == pytest.approx(0.0232, abs=1e-3)
    assert not (0.018 <= lo), "the old band's lower edge is above this one's"


def test_sd_band_is_undefined_below_three_seeds():
    """An sd from two samples has a band wider than the number it qualifies."""
    assert m.sd_band(0.02, 2) is None
    assert m.sd_band(None, 20) is None


def test_duplicate_record_signature_is_not_the_correct_both_orders_sd():
    """The alarm and the answer must be distinguishable. Scaled off the run's own
    expected sd, the m=2 duplicate-record signature is 0.0125 against a correct
    0.0176; scaled off the single-order 0.0253 it would be 0.0179 -- i.e. the
    alarm would fire on every clean both-orders run."""
    exp = m.expected_sd(640, 0.205)
    signature = exp / (2 ** 0.5)
    assert signature == pytest.approx(0.0125, abs=5e-4)
    stale_alarm = 0.0253 / (2 ** 0.5)
    assert abs(stale_alarm - exp) < 5e-4, "the stale alarm sat on top of the answer"


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


# --- what --both-orders can and cannot see -----------------------------------

MATCH = 300          # a short "match" so the whole run stays a unit test
ARENAS = 8
BLUE_EDGE = 0.30     # injected P(blue scores) - P(orange scores), by construction


class _FakeEngine:
    """A scoring tape with a DELIBERATE blue advantage. Self-play in this script
    means the same net drives both sides, so the only thing an order can differ
    in is the arena itself -- which is exactly what this fakes."""

    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def set_weights(self, sd):
        pass

    def set_opponents(self, sds):
        pass

    def collect(self, steps, arena_opponents=None):
        ncol = ARENAS * 2
        rew = np.zeros((steps, ncol), dtype=np.float32)
        term = np.zeros((steps, ncol), dtype=bool)
        term[MATCH - 1::MATCH] = True
        for k in range(ARENAS):
            cols = slice(k * 2, (k + 1) * 2)
            for t0 in range(0, steps, MATCH):
                u = self.rng.random()
                if u < 0.45 + BLUE_EDGE / 2:
                    rew[t0 + 10, cols] = 10.0
                elif u < 0.9:
                    rew[t0 + 10, cols] = -10.0
        return {"rewards": rew, "terminated": term}


class _FakeRunner:
    def __init__(self, num_arenas, seed, mode, **kw):
        self.eng = _FakeEngine(seed)
        self.assignment = [0] * num_arenas


_CACHE = {}


def _run_null(monkeypatch, capsys, *extra):
    """Cached across tests: split_matches walks the tape a step at a time, so a
    stubbed run is seconds rather than milliseconds and four tests read the same
    two runs."""
    if extra in _CACHE:
        return _CACHE[extra]
    import construct.league.matches as matches
    monkeypatch.setattr(matches, "MatchRunner", _FakeRunner)
    monkeypatch.setattr(matches, "load_sd", lambda p: {})
    m.main(["--champion", "fake.pt", "--mode", "2", "--arenas", str(ARENAS),
            "--steps", "1500", "--seeds", *[str(s) for s in range(11, 23)], *extra])
    _CACHE[extra] = capsys.readouterr().out
    return _CACHE[extra]


def test_both_orders_mean_cannot_see_an_arena_asymmetry(monkeypatch, capsys):
    """The bug this pins. --both-orders is self-play in BOTH iterations, so the
    flip makes a systematic blue edge b cancel: order 0 contributes 0.5+b and
    flipped order 1 contributes 0.5-b. A 0.30 blue edge -- enormous -- still
    leaves the combined mean sitting on 0.5, so 'mean within 0.5 +/- 0.011' is
    not a symmetry criterion, it is a tautology."""
    out = _run_null(monkeypatch, capsys, "--both-orders")
    mean = float(out.split("both orders, fake.pt]: mean=")[1].split()[0])
    assert abs(mean - 0.5) < 0.02, "the flip cancels b, so this cannot detect it"
    assert "BY CONSTRUCTION" in out, "and the output has to say so"


def test_mirror_symmetry_block_does_see_it(monkeypatch, capsys):
    """The same run, read off the UNFLIPPED blue-perspective shares -- where the
    edge survives instead of cancelling. This is the asymmetry test the docstring
    used to attribute to --both-orders."""
    out = _run_null(monkeypatch, capsys, "--both-orders")
    # split on the BLOCK header: "MIRROR SYMMETRY" also appears in the pointer
    # printed next to the combined mean.
    block = out.split("MIRROR SYMMETRY [")[1]
    edge = float(block.split("blue share = ")[1].split()[0]) - 0.5
    assert edge == pytest.approx(BLUE_EDGE / 2, abs=0.05), (
        "P(blue) - P(orange) = 0.30 is a win-share edge of 0.15")
    assert "ASYMMETRIC" in block


def test_expected_sd_is_quoted_on_the_n_the_run_measured(monkeypatch, capsys):
    """Both orders doubles n, so the correct sd is smaller by sqrt(2). The
    reference printed next to it must move with the protocol -- a hardcoded
    single-order constant is what made a clean both-orders run look like the
    duplicate-record bug."""
    one = _run_null(monkeypatch, capsys)
    both = _run_null(monkeypatch, capsys, "--both-orders")
    exp_one = float(one.split("expected sd for this run ~")[1].split()[0])
    exp_both = float(both.split("expected sd for this run ~")[1].split()[0])
    assert exp_both == pytest.approx(exp_one / (2 ** 0.5), rel=0.05)
    assert "single order" in one and "both orders" in both


def test_duplicate_record_alarm_sits_below_the_correct_sd(monkeypatch, capsys):
    """The alarm has to be distinguishable from the answer. Scaled off this run's
    own expected sd it is expected/sqrt(2); scaled off the old single-order 0.0253
    it would have landed ON the correct both-orders value."""
    out = _run_null(monkeypatch, capsys, "--both-orders")
    exp = float(out.split("expected sd for this run ~")[1].split()[0])
    alarm = float(out.split("duplicate-record signature: sd near ")[1].split()[0])
    assert alarm == pytest.approx(exp / (2 ** 0.5), abs=1e-3)
    assert alarm < exp
