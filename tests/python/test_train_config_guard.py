import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from construct.learn.train import check_win_prob_gamma  # noqa: E402


def test_matching_gamma_passes():
    check_win_prob_gamma({"win_prob_weight": 10.0, "win_prob_gamma": 0.9954},
                         {"gamma": 0.9954})


def test_mismatched_gamma_raises_with_both_values_named():
    """A silent mismatch trains a different objective and looks normal, so the
    error must name both numbers and say why it matters."""
    with pytest.raises(ValueError) as e:
        check_win_prob_gamma({"win_prob_weight": 10.0, "win_prob_gamma": 0.99},
                             {"gamma": 0.9954})
    msg = str(e.value)
    assert "0.99" in msg and "0.9954" in msg
    assert "potential" in msg.lower()


def test_guard_is_inert_when_shaping_is_off():
    """Historical configs have no win_prob keys; they must not trip the guard."""
    check_win_prob_gamma({}, {"gamma": 0.9954})
    check_win_prob_gamma({"win_prob_weight": 0.0, "win_prob_gamma": 0.0},
                         {"gamma": 0.9954})


def test_shaping_without_match_mode_raises():
    """Win-prob shaping needs a score and a clock. Without match_mode there is
    neither, PHI never moves, and the run trains on a constant while looking
    perfectly healthy."""
    from construct.learn.train import check_match_mode_required
    with pytest.raises(ValueError) as e:
        check_match_mode_required({"win_prob_weight": 10.0}, {"match_mode": False})
    assert "match_mode" in str(e.value)


def test_match_mode_without_shaping_is_allowed():
    """Full matches with a legacy reward is a legitimate ablation -- it isolates
    the match layer from the objective change."""
    from construct.learn.train import check_match_mode_required
    check_match_mode_required({"win_prob_weight": 0.0}, {"match_mode": True})


def test_shaping_with_match_mode_passes():
    from construct.learn.train import check_match_mode_required
    check_match_mode_required({"win_prob_weight": 10.0}, {"match_mode": True})
