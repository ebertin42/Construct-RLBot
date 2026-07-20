"""Tests for scripts/decompose_update.py.

The maths here decides whether "the harmful part of a PPO update is its
seed-shared component" is answerable, so the algebra gets pinned directly:
residuals must be orthogonal-ish to the mean, rescaling must preserve direction
exactly, and a decomposition of identical updates must have no residual at all.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import decompose_update as du  # noqa: E402

KEYS = ["a", "b"]


def mk(a, b):
    return {"a": torch.tensor(a, dtype=torch.float32),
            "b": torch.tensor(b, dtype=torch.float32)}


def test_mean_of_identical_updates_is_that_update():
    d = mk([1.0, 2.0], [3.0])
    m = du.mean_delta([d, d, d], KEYS)
    assert torch.allclose(m["a"], d["a"]) and torch.allclose(m["b"], d["b"])


def test_identical_updates_leave_no_residual():
    """If every seed did the same thing there is nothing seed-specific, and the
    residual probe would be meaningless -- the script must not manufacture one."""
    d = mk([1.0, 2.0], [3.0])
    m = du.mean_delta([d, d], KEYS)
    resid = {k: d[k] - m[k] for k in KEYS}
    assert du.norm_of(resid, KEYS) == pytest.approx(0.0, abs=1e-6)


def test_residuals_sum_to_zero_across_runs():
    """Definitional: sum(u_i - mean) = 0. If this ever fails the decomposition
    is not a decomposition."""
    ds = [mk([1.0, 0.0], [2.0]), mk([0.0, 3.0], [-1.0]), mk([2.0, 1.0], [0.5])]
    m = du.mean_delta(ds, KEYS)
    total = {k: sum(d[k] - m[k] for d in ds) for k in KEYS}
    assert du.norm_of(total, KEYS) == pytest.approx(0.0, abs=1e-5)


def test_rescale_hits_the_target_norm():
    d = mk([3.0, 4.0], [0.0])
    out = du.rescale(d, KEYS, 10.0)
    assert du.norm_of(out, KEYS) == pytest.approx(10.0, abs=1e-5)


def test_rescale_preserves_direction_exactly():
    """Only the length may change -- the per-layer profile IS part of what makes
    it PPO's direction, so a rescale that reshaped it would destroy the probe."""
    d = mk([3.0, 4.0], [1.0])
    out = du.rescale(d, KEYS, 7.0)
    assert du.cosine(d, out, KEYS) == pytest.approx(1.0, abs=1e-6)


def test_rescale_refuses_a_zero_direction():
    with pytest.raises(SystemExit):
        du.rescale(mk([0.0, 0.0], [0.0]), KEYS, 1.0)


def test_cosine_of_orthogonal_directions_is_zero():
    assert du.cosine(mk([1.0, 0.0], [0.0]), mk([0.0, 1.0], [0.0]), KEYS) == pytest.approx(0.0)


def test_apply_direction_adds_and_leaves_other_entries_alone():
    champ = {"a": torch.zeros(2), "b": torch.zeros(1), "action_table": "opaque"}
    out = du.apply_direction(champ, mk([1.0, 2.0], [3.0]), KEYS)
    assert torch.allclose(out["a"], torch.tensor([1.0, 2.0]))
    assert out["action_table"] == "opaque", "non-float entries must survive untouched"


def test_apply_direction_does_not_mutate_the_champion():
    champ = {"a": torch.zeros(2), "b": torch.zeros(1)}
    du.apply_direction(champ, mk([5.0, 5.0], [5.0]), KEYS)
    assert float(champ["a"].abs().sum()) == 0.0, "champion must not be modified in place"


def test_anticorrelated_updates_have_a_near_zero_shared_component():
    """Two updates pointing opposite ways share nothing; the shared probe should
    be tiny and the script's rescale would then be extrapolating from noise --
    which is exactly the case the printed ||shared mean|| is there to expose."""
    ds = [mk([1.0, 0.0], [0.0]), mk([-1.0, 0.0], [0.0])]
    m = du.mean_delta(ds, KEYS)
    assert du.norm_of(m, KEYS) == pytest.approx(0.0, abs=1e-6)


def test_float_keys_skips_non_float_entries():
    sd = {"w": torch.zeros(2), "idx": torch.zeros(2, dtype=torch.long), "s": "x"}
    assert du.float_keys(sd) == ["w"]
