"""Tests for the foreign-teacher distillation loss.

The load-bearing property is the equivalence-class collapse. Without it the loss carries an
irreducible ln(3) floor that no student can pay down, which would read as "distillation is
not working" when in fact it had converged.
"""

import numpy as np
import pytest
import torch

from construct.learn.foreign_distill import (
    collapse_logits,
    equivalence_classes,
    foreign_distill_loss,
)
from construct._engine import action_table_v1_air

# The table is generated in Rust (`actions::make_lookup_table_v1_air`) and reaches the python
# side either through this export or through a checkpoint's registered `action_table` buffer.
# Taking it from the engine keeps these tests independent of any checkpoint file existing.
ACTION_TABLE_V1_AIR = action_table_v1_air()


def test_equivalence_classes_match_the_known_duplicate_groups():
    cls = equivalence_classes(ACTION_TABLE_V1_AIR)
    assert cls.shape[0] == len(ACTION_TABLE_V1_AIR) == 104
    assert int(cls.max()) + 1 == 96, "104 rows, 8 duplicates -> 96 distinct control vectors"
    # The six groups documented in engine/src/actions.rs.
    for group in ([56, 94, 100], [57, 95, 101], [92, 98], [93, 99], [96, 102], [97, 103]):
        ids = {int(cls[r]) for r in group}
        assert len(ids) == 1, f"rows {group} are byte-identical and must share a class"
    # Nothing ELSE is merged: the 8 duplicate rows above are the only ones sharing a class,
    # so exactly 96 rows are the first appearance of their class and the map is onto [0,96).
    firsts = [r for r in range(104) if (cls[:r] != cls[r]).all()]
    assert len(firsts) == 96
    assert sorted(int(cls[r]) for r in firsts) == list(range(96))


def test_collapse_sums_probability_not_logits():
    # Three rows in one class, one alone. Equal logits -> the 3-row class must hold 3/4 of
    # the mass. Summing logits (rather than logsumexp) would give 1/2, and averaging 1/2 too;
    # only probability-summing gives 3/4.
    cls = torch.tensor([0, 0, 0, 1])
    logits = torch.zeros(1, 4)
    out = collapse_logits(logits, cls, 2)
    p = out.softmax(-1)[0]
    assert p[0] == pytest.approx(0.75, abs=1e-6)
    assert p[1] == pytest.approx(0.25, abs=1e-6)


def test_collapse_is_stable_at_large_logits():
    cls = torch.tensor([0, 0, 1])
    out = collapse_logits(torch.tensor([[300.0, 300.0, -300.0]]), cls, 2)
    assert torch.isfinite(out).all()
    assert float(out[0, 0]) == pytest.approx(300.0 + np.log(2), abs=1e-3)


def test_collapse_matches_a_naive_logsumexp_reference():
    torch.manual_seed(0)
    cls = equivalence_classes(ACTION_TABLE_V1_AIR)
    logits = torch.randn(7, 104) * 5
    out = collapse_logits(logits, cls, 96)
    for c in range(96):
        members = (cls == c).nonzero().flatten()
        assert torch.allclose(out[:, c], torch.logsumexp(logits[:, members], dim=1), atol=1e-5)


def test_duplicate_rows_cost_the_student_nothing():
    """The whole point. A student that spreads mass over a class's members must score the
    same as one that concentrates it, because the arena cannot tell them apart."""
    cls = equivalence_classes(ACTION_TABLE_V1_AIR)
    spread = torch.zeros(1, 104)
    spread[0, [56, 94, 100]] = 5.0            # class mass split three ways: logsumexp = 5+ln3
    conc = torch.zeros(1, 104)
    conc[0, [94, 100]] = -1e4                 # siblings emptied...
    conc[0, 56] = 5.0 + float(np.log(3))      # ...and their mass folded into row 56
    ta = torch.tensor([56])
    a, _ = foreign_distill_loss(spread, ta, cls, 96)
    b, _ = foreign_distill_loss(conc, ta, cls, 96)
    assert float(a) == pytest.approx(float(b), abs=1e-4)


def test_a_noncanonical_label_would_be_scored_identically():
    """The engine canonicalises to the first match, but if a label ever arrived as 100
    instead of 56 the loss must be unchanged -- they are the same control vector."""
    cls = equivalence_classes(ACTION_TABLE_V1_AIR)
    torch.manual_seed(1)
    logits = torch.randn(1, 104)
    a, _ = foreign_distill_loss(logits, torch.tensor([56]), cls, 96)
    b, _ = foreign_distill_loss(logits, torch.tensor([100]), cls, 96)
    assert float(a) == pytest.approx(float(b), abs=1e-6)


def test_sentinel_rows_are_masked_not_indexed():
    """-1 must never reach the table: action_table[-1] is row 103 in torch, a real action."""
    cls = equivalence_classes(ACTION_TABLE_V1_AIR)
    torch.manual_seed(2)
    logits = torch.randn(4, 104)
    ta = torch.tensor([-1, 5, -1, 7])
    loss, info = foreign_distill_loss(logits, ta, cls, 96)
    assert info["fd_label_frac"] == pytest.approx(0.5)
    kept, _ = foreign_distill_loss(logits[[1, 3]], torch.tensor([5, 7]), cls, 96)
    assert float(loss) == pytest.approx(float(kept), abs=1e-6)


def test_all_masked_minibatch_is_zero_and_still_differentiable():
    cls = equivalence_classes(ACTION_TABLE_V1_AIR)
    logits = torch.randn(3, 104, requires_grad=True)
    loss, info = foreign_distill_loss(logits, torch.tensor([-1, -1, -1]), cls, 96)
    assert loss.item() == 0.0
    assert info["fd_label_frac"] == 0.0
    loss.backward()  # must not raise: the graph stays connected via the *0 no-op
    assert torch.count_nonzero(logits.grad) == 0


def test_loss_is_minimised_by_matching_the_teacher():
    """Sanity: gradient descent on this loss actually moves mass onto the teacher's class."""
    cls = equivalence_classes(ACTION_TABLE_V1_AIR)
    torch.manual_seed(3)
    logits = torch.zeros(64, 104, requires_grad=True)
    ta = torch.full((64,), 7)
    # Adam, not SGD: cross_entropy averages over the batch, so a plain SGD step moves each
    # logit by ~1/64 of the gradient and 50 steps is nowhere near convergence -- which is a
    # fact about the optimiser, not about the loss.
    opt = torch.optim.Adam([logits], lr=0.1)
    first = None
    for _ in range(200):
        loss, _ = foreign_distill_loss(logits, ta, cls, 96)
        if first is None:
            first = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final, _ = foreign_distill_loss(logits, ta, cls, 96)
    assert first == pytest.approx(np.log(96), abs=0.1), "zero logits => uniform over classes"
    assert final.item() < first * 0.1
    assert int(collapse_logits(logits.detach(), cls, 96)[0].argmax()) == int(cls[7])
