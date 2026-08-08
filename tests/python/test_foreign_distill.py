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


def _tiny_batch(n=64, a=8, seed=0):
    """A minimal v0-shaped PPO batch: obs/actions/logprobs/advantages/returns."""
    g = torch.Generator().manual_seed(seed)
    return {
        "obs": torch.randn(n, 12, generator=g),
        "actions": torch.randint(0, a, (n,), generator=g),
        "logprobs": torch.full((n,), -float(np.log(a))),
        "advantages": torch.randn(n, generator=g),
        "returns": torch.randn(n, generator=g),
    }


def _tiny_net(a=8):
    from construct.learn.model import PolicyValueNet
    torch.manual_seed(0)
    return PolicyValueNet(obs_size=12, action_count=a, hidden=(16,))


def test_policy_coef_zero_removes_the_policy_gradient():
    """The distillation loss only ADDS to PPO's. Without policy_coef=0 the policy gradient
    competes with the teacher for the whole run, so a 'pure distillation' gate would in fact
    measure a mixture. Assert the term is genuinely gone, not just small."""
    from construct.learn.ppo import ppo_update
    batch = _tiny_batch()
    net = _tiny_net()
    opt = torch.optim.SGD(net.parameters(), lr=0.0)  # lr 0: stats only, weights frozen
    stats = ppo_update(net, opt, batch, epochs=1, minibatch_size=64,
                       entropy_coef=0.0, value_coef=0.0, policy_coef=0.0)
    # policy_loss is still REPORTED (it is computed for the stat) but must not have moved the
    # weights: with every coefficient zero there is no loss at all.
    before = [p.detach().clone() for p in net.parameters()]
    opt2 = torch.optim.SGD(net.parameters(), lr=1.0)
    ppo_update(net, opt2, batch, epochs=1, minibatch_size=64,
               entropy_coef=0.0, value_coef=0.0, policy_coef=0.0)
    for b, p in zip(before, net.parameters()):
        assert torch.equal(b, p), "policy_coef=0 with no other term must not change weights"
    assert "policy_loss" in stats


def test_policy_coef_one_is_unchanged_from_the_default():
    """Byte-identity guard: 1.0 takes the identity branch, so passing it explicitly must give
    exactly the same weights as not passing it at all."""
    from construct.learn.ppo import ppo_update
    outs = []
    for kwargs in ({}, {"policy_coef": 1.0}):
        net = _tiny_net()
        opt = torch.optim.SGD(net.parameters(), lr=0.1)
        ppo_update(net, opt, _tiny_batch(), epochs=2, minibatch_size=32, **kwargs)
        outs.append([p.detach().clone() for p in net.parameters()])
    for a, b in zip(*outs):
        assert torch.equal(a, b), "policy_coef=1.0 must be byte-identical to the default"


# --- multi-teacher mixture sampler ------------------------------------------------------
#
# The engine holds exactly ONE teacher, so a weighted mixture is realised in time: one
# teacher sampled per iteration. Three properties have to hold or the arm measures something
# other than the mixture it claims. `_pick_teacher` touches only a handful of attributes, so
# it is exercised through a stand-in rather than a real Trainer (which would want an engine,
# a checkpoint and a GPU to say anything about six lines of arithmetic).


class _FakeEngine:
    def __init__(self):
        self.calls = []

    def set_teacher(self, weights, kind):
        self.calls.append(kind)


class _Stub:
    """Minimal surface of Trainer that _pick_teacher reads."""

    def __init__(self, mix, seed=41, steps=0):
        from types import SimpleNamespace
        self.cfg = SimpleNamespace(env={"seed": seed})
        self.total_steps = steps
        self.engine = _FakeEngine()
        self._teacher_mix = mix
        self._foreign_teacher = mix[0][0] if mix else None


def _pick(stub):
    from construct.learn.train import Trainer
    Trainer._pick_teacher(stub)


def _draw(mix, n, seed=41, step=1_000_000):
    """n independent draws, one per distinct total_steps, as run() would make them."""
    out = []
    for i in range(n):
        s = _Stub(mix, seed=seed, steps=i * step)
        s._foreign_teacher = None  # force a set_teacher on every pick so we can read it
        _pick(s)
        out.append(s._foreign_teacher)
    return out


def test_mixture_respects_the_weights():
    """A 0.75/0.25 split has to actually be 0.75/0.25. If the sampler were uniform the arm
    would be a half-and-half mixture wearing a nexto-dominant label, and the whole reason
    nexto is dominant is that its argmax is the one we want preserved."""
    mix = [("nexto", {}, 0.75), ("immortal", {}, 0.25)]
    draws = _draw(mix, 4000)
    frac = draws.count("nexto") / len(draws)
    # se at n=4000 is ~0.0068; 5 se is a wide enough band to never flake and still catch a
    # uniform sampler (which would land at 0.50, 37 se away).
    assert 0.75 - 0.034 < frac < 0.75 + 0.034, f"nexto share {frac:.3f} != 0.75"
    assert set(draws) == {"nexto", "immortal"}


def test_pick_is_keyed_on_total_steps_not_on_iteration():
    """`it` restarts at 0 on every resume. An it-keyed sequence would replay the same teacher
    order after each restart, correlating teacher choice with restarts -- the same trap the lr
    anneal already had to be moved off. Same total_steps must give the same teacher; different
    total_steps must not all give one."""
    mix = [("nexto", {}, 0.5), ("immortal", {}, 0.5)]
    a = _Stub(mix, steps=7_000_000); a._foreign_teacher = None; _pick(a)
    b = _Stub(mix, steps=7_000_000); b._foreign_teacher = None; _pick(b)
    assert a._foreign_teacher == b._foreign_teacher
    assert len(set(_draw(mix, 50))) == 2


def test_pick_skips_the_engine_call_when_the_teacher_is_unchanged():
    """set_teacher REBUILDS the ForeignPolicy and drops its per-car prev-action map, which
    the teacher's own observation carries. Re-installing the teacher it already has would pay
    that reset for nothing."""
    s = _Stub([("nexto", {}, 1.0)])
    for _ in range(5):
        _pick(s)
    assert s.engine.calls == [], "an unchanged teacher must not be re-installed"


def test_empty_mixture_is_a_no_op():
    """Single-teacher runs go through the pre-mixture path untouched: _pick_teacher must not
    so much as look at the engine, or every existing distillation arm changes behaviour."""
    s = _Stub([])
    _pick(s)
    assert s.engine.calls == [] and s._foreign_teacher is None
