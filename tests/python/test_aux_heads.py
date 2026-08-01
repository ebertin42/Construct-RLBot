"""Aux heads (Lucy-SKG): reconstruction + multi-horizon return prediction.

WHY THESE EXIST AND WHY THEY WERE DEAD. `aux_reward`/`aux_recon` have been declared in
model_v1.py since the entity-transformer plan, but `aux_outputs()` was called nowhere, no
loss term used them, and train.py built the net WITHOUT passing `aux` at all. Turning the
flag on would therefore have added ~57k parameters that receive no gradient -- a guaranteed
null. These tests pin the wiring so that cannot silently regress.

WHAT THEY PREDICT (chosen here, not in the original spec, which never said):
  * aux_recon: reconstruct the entity matrix from `pooled`, MASKED -- absent entity slots
    are all-zero rows and would otherwise dominate an unmasked MSE and teach the net to
    predict zeros.
  * aux_reward (d->3): the discounted return at THREE horizons (1, 15, 150 decisions =
    0.067s / 1s / 10s at tick_skip 8). 150 is ~gamma's half-life. This targets the measured
    deficiency directly: the critic explains 84% of short-horizon GAE returns but only ~35%
    of true long-horizon returns.
"""
from pathlib import Path

import numpy as np
import pytest
import torch

from construct.learn.aux import aux_losses, horizon_returns
from construct.learn.model_v1 import EntityPolicyNet


def _table(n=104):
    g = torch.Generator().manual_seed(0)
    return torch.rand(n, 8, generator=g).numpy()


def _net(aux=True, d=32):
    return EntityPolicyNet(d_model=d, layers=1, heads=2, ff=64,
                           action_table=_table(), aux=aux)


# ---------------------------------------------------------------- horizon_returns


def test_horizon_return_at_h1_is_reward_plus_discounted_bootstrap():
    """h=1 is the one-step TD target r + gamma*V(s'), which is the definition every
    other horizon generalises. Getting this wrong scales every aux target."""
    T, N = 4, 2
    rew = np.zeros((T, N), np.float32); rew[:] = 1.0
    val = np.zeros((T + 1, N), np.float32); val[:] = 10.0
    term = np.zeros((T, N), bool); trunc = np.zeros((T, N), bool)
    out = horizon_returns(rew, val, term, trunc, gamma=0.9, horizons=(1,))
    assert out.shape == (T, N, 1)
    np.testing.assert_allclose(out[..., 0], 1.0 + 0.9 * 10.0, rtol=1e-6)


def test_termination_truncates_the_horizon_and_drops_the_bootstrap():
    """A terminated step has NO successor value: the return must stop at the reward.
    Bootstrapping past a termination leaks the next episode's value backwards."""
    rew = np.array([[1.0], [1.0], [1.0]], np.float32)
    val = np.array([[5.0], [5.0], [5.0], [5.0]], np.float32)
    term = np.array([[False], [True], [False]])
    trunc = np.zeros((3, 1), bool)
    out = horizon_returns(rew, val, term, trunc, gamma=0.5, horizons=(3,))
    # t=0: r0 + g*r1, then STOP (t=1 terminated) -> 1 + 0.5*1 = 1.5
    assert out[0, 0, 0] == pytest.approx(1.5)
    # t=1: terminated on its own step -> just r1
    assert out[1, 0, 0] == pytest.approx(1.0)


def test_longer_horizon_sees_more_reward():
    """Monotonicity with all-positive rewards -- catches an off-by-one that would make
    a 'longer' horizon actually shorter."""
    T = 8
    rew = np.ones((T, 1), np.float32)
    val = np.zeros((T + 1, 1), np.float32)
    z = np.zeros((T, 1), bool)
    r1 = horizon_returns(rew, val, z, z, gamma=1.0, horizons=(1,))[0, 0, 0]
    r4 = horizon_returns(rew, val, z, z, gamma=1.0, horizons=(4,))[0, 0, 0]
    assert r4 > r1 and r4 == pytest.approx(4.0) and r1 == pytest.approx(1.0)


def test_horizons_are_returned_in_the_order_requested():
    """The 3 columns feed a 3-output head; a permutation here silently trains the head
    on the wrong targets and nothing downstream would notice."""
    T = 6
    rew = np.ones((T, 1), np.float32)
    val = np.zeros((T + 1, 1), np.float32)
    z = np.zeros((T, 1), bool)
    out = horizon_returns(rew, val, z, z, gamma=1.0, horizons=(1, 3, 5))
    assert out.shape == (T, 1, 3)
    assert out[0, 0, 0] < out[0, 0, 1] < out[0, 0, 2]


# ---------------------------------------------------------------- aux_losses


def test_recon_loss_ignores_masked_entity_slots():
    """Absent slots are all-zero rows. An UNMASKED mse would be dominated by them and
    teach the trunk to predict zeros -- the loss must not move when a masked slot's
    target changes."""
    torch.manual_seed(0)
    B, E, F = 4, 17, 26
    pooled = torch.randn(B, 32)
    net = _net()
    ents = torch.randn(B, E, F)
    mask = torch.zeros(B, E, dtype=torch.bool)
    mask[:, 8:] = True                       # slots 8.. absent
    tgt = torch.zeros(B, 3)
    l1, _ = aux_losses(net, pooled, ents, mask, tgt, w_recon=1.0, w_reward=0.0)
    ents2 = ents.clone()
    ents2[:, 8:] = 999.0                     # change ONLY masked slots
    l2, _ = aux_losses(net, pooled, ents2, mask, tgt, w_recon=1.0, w_reward=0.0)
    assert torch.allclose(l1, l2), "masked slots must not contribute to the recon loss"


def test_recon_loss_does_move_when_a_visible_slot_changes():
    """The mirror of the above -- a mask that swallowed everything would also pass
    the previous test."""
    torch.manual_seed(0)
    pooled = torch.randn(4, 32)
    net = _net()
    ents = torch.randn(4, 17, 26)
    mask = torch.zeros(4, 17, dtype=torch.bool); mask[:, 8:] = True
    tgt = torch.zeros(4, 3)
    l1, _ = aux_losses(net, pooled, ents, mask, tgt, w_recon=1.0, w_reward=0.0)
    ents2 = ents.clone(); ents2[:, 0] += 5.0
    l2, _ = aux_losses(net, pooled, ents2, mask, tgt, w_recon=1.0, w_reward=0.0)
    assert not torch.allclose(l1, l2)


def test_zero_weights_give_exactly_zero_loss_and_keep_the_graph():
    """Both weights 0.0 must contribute nothing NUMERICALLY while still returning a
    tensor that can be added to the PPO loss without breaking autograd."""
    net = _net()
    pooled = torch.randn(3, 32, requires_grad=True)
    loss, info = aux_losses(net, pooled, torch.randn(3, 17, 26),
                            torch.zeros(3, 17, dtype=torch.bool), torch.zeros(3, 3),
                            w_recon=0.0, w_reward=0.0)
    assert float(loss) == 0.0
    assert loss.requires_grad


def test_aux_losses_reach_both_heads_by_gradient():
    """The whole point: the aux params must RECEIVE GRADIENT. This is the test that
    would have caught the dead-scaffolding state."""
    net = _net()
    pooled = torch.randn(5, 32)
    loss, _ = aux_losses(net, pooled, torch.randn(5, 17, 26),
                         torch.zeros(5, 17, dtype=torch.bool), torch.randn(5, 3),
                         w_recon=1.0, w_reward=1.0)
    loss.backward()
    assert net.aux_recon.weight.grad is not None
    assert net.aux_reward.weight.grad is not None
    assert net.aux_recon.weight.grad.abs().sum() > 0
    assert net.aux_reward.weight.grad.abs().sum() > 0


def test_aux_heads_absent_when_flag_off_and_calling_them_raises():
    net = _net(aux=False)
    assert not hasattr(net, "aux_recon")
    with pytest.raises(AssertionError):
        net.aux_outputs(torch.randn(2, 32))


# ---------------------------------------------------------------- model plumbing


def test_trunk_returns_the_same_pooled_the_value_head_sees():
    """aux heads consume `pooled`, which forward() does not return. trunk() must be the
    EXACT tensor forward() uses, or the aux losses shape a different representation than
    the one the value head reads."""
    net = _net(aux=True, d=32).eval()
    B = 3
    ents = torch.randn(B, 17, 26)
    mask = torch.zeros(B, 17, dtype=torch.bool)
    query = torch.randn(B, 64)
    prev = torch.zeros(B, 5, dtype=torch.long)
    with torch.no_grad():
        pooled = net.trunk(ents, mask, query, prev)
        _, value = net(ents, mask, query, prev)
        assert torch.allclose(net.value_head(pooled), value, atol=1e-6)


def test_enabling_aux_adds_only_aux_keys_so_the_engine_still_loads():
    """The Rust engine looks weights up BY NAME (policy_v1.rs) and ignores aux_*, but
    only if enabling aux changes NOTHING else in the state_dict. A renamed or reshaped
    shared tensor would break set_weights on the live box."""
    off = set(_net(aux=False).state_dict())
    on = set(_net(aux=True).state_dict())
    assert on - off == {"aux_reward.weight", "aux_reward.bias",
                        "aux_recon.weight", "aux_recon.bias"}
    assert off - on == set()


# ---------------------------------------------------------------- trainer wiring
#
# These cover the ways this feature could be "on" and still do nothing -- the failure
# mode it actually shipped in for weeks.


def test_compose_extra_loss_refuses_aux_weights_on_a_net_without_aux_heads():
    """The silent-null guard. Asking for an aux loss on an aux=False net must RAISE,
    not quietly train with dead heads."""
    from construct.learn.train import compose_extra_loss
    net = _net(aux=False)
    with pytest.raises(ValueError, match="aux=False"):
        compose_extra_loss(net, {}, kickstart=None, lambda_k=0.0, lambda_v=0.0,
                           prior_logits=None, lambda_p=0.0, w_recon=0.1, w_reward=0.0)


def test_compose_extra_loss_returns_none_when_nothing_is_active():
    """The no-hook path must stay byte-identical for runs that configure none of
    kickstart / prior / aux."""
    from construct.learn.train import compose_extra_loss
    assert compose_extra_loss(_net(aux=False), {}, kickstart=None, lambda_k=0.0,
                              lambda_v=0.0, prior_logits=None, lambda_p=0.0) is None


def test_aux_alone_activates_the_hook():
    from construct.learn.train import compose_extra_loss
    fn = compose_extra_loss(_net(aux=True), {}, kickstart=None, lambda_k=0.0, lambda_v=0.0,
                            prior_logits=None, lambda_p=0.0, w_recon=0.1, w_reward=0.0)
    assert fn is not None


def test_resume_train_exposes_aux_override_because_the_checkpoint_overwrites_net_cfg():
    """resume_train.py does `cfg.net = state['config']['net']`, so `aux = true` in a toml
    is DISCARDED on a resume. A CLI override is the only way to enable it, and its absence
    would make the forked run a guaranteed null -- same trap as --entropy-coef."""
    src = (Path(__file__).resolve().parents[2] / "scripts" / "resume_train.py").read_text()
    assert '"--aux"' in src
    assert '"--aux-recon-coef"' in src and '"--aux-reward-coef"' in src
    # the override must come AFTER the line that clobbers cfg.net, or it is a no-op
    assert src.index('cfg.net = state["config"]["net"]') < src.index("if args.aux:")


def test_zero_init_head_gives_zero_gradient_to_the_trunk_on_step_one():
    """Why the aux heads must be zero-initialised on a resume.

    Attaching a RANDOM head to a 2.8B-step trunk is a large gradient aimed at a
    well-trained representation: measured on the box, the first update drove the raw
    recon loss to 1.7e24, clip_frac to 0.356 (normal ~0.05) and ep_rew 2297 -> 2041.

    With zero weights the head predicts 0, so dL/d(pooled) = 0 at step one -- the trunk
    sees NOTHING until the head has learned something worth propagating, and the ramp is
    automatic. The head itself still gets gradient, or it could never learn.
    """
    net = _net(aux=True)
    with torch.no_grad():
        for p_ in (net.aux_recon.weight, net.aux_recon.bias,
                   net.aux_reward.weight, net.aux_reward.bias):
            p_.zero_()
    pooled = torch.randn(6, 32, requires_grad=True)
    loss, _ = aux_losses(net, pooled, torch.randn(6, 17, 26),
                         torch.zeros(6, 17, dtype=torch.bool), torch.randn(6, 3),
                         w_recon=0.1, w_reward=0.003)
    loss.backward()
    assert torch.allclose(pooled.grad, torch.zeros_like(pooled.grad), atol=1e-12), \
        "a zero-init aux head must send NO gradient into the shared trunk"
    assert net.aux_recon.weight.grad.abs().sum() > 0, "the head itself must still learn"
    assert net.aux_reward.weight.grad.abs().sum() > 0


def test_per_horizon_explained_variance_separates_the_easy_horizon_from_the_hard_one():
    """The aggregate aux_reward MSE is a TRAP: it averages horizons whose targets have
    very different variance, so a large drop is consistent with only the trivial h=1
    improving while h=150 -- the one ev_mc says is the real deficiency -- does not move.

    Constructed so horizon 0 is predicted perfectly and horizon 2 not at all; the
    per-horizon ev must show that, and the aggregate MSE must NOT.
    """
    net = _net(aux=True)
    B = 256
    torch.manual_seed(0)
    target = torch.randn(B, 3) * torch.tensor([1.0, 1.0, 10.0])
    pooled = torch.randn(B, 32)
    with torch.no_grad():          # make the head reproduce column 0 and zero elsewhere
        net.aux_reward.weight.zero_(); net.aux_reward.bias.zero_()
    _, info = aux_losses(net, pooled, torch.randn(B, 17, 26),
                         torch.zeros(B, 17, dtype=torch.bool), target,
                         w_recon=0.0, w_reward=1.0)
    # a zero head predicts the mean-ish (0), so ev ~ 0 on EVERY horizon...
    assert all(abs(info[f"aux_ev{i}"]) < 0.1 for i in range(3))
    # ...while the aggregate MSE is dominated by the high-variance horizon alone,
    # which is exactly why it cannot be read as "the critic improved".
    assert info["aux_reward"] > 30


def test_degenerate_target_column_reports_nan_not_a_perfect_fit():
    """A constant target has zero variance; 1 - MSE/0 must not render as a great score."""
    import math
    net = _net(aux=True)
    target = torch.zeros(64, 3)          # every column constant
    _, info = aux_losses(net, torch.randn(64, 32), torch.randn(64, 17, 26),
                         torch.zeros(64, 17, dtype=torch.bool), target,
                         w_recon=0.0, w_reward=1.0)
    assert all(math.isnan(info[f"aux_ev{i}"]) for i in range(3))
