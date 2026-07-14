import numpy as np
import torch
from construct.learn.model import PolicyValueNet
from construct.learn.ppo import ppo_update


def test_ppo_solves_two_armed_bandit():
    torch.manual_seed(0)
    net = PolicyValueNet(obs_size=4, action_count=2, hidden=(32,))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    obs = torch.ones(512, 4)
    for _ in range(40):
        with torch.no_grad():
            actions, logprobs, values = net.act(obs)
        rewards = (actions == 0).float()          # arm 0 pays 1
        adv = rewards - rewards.mean()
        batch = {
            "obs": obs, "actions": actions, "logprobs": logprobs,
            "advantages": adv, "returns": rewards, "values": values,
        }
        ppo_update(net, opt, batch, epochs=2, minibatch_size=256)
    with torch.no_grad():
        logits, _ = net(obs[:1])
        p0 = torch.softmax(logits, -1)[0, 0].item()
    assert p0 > 0.9, f"P(arm0)={p0}"


def test_ppo_survives_extreme_logprob_gap():
    """Regression: reward-swap NaN cascade. Stale old_logprobs far from the
    net's current logprobs must not inf/NaN the update (logratio clamp), and
    weights must stay finite afterward."""
    torch.manual_seed(0)
    net = PolicyValueNet(obs_size=4, action_count=2, hidden=(32,))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    obs = torch.ones(256, 4)
    with torch.no_grad():
        actions, _, values = net.act(obs)
    batch = {
        "obs": obs, "actions": actions,
        "logprobs": torch.full((256,), -200.0),   # absurdly stale -> exp(+199) = inf if unclamped
        "advantages": torch.randn(256),
        "returns": torch.randn(256), "values": values,
    }
    stats = ppo_update(net, opt, batch, epochs=1, minibatch_size=128)
    assert stats["skipped"] == 0, "clamped ratio should make updates finite, not skipped"
    for p in net.parameters():
        assert torch.isfinite(p).all(), "weights must stay finite"


def test_ppo_skips_nonfinite_loss_and_keeps_weights():
    """If a loss still manages to go nonfinite (e.g. nan advantages), the
    minibatch is skipped and weights are untouched."""
    torch.manual_seed(0)
    net = PolicyValueNet(obs_size=4, action_count=2, hidden=(32,))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    before = [p.clone() for p in net.parameters()]
    obs = torch.ones(64, 4)
    with torch.no_grad():
        actions, logprobs, values = net.act(obs)
    batch = {
        "obs": obs, "actions": actions, "logprobs": logprobs,
        "advantages": torch.full((64,), float("nan")),
        "returns": torch.zeros(64), "values": values,
    }
    stats = ppo_update(net, opt, batch, epochs=1, minibatch_size=64)
    assert stats["skipped"] == 1
    for a, b in zip(before, net.parameters()):
        assert torch.equal(a, b), "weights changed on a skipped minibatch"
