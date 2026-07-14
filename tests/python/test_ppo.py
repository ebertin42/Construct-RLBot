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
