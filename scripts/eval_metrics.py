"""Headless behavior eval: touches/min and mean dist-to-ball over N eval steps."""
import sys

import numpy as np
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
eng = Engine(num_arenas=16, blue=1, orange=1, schema_path="schema/v0.toml",
             reward_config_path="configs/reward_v0.toml", seed=1234)
net = PolicyValueNet(eng.obs_size, eng.action_count, tuple(ck["config"]["net"]["hidden"]))
net.load_state_dict(ck["model"])
net.eval()

obs = torch.as_tensor(eng.reset())
STEPS = 4500  # 5 min of game time
touches = 0
goals = 0
dist_sum = 0.0
torch.manual_seed(0)  # sampled actions, reproducible: argmax play deadlocks in
# mirror-symmetric standoffs (both cars park at the ball) and zeroes the metric
for _ in range(STEPS):
    with torch.no_grad():
        logits = net(obs)[0]
        acts = torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int64)
    nobs, rew, term, trunc, _ = eng.step(acts)
    touches += int((rew >= 0.5).sum())  # touch weight fires at >= 0.5
    goals += int((rew >= 9.5).sum())    # goal pays +10 to the scorer, unambiguous
    # obs[28:31] is (ball - car) * pos_norm
    dist_sum += float(np.linalg.norm(nobs[:, 28:31], axis=1).mean() / (1 / 2300))
    obs = torch.as_tensor(nobs)

minutes = STEPS / 15 / 60 * eng.num_agents
match_minutes = STEPS / 15 / 60 * 16  # 16 arenas = 16 concurrent matches
print(f"touches/min/agent: {touches / minutes:.2f}")
print(f"mean dist to ball: {dist_sum / STEPS:.0f} uu")
print(f"goals/min/match: {goals / match_minutes:.2f}")
