"""Watch a checkpoint play in RLViser.
1) Download rlviser binary (github.com/VirxEC/rlviser/releases) into repo root
2) ./rlviser   (Linux/WSLg)  — or run rlviser on Windows and adjust target IP
3) python scripts/watch.py checkpoints/ck_XXXX.pt [--argmax]

Actions are sampled by default (matches training behavior; argmax play
deadlocks in mirror-symmetric standoffs). Pass --argmax for deterministic.
"""
import sys

import numpy as np
import torch

from construct._engine import RenderSession
from construct.learn.model import PolicyValueNet

deterministic = "--argmax" in sys.argv
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
sess = RenderSession(blue=1, orange=1, schema_path="schema/v0.toml",
                     reward_config_path="configs/reward_v0.toml", seed=42)
net = PolicyValueNet(sess.obs_size, sess.action_count, tuple(ck["config"]["net"]["hidden"]))
net.load_state_dict(ck["model"])
net.eval()

obs = torch.as_tensor(sess.reset())
try:
    while True:
        with torch.no_grad():
            logits = net(obs)[0]
            if deterministic:
                actions = logits.argmax(-1)
            else:
                actions = torch.distributions.Categorical(logits=logits).sample()
        nobs, *_ = sess.step(actions.numpy().astype(np.int64))
        obs = torch.as_tensor(nobs)
except KeyboardInterrupt:
    sess.close()
