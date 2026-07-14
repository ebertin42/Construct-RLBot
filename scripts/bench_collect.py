# scripts/bench_collect.py
"""Collection throughput: rust in-engine path, by arena count."""
import time

import numpy as np
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet

torch.manual_seed(0)
net = PolicyValueNet(94, 90, (512, 512))
sd = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}

for arenas in (64, 96, 192, 256):
    eng = Engine(num_arenas=arenas, blue=1, orange=1, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=0)
    eng.set_weights(sd)
    eng.collect(16)  # warmup
    t0 = time.perf_counter()
    T = 128
    eng.collect(T)
    dt = time.perf_counter() - t0
    print(f"arenas={arenas:4d} agents={eng.num_agents:4d}: {T * arenas / dt:>10,.0f} env-steps/s")
