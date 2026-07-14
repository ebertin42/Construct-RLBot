import numpy as np
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet


def test_engine_accepts_v1_config():
    eng = Engine(num_arenas=2, blue=1, orange=1, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v1.toml", seed=5)
    torch.manual_seed(0)
    sd = {k: v.detach().numpy().astype(np.float32)
          for k, v in PolicyValueNet(94, 90, (64, 64)).state_dict().items()}
    eng.set_weights(sd)
    out = eng.collect(32)
    assert np.isfinite(out["rewards"]).all()
    # goal weight is 20: nothing on a random-ish rollout should exceed the
    # theoretical per-step bound |goal| + sum of shaping weights
    assert np.abs(out["rewards"]).max() <= 20.0 + 2.0 + 0.5 + 0.3 + 0.02 + 1e-4


def test_v0_config_still_loads():
    eng = Engine(num_arenas=1, blue=1, orange=1, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=5)
    assert eng.num_agents == 2
