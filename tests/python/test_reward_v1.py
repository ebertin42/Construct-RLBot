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
    # Post-blend per-step bound (1v1, reward_v1.toml): raw max ≈ 22.82
    # (goal 20 + shaping ≈ 2.82); blend adds -opp_spirit*r_opp where the
    # conceder can be ≈ -16.5 (concede -16 + worst shaping) →
    # 22.82 + 0.3*16.5 ≈ 27.8. Assert with headroom:
    assert np.abs(out["rewards"]).max() <= 28.5


def test_v0_config_still_loads():
    eng = Engine(num_arenas=1, blue=1, orange=1, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=5)
    assert eng.num_agents == 2
