import numpy as np
import pytest
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet


def weights(seed=0):
    torch.manual_seed(seed)
    net = PolicyValueNet(94, 90, (64, 64))
    return {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def test_mixed_team_sizes_agent_count_and_shapes():
    eng = Engine(num_arenas=4, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=0,
                 team_size_weights=[0.5, 0.25, 0.25])
    # sizes [1,1,2,3] -> agents 2+2+4+6 = 14
    assert eng.num_agents == 14
    eng.set_weights(weights())
    out = eng.collect(8)
    assert out["obs"].shape == (8, 14, 94)
    assert np.isfinite(out["obs"]).all() and np.isfinite(out["logprobs"]).all()


def test_mixed_sizes_deterministic_fixed_config():
    w = weights(3)
    mk = lambda: Engine(num_arenas=6, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", seed=11,
                        num_threads=2, team_size_weights=[1.0, 1.0, 1.0])
    a, b = mk(), mk()
    a.set_weights(w); b.set_weights(w)
    oa, ob = a.collect(16), b.collect(16)
    for k in oa:
        np.testing.assert_array_equal(oa[k], ob[k], err_msg=k)


def test_bad_weights_rejected():
    with pytest.raises(Exception):
        Engine(num_arenas=4, schema_path="schema/v0.toml",
               reward_config_path="configs/reward_v0.toml",
               team_size_weights=[0.0, 0.0, 0.0])
    with pytest.raises(Exception):
        Engine(num_arenas=4, schema_path="schema/v0.toml",
               reward_config_path="configs/reward_v0.toml",
               team_size_weights=[1.0, 2.0])


def test_default_none_matches_legacy():
    mk_old = lambda: Engine(num_arenas=2, blue=1, orange=1, schema_path="schema/v0.toml",
                            reward_config_path="configs/reward_v0.toml", seed=4)
    a, b = mk_old(), mk_old()
    np.testing.assert_array_equal(a.reset(), b.reset())


from construct.learn.config import TrainConfig
from construct.learn.train import Trainer


def test_trainer_runs_mixed_sizes_with_curriculum(tmp_path):
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=4, team_size_weights=[0.5, 0.25, 0.25])
    cfg.curriculum_config_path = "configs/curriculum_v1.toml"
    cfg.ppo.update(rollout_steps=16, minibatch_size=128)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=1)
    t = Trainer(cfg)
    assert t.engine.num_agents == 14
    t.run(max_iterations=1)
    assert t.total_steps == 16 * 14
    import torch
    ck = torch.load(f"{tmp_path}/ck_{t.total_steps:012d}.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["env"]["team_size_weights"] == [0.5, 0.25, 0.25]
    assert ck["curriculum_config_path"] == "configs/curriculum_v1.toml"
