import numpy as np
import pytest
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet


def mk(n=2, seed=0, threads=0):
    return Engine(num_arenas=n, blue=1, orange=1, schema_path="schema/v0.toml",
                  reward_config_path="configs/reward_v0.toml", seed=seed, num_threads=threads)


def state_dict_np(net):
    return {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}


def test_set_weights_accepts_real_state_dict():
    torch.manual_seed(0)
    net = PolicyValueNet(94, 90, (512, 512))
    eng = mk()
    eng.set_weights(state_dict_np(net))  # must not raise


def test_full_path_forward_parity():
    torch.manual_seed(1)
    net = PolicyValueNet(94, 90, (64, 64)).eval()
    eng = mk()
    eng.set_weights(state_dict_np(net))
    obs = np.random.default_rng(2).standard_normal((6, 94)).astype(np.float32)
    logits_r, values_r = eng.debug_policy_forward(obs)
    with torch.no_grad():
        logits_t, values_t = net(torch.from_numpy(obs))
    assert np.abs(logits_r - logits_t.numpy()).max() < 1e-4
    assert np.abs(values_r - values_t.numpy()).max() < 1e-4


def test_set_weights_rejects_garbage():
    eng = mk()
    with pytest.raises(Exception):
        eng.set_weights({"trunk.0.weight": np.zeros((3, 3), dtype=np.float32)})
