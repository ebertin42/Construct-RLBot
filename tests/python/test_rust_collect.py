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


def _weights(hidden=(64, 64), seed=3):
    torch.manual_seed(seed)
    return state_dict_np(PolicyValueNet(94, 90, hidden))


def test_collect_shapes_dtypes():
    eng = mk(n=4)
    eng.set_weights(_weights())
    out = eng.collect(16)
    N = eng.num_agents
    assert out["obs"].shape == (16, N, 94) and out["obs"].dtype == np.float32
    assert out["actions"].shape == (16, N) and out["actions"].dtype == np.int64
    for k in ("logprobs", "values", "rewards", "final_values"):
        assert out[k].shape == (16, N) and out[k].dtype == np.float32, k
    for k in ("terminated", "truncated"):
        assert out[k].shape == (16, N) and out[k].dtype == np.bool_, k
    assert out["last_values"].shape == (N,)
    assert (out["actions"] >= 0).all() and (out["actions"] < 90).all()
    assert np.isfinite(out["logprobs"]).all() and (out["logprobs"] <= 0).all()


def test_collect_requires_weights():
    eng = mk()
    with pytest.raises(Exception):
        eng.collect(4)


def test_collect_deterministic_across_thread_counts():
    w = _weights()
    a, b = mk(n=4, seed=9, threads=1), mk(n=4, seed=9, threads=4)
    a.set_weights(w); b.set_weights(w)
    oa, ob = a.collect(32), b.collect(32)
    for k in oa:
        np.testing.assert_array_equal(oa[k], ob[k], err_msg=k)


def test_collect_logprob_matches_torch_recompute():
    torch.manual_seed(5)
    net = PolicyValueNet(94, 90, (64, 64)).eval()
    eng = mk(n=2, seed=4)
    eng.set_weights(state_dict_np(net))
    out = eng.collect(8)
    obs = torch.from_numpy(out["obs"].reshape(-1, 94))
    acts = torch.from_numpy(out["actions"].reshape(-1))
    with torch.no_grad():
        lp, _, vals = net.evaluate(obs, acts)
    assert np.abs(lp.numpy() - out["logprobs"].reshape(-1)).max() < 1e-4
    assert np.abs(vals.numpy() - out["values"].reshape(-1)).max() < 1e-4
