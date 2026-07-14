import numpy as np
import pytest
from construct._engine import Engine

def mk(n=4, seed=0):
    return Engine(num_arenas=n, blue=1, orange=1, schema_path="schema/v0.toml",
                  reward_config_path="configs/reward_v0.toml", seed=seed)

def test_shapes_and_dtypes():
    eng = mk()
    assert eng.num_agents == 8 and eng.obs_size == 94 and eng.action_count == 90
    obs = eng.reset()
    assert obs.shape == (8, 94) and obs.dtype == np.float32
    acts = np.zeros(8, dtype=np.int64)
    obs, rew, term, trunc, final_obs = eng.step(acts)
    assert obs.shape == (8, 94) and rew.shape == (8,)
    assert term.dtype == np.bool_ and trunc.dtype == np.bool_
    assert final_obs.shape == (8, 94)

def test_deterministic_with_seed():
    a, b = mk(seed=7), mk(seed=7)
    a.reset(); b.reset()
    rng = np.random.default_rng(0)
    for _ in range(20):
        acts = rng.integers(0, 90, size=8).astype(np.int64)
        oa, ra, *_ = a.step(acts)
        ob, rb, *_ = b.step(acts)
        np.testing.assert_array_equal(oa, ob)
        np.testing.assert_array_equal(ra, rb)

def test_rejects_bad_actions():
    eng = mk()
    eng.reset()
    with pytest.raises(Exception):
        eng.step(np.full(8, 90, dtype=np.int64))  # out of range
    with pytest.raises(Exception):
        eng.step(np.zeros(3, dtype=np.int64))     # wrong length

def test_episodes_eventually_end():
    eng = mk(n=2)
    eng.reset()
    acts = np.full(4, 4, dtype=np.int64)  # idle-ish action -> no-touch truncation
    done_seen = False
    for _ in range(500):
        _, _, term, trunc, _ = eng.step(acts)
        if term.any() or trunc.any():
            done_seen = True
            break
    assert done_seen
