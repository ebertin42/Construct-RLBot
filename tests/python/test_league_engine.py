import numpy as np
import pytest
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet


def sd(seed):
    torch.manual_seed(seed)
    return {k: v.detach().numpy().astype(np.float32)
            for k, v in PolicyValueNet(94, 90, (64, 64)).state_dict().items()}


def mk(n=4, seed=0, threads=2):
    return Engine(num_arenas=n, blue=1, orange=1, schema_path="schema/v0.toml",
                  reward_config_path="configs/reward_v0.toml", seed=seed, num_threads=threads)


def test_no_assignment_is_byte_identical_to_legacy():
    w = sd(1)
    a, b = mk(seed=5), mk(seed=5)
    a.set_weights(w); b.set_weights(w)
    oa = a.collect(16)
    ob = b.collect(16, arena_opponents=None)
    for k in oa:
        np.testing.assert_array_equal(np.asarray(oa[k]), np.asarray(ob[k]), err_msg=k)


def test_opponent_arena_shrinks_buffers_and_orders_learners():
    eng = mk(n=4, seed=3)
    eng.set_weights(sd(1))
    eng.set_opponents([sd(2)])
    out = eng.collect(8, arena_opponents=[-1, 0, -1, 0])
    # arenas 0,2 self-play (2 learners each), arenas 1,3 opponent (1 learner each) = 6
    assert out["learner_agents"] == 6
    assert out["obs"].shape == (8, 6, 94)
    assert out["actions"].shape == (8, 6)
    assert np.isfinite(out["logprobs"]).all()


def test_opponent_actually_plays_differently():
    eng1, eng2 = mk(n=2, seed=9), mk(n=2, seed=9)
    for e in (eng1, eng2):
        e.set_weights(sd(1))
    eng1.set_opponents([sd(1)])   # opponent = same weights as learner
    eng2.set_opponents([sd(42)])  # opponent = different net
    o1 = eng1.collect(32, arena_opponents=[0, 0])
    o2 = eng2.collect(32, arena_opponents=[0, 0])
    # same learner weights + same seeds; only the opponent differs. If the opponent
    # net is actually driving orange, trajectories must diverge.
    assert not np.array_equal(o1["obs"], o2["obs"])


def test_bad_assignment_rejected():
    eng = mk()
    eng.set_weights(sd(1))
    eng.set_opponents([sd(2)])
    with pytest.raises(Exception):
        eng.collect(4, arena_opponents=[0, 0])            # wrong length (4 arenas)
    with pytest.raises(Exception):
        eng.collect(4, arena_opponents=[-1, -1, -1, 3])   # unset slot
    with pytest.raises(Exception):
        eng.set_opponents([sd(i) for i in range(9)])      # > 8 slots


def test_determinism_with_opponents():
    w, opp = sd(1), sd(7)
    mk2 = lambda: mk(n=4, seed=11, threads=2)
    a, b = mk2(), mk2()
    for e in (a, b):
        e.set_weights(w); e.set_opponents([opp])
    oa = a.collect(16, arena_opponents=[-1, 0, 0, -1])
    ob = b.collect(16, arena_opponents=[-1, 0, 0, -1])
    for k in oa:
        np.testing.assert_array_equal(np.asarray(oa[k]), np.asarray(ob[k]), err_msg=k)
