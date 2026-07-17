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
    # Root-cause note (task #46): arenas with 2+ cars on a team are NOT
    # bit-reproducible across independently-constructed engines, even with an
    # identical seed. RocketSim's Arena::ResetToRandomKickoff (rocketsim_rs,
    # a pinned vendored dependency) groups same-team cars by iterating an
    # internal `std::unordered_set<Car*>`, whose order depends on the cars'
    # heap addresses -- not on car id, add order, or any seed we control.
    # Two fresh engines can therefore hand a same-team car pair/triple their
    # kickoff spawn slots in a different order purely as a function of
    # process allocation history (confirmed via engine.debug_state_and_obs:
    # identical seeds, differing kickoff car->slot assignment for 2v2/3v3
    # arenas only, never 1v1). It doesn't permute cleanly back out of obs
    # either -- teammate/opponent features are keyed by identity, not by
    # kickoff slot -- so re-sorting agent rows can't undo it. team_size==1
    # arenas have exactly one car per team, so there is no grouping
    # ambiguity there: assert full byte-for-byte reproducibility for those,
    # matching the (seed, num_arenas, num_threads) contract documented in
    # test_rust_collect.py::test_collect_deterministic_fixed_config. For
    # team_size>=2 arenas, only assert well-formedness (finite values);
    # exact cross-instance reproducibility isn't a guarantee this engine
    # (via its vendored physics dependency) actually provides for those.
    w = weights(3)
    mk = lambda: Engine(num_arenas=6, schema_path="schema/v0.toml",
                        reward_config_path="configs/reward_v0.toml", seed=11,
                        num_threads=2, team_size_weights=[1.0, 1.0, 1.0])
    a, b = mk(), mk()
    a.set_weights(w); b.set_weights(w)
    oa, ob = a.collect(16), b.collect(16)

    # allocate_team_sizes(6, [1,1,1]) yields two arenas each of team-size
    # 1/2/3 (see engine/src/engine.rs::allocate_team_sizes), ordered
    # 1s-block, 2s-block, 3s-block -> per-arena team sizes [1, 1, 2, 2, 3, 3].
    sizes = [1, 1, 2, 2, 3, 3]
    n_agents = sum(2 * s for s in sizes)
    assert oa["obs"].shape[1] == n_agents, "team-size allocation changed; update `sizes` above"
    safe = np.zeros(n_agents, dtype=bool)
    off = 0
    for s in sizes:
        n = 2 * s
        if s == 1:
            safe[off:off + n] = True
        off += n
    safe_idx = np.flatnonzero(safe)

    for k in oa:
        arr_a, arr_b = np.asarray(oa[k]), np.asarray(ob[k])
        if arr_a.ndim == 0:
            # scalar metadata (e.g. learner_agents) -- fully deterministic
            np.testing.assert_array_equal(arr_a, arr_b, err_msg=k)
            continue
        agent_axis = 0 if arr_a.ndim == 1 else 1
        sl_a = np.take(arr_a, safe_idx, axis=agent_axis)
        sl_b = np.take(arr_b, safe_idx, axis=agent_axis)
        np.testing.assert_array_equal(sl_a, sl_b, err_msg=f"{k} (team_size==1 arenas)")
        if np.issubdtype(arr_a.dtype, np.floating):
            assert np.isfinite(arr_a).all() and np.isfinite(arr_b).all(), k


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
