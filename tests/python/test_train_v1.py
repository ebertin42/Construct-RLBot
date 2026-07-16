"""T8: v1 (entity-transformer) trainer integration — collect->GAE->PPO->
checkpoint end-to-end on the real engine, plus the kickstart e2e and the
v0-checkpoint/v1-config guard. This is the plan's binding test for T8."""
import math
import re

import numpy as np
import pytest
import torch

from construct._engine import action_table, action_table_v1
from construct.learn.config import TrainConfig
from construct.learn.model import PolicyValueNet
from construct.learn.model_v1 import ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT, EntityPolicyNet
from construct.learn.ppo import ppo_update
from construct.learn.train import Trainer


def v1_cfg(tmp_path, arenas=4, rollout=128):
    """train_v1.toml shrunk to test size: 1v1, tiny net, cpu."""
    cfg = TrainConfig.load("configs/train_v1.toml")
    cfg.env.update(num_arenas=arenas, blue=1, orange=1)
    cfg.env.pop("team_size_weights", None)  # pure 1v1 for exact agent counts
    cfg.curriculum_config_path = ""
    cfg.net = {"d_model": 32, "layers": 1, "heads": 2, "ff": 64}
    cfg.ppo.update(rollout_steps=rollout, minibatch_size=256)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=100)
    cfg.league = {}
    cfg.kickstart = {}
    return cfg


def test_action_table_v1_is_92_rows_appending_v0():
    t0, t1 = action_table(), action_table_v1()
    assert t1.shape == (92, 8) and t1.dtype == np.float32
    np.testing.assert_array_equal(t1[:90], t0)


def test_v1_end_to_end_two_iters_and_checkpoint_resume(tmp_path):
    cfg = v1_cfg(tmp_path)
    t = Trainer(cfg)
    assert t.engine.obs_mode == "v1"
    assert isinstance(t.net, EntityPolicyNet)
    t.run(max_iterations=2)
    assert t.total_steps == 2 * 128 * 8  # T * num_agents (4 arenas, 1v1)

    # losses finite on a fresh batch through the real obs-dict PPO seam
    batch = t.collect(16)
    stats = ppo_update(t.net, t.opt, batch, clip=0.2, entropy_coef=0.01,
                       value_coef=1.0, epochs=1, minibatch_size=64)
    assert stats["updates"] > 0 and stats["skipped"] == 0
    for k in ("policy_loss", "value_loss", "entropy"):
        assert math.isfinite(stats[k]), (k, stats)

    # checkpoint roundtrip: schema_version=1 + net dims recorded, resume works
    p = f"{tmp_path}/ck_v1.pt"
    t.save_checkpoint(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["schema_version"] == 1
    assert ck["config"]["net"] == {"d_model": 32, "layers": 1, "heads": 2, "ff": 64}

    t2 = Trainer.load_checkpoint(p, cfg_path="configs/train_v1.toml")
    assert isinstance(t2.net, EntityPolicyNet)
    assert t2.total_steps == t.total_steps
    before = [x.clone() for x in t2.net.parameters()]
    t2.run(max_iterations=1)
    assert t2.total_steps == t.total_steps + 128 * 8
    assert any(not torch.equal(a, b) for a, b in zip(before, t2.net.parameters()))


def test_v1_collect_returns_obs_dict_with_engine_consistent_logprobs(tmp_path):
    t = Trainer(v1_cfg(tmp_path))
    T = 8
    batch = t.collect(T)
    n = t.engine.num_agents
    obs = batch["obs"]
    assert isinstance(obs, dict)
    assert obs["ents"].shape == (T * n, MAX_ENT, ENT_FEAT) and obs["ents"].dtype == torch.float32
    assert obs["mask"].shape == (T * n, MAX_ENT) and obs["mask"].dtype == torch.bool
    assert obs["query"].shape == (T * n, Q_FEAT) and obs["query"].dtype == torch.float32
    assert obs["prev"].shape == (T * n, PREV_ACTIONS) and obs["prev"].dtype == torch.int64
    assert batch["actions"].shape == (T * n,)
    assert (batch["actions"] >= 0).all() and (batch["actions"] < 92).all()
    # candle (in-engine) logprobs must be consistent with the torch net
    with torch.no_grad():
        lp, _, vals = t.net.evaluate(**obs, actions=batch["actions"])
    assert (lp - batch["logprobs"]).abs().max().item() < 1e-3
    assert (vals - batch["values"]).abs().max().item() < 1e-3


def _synthetic_v0_checkpoint(path, seed=0):
    torch.manual_seed(seed)
    ref = PolicyValueNet(94, 90, (32,))
    torch.save({
        "model": ref.state_dict(),
        "optimizer": None,
        "total_steps": 0,
        "schema_version": 0,
        "config": {"net": {"hidden": [32]},
                   "ppo": {"rollout_steps": 16},
                   "env": {"num_arenas": 4, "blue": 1, "orange": 1, "seed": 0}},
    }, path)
    return str(path)


def test_kickstart_e2e_one_iter_kick_kl_finite(tmp_path, capsys):
    teacher_ck = _synthetic_v0_checkpoint(tmp_path / "teacher.pt")
    cfg = v1_cfg(tmp_path, arenas=2, rollout=32)
    cfg.kickstart = {"teacher": teacher_ck, "steps": 10_000_000}
    t = Trainer(cfg)
    assert t.kickstart is not None
    batch = t.collect(8)
    assert "obs_v0" in batch and batch["obs_v0"].shape == (8 * 4, 94)
    t.run(max_iterations=1)
    out = capsys.readouterr().out
    m = re.search(r"kick_kl (\S+)", out)
    assert m, f"kick_kl missing from training log: {out!r}"
    assert math.isfinite(float(m.group(1))), out


def test_v0_checkpoint_with_v1_config_errors_clearly(tmp_path):
    ck = _synthetic_v0_checkpoint(tmp_path / "v0.pt")
    with pytest.raises(ValueError, match="schema"):
        Trainer.load_checkpoint(ck, cfg_path="configs/train_v1.toml")
