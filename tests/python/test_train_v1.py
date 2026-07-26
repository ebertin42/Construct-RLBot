"""T8: v1 (entity-transformer) trainer integration — collect->GAE->PPO->
checkpoint end-to-end on the real engine, plus the kickstart e2e and the
v0-checkpoint/v1-config guard. This is the plan's binding test for T8."""
import math
import re

import numpy as np
import pytest
import torch

from construct._engine import action_table, action_table_v1, action_table_v1_air
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


def test_action_table_v1_air_is_104_rows_appending_v1():
    """v1-air must be a strict APPEND to v1.1, so every v1.1 index keeps its
    meaning and the two tables can coexist in one engine binary."""
    t1, ta = action_table_v1(), action_table_v1_air()
    assert ta.shape == (104, 8) and ta.dtype == np.float32
    np.testing.assert_array_equal(ta[:92], t1)


def test_v1_air_makes_the_aerial_primitive_reachable():
    """The measurement the table change exists to move.

    v1.1 derives handbrake from rotation (`jump and (pitch or yaw or roll)`)
    and skips `jump and yaw`, so "jump without air roll" is ABSENT at every
    index, not merely rare: 2/92 clean-jump rows, and a fresh policy holds one
    for the 3 consecutive decisions a full-height takeoff needs with
    probability 1.03e-5 (~0.96 times per 93,440-row iteration). No reward can
    shape a sequence that is never sampled. v1-air takes that to 2.44e-3, a
    237x change.
    """
    def clean(t):   # jump, no yaw, no roll, no handbrake
        return int(((t[:, 5] != 0) & (t[:, 3] == 0) & (t[:, 4] == 0) & (t[:, 7] == 0)).sum())

    t1, ta = action_table_v1(), action_table_v1_air()
    assert clean(t1) == 2 and clean(ta) == 14
    p1, pa = clean(t1) / len(t1), clean(ta) / len(ta)
    assert abs(p1 - 0.0217) < 1e-3 and abs(pa - 0.1346) < 1e-3
    # P(hold 3) -- the load-bearing number.
    assert abs(p1**3 - 1.027e-5) < 1e-7
    assert abs(pa**3 - 2.439e-3) < 1e-5
    assert pa**3 / p1**3 > 200
    # P(clean jump | jump): 10% -> 43.8%. The other 90% of v1.1's jump rows
    # force handbrake, which airborne IS air roll.
    assert clean(t1) / int((t1[:, 5] != 0).sum()) == 0.10
    assert abs(clean(ta) / int((ta[:, 5] != 0).sum()) - 0.4375) < 1e-9
    # Append-only never mutates the forced-air-roll rows; only their share falls.
    rollers = lambda t: int(((t[:, 5] != 0) & (t[:, 7] != 0)).sum())
    assert rollers(t1) == rollers(ta) == 18


def test_v1_air_end_to_end_two_iters_and_checkpoint_roundtrip(tmp_path):
    """The whole 104-row path, live: schema -> engine action_count -> net
    output dim -> sampled indices -> PPO -> checkpoint.

    The action table sets the policy's OUTPUT DIMENSION, so it can only be
    chosen at the start of a lineage -- there is no migration. This is the test
    that says v9 can actually launch on it.
    """
    cfg = v1_cfg(tmp_path)
    cfg.schema_path = "schema/v1_air.toml"
    t = Trainer(cfg)
    assert t.engine.obs_mode == "v1"
    assert t.engine.action_count == 104
    # The net's non-trainable buffer IS the 104-row table, appended to v1.1.
    tbl = t.net.action_table
    assert tuple(tbl.shape) == (104, 8)
    np.testing.assert_array_equal(tbl.cpu().numpy()[:92], action_table_v1())

    t.run(max_iterations=2)
    assert t.total_steps == 2 * 128 * 8

    batch = t.collect(16)
    # Sampled indices must span the WIDER table, not silently stay under 92.
    acts = batch["actions"]
    assert int(acts.max()) < 104
    assert int(acts.max()) >= 92, (
        "no action index landed in the appended block -- the wider table is "
        "not actually reaching the sampler"
    )
    stats = ppo_update(t.net, t.opt, batch, clip=0.2, entropy_coef=0.01,
                       value_coef=1.0, epochs=1, minibatch_size=64)
    assert stats["updates"] > 0 and stats["skipped"] == 0
    for k in ("policy_loss", "value_loss", "entropy"):
        assert math.isfinite(stats[k]), (k, stats)

    # The checkpoint must RECORD its table: schema_version is 1 for both, so
    # the name is the only thing that tells deploy/league which one decodes it.
    p = f"{tmp_path}/ck_v1air.pt"
    t.save_checkpoint(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["schema_version"] == 1
    assert ck["action_table"] == "construct_104_v1air"
    assert tuple(ck["model"]["action_table"].shape) == (104, 8)


def test_v1_checkpoint_records_the_92_row_table_name(tmp_path):
    """The same field on the frozen lineage, so readers never have to guess."""
    t = Trainer(v1_cfg(tmp_path))
    p = f"{tmp_path}/ck_v1.pt"
    t.save_checkpoint(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["action_table"] == "construct_92_v1"
    assert tuple(ck["model"]["action_table"].shape) == (92, 8)


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


def _synthetic_v1_prior_checkpoint(path, net_dims, seed=2):
    """A tiny v1 EntityPolicyNet checkpoint, dims matching net_dims exactly
    (KLPrior's expect_net assert requires this against the student's cfg.net),
    saved with schema_version=1 as kl_prior.py's KLPrior expects."""
    torch.manual_seed(seed)
    net = EntityPolicyNet(**net_dims, action_table=action_table_v1())
    torch.save({
        "model": net.state_dict(),
        "schema_version": 1,
        "config": {"net": dict(net_dims)},
        "total_steps": 0,
    }, path)
    return str(path)


def test_kl_prior_e2e_one_iter_gate_and_log(tmp_path, capsys):
    """K3 e2e: [kl_prior] config block gates Trainer.kl_prior on, prints the
    init anchor line, and one run() iteration logs a finite kl_pri stat +
    lambda_p. Mirrors test_kickstart_e2e_one_iter_kick_kl_finite's harness
    (real engine, tiny net, cpu) since compose_extra_loss's prior_on path
    needs a genuine v1 collect() batch (obs dict) exactly like kickstart's
    obs_v0 path does."""
    cfg = v1_cfg(tmp_path, arenas=2, rollout=32)
    prior_ck = _synthetic_v1_prior_checkpoint(tmp_path / "prior.pt", cfg.net)
    cfg.kl_prior = {"ck": prior_ck, "lambda": 0.25}

    t = Trainer(cfg)
    assert t.kl_prior is not None
    init_out = capsys.readouterr().out
    assert f"kl_prior: anchored to {prior_ck}" in init_out
    assert "lambda_p=0.25" in init_out

    t.run(max_iterations=1)
    out = capsys.readouterr().out
    m = re.search(r"kl_pri (\S+)", out)
    assert m, f"kl_pri missing from training log: {out!r}"
    assert math.isfinite(float(m.group(1))), out
    assert "lambda_p" in out


def test_v0_checkpoint_with_v1_config_errors_clearly(tmp_path):
    ck = _synthetic_v0_checkpoint(tmp_path / "v0.pt")
    with pytest.raises(ValueError, match="schema"):
        Trainer.load_checkpoint(ck, cfg_path="configs/train_v1.toml")
