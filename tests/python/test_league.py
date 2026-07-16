import os

from construct.league.registry import Registry


def test_registry_roundtrip(tmp_path):
    r = Registry(path=str(tmp_path / "reg.jsonl"))
    r.add("ck_a.pt", steps=100, run="main", reward_config="configs/reward_v1.toml")
    r.add("ck_b.pt", steps=200, run="b", reward_config="configs/reward_v2.toml")
    r.add("ck_a.pt", steps=100, run="main", reward_config="configs/reward_v1.toml")  # dup no-op
    assert len(r.entries()) == 2
    r2 = Registry(path=str(tmp_path / "reg.jsonl"))  # reload from disk
    assert [e["ck"] for e in r2.entries()] == ["ck_a.pt", "ck_b.pt"]
    mu, sigma = r2.rating("ck_a.pt")
    assert mu == 25.0 and abs(sigma - 25 / 3) < 1e-9


def test_match_updates_ratings(tmp_path):
    r = Registry(path=str(tmp_path / "reg.jsonl"))
    r.add("A", 1, "main", "x"); r.add("B", 2, "main", "x")
    r.record_match("A", "B", goals_a=3, goals_b=0)
    mu_a, _ = r.rating("A"); mu_b, _ = r.rating("B")
    assert mu_a > 25.0 > mu_b
    r.record_match("A", "B", goals_a=1, goals_b=1)  # draw: ratings converge, order kept
    assert r.rating("A")[0] > r.rating("B")[0]
    ladder = r.ladder()
    assert ladder[0]["ck"] == "A" and "skill" in ladder[0]
    assert ladder[0]["games"] == 2


def test_atomic_persistence(tmp_path):
    p = tmp_path / "reg.jsonl"
    r = Registry(path=str(p))
    r.add("A", 1, "main", "x")
    # file exists and is valid jsonl after every mutation
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 1
    import json
    e = json.loads(lines[0])
    assert e["ck"] == "A" and e["games"] == 0


import numpy as np
import torch

from construct.league.matches import MatchRunner, load_sd
from construct.learn.model import PolicyValueNet


def _sd(seed):
    torch.manual_seed(seed)
    return {k: v.detach().numpy().astype(np.float32)
            for k, v in PolicyValueNet(94, 90, (64, 64)).state_dict().items()}


def test_match_runs_and_counts(tmp_path):
    mr = MatchRunner(num_arenas=4, seed=2)
    ga, gb = mr.play(_sd(1), _sd(2), steps=600)
    assert ga >= 0 and gb >= 0  # random-ish nets rarely score in 40s; counting must not crash


def test_match_deterministic(tmp_path):
    a = MatchRunner(num_arenas=4, seed=7).play(_sd(1), _sd(2), steps=400)
    b = MatchRunner(num_arenas=4, seed=7).play(_sd(1), _sd(2), steps=400)
    assert a == b


def test_load_sd_roundtrip(tmp_path):
    net = PolicyValueNet(94, 90, (64, 64))
    ck = {"model": net.state_dict(), "config": {"net": {"hidden": [64, 64]}}, "total_steps": 1,
          "schema_version": 0}
    p = str(tmp_path / "ck.pt")
    torch.save(ck, p)
    sd = load_sd(p)
    assert sd["policy_head.weight"].dtype == np.float32


def test_goal_threshold_covers_shaped_concede():
    from construct.league.matches import GOAL_THRESHOLD
    assert GOAL_THRESHOLD <= 9.45 - 1e-6   # concede magnitude floor
    assert GOAL_THRESHOLD >= 0.55 + 1e-6   # non-goal shaping ceiling


import random

from construct.league.sampling import choose_opponents


def test_choose_opponents_mix(tmp_path):
    r = Registry(path=str(tmp_path / "reg.jsonl"))
    for i in range(10):
        r.add(f"ck{i}", steps=i, run="main", reward_config="x")
    # give ck0 a huge rating so it must appear as a top pick
    r.record_match("ck0", "ck5", 5, 0)
    r.record_match("ck0", "ck6", 5, 0)
    picks = choose_opponents(r, k=4, recent=5, rng=random.Random(3))
    names = [p["ck"] for p in picks]
    assert "ck0" in names
    assert len(names) == len(set(names)) <= 4
    # deterministic
    again = [p["ck"] for p in choose_opponents(r, k=4, recent=5, rng=random.Random(3))]
    assert names == again


import pytest

from construct.learn.config import TrainConfig
from construct.learn.train import Trainer


def _league_cfg(tmp_path, reg_path, **league_overrides):
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=4)
    cfg.ppo.update(rollout_steps=8, minibatch_size=64)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=100)
    cfg.league = {"enabled": True, "opponent_frac": 0.5, "registry": reg_path,
                  "refresh_iters": 1, "slots": 2, **league_overrides}
    return cfg


def test_refresh_skips_missing_checkpoint(tmp_path, capsys):
    # A registry entry whose checkpoint file is gone (pruned/corrupted on disk)
    # must not crash the trainer at a refresh boundary.
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    reg.add(str(tmp_path / "does_not_exist.pt"), steps=1, run="main", reward_config="x")

    cfg = _league_cfg(tmp_path, reg_path)
    t = Trainer(cfg)
    t._refresh_opponents()  # must not raise
    assert t._assignment is None  # all picks failed -> no prior assignment to fall back to
    out = capsys.readouterr().out
    assert "league" in out and "does_not_exist.pt" in out


def test_refresh_keeps_previous_assignment_when_all_picks_fail(tmp_path, capsys):
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    reg.add(str(tmp_path / "missing.pt"), steps=1, run="main", reward_config="x")

    cfg = _league_cfg(tmp_path, reg_path)
    t = Trainer(cfg)
    t._assignment = [-1, -1, 0, 0]  # simulate a prior good assignment
    t._refresh_opponents()
    assert t._assignment == [-1, -1, 0, 0]  # unchanged, not wiped to None
    out = capsys.readouterr().out
    assert "league" in out


def test_assignment_layout_tail_round_robin(tmp_path):
    import torch
    from construct.learn.model import PolicyValueNet
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    for i in (1, 2):
        net = PolicyValueNet(94, 90, (512, 512))
        p = str(tmp_path / f"opp{i}.pt")
        torch.save({"model": net.state_dict(), "total_steps": i,
                    "config": {"net": {"hidden": [512, 512]}}, "schema_version": 0,
                    "reward_config_path": "x"}, p)
        reg.add(p, steps=i, run="main", reward_config="x")

    cfg = _league_cfg(tmp_path, reg_path)
    t = Trainer(cfg)
    t._refresh_opponents()
    # 4 arenas, frac 0.5 -> n_opp=2 -> tail-assigned, slot round-robin over 2 opponents
    assert t._assignment == [-1, -1, 0, 1]


def test_opponent_frac_bounds_validated():
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.run.update(device="cpu")
    cfg.league = {"enabled": True, "opponent_frac": 1.0}
    with pytest.raises(AssertionError, match="opponent_frac"):
        Trainer(cfg)


def test_opponent_frac_negative_rejected():
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.run.update(device="cpu")
    cfg.league = {"enabled": True, "opponent_frac": -0.1}
    with pytest.raises(AssertionError, match="opponent_frac"):
        Trainer(cfg)


def test_refresh_iters_floors_to_one(tmp_path):
    reg_path = str(tmp_path / "reg.jsonl")
    Registry(path=reg_path)  # empty registry is fine, just checking init doesn't div-by-zero
    cfg = _league_cfg(tmp_path, reg_path, refresh_iters=0)
    t = Trainer(cfg)
    assert t._league["refresh"] == 1
    it = 0
    it % t._league["refresh"]  # would raise ZeroDivisionError pre-fix


def test_trainer_league_refresh(tmp_path):
    # seed a registry with two members built from tiny checkpoints
    import torch
    from construct.learn.model import PolicyValueNet
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    for i in (1, 2):
        net = PolicyValueNet(94, 90, (512, 512))
        p = str(tmp_path / f"opp{i}.pt")
        torch.save({"model": net.state_dict(), "total_steps": i,
                    "config": {"net": {"hidden": [512, 512]}}, "schema_version": 0,
                    "reward_config_path": "x"}, p)
        reg.add(p, steps=i, run="main", reward_config="x")

    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=4)
    cfg.ppo.update(rollout_steps=8, minibatch_size=64)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=100)
    cfg.league = {"enabled": True, "opponent_frac": 0.5, "registry": reg_path,
                  "refresh_iters": 1, "slots": 2}
    t = Trainer(cfg)
    t.run(max_iterations=2)
    # 4 arenas, frac 0.5 -> 2 opponent arenas -> learner agents = 4*2 - 2 = 6
    assert t.total_steps == 2 * 8 * 6


def _save_tiny_ck(tmp_path, name):
    import torch
    from construct.learn.model import PolicyValueNet
    net = PolicyValueNet(94, 90, (512, 512))
    p = str(tmp_path / name)
    torch.save({"model": net.state_dict(), "total_steps": 1,
                "config": {"net": {"hidden": [512, 512]}}, "schema_version": 0,
                "reward_config_path": "x"}, p)
    return p


def test_refresh_opponents_reloads_registry_from_disk(tmp_path, capsys):
    # scripts/league_tick.py runs out-of-process and appends to the registry
    # via its own Registry instance. The Trainer's in-memory snapshot (taken
    # once at __init__) must not go stale -- a later refresh must see entries
    # added to the jsonl after the Trainer was constructed.
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    reg.add(_save_tiny_ck(tmp_path, "opp1.pt"), steps=1, run="main", reward_config="x")

    cfg = _league_cfg(tmp_path, reg_path, slots=2)
    t = Trainer(cfg)
    t._refresh_opponents()
    capsys.readouterr()  # drain first refresh's output

    # a second, independent Registry instance (standing in for league_tick.py)
    # appends a new entry to the same on-disk jsonl.
    p2 = _save_tiny_ck(tmp_path, "opp2.pt")
    Registry(path=reg_path).add(p2, steps=2, run="main", reward_config="x")

    t._refresh_opponents()
    out = capsys.readouterr().out
    # k=2, only 2 entries total in the reloaded registry -> both must be picked
    assert p2 in out


def test_refresh_opponents_seeded_rng_is_deterministic(tmp_path):
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    for i in range(6):
        reg.add(_save_tiny_ck(tmp_path, f"opp{i}.pt"), steps=i, run="main", reward_config="x")

    cfg = _league_cfg(tmp_path, reg_path, slots=3)
    t1 = Trainer(cfg)
    t1._refresh_opponents(it=5)
    t2 = Trainer(cfg)
    t2._refresh_opponents(it=5)
    # same config + registry + iteration -> identical opponent assignment
    # (fixed-config determinism contract for league runs)
    assert t1._assignment == t2._assignment


def test_league_slots_upper_bound_validated():
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.run.update(device="cpu")
    cfg.league = {"enabled": True, "slots": 9}
    with pytest.raises(AssertionError, match="slots"):
        Trainer(cfg)


def test_refresh_set_opponents_failure_keeps_previous_assignment(tmp_path, capsys):
    # A future league-on-v1 misconfig (or any other set_opponents-time
    # failure, e.g. an opponent state dict incompatible with the engine's
    # obs mode) must degrade to "keep the prior assignment" like the other
    # refresh failure modes, not propagate and kill the run.
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    reg.add(_save_tiny_ck(tmp_path, "opp1.pt"), steps=1, run="main", reward_config="x")

    cfg = _league_cfg(tmp_path, reg_path, slots=1)
    t = Trainer(cfg)
    t._assignment = [-1, -1, 0, 0]  # simulate a prior good assignment

    class _FailingEngine:
        def set_opponents(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    t.engine = _FailingEngine()
    t._refresh_opponents()  # must not raise
    assert t._assignment == [-1, -1, 0, 0]  # unchanged
    out = capsys.readouterr().out
    assert "league" in out and "set_opponents" in out and "boom" in out
