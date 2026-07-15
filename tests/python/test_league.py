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
