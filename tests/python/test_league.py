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
