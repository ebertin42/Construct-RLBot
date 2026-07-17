"""Tick-level league behavior: schema-pure pairing, match-budget fairness
across schema pools, rating updates through the Registry API, and one real
v1-engine smoke.

Logic tests mock the match runner (construct.league.tick's make_runner/play
injection seams) -- no engine is ever built, no checkpoint file is ever read.
The single real smoke at the bottom drives two actual checkpoints_entity/
checkpoints (read-only) through the real v1 engine path.
"""
import random
import sys
from pathlib import Path

import pytest

from construct.league.registry import Registry
from construct.league.tick import play_rating_matches, run_tick, split_budget

REPO = Path(__file__).resolve().parents[2]


def _seed_pool(reg, schema_version, n, prefix):
    for i in range(n):
        reg.add(f"{prefix}{i}.pt", steps=i, run="x", reward_config="x",
                schema_version=schema_version)


def _fake_play(pairs, score=(1, 0)):
    def play(mr, entry_a, entry_b, steps=2700):
        pairs.append((entry_a, entry_b))
        return score
    return play


# --- budget fairness --------------------------------------------------------

def test_split_budget_even_remainder_and_sum():
    assert split_budget(6, 2) == [3, 3]
    assert split_budget(7, 2) == [4, 3]
    assert split_budget(1, 2) == [1, 0]
    assert split_budget(0, 2) == [0, 0]
    assert sum(split_budget(11, 3)) == 11


def test_both_pools_get_matches_per_tick(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # discovery globs find nothing
    reg_path = str(tmp_path / "mixed.jsonl")
    reg = Registry(path=reg_path)
    _seed_pool(reg, 0, 3, "v0_")
    _seed_pool(reg, 1, 3, "v1_")

    pairs = []
    played = run_tick([0, 1], matches=6, registry_path=reg_path,
                      rng=random.Random(0), make_runner=lambda: None,
                      play=_fake_play(pairs))
    # 6-match budget split fairly: each schema pool gets exactly its share
    assert played == {0: 3, 1: 3}
    assert len(pairs) == 6


def test_budget_rolls_over_when_pool_cannot_play(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg_path = str(tmp_path / "mixed.jsonl")
    reg = Registry(path=reg_path)
    _seed_pool(reg, 0, 1, "v0_")  # one entry: cannot pair, share must roll over
    _seed_pool(reg, 1, 4, "v1_")

    pairs = []
    played = run_tick([0, 1], matches=6, registry_path=reg_path,
                      rng=random.Random(0), make_runner=lambda: None,
                      play=_fake_play(pairs))
    assert played[0] == 0
    # v1 exceeded its even share of 3: v0's unusable budget flowed to it
    assert played[1] >= 4


def test_zero_budget_plays_nothing_and_builds_no_runner(tmp_path):
    reg = Registry(path=str(tmp_path / "reg.jsonl"))
    _seed_pool(reg, 1, 3, "v1_")

    def exploding_runner():
        raise AssertionError("runner must not be built for a zero budget")

    assert play_rating_matches(reg, 1, budget=0, make_runner=exploding_runner) == 0
    assert all(e["games"] == 0 for e in reg.entries())


# --- schema-pure pairing ----------------------------------------------------

def test_tick_never_pairs_across_schemas(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg_path = str(tmp_path / "mixed.jsonl")
    reg = Registry(path=reg_path)
    _seed_pool(reg, 0, 3, "v0_")
    _seed_pool(reg, 1, 3, "v1_")

    pairs = []
    run_tick([0, 1], matches=6, registry_path=reg_path, rng=random.Random(7),
             make_runner=lambda: None, play=_fake_play(pairs))
    assert pairs, "tick must actually pair members with opponents"
    for a, b in pairs:
        assert a["schema_version"] == b["schema_version"], f"cross-schema pair {a['ck']} vs {b['ck']}"
    # and both schemas were represented, so purity wasn't vacuous
    assert {a["schema_version"] for a, _ in pairs} == {0, 1}


# --- rating updates ---------------------------------------------------------

def test_match_result_updates_ratings_and_persists(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    _seed_pool(reg, 1, 2, "v1_")

    pairs = []
    played = run_tick([1], matches=1, registry_path=reg_path, rng=random.Random(0),
                      make_runner=lambda: None, play=_fake_play(pairs, score=(3, 0)))
    assert played == {1: 1}
    (winner, loser) = pairs[0][0]["ck"], pairs[0][1]["ck"]

    fresh = Registry(path=reg_path)  # reload from disk: updates must persist
    mu_w, _ = fresh.rating(winner)
    mu_l, _ = fresh.rating(loser)
    assert mu_w > 25.0 > mu_l
    assert all(e["games"] == 1 for e in fresh.entries())


def test_shared_registry_file_keeps_both_pools_updates(tmp_path, monkeypatch):
    # 'all' mode on one mixed file (the remote-box topology) must accumulate
    # BOTH pools' rating updates -- a second Registry instance on the same path
    # would clobber the first pool's results at _save() time.
    monkeypatch.chdir(tmp_path)
    reg_path = str(tmp_path / "mixed.jsonl")
    reg = Registry(path=reg_path)
    _seed_pool(reg, 0, 3, "v0_")
    _seed_pool(reg, 1, 3, "v1_")

    played = run_tick([0, 1], matches=6, registry_path=reg_path,
                      rng=random.Random(0), make_runner=lambda: None,
                      play=_fake_play([], score=(2, 0)))
    fresh = Registry(path=reg_path)
    for sv in (0, 1):
        pool_games = sum(e["games"] for e in fresh.entries(schema_version=sv))
        assert pool_games == 2 * played[sv] > 0


# --- CLI wiring -------------------------------------------------------------

def _load_cli():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "league_tick_cli", REPO / "scripts" / "league_tick.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_cli_all_mode_dispatches_both_schemas(monkeypatch):
    mod = _load_cli()
    calls = {}
    monkeypatch.setattr(mod, "run_tick",
                        lambda svs, matches, **kw: calls.update(svs=svs, matches=matches, kw=kw))
    monkeypatch.setattr(sys, "argv",
                        ["league_tick.py", "--schema-version", "all", "--matches", "5"])
    mod.main()
    assert calls["svs"] == [0, 1]
    assert calls["matches"] == 5


def test_cli_single_schema_stays_single(monkeypatch):
    mod = _load_cli()
    calls = {}
    monkeypatch.setattr(mod, "run_tick",
                        lambda svs, matches, **kw: calls.update(svs=svs, matches=matches, kw=kw))
    monkeypatch.setattr(sys, "argv", ["league_tick.py", "--schema-version", "1"])
    mod.main()
    assert calls["svs"] == [1]


# --- real v1 engine smoke ---------------------------------------------------

REAL_CK_DIR = REPO / "checkpoints_entity"
REAL_CKS = sorted(REAL_CK_DIR.glob("ck_*.pt")) if REAL_CK_DIR.is_dir() else []


@pytest.mark.skipif(len(REAL_CKS) < 2,
                    reason="needs two real v1 checkpoints in checkpoints_entity/")
def test_real_v1_checkpoints_play_short_match(tmp_path):
    # The ONE real-engine smoke: two actual entity checkpoints (read-only) play
    # a short match through the real v1 MatchRunner path; the result -- win,
    # loss, or draw -- must land in the registry (games up, sigma shrunk).
    import torch
    reg = Registry(path=str(tmp_path / "reg.jsonl"))
    heads = None
    for ck in (REAL_CKS[0], REAL_CKS[-1]):
        meta = torch.load(ck, map_location="cpu", weights_only=False)
        assert meta.get("schema_version") == 1
        heads = meta["config"]["net"]["heads"]
        reg.add(str(ck), steps=meta["total_steps"], run="entity",
                reward_config=meta.get("reward_config_path", "x"), schema_version=1)

    played = play_rating_matches(reg, 1, budget=1, net_heads=heads,
                                 rng=random.Random(0), steps=120, num_arenas=2)
    assert played == 1
    entries = reg.entries(schema_version=1)
    assert [e["games"] for e in entries] == [1, 1]
    assert all(e["sigma"] < 25 / 3 for e in entries)  # any result shrinks sigma
