"""v1-schema league support: schema-tagged registry entries, schema-gated
matches, and v1 checkpoint discovery for league_tick.py.

v0 and v1 policies can NEVER play each other (different obs contracts) --
every test here ultimately serves that invariant. Never touches the real
checkpoints*/ dirs or the live league/registry.jsonl -- everything runs
against tmp_path fixtures.
"""
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from construct._engine import action_table_v1
from construct.learn.model_v1 import EntityPolicyNet
from construct.league.matches import MatchRunner, load_sd, play_entries
from construct.league.registry import Registry


def _v1_sd(seed, heads=2, d_model=32, ff=64):
    torch.manual_seed(seed)
    net = EntityPolicyNet(d_model=d_model, layers=1, heads=heads, ff=ff,
                           action_table=action_table_v1())
    return {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}


# --- registry: schema_version tagging -------------------------------------

def test_legacy_registry_line_reads_as_schema_v0(tmp_path):
    p = tmp_path / "reg.jsonl"
    # hand-written legacy line, exactly as pre-v1 registries look on disk: no
    # schema_version key at all.
    legacy = {"ck": "old.pt", "steps": 10, "run": "main", "reward_config": "x",
              "added_ts": 1, "mu": 25.0, "sigma": 25 / 3, "games": 0}
    p.write_text(json.dumps(legacy) + "\n")
    r = Registry(path=str(p))
    (entry,) = r.entries()
    assert entry["schema_version"] == 0


def test_new_registry_line_round_trips_schema_v1(tmp_path):
    p = tmp_path / "reg.jsonl"
    r = Registry(path=str(p))
    r.add("ck_v1.pt", steps=1, run="entity", reward_config="x", schema_version=1)
    r2 = Registry(path=str(p))  # reload from disk
    (entry,) = r2.entries()
    assert entry["schema_version"] == 1
    # on-disk line itself carries the field (not just an in-memory default)
    line = json.loads(p.read_text().strip().splitlines()[0])
    assert line["schema_version"] == 1


def test_add_defaults_schema_version_zero(tmp_path):
    r = Registry(path=str(tmp_path / "reg.jsonl"))
    r.add("ck.pt", steps=1, run="main", reward_config="x")  # no schema_version kwarg
    assert r.entries()[0]["schema_version"] == 0


def test_entries_filters_by_schema_version(tmp_path):
    r = Registry(path=str(tmp_path / "reg.jsonl"))
    r.add("v0.pt", steps=1, run="main", reward_config="x", schema_version=0)
    r.add("v1.pt", steps=1, run="entity", reward_config="x", schema_version=1)
    assert [e["ck"] for e in r.entries(schema_version=0)] == ["v0.pt"]
    assert [e["ck"] for e in r.entries(schema_version=1)] == ["v1.pt"]
    assert len(r.entries()) == 2  # unfiltered: both


# --- matches: v1 smoke + cross-schema guard --------------------------------

def test_match_runner_v1_smoke(tmp_path):
    mr = MatchRunner(num_arenas=2, seed=3, schema_version=1, net_heads=2)
    ga, gb = mr.play(_v1_sd(1), _v1_sd(2), steps=200)
    assert ga >= 0 and gb >= 0  # random-ish nets rarely score; counting must not crash


def test_match_runner_v1_deterministic():
    a = MatchRunner(num_arenas=2, seed=7, schema_version=1, net_heads=2).play(
        _v1_sd(1), _v1_sd(2), steps=150)
    b = MatchRunner(num_arenas=2, seed=7, schema_version=1, net_heads=2).play(
        _v1_sd(1), _v1_sd(2), steps=150)
    assert a == b


def test_load_sd_v1_roundtrip_includes_action_table(tmp_path):
    net = EntityPolicyNet(d_model=32, layers=1, heads=2, ff=64, action_table=action_table_v1())
    ck = {"model": net.state_dict(), "total_steps": 1, "schema_version": 1}
    p = str(tmp_path / "ck_v1.pt")
    torch.save(ck, p)
    sd = load_sd(p)
    assert sd["action_table"].shape == (92, 8)
    assert sd["action_table"].dtype == np.float32


def test_match_runner_rejects_unsupported_schema_version():
    with pytest.raises(AssertionError, match="schema_version"):
        MatchRunner(num_arenas=2, seed=0, schema_version=2)


def test_cross_schema_match_refused():
    entry_a = {"ck": "a.pt", "schema_version": 0}
    entry_b = {"ck": "b.pt", "schema_version": 1}
    with pytest.raises(ValueError, match="schema"):
        play_entries(None, entry_a, entry_b)  # must refuse before touching mr or disk


def test_same_schema_match_not_refused_by_guard(tmp_path):
    # play_entries's schema check must not false-positive on matched schemas;
    # drive it end to end with real tiny checkpoints on disk.
    p_a, p_b = tmp_path / "a.pt", tmp_path / "b.pt"
    for p, seed in ((p_a, 1), (p_b, 2)):
        net = EntityPolicyNet(d_model=32, layers=1, heads=2, ff=64, action_table=action_table_v1())
        torch.manual_seed(seed)
        torch.save({"model": net.state_dict(), "total_steps": 1, "schema_version": 1}, p)
    mr = MatchRunner(num_arenas=2, seed=1, schema_version=1, net_heads=2)
    entry_a = {"ck": str(p_a), "schema_version": 1}
    entry_b = {"ck": str(p_b), "schema_version": 1}
    ga, gb = play_entries(mr, entry_a, entry_b, steps=100)
    assert ga >= 0 and gb >= 0


# --- Trainer._refresh_opponents: schema filtering --------------------------

def _save_tiny_v1_ck(tmp_path, name, seed=0):
    torch.manual_seed(seed)
    net = EntityPolicyNet(d_model=32, layers=1, heads=2, ff=64, action_table=action_table_v1())
    p = str(tmp_path / name)
    torch.save({"model": net.state_dict(), "total_steps": 1,
                "config": {"net": {"d_model": 32, "layers": 1, "heads": 2, "ff": 64}},
                "schema_version": 1, "reward_config_path": "x"}, p)
    return p


def _save_tiny_v0_ck(tmp_path, name, seed=0):
    from construct.learn.model import PolicyValueNet
    torch.manual_seed(seed)
    net = PolicyValueNet(94, 90, (32,))
    p = str(tmp_path / name)
    torch.save({"model": net.state_dict(), "total_steps": 1,
                "config": {"net": {"hidden": [32]}}, "schema_version": 0,
                "reward_config_path": "x"}, p)
    return p


def _v1_trainer_cfg(tmp_path, reg_path):
    from construct.learn.config import TrainConfig
    cfg = TrainConfig.load("configs/train_v1.toml")
    cfg.env.update(num_arenas=2, blue=1, orange=1)
    cfg.env.pop("team_size_weights", None)
    cfg.curriculum_config_path = ""
    cfg.net = {"d_model": 32, "layers": 1, "heads": 2, "ff": 64}
    cfg.ppo.update(rollout_steps=8, minibatch_size=64)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=100)
    cfg.kickstart = {}
    cfg.league = {"enabled": True, "opponent_frac": 0.5, "registry": reg_path,
                  "refresh_iters": 1, "slots": 2}
    return cfg


def test_refresh_opponents_filters_to_trainer_schema(tmp_path, capsys):
    from construct.learn.train import Trainer
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    reg.add(_save_tiny_v0_ck(tmp_path, "v0.pt"), steps=1, run="main", reward_config="x",
            schema_version=0)
    reg.add(_save_tiny_v1_ck(tmp_path, "v1.pt"), steps=1, run="entity", reward_config="x",
            schema_version=1)

    cfg = _v1_trainer_cfg(tmp_path, reg_path)
    t = Trainer(cfg)
    assert t.is_v1
    t._refresh_opponents()
    out = capsys.readouterr().out
    assert "v1.pt" in out
    assert "v0.pt" not in out


def test_refresh_opponents_v0_trainer_ignores_v1_entries(tmp_path, capsys):
    from construct.learn.config import TrainConfig
    from construct.learn.train import Trainer
    reg_path = str(tmp_path / "reg.jsonl")
    reg = Registry(path=reg_path)
    reg.add(_save_tiny_v0_ck(tmp_path, "v0.pt"), steps=1, run="main", reward_config="x",
            schema_version=0)
    reg.add(_save_tiny_v1_ck(tmp_path, "v1.pt"), steps=1, run="entity", reward_config="x",
            schema_version=1)

    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=2)
    cfg.ppo.update(rollout_steps=8, minibatch_size=64)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=100)
    cfg.league = {"enabled": True, "opponent_frac": 0.5, "registry": reg_path,
                  "refresh_iters": 1, "slots": 2}
    t = Trainer(cfg)
    assert not t.is_v1
    t._refresh_opponents()
    out = capsys.readouterr().out
    assert "v0.pt" in out
    assert "v1.pt" not in out


# --- league_tick.py: v1 checkpoint discovery -------------------------------

def test_league_tick_registers_from_checkpoints_entity(tmp_path):
    # synthetic v1 checkpoint under a fake repo root's checkpoints_entity/ --
    # never touches the real repo's checkpoints_entity/ or league/registry.jsonl.
    ck_dir = tmp_path / "checkpoints_entity"
    ck_dir.mkdir()
    net = EntityPolicyNet(d_model=32, layers=1, heads=2, ff=64, action_table=action_table_v1())
    torch.save({"model": net.state_dict(), "total_steps": 42,
                "config": {"net": {"d_model": 32, "layers": 1, "heads": 2, "ff": 64}},
                "schema_version": 1, "reward_config_path": "x"},
               ck_dir / "ck_000000000042.pt")

    reg_path = tmp_path / "league" / "registry_v1.jsonl"
    result = subprocess.run(
        [sys.executable, str(__import__("pathlib").Path(__file__).resolve().parents[2]
                              / "scripts" / "league_tick.py"),
         "--schema-version", "1", "--registry", str(reg_path), "--matches", "0"],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert reg_path.exists()
    lines = [json.loads(l) for l in reg_path.read_text().strip().splitlines()]
    assert len(lines) == 1
    assert lines[0]["schema_version"] == 1
    assert lines[0]["ck"].endswith("ck_000000000042.pt")


def test_league_tick_default_registry_path_depends_on_schema_version(tmp_path):
    # --registry omitted: v0 keeps the legacy default path, v1 gets its own.
    # A checkpoint must actually be discovered for a registry file to get
    # written at all (Registry only creates the file on its first add()), so
    # seed each fake repo root with one checkpoint at the schema-appropriate
    # discovery location.
    script = str(__import__("pathlib").Path(__file__).resolve().parents[2]
                 / "scripts" / "league_tick.py")
    cases = (
        (0, "checkpoints", "league/registry.jsonl"),
        (1, "checkpoints_entity", "league/registry_v1.jsonl"),
    )
    for schema_version, ck_dir_name, expected_rel in cases:
        root = tmp_path / f"v{schema_version}"
        ck_dir = root / ck_dir_name
        ck_dir.mkdir(parents=True)
        net = EntityPolicyNet(d_model=32, layers=1, heads=2, ff=64, action_table=action_table_v1())
        torch.save({"model": net.state_dict(), "total_steps": 1, "schema_version": schema_version,
                    "reward_config_path": "x"}, ck_dir / "ck_000000000001.pt")
        result = subprocess.run(
            [sys.executable, script, "--schema-version", str(schema_version), "--matches", "0"],
            cwd=str(root), capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr
        assert (root / expected_rel).exists()
