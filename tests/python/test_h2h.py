"""Tests for scripts/h2h_eval.py, the head-to-head skill-eval harness.

Logic tests use a MOCKED MatchRunner (a tiny fake with a .play() method) --
no engine is ever built, no real checkpoint file is ever read, for the
aggregation math / share / disagreement / jsonl / config-parsing tests. The
one real-engine smoke at the bottom mirrors test_league_tick.py's pattern:
two actual checkpoints_entity/ checkpoints (read-only) play a short match
through the real v1 MatchRunner path, kept under ~5s with tiny steps."""
import json
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package, so import it by adding it to sys.path (same
# pattern as tests/python/test_ctl.py).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import h2h_eval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# goal_share
# ---------------------------------------------------------------------------

def test_goal_share_basic():
    assert h2h_eval.goal_share(35, 119) == pytest.approx(35 / 154)


def test_goal_share_zero_zero_is_none():
    assert h2h_eval.goal_share(0, 0) is None


def test_goal_share_shutout():
    assert h2h_eval.goal_share(10, 0) == 1.0
    assert h2h_eval.goal_share(0, 10) == 0.0


# ---------------------------------------------------------------------------
# aggregate_sides: both-side-order aggregation math + disagreement flagging
# ---------------------------------------------------------------------------

def test_aggregate_sides_sums_both_orders():
    # journal 2026-07-19 ~14:50: anchored 909M vs unanchored 1.38B, 19-60 / 16-59
    result = h2h_eval.aggregate_sides((19, 60), (16, 59))
    assert result["a"] == 35
    assert result["b"] == 119
    assert result["share"] == pytest.approx(35 / 154)


def test_aggregate_sides_preserves_per_side_pairs():
    result = h2h_eval.aggregate_sides((19, 60), (16, 59))
    assert result["side1"] == (19, 60)
    assert result["side2"] == (16, 59)
    assert result["share_side1"] == pytest.approx(19 / 79)
    assert result["share_side2"] == pytest.approx(16 / 75)


def test_aggregate_sides_agreement_not_flagged():
    # side1 share 50%, side2 share 50% -- perfect agreement
    result = h2h_eval.aggregate_sides((10, 10), (10, 10))
    assert result["disagreement"] == pytest.approx(0.0)
    assert result["unstable"] is False


def test_aggregate_sides_small_disagreement_not_flagged():
    # side1 share 60%, side2 share 50% -- 10% split, under the 15% threshold
    result = h2h_eval.aggregate_sides((6, 4), (5, 5))
    assert result["disagreement"] == pytest.approx(0.10)
    assert result["unstable"] is False


def test_aggregate_sides_large_disagreement_flagged():
    # journal: 19-60 (share 24.1%) / 16-59 (share 21.3%) -- actually a SMALL
    # split; use the bigger swing pair from the same entry to force >15%.
    # side1 share 90%, side2 share 10% -- 80% split, well past 15%.
    result = h2h_eval.aggregate_sides((9, 1), (1, 9))
    assert result["disagreement"] == pytest.approx(0.8)
    assert result["unstable"] is True


def test_is_unstable_threshold_boundary():
    assert h2h_eval.DISAGREEMENT_THRESHOLD == 0.15
    assert h2h_eval.is_unstable(0.15) is False        # exactly at: not flagged
    assert h2h_eval.is_unstable(0.150001) is True      # a hair past: flagged
    assert h2h_eval.is_unstable(0.0) is False
    assert h2h_eval.is_unstable(None) is False         # no share to disagree about


def test_aggregate_sides_uses_is_unstable_for_the_flag():
    result = h2h_eval.aggregate_sides((651, 349), (500, 500))
    assert result["disagreement"] > 0.15
    assert result["unstable"] is True


def test_aggregate_sides_zero_goals_share_is_none_not_unstable():
    result = h2h_eval.aggregate_sides((0, 0), (0, 0))
    assert result["share"] is None
    assert result["disagreement"] is None
    assert result["unstable"] is False


# ---------------------------------------------------------------------------
# play_h2h: drives a mocked runner, exercises the side-swap plumbing
# ---------------------------------------------------------------------------

class _FakeRunner:
    """Records calls and returns scripted (a, b) results in order, keyed by
    which state dict came first -- mirrors MatchRunner.play's signature
    (sd_a, sd_b, steps=...) -> (goals_a, goals_b)."""

    def __init__(self, results):
        self._results = list(results)
        self.calls = []

    def play(self, sd_a, sd_b, steps=2700):
        self.calls.append((sd_a, sd_b, steps))
        return self._results.pop(0)


def test_play_h2h_calls_both_side_orders_with_swapped_args():
    mr = _FakeRunner([(19, 60), (59, 16)])  # side2 call returns (b, a) raw
    result = h2h_eval.play_h2h(mr, "SD_A", "SD_B", steps=5400)

    assert len(mr.calls) == 2
    assert mr.calls[0] == ("SD_A", "SD_B", 5400)   # side1: A first
    assert mr.calls[1] == ("SD_B", "SD_A", 5400)   # side2: B first (swapped)

    # side2's raw (59, 16) is (goals_for_B, goals_for_A) since B was passed
    # first -- play_h2h must flip it back to (a, b) order before aggregating.
    assert result["side1"] == (19, 60)
    assert result["side2"] == (16, 59)
    assert result["a"] == 35 and result["b"] == 119


# ---------------------------------------------------------------------------
# format_result: sanity on the human-readable report (no engine involved)
# ---------------------------------------------------------------------------

def test_format_result_includes_totals_and_share():
    result = h2h_eval.aggregate_sides((19, 60), (16, 59))
    text = h2h_eval.format_result("A", "B", result)
    assert "19-60" in text and "16-59" in text
    assert "A=35" in text and "B=119" in text
    assert "22.7%" in text  # 35/154


def test_format_result_flags_unstable():
    result = h2h_eval.aggregate_sides((9, 1), (1, 9))
    text = h2h_eval.format_result("A", "B", result)
    assert "unstable" in text.lower()


def test_format_result_stable_omits_unstable_note():
    result = h2h_eval.aggregate_sides((6, 4), (5, 5))
    text = h2h_eval.format_result("A", "B", result)
    assert "unstable" not in text.lower()


def test_format_result_zero_zero_share_na():
    result = h2h_eval.aggregate_sides((0, 0), (0, 0))
    text = h2h_eval.format_result("A", "B", result)
    assert "n/a" in text


# ---------------------------------------------------------------------------
# jsonl append / round-trip schema
# ---------------------------------------------------------------------------

def test_append_h2h_history_writes_expected_schema(tmp_path):
    path = tmp_path / "h2h_history.jsonl"
    row = h2h_eval.append_h2h_history(
        path, "checkpoints_entity/ck_000909000000.pt",
        "checkpoints_entity/ck_000562083840.pt", "peak-562M",
        goals_ck=35, goals_ref=119, steps=5400, seed=11, ts=1784200000,
    )
    assert row == {
        "ts": 1784200000, "ck": "ck_000909000000.pt", "ref": "ck_000562083840.pt",
        "ref_label": "peak-562M", "goals_ck": 35, "goals_ref": 119,
        "share": pytest.approx(35 / 154), "steps": 5400, "seed": 11,
    }
    lines = path.read_text().splitlines()
    assert len(lines) == 1
    on_disk = json.loads(lines[0])
    assert on_disk["ck"] == "ck_000909000000.pt"
    assert on_disk["share"] == pytest.approx(35 / 154)


def test_append_h2h_history_appends_not_overwrites(tmp_path):
    path = tmp_path / "nested" / "h2h_history.jsonl"  # also: parent dir creation
    h2h_eval.append_h2h_history(path, "a.pt", "ref.pt", "L1", 1, 0, 100, 1, ts=1)
    h2h_eval.append_h2h_history(path, "a.pt", "ref.pt", "L1", 2, 0, 100, 1, ts=2)
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert [json.loads(ln)["ts"] for ln in lines] == [1, 2]


def test_append_h2h_history_defaults_ts_to_now(tmp_path):
    path = tmp_path / "h.jsonl"
    row = h2h_eval.append_h2h_history(path, "a.pt", "b.pt", "L", 1, 1, 10, 1)
    assert isinstance(row["ts"], int) and row["ts"] > 0


# ---------------------------------------------------------------------------
# reference-config parsing
# ---------------------------------------------------------------------------

def test_load_references_parses_toml(tmp_path):
    cfg = tmp_path / "refs.toml"
    cfg.write_text(
        '[[reference]]\nck = "checkpoints_entity/ck_000562083840.pt"\nlabel = "peak-562M"\n'
        '\n[[reference]]\nck = "checkpoints_entity/ck_000100000000.pt"\nlabel = "old-100M"\n'
    )
    refs = h2h_eval.load_references(cfg)
    assert refs == [
        {"ck": "checkpoints_entity/ck_000562083840.pt", "label": "peak-562M"},
        {"ck": "checkpoints_entity/ck_000100000000.pt", "label": "old-100M"},
    ]


def test_load_references_missing_label_defaults_to_ck(tmp_path):
    cfg = tmp_path / "refs.toml"
    cfg.write_text('[[reference]]\nck = "some/ck.pt"\n')
    refs = h2h_eval.load_references(cfg)
    assert refs == [{"ck": "some/ck.pt", "label": "some/ck.pt"}]


def test_load_references_missing_file_returns_empty(tmp_path):
    assert h2h_eval.load_references(tmp_path / "nonexistent.toml") == []


def test_load_references_empty_reference_table_returns_empty(tmp_path):
    cfg = tmp_path / "refs.toml"
    cfg.write_text("# no references yet\n")
    assert h2h_eval.load_references(cfg) == []


def test_project_h2h_references_config_seeded_with_peak_562m():
    # the real, committed configs/h2h_references.toml -- guards against the
    # seeded reference silently drifting or being deleted.
    refs = h2h_eval.load_references(REPO / "configs" / "h2h_references.toml")
    assert any(r["label"] == "peak-562M" for r in refs)
    peak = next(r for r in refs if r["label"] == "peak-562M")
    assert peak["ck"] == "checkpoints_entity/ck_000562083840.pt"


# ---------------------------------------------------------------------------
# cross-schema refusal
# ---------------------------------------------------------------------------

def test_require_compatible_allows_same_schema_v1():
    meta_a = {"path": "a.pt", "schema_version": 1, "heads": 4}
    meta_b = {"path": "b.pt", "schema_version": 1, "heads": 4}
    h2h_eval.require_compatible(meta_a, meta_b)  # must not raise


def test_require_compatible_allows_same_schema_v0():
    meta_a = {"path": "a.pt", "schema_version": 0, "heads": None}
    meta_b = {"path": "b.pt", "schema_version": 0, "heads": None}
    h2h_eval.require_compatible(meta_a, meta_b)  # must not raise


def test_require_compatible_refuses_cross_schema():
    meta_a = {"path": "v0.pt", "schema_version": 0, "heads": None}
    meta_b = {"path": "v1.pt", "schema_version": 1, "heads": 4}
    with pytest.raises(h2h_eval.SchemaMismatchError, match="cross-schema"):
        h2h_eval.require_compatible(meta_a, meta_b)


def test_require_compatible_refuses_cross_schema_error_names_both_checkpoints():
    meta_a = {"path": "checkpoints/v0.pt", "schema_version": 0, "heads": None}
    meta_b = {"path": "checkpoints_entity/v1.pt", "schema_version": 1, "heads": 4}
    with pytest.raises(h2h_eval.SchemaMismatchError) as exc:
        h2h_eval.require_compatible(meta_a, meta_b)
    assert "checkpoints/v0.pt" in str(exc.value)
    assert "checkpoints_entity/v1.pt" in str(exc.value)


def test_require_compatible_refuses_mismatched_v1_heads():
    meta_a = {"path": "a.pt", "schema_version": 1, "heads": 4}
    meta_b = {"path": "b.pt", "schema_version": 1, "heads": 8}
    with pytest.raises(h2h_eval.SchemaMismatchError, match="heads"):
        h2h_eval.require_compatible(meta_a, meta_b)


# ---------------------------------------------------------------------------
# run_vs_references: schema-mismatch refs are skipped, not fatal
# ---------------------------------------------------------------------------

def test_run_vs_references_skips_incompatible_ref_and_continues(tmp_path, monkeypatch, capsys):
    cfg = tmp_path / "refs.toml"
    cfg.write_text(
        '[[reference]]\nck = "bad_ref.pt"\nlabel = "wrong-schema"\n'
        '\n[[reference]]\nck = "good_ref.pt"\nlabel = "ok"\n'
    )
    history = tmp_path / "h2h_history.jsonl"

    metas = {
        "ck.pt": {"path": "ck.pt", "schema_version": 1, "heads": 4, "steps": 1},
        "bad_ref.pt": {"path": "bad_ref.pt", "schema_version": 0, "heads": None, "steps": 1},
        "good_ref.pt": {"path": "good_ref.pt", "schema_version": 1, "heads": 4, "steps": 1},
    }
    monkeypatch.setattr(h2h_eval, "checkpoint_meta", lambda p: metas[str(p)])
    monkeypatch.setattr(h2h_eval, "load_sd", lambda p: f"SD[{p}]")

    built = []

    def fake_build_runner(meta, arenas, seed):
        built.append(meta)
        return _FakeRunner([(3, 1), (2, 2)])

    monkeypatch.setattr(h2h_eval, "_build_runner", fake_build_runner)

    results = h2h_eval.run_vs_references("ck.pt", steps=100, arenas=2, seed=1,
                                         refs_config=cfg, history_path=history)

    # only the compatible ref actually played
    assert len(results) == 1
    assert results[0]["ref"]["label"] == "ok"
    assert len(built) == 1  # runner only built once, lazily, after the skip

    err = capsys.readouterr().err
    assert "wrong-schema" in err or "bad_ref.pt" in err

    lines = history.read_text().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["ref_label"] == "ok"
    assert row["goals_ck"] == 5 and row["goals_ref"] == 3  # (3+2, 1+2)


# ---------------------------------------------------------------------------
# real v1 engine smoke (tiny steps, mirrors test_league_tick.py's smoke)
# ---------------------------------------------------------------------------

REAL_CK_DIR = REPO / "checkpoints_entity"
REAL_CKS = sorted(REAL_CK_DIR.glob("ck_*.pt")) if REAL_CK_DIR.is_dir() else []


@pytest.mark.skipif(len(REAL_CKS) < 2,
                    reason="needs two real v1 checkpoints in checkpoints_entity/")
def test_real_v1_checkpoints_play_h2h_both_sides(tmp_path):
    ck_a, ck_b = str(REAL_CKS[0]), str(REAL_CKS[-1])
    meta_a = h2h_eval.checkpoint_meta(ck_a)
    meta_b = h2h_eval.checkpoint_meta(ck_b)
    assert meta_a["schema_version"] == 1 and meta_b["schema_version"] == 1
    h2h_eval.require_compatible(meta_a, meta_b)  # real checkpoints, must not raise

    mr = h2h_eval._build_runner(meta_a, arenas=2, seed=0)
    result = h2h_eval.play_h2h(mr, h2h_eval.load_sd(ck_a), h2h_eval.load_sd(ck_b), steps=60)

    # sanity: real engine ran, goals are non-negative ints, totals are
    # internally consistent -- not asserting a specific score (nondeterminism
    # across engine builds isn't the point of this smoke).
    a1, b1 = result["side1"]
    a2, b2 = result["side2"]
    assert all(isinstance(x, int) and x >= 0 for x in (a1, b1, a2, b2))
    assert result["a"] == a1 + a2 and result["b"] == b1 + b2


@pytest.mark.skipif(len(REAL_CKS) < 2,
                    reason="needs two real v1 checkpoints in checkpoints_entity/")
def test_real_v1_h2h_deterministic_with_fixed_seed():
    # Determinism claim from the module docstring: fixed seed -> identical
    # result. Two independent MatchRunners, same seed, same tiny tape.
    ck_a, ck_b = str(REAL_CKS[0]), str(REAL_CKS[-1])
    meta_a = h2h_eval.checkpoint_meta(ck_a)
    sd_a, sd_b = h2h_eval.load_sd(ck_a), h2h_eval.load_sd(ck_b)

    mr1 = h2h_eval._build_runner(meta_a, arenas=2, seed=7)
    result1 = h2h_eval.play_h2h(mr1, sd_a, sd_b, steps=60)

    mr2 = h2h_eval._build_runner(meta_a, arenas=2, seed=7)
    result2 = h2h_eval.play_h2h(mr2, sd_a, sd_b, steps=60)

    assert result1["side1"] == result2["side1"]
    assert result1["side2"] == result2["side2"]
