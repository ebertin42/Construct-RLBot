"""Tests for scripts/champion_gate.py, the champion promotion gate.

Everything here is PURE: the match runner (champion_gate.run_match) is
monkeypatched to return canned aggregates, so no engine is ever built, no real
checkpoint is ever loaded, and the whole file runs in well under a second.
The numbers used as fixtures are the measured ones from
docs/training-journal.md 2026-07-19 ~17:40 -> 2026-07-20 ~00:30 (arms A-D, the
null control) so the thresholds are tested against reality, not invented data.
"""
import json
import sys
import tomllib
from pathlib import Path

import pytest

# scripts/ isn't a package (same pattern as tests/python/test_h2h.py).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import champion_gate  # noqa: E402
import h2h_eval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

CHAMPION = "checkpoints_entity/ck_000320471040.pt"


# ---------------------------------------------------------------------------
# fixtures: a writable copy of the real config
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_path(tmp_path):
    """A copy of the REAL configs/champion.toml in a tmp dir -- so the tests
    exercise the committed file's actual shape, but never mutate it."""
    dst = tmp_path / "champion.toml"
    dst.write_text((REPO / "configs" / "champion.toml").read_text())
    return dst


@pytest.fixture
def cfg(cfg_path, tmp_path):
    c = champion_gate.load_config(cfg_path)
    c["history"] = str(tmp_path / "champion_history.jsonl")
    c["candidate_dir"] = str(tmp_path / "candidates")
    return c


def _agg(side1, side2):
    """Build a real h2h_eval aggregate from two side-order scores."""
    return h2h_eval.aggregate_sides(side1, side2)


def _mock_match(monkeypatch, side1, side2, record=None):
    def fake(candidate, champion, steps, arenas, seed):
        if record is not None:
            record.append((candidate, champion, steps, arenas, seed))
        r = _agg(side1, side2)
        r["meta_candidate"] = {"path": candidate, "schema_version": 1, "heads": 4, "steps": 999}
        r["meta_champion"] = {"path": champion, "schema_version": 1, "heads": 4, "steps": 320471040}
        return r
    monkeypatch.setattr(champion_gate, "run_match", fake)


# ---------------------------------------------------------------------------
# config loading: fail loudly, never default
# ---------------------------------------------------------------------------

def test_load_config_reads_real_committed_config():
    c = champion_gate.load_config(REPO / "configs" / "champion.toml")
    assert c["champion_ck"] == CHAMPION          # the measured strongest ck
    assert c["promote_threshold"] == 0.52
    assert c["steps"] == 5400 and c["arenas"] == 8   # matches the h2h harness
    assert c["registry"].endswith("registry_armc.jsonl")


def test_load_config_missing_file_raises_not_defaults(tmp_path):
    with pytest.raises(champion_gate.ChampionConfigError, match="not found"):
        champion_gate.load_config(tmp_path / "nope.toml")


def test_load_config_corrupt_toml_raises(tmp_path):
    bad = tmp_path / "champion.toml"
    bad.write_text('champion_ck = "unterminated\npromote_threshold = ')
    with pytest.raises(champion_gate.ChampionConfigError, match="corrupt"):
        champion_gate.load_config(bad)


def test_load_config_missing_required_key_raises(tmp_path):
    bad = tmp_path / "champion.toml"
    bad.write_text('promote_threshold = 0.52\n')
    with pytest.raises(champion_gate.ChampionConfigError, match="champion_ck"):
        champion_gate.load_config(bad)


def test_load_config_missing_section_raises(tmp_path, cfg_path):
    text = cfg_path.read_text().split("[min_games]")[0]
    bad = tmp_path / "trunc.toml"
    bad.write_text(text)
    with pytest.raises(champion_gate.ChampionConfigError, match=r"\[min_games\]"):
        champion_gate.load_config(bad)


def test_load_config_rejects_threshold_at_or_below_parity(tmp_path, cfg_path):
    bad = tmp_path / "champion.toml"
    bad.write_text(cfg_path.read_text().replace("promote_threshold = 0.52",
                                                "promote_threshold = 0.50"))
    with pytest.raises(champion_gate.ChampionConfigError, match="parity"):
        champion_gate.load_config(bad)


def test_load_config_rejects_out_of_range_threshold(tmp_path, cfg_path):
    bad = tmp_path / "champion.toml"
    bad.write_text(cfg_path.read_text().replace("promote_threshold = 0.52",
                                                "promote_threshold = 1.7"))
    with pytest.raises(champion_gate.ChampionConfigError, match="goal share"):
        champion_gate.load_config(bad)


# ---------------------------------------------------------------------------
# threshold logic: below / at / above, tie, zero-goal
# ---------------------------------------------------------------------------

def test_evaluate_above_threshold_passes():
    v, why = champion_gate.evaluate(0.55, 200, 0.52, 20)
    assert v == champion_gate.PASS
    assert "55.0%" in why and "52.0%" in why


def test_evaluate_exactly_at_threshold_passes():
    # >= is deliberate: the bar is "reach 52%", not "beat 52%".
    v, _ = champion_gate.evaluate(0.52, 200, 0.52, 20)
    assert v == champion_gate.PASS


def test_evaluate_just_below_threshold_fails():
    v, why = champion_gate.evaluate(0.5199, 200, 0.52, 20)
    assert v == champion_gate.FAIL
    assert "<" in why


def test_evaluate_tie_fails_incumbent_keeps_belt():
    v, why = champion_gate.evaluate(0.5, 200, 0.52, 20)
    assert v == champion_gate.FAIL
    assert "tie" in why.lower()


def test_evaluate_tie_fails_even_if_threshold_were_at_parity():
    # guard the rule itself, independent of the threshold arithmetic
    v, why = champion_gate.evaluate(0.5, 200, 0.5, 20)
    assert v == champion_gate.FAIL
    assert "tie" in why.lower()


def test_evaluate_zero_goal_match_fails():
    v, why = champion_gate.evaluate(None, 0, 0.52, 20)
    assert v == champion_gate.FAIL
    assert "0-0" in why


def test_evaluate_too_few_goals_fails_even_at_100_percent_share():
    # a 3-0 shutout is 100% share and still not a promotion: three goals is
    # not a measurement of anything.
    v, why = champion_gate.evaluate(1.0, 3, 0.52, 20)
    assert v == champion_gate.FAIL
    assert "min_total_goals" in why


def test_evaluate_null_control_share_fails():
    # journal 2026-07-19 ~22:00: 320M vs ITSELF summed to 51.1%. The threshold
    # exists precisely so that number does NOT promote.
    v, _ = champion_gate.evaluate(0.511, 178, 0.52, 20)
    assert v == champion_gate.FAIL


@pytest.mark.parametrize("share", [0.104, 0.220, 0.255, 0.271, 0.492])
def test_evaluate_every_measured_arm_fails(share):
    # arms A-E as measured; none of them earned the belt.
    v, _ = champion_gate.evaluate(share, 180, 0.52, 20)
    assert v == champion_gate.FAIL


# ---------------------------------------------------------------------------
# both-orders aggregation (imported from h2h_eval, not reimplemented)
# ---------------------------------------------------------------------------

def test_gate_uses_h2h_eval_aggregation_not_a_copy():
    assert champion_gate.aggregate_sides is h2h_eval.aggregate_sides
    assert champion_gate.play_h2h is h2h_eval.play_h2h
    assert champion_gate.require_compatible is h2h_eval.require_compatible


def test_both_orders_are_summed_before_the_verdict(monkeypatch, cfg):
    # journal arm D: 50-43 / 43-53 -> 93/96 = 49.2%. Order 1 alone is 53.8%
    # (would PASS); order 2 alone is 44.8% (would FAIL). Summed: FAIL.
    assert champion_gate.evaluate(50 / 93, 93, 0.52, 20)[0] == champion_gate.PASS
    _mock_match(monkeypatch, (50, 43), (43, 53))
    row = champion_gate.gate_one(cfg, "armD.pt", promote_if_pass=False, quiet=True)
    assert row["goals_c"] == 93 and row["goals_champ"] == 96
    assert row["share"] == pytest.approx(93 / 189)
    assert row["verdict"] == champion_gate.FAIL


def test_gate_passes_the_configured_match_size_to_the_runner(monkeypatch, cfg):
    calls = []
    _mock_match(monkeypatch, (100, 50), (100, 50), record=calls)
    champion_gate.gate_one(cfg, "cand.pt", promote_if_pass=False, quiet=True)
    assert calls == [("cand.pt", cfg["champion_ck"], 5400, 8, 11)]


# ---------------------------------------------------------------------------
# "does not promote without --promote-if-pass"
# ---------------------------------------------------------------------------

def test_pass_without_promote_flag_does_not_move_the_pointer(monkeypatch, cfg, cfg_path):
    _mock_match(monkeypatch, (100, 50), (100, 50))   # 66.7% -- a clear PASS
    row = champion_gate.gate_one(cfg, "winner.pt", promote_if_pass=False, quiet=True)
    assert row["verdict"] == champion_gate.PASS
    assert row["promoted"] is False
    assert "--promote-if-pass" in row["reason"]
    # champion pointer on disk is untouched
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == CHAMPION


def test_pass_with_promote_flag_moves_the_pointer(monkeypatch, cfg, cfg_path):
    _mock_match(monkeypatch, (100, 50), (100, 50))
    monkeypatch.setattr(champion_gate, "add_to_league", lambda *a, **k: None)
    row = champion_gate.gate_one(cfg, "winner.pt", promote_if_pass=True, quiet=True)
    assert row["promoted"] is True
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == "winner.pt"


def test_fail_with_promote_flag_still_does_not_promote(monkeypatch, cfg, cfg_path):
    _mock_match(monkeypatch, (28, 70), (28, 94))     # arm A, 25.5%
    row = champion_gate.gate_one(cfg, "armA.pt", promote_if_pass=True, quiet=True)
    assert row["verdict"] == champion_gate.FAIL and row["promoted"] is False
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == CHAMPION


def test_zero_goal_match_never_promotes(monkeypatch, cfg, cfg_path):
    _mock_match(monkeypatch, (0, 0), (0, 0))
    row = champion_gate.gate_one(cfg, "dud.pt", promote_if_pass=True, quiet=True)
    assert row["verdict"] == champion_gate.FAIL and row["promoted"] is False
    assert row["share"] is None
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == CHAMPION


def test_should_promote_truth_table():
    assert champion_gate.should_promote(champion_gate.PASS, True) is True
    assert champion_gate.should_promote(champion_gate.PASS, False) is False
    assert champion_gate.should_promote(champion_gate.FAIL, True) is False
    assert champion_gate.should_promote(champion_gate.FAIL, False) is False


# ---------------------------------------------------------------------------
# promotion atomicity: a crash between write and rename leaves the old
# champion intact
# ---------------------------------------------------------------------------

def test_stage_does_not_touch_the_live_config(cfg_path):
    tmp = champion_gate.stage_champion_ck(cfg_path, "new_champion.pt")
    # simulated crash here: process dies before commit_staged
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == CHAMPION
    assert tomllib.loads(tmp.read_text())["champion_ck"] == "new_champion.pt"
    assert tmp != cfg_path


def test_commit_after_stage_swaps_in_the_new_champion(cfg_path):
    tmp = champion_gate.stage_champion_ck(cfg_path, "new_champion.pt")
    champion_gate.commit_staged(tmp, cfg_path)
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == "new_champion.pt"
    assert not tmp.exists()          # os.replace consumed it


def test_crash_during_league_append_leaves_champion_promoted(monkeypatch, cfg, cfg_path, capsys):
    """Pointer is written BEFORE the league append, so a failing registry
    write can't leave a half-promoted state -- it warns and moves on."""
    _mock_match(monkeypatch, (100, 50), (100, 50))

    def boom(*a, **k):
        raise OSError("registry disk full")
    monkeypatch.setattr(champion_gate, "add_to_league", boom)
    row = champion_gate.gate_one(cfg, "winner.pt", promote_if_pass=True, quiet=True)
    assert row["promoted"] is True
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == "winner.pt"
    assert "league registry append failed" in capsys.readouterr().err


def test_rewrite_preserves_comments_and_other_keys(cfg_path):
    original = cfg_path.read_text()
    champion_gate.write_champion_ck(cfg_path, "checkpoints_entity/ck_999.pt")
    after = cfg_path.read_text()
    # the justification comment (the reason 0.52 is 0.52) must survive
    assert "null control" in after
    assert "51.1%" in after
    assert after.count("promote_threshold = 0.52") == 1
    reparsed = champion_gate.load_config(cfg_path)
    assert reparsed["champion_ck"] == "checkpoints_entity/ck_999.pt"
    assert reparsed["promote_threshold"] == 0.52
    assert len(after.splitlines()) == len(original.splitlines())


def test_rewrite_refuses_config_without_champion_key():
    with pytest.raises(champion_gate.ChampionConfigError, match="champion_ck"):
        champion_gate._rewrite_champion_text("promote_threshold = 0.52\n", "x.pt")


# ---------------------------------------------------------------------------
# manual promote / reject
# ---------------------------------------------------------------------------

def test_manual_promote_records_reason_and_manual_flag(monkeypatch, cfg, cfg_path):
    monkeypatch.setattr(champion_gate, "add_to_league", lambda *a, **k: None)
    monkeypatch.setattr(champion_gate, "checkpoint_meta",
                        lambda p: {"path": p, "schema_version": 1, "heads": 4, "steps": 42})
    rc = champion_gate.cmd_manual(cfg, "handpicked.pt", "Elliot: BC prior repertoire", promote=True)
    assert rc == 0
    row = champion_gate.read_history(cfg["history"])[-1]
    assert row["manual"] is True
    assert row["promoted"] is True
    assert row["verdict"] == champion_gate.PASS
    assert row["reason"] == "Elliot: BC prior repertoire"
    assert row["candidate"] == "handpicked.pt"
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == "handpicked.pt"


def test_manual_reject_records_reason_and_leaves_champion(cfg, cfg_path):
    rc = champion_gate.cmd_manual(cfg, "suspect.pt", "boost-masher, 16% repertoire overlap",
                                  promote=False)
    assert rc == 0
    row = champion_gate.read_history(cfg["history"])[-1]
    assert row["manual"] is True and row["promoted"] is False
    assert row["verdict"] == champion_gate.FAIL
    assert row["reason"] == "boost-masher, 16% repertoire overlap"
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == CHAMPION


def test_manual_requires_a_reason(cfg, cfg_path):
    assert champion_gate.cmd_manual(cfg, "x.pt", "", promote=True) == 1
    assert champion_gate.read_history(cfg["history"]) == []
    assert tomllib.loads(cfg_path.read_text())["champion_ck"] == CHAMPION


def test_measured_rows_are_distinguishable_from_manual_ones(monkeypatch, cfg):
    _mock_match(monkeypatch, (28, 70), (28, 94))
    champion_gate.gate_one(cfg, "armA.pt", promote_if_pass=False, quiet=True)
    champion_gate.cmd_manual(cfg, "armA.pt", "keeping it around anyway", promote=False)
    rows = champion_gate.read_history(cfg["history"])
    assert [r["manual"] for r in rows] == [False, True]


def test_cli_promote_requires_reason():
    with pytest.raises(SystemExit):
        champion_gate.build_parser().parse_args(["promote", "ck.pt"])


def test_cli_gate_defaults_to_measure_only():
    args = champion_gate.build_parser().parse_args(["gate", "ck.pt"])
    assert args.promote_if_pass is False


# ---------------------------------------------------------------------------
# history jsonl round-trip + h2h schema overlap
# ---------------------------------------------------------------------------

def test_history_row_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "logs" / "champion_history.jsonl"   # also: parent creation
    row = champion_gate.history_row(
        "cand.pt", CHAMPION, 93, 96, champion_gate.FAIL, False,
        "share 49.2% < threshold 52.0%", steps=5400, seed=11, arenas=8,
        threshold=0.52, ts=1784200000)
    champion_gate.append_history(path, row)
    back = champion_gate.read_history(path)
    assert back == [row]
    assert json.loads(path.read_text().splitlines()[0]) == row


def test_history_row_has_the_required_gate_keys():
    row = champion_gate.history_row("c.pt", "ch.pt", 10, 20, champion_gate.FAIL,
                                    False, "why", ts=1)
    for key in ("ts", "candidate", "champion", "goals_c", "goals_champ",
                "share", "verdict", "promoted", "reason"):
        assert key in row


def test_history_row_is_h2h_schema_compatible(tmp_path):
    """The dashboard's parse_h2h_history must read champion history unchanged
    (schema: ts, ck, ref, ref_label, goals_ck, goals_ref, share, steps, seed)."""
    sys.path.insert(0, str(_SCRIPTS_DIR))
    import dashboard  # noqa: PLC0415

    path = tmp_path / "champion_history.jsonl"
    champion_gate.append_history(path, champion_gate.history_row(
        "checkpoints_entity/ck_999.pt", CHAMPION, 93, 96, champion_gate.FAIL,
        False, "why", steps=5400, seed=11, arenas=8, threshold=0.52, ts=1784200000))
    rows = dashboard.parse_h2h_history(path.read_text())
    assert len(rows) == 1
    assert rows[0]["ck"] == "ck_999.pt"
    assert rows[0]["ref"] == "ck_000320471040.pt"
    assert rows[0]["ref_label"] == "champion"
    assert rows[0]["goals_ck"] == 93 and rows[0]["goals_ref"] == 96
    assert rows[0]["share"] == pytest.approx(93 / 189)
    assert rows[0]["steps"] == 5400 and rows[0]["seed"] == 11


def test_read_history_skips_truncated_tail(tmp_path):
    path = tmp_path / "h.jsonl"
    good = champion_gate.history_row("a.pt", "b.pt", 1, 2, champion_gate.FAIL,
                                     False, "r", ts=1)
    path.write_text(json.dumps(good) + "\n" + '{"ts": 2, "candi')
    assert champion_gate.read_history(path) == [good]


def test_read_history_missing_file_is_empty(tmp_path):
    assert champion_gate.read_history(tmp_path / "nope.jsonl") == []


def test_promotions_filters_to_promoted_rows():
    rows = [
        champion_gate.history_row("a.pt", "c.pt", 1, 9, champion_gate.FAIL, False, "r", ts=1),
        champion_gate.history_row("b.pt", "c.pt", 9, 1, champion_gate.PASS, True, "r", ts=2),
    ]
    assert [r["candidate"] for r in champion_gate.promotions(rows)] == ["b.pt"]


# ---------------------------------------------------------------------------
# cross-schema refusal
# ---------------------------------------------------------------------------

def test_gate_refuses_cross_schema_pair(monkeypatch, cfg, capsys):
    def refuse(*a, **k):
        raise h2h_eval.SchemaMismatchError("cross-schema h2h refused: v0 vs v1")
    monkeypatch.setattr(champion_gate, "run_match", refuse)
    rc = champion_gate.cmd_gate(cfg, "v0_ck.pt", promote_if_pass=True)
    assert rc == 1
    assert "cross-schema" in capsys.readouterr().err
    # nothing recorded, nothing promoted
    assert champion_gate.read_history(cfg["history"]) == []


def test_run_match_refuses_before_building_an_engine(monkeypatch, cfg):
    """require_compatible must fire before _build_runner is ever called."""
    metas = {
        "v1.pt": {"path": "v1.pt", "schema_version": 1, "heads": 4, "steps": 1},
        "v0.pt": {"path": "v0.pt", "schema_version": 0, "heads": None, "steps": 1},
    }
    monkeypatch.setattr(champion_gate, "checkpoint_meta", lambda p: metas[str(p)])

    def never(*a, **k):
        raise AssertionError("engine was built for a cross-schema pair")
    monkeypatch.setattr(h2h_eval, "_build_runner", never)
    with pytest.raises(h2h_eval.SchemaMismatchError, match="cross-schema"):
        champion_gate.run_match("v0.pt", "v1.pt", 100, 2, 1)


def test_watch_records_schema_mismatch_as_a_fail_and_keeps_going(monkeypatch, cfg, tmp_path):
    cand_dir = Path(cfg["candidate_dir"])
    cand_dir.mkdir(parents=True)
    (cand_dir / "ck_000000000001.pt").write_text("x")

    def refuse(*a, **k):
        raise h2h_eval.SchemaMismatchError("cross-schema h2h refused")
    monkeypatch.setattr(champion_gate, "run_match", refuse)
    monkeypatch.setattr(champion_gate, "load_config", lambda p: cfg)
    champion_gate.cmd_watch(cfg, auto_promote=True, once=True)

    rows = champion_gate.read_history(cfg["history"])
    assert len(rows) == 1
    assert rows[0]["verdict"] == champion_gate.FAIL
    assert rows[0]["promoted"] is False
    assert "schema mismatch" in rows[0]["reason"]


# ---------------------------------------------------------------------------
# candidate discovery + reject alerting
# ---------------------------------------------------------------------------

def test_pending_candidates_excludes_gated_and_the_champion(tmp_path):
    d = tmp_path / "cks"
    d.mkdir()
    for name in ("ck_000000000001.pt", "ck_000000000002.pt", "ck_000000000003.pt"):
        (d / name).write_text("x")
    champ = d / "ck_000000000003.pt"
    rows = [champion_gate.history_row(str(d / "ck_000000000001.pt"), str(champ),
                                      1, 9, champion_gate.FAIL, False, "r", ts=1)]
    pend = champion_gate.pending_candidates(d, rows, str(champ))
    assert [p.name for p in pend] == ["ck_000000000002.pt"]


def test_pending_candidates_missing_dir_is_empty(tmp_path):
    assert champion_gate.pending_candidates(tmp_path / "nope", [], None) == []


def test_watch_gates_each_new_candidate_once(monkeypatch, cfg):
    cand_dir = Path(cfg["candidate_dir"])
    cand_dir.mkdir(parents=True)
    for name in ("ck_000000000001.pt", "ck_000000000002.pt"):
        (cand_dir / name).write_text("x")
    _mock_match(monkeypatch, (10, 90), (10, 90))     # 10% -- arm-B-grade
    monkeypatch.setattr(champion_gate, "load_config", lambda p: cfg)
    champion_gate.cmd_watch(cfg, auto_promote=True, once=True)
    rows = champion_gate.read_history(cfg["history"])
    assert len(rows) == 2
    assert all(r["verdict"] == champion_gate.FAIL and not r["promoted"] for r in rows)
    # a second pass has nothing left to do
    champion_gate.cmd_watch(cfg, auto_promote=True, once=True)
    assert len(champion_gate.read_history(cfg["history"])) == 2


def test_consecutive_rejects_counts_the_tail():
    def r(verdict):
        return champion_gate.history_row("a.pt", "b.pt", 1, 1, verdict, False, "x", ts=1)
    assert champion_gate.consecutive_rejects([]) == 0
    assert champion_gate.consecutive_rejects([r(champion_gate.FAIL)] * 3) == 3
    assert champion_gate.consecutive_rejects(
        [r(champion_gate.FAIL), r(champion_gate.PASS), r(champion_gate.FAIL)]) == 1


def test_alert_fires_after_configured_consecutive_rejects(cfg, capsys):
    rows = [champion_gate.history_row("a.pt", "b.pt", 1, 9, champion_gate.FAIL,
                                      False, "x", ts=i) for i in range(3)]
    assert champion_gate._print_alert(cfg, rows) is True
    assert "ALERT" in capsys.readouterr().err
    assert champion_gate._print_alert(cfg, rows[:2]) is False


# ---------------------------------------------------------------------------
# status renders without an engine
# ---------------------------------------------------------------------------

def test_status_runs_and_shows_champion_and_threshold(monkeypatch, cfg, capsys):
    _mock_match(monkeypatch, (28, 70), (28, 94))
    champion_gate.gate_one(cfg, "armA.pt", promote_if_pass=False, quiet=True)
    capsys.readouterr()
    assert champion_gate.cmd_status(cfg) == 0
    out = capsys.readouterr().out
    assert "ck_000320471040.pt" in out
    assert "52.0%" in out
    assert "armA.pt" in out
    assert "FAIL" in out
