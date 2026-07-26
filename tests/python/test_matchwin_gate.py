"""Tests for scripts/matchwin_gate.py — the match-win promotion gate.

This gate DECIDES PROMOTIONS, so the two error-prone pure pieces are pinned
here: the side-order flip (order 2 plays the champion, so its records must be
inverted to the candidate's perspective before summing) and the win-share /
threshold arithmetic. A bug in either silently promotes or rejects.
"""
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import matchwin_gate as mg  # noqa: E402


# --- the side-order flip ----------------------------------------------------

def test_flip_swaps_each_pair_to_candidate_perspective():
    # order 2 records are (champion_goals, candidate_goals); flipped they must
    # become (candidate_goals, champion_goals).
    assert mg.flip_to_candidate([(3, 1), (0, 2)]) == [(1, 3), (2, 0)]


def test_flip_of_empty_is_empty():
    assert mg.flip_to_candidate([]) == []


def test_flip_is_its_own_inverse():
    m = [(5, 2), (0, 0), (1, 4)]
    assert mg.flip_to_candidate(mg.flip_to_candidate(m)) == m


def test_a_champion_win_in_order2_becomes_a_candidate_loss():
    # (champion 2, candidate 0) -> candidate perspective (0, 2): a loss.
    flipped = mg.flip_to_candidate([(2, 0)])
    a, b = flipped[0]
    assert a < b, "a champion win must read as a candidate loss after the flip"


# --- the aggregation arithmetic ---------------------------------------------

def test_aggregate_sums_both_orders():
    r = mg.aggregate((10, 2, 8), (12, 4, 6), 0.55)
    assert r["wins"] == 22 and r["draws"] == 6 and r["losses"] == 14
    assert r["n"] == 42


def test_draws_count_half():
    # 0W, 4D, 0L over both orders -> exactly 0.5.
    assert mg.aggregate((0, 2, 0), (0, 2, 0), 0.55)["share"] == pytest.approx(0.5)


def test_pass_requires_share_at_or_above_threshold():
    # 60% win share clears 0.55.
    r = mg.aggregate((30, 0, 20), (30, 0, 20), 0.55)
    assert r["share"] == pytest.approx(0.6) and r["verdict"] == "PASS"


def test_parity_is_a_fail_not_a_promotion():
    # Exactly the match-win arm's outcome: ~parity must NOT promote.
    r = mg.aggregate((25, 0, 25), (25, 0, 25), 0.55)
    assert r["share"] == pytest.approx(0.5) and r["verdict"] == "FAIL"


def test_threshold_boundary_is_inclusive():
    # share exactly == threshold passes (>=).
    r = mg.aggregate((55, 0, 45), (55, 0, 45), 0.55)
    assert r["share"] == pytest.approx(0.55) and r["verdict"] == "PASS"


def test_just_below_threshold_fails():
    r = mg.aggregate((54, 0, 46), (55, 0, 45), 0.55)
    assert r["share"] < 0.55 and r["verdict"] == "FAIL"


def test_no_completed_matches_is_none_share_not_zero():
    # 0 of 0 is not a total loss; share must be None, never 0.0, so the gate
    # never rejects on zero evidence.
    r = mg.aggregate((0, 0, 0), (0, 0, 0), 0.55)
    assert r["share"] is None and r["verdict"] == "FAIL"
    assert r["n"] == 0


def test_se_matches_binomial_at_half():
    r = mg.aggregate((160, 0, 160), (160, 0, 160), 0.55)
    assert r["se"] == pytest.approx(math.sqrt(0.25 / 640))


def test_draws_do_not_move_share_off_parity():
    # A draw-heavy split with equal W and L stays at 0.5 -- draws are neutral,
    # exactly why the iter-580 draw-degeneration read as ~parity-to-below, not
    # a win.
    r = mg.aggregate((100, 199, 359), (0, 0, 0), 0.55)
    # wins 100, draws 199, losses 359, n=658
    expected = (100 + 0.5 * 199) / 658
    assert r["share"] == pytest.approx(expected)
    assert r["verdict"] == "FAIL"


# --- the champion pointer (it MOVES) ----------------------------------------
#
# Gating against a net we already beat 0.837 always passes and measures nothing,
# so the champion has to follow configs/champion.toml rather than a hardcoded
# path. These pin the resolve/promote pair, because a bug here either measures
# against the wrong reference or corrupts the project's single champion pointer.

def _cfg(tmp_path, ck):
    p = tmp_path / "champion.toml"
    p.write_text(
        'schema_version = 1\n'
        f'champion_ck = "{ck}"\n'
        'promote_threshold = 0.52\n'
        '[min_games]\nsteps = 5400\narenas = 8\nseed = 11\nmin_total_goals = 20\n'
        '[watch]\ncandidate_dir = "checkpoints_entity"\npoll_seconds = 300\n'
        'consecutive_rejects_before_alert = 3\n'
        '[league]\nregistry = "league/r.jsonl"\nrun = "champion"\n'
        'reward_config = "configs/reward_v3.toml"\n'
        '[history]\npath = "logs/champion_history.jsonl"\n'
    )
    return p


def test_resolve_champion_reads_the_pointer(tmp_path):
    cfg = _cfg(tmp_path, "checkpoints_scratch/ck_001171502080.pt")
    assert mg.resolve_champion(cfg) == "checkpoints_scratch/ck_001171502080.pt"


def test_resolve_champion_falls_back_when_pointer_unreadable(tmp_path, capsys):
    """A gate is a MEASUREMENT: losing the ability to measure because a pointer
    file moved is worse than measuring against the seeded reference. (The
    mutating path, champion_gate.load_config, is strict instead -- see there.)"""
    got = mg.resolve_champion(tmp_path / "does_not_exist.toml")
    assert got == mg.FALLBACK_CHAMPION
    assert "falling back" in capsys.readouterr().out


def test_promote_moves_the_pointer_and_returns_the_previous(tmp_path):
    cfg = _cfg(tmp_path, "old/champ.pt")
    previous = mg.promote("new/better.pt", cfg)
    assert previous == "old/champ.pt"
    assert mg.resolve_champion(cfg) == "new/better.pt"


def test_promote_preserves_every_other_key(tmp_path):
    """The pointer rewrite must be surgical -- thresholds, min_games and the
    file's long rationale comments have to survive a promotion."""
    cfg = _cfg(tmp_path, "old/champ.pt")
    before = cfg.read_text()
    mg.promote("new/better.pt", cfg)
    after = cfg.read_text()
    assert after.count("\n") == before.count("\n"), "no lines added or dropped"
    for keep in ("promote_threshold = 0.52", "min_total_goals = 20",
                 "consecutive_rejects_before_alert = 3", "schema_version = 1"):
        assert keep in after, f"promotion clobbered {keep!r}"
    assert "old/champ.pt" not in after


def test_promote_is_idempotent(tmp_path):
    cfg = _cfg(tmp_path, "a.pt")
    mg.promote("b.pt", cfg)
    assert mg.promote("b.pt", cfg) == "b.pt"
    assert mg.resolve_champion(cfg) == "b.pt"


# --- the champion must rotate into the OPPONENT POOL, not just the pointer ----

def test_promote_enrols_the_new_champion_in_the_league(tmp_path, monkeypatch):
    """Moving the pointer alone would be 'rotating champion' in name only -- the
    trainer plays whatever is in the registry, not whatever the pointer says."""
    cfg = _cfg(tmp_path, "old/champ.pt")
    reg = tmp_path / "pool.jsonl"
    cfg.write_text(cfg.read_text().replace('registry = "league/r.jsonl"',
                                           f'registry = "{reg}"'))
    monkeypatch.setattr(mg, "_ck_provenance",
                        lambda ck: (1_171_502_080, 1, "construct_92_v1"))
    mg.promote("new/better.pt", cfg)

    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    assert [r["ck"] for r in rows] == ["new/better.pt"]
    assert rows[0]["steps"] == 1_171_502_080
    # schema_version MUST match the trainer's, or the entry is silently never
    # selected (v0 and v1 policies cannot play each other -- different obs).
    assert rows[0]["schema_version"] == 1
    assert rows[0]["run"] == "champion"
    assert rows[0]["action_table"] == "construct_92_v1"


def test_promote_records_the_candidates_action_table_not_the_default(tmp_path, monkeypatch):
    """The SAME hazard as schema_version, one level down and invisible in it:
    schema_version is 1 for both v1 tables. Registry.add defaults action_table
    to the 92-row one, so a promoted 104-row v1-air champion stamped with the
    default would (a) sail through play_entries' cross-table refusal, which
    compares exactly this field, and (b) be filtered out of its own run's
    league by choose_opponents -- a league that stays dead and never says
    why."""
    cfg = _cfg(tmp_path, "old/champ.pt")
    reg = tmp_path / "pool.jsonl"
    cfg.write_text(cfg.read_text().replace('registry = "league/r.jsonl"',
                                           f'registry = "{reg}"'))
    monkeypatch.setattr(mg, "_ck_provenance", lambda ck: (9, 1, "construct_104_v1air"))
    mg.promote("v9/air.pt", cfg)

    rows = [json.loads(l) for l in reg.read_text().splitlines() if l.strip()]
    assert rows[0]["action_table"] == "construct_104_v1air"

    # ...and that entry is then refused against a 92-row champion instead of
    # being handed to an engine that cannot decode one of the two.
    from construct.league.matches import play_entries
    champ = {"ck": "old/champ.pt", "schema_version": 1}  # pre-v9: no field at all
    with pytest.raises(ValueError, match="cross-action-table"):
        play_entries(object(), rows[0], champ)


def test_promote_still_moves_the_pointer_if_league_enrolment_fails(tmp_path, monkeypatch, capsys):
    """The pointer is the authoritative record; the pool is a convenience. A
    registry problem must not leave the gate measuring against a stale champion."""
    cfg = _cfg(tmp_path, "old/champ.pt")

    def boom(ck):
        raise RuntimeError("unreadable checkpoint")
    monkeypatch.setattr(mg, "_ck_provenance", boom)

    previous = mg.promote("new/better.pt", cfg)
    assert previous == "old/champ.pt"
    assert mg.resolve_champion(cfg) == "new/better.pt", "pointer must still move"
    assert "league enrolment failed" in capsys.readouterr().out


def test_promote_can_skip_the_league(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, "old/champ.pt")
    called = []
    monkeypatch.setattr(mg, "_ck_provenance",
                        lambda ck: called.append(ck) or (1, 1, "construct_92_v1"))
    mg.promote("new/better.pt", cfg, add_to_league=False)
    assert called == [], "add_to_league=False must not touch the pool"
    assert mg.resolve_champion(cfg) == "new/better.pt"


# --- the action table both sides must share ----------------------------------

def _tiny_ck(tmp_path, name, table):
    import torch
    from construct.learn.model_v1 import EntityPolicyNet

    net = EntityPolicyNet(d_model=16, layers=1, heads=2, ff=32, action_table=table)
    p = tmp_path / name
    torch.save({"model": net.state_dict(), "total_steps": 1, "schema_version": 1,
                "config": {"net": {"d_model": 16, "layers": 1, "heads": 2, "ff": 32}}}, p)
    return str(p)


def test_gate_refuses_a_cross_action_table_pair(tmp_path):
    """The gate drives both sides through ONE engine and one engine binds one
    decode table, so a 104-row v1-air candidate against the 92-row champion is
    impossible, not merely hard. schema_version is 1 for both and cannot catch
    it; unrefused, it is an out-of-bounds index in a worker thread."""
    from construct._engine import action_table_v1, action_table_v1_air

    cand = _tiny_ck(tmp_path, "cand_air.pt", action_table_v1_air())
    champ = _tiny_ck(tmp_path, "champ_92.pt", action_table_v1())
    with pytest.raises(SystemExit, match="gate refused"):
        mg._gate_action_table(cand, champ)
    # a matched pair returns the shared name, which selects the schema file
    assert mg._gate_action_table(cand, cand) == "construct_104_v1air"
    assert mg._gate_action_table(champ, champ) == "construct_92_v1"


# --- team sizes (--mode) -----------------------------------------------------
#
# The gate is NET vs NET, so it works at any team size; the hazards are all about
# not letting a 2v2 number stand in for the 1v1 ruler. These pin the two that
# would be silent: promotion at m>1, and the history row's scale.

def _parse(argv):
    """Build the parsed args without running a gate (main() only reaches the
    engine after parse + the promotion refusal)."""
    import contextlib
    import io

    seen = {}

    def spy(candidate, champion, arenas, steps, seed, threshold, mode=1):
        seen.update(locals())
        raise SystemExit(99)                 # stop before touching an engine

    old = mg.gate
    mg.gate = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            mg.main(argv)
    except SystemExit as e:
        if e.code != 99:
            raise
    finally:
        mg.gate = old
    return seen


def test_mode_defaults_to_1(tmp_path):
    """Every existing invocation must be unchanged, so the flag's absence has to
    mean exactly what the code did before it existed."""
    cfg = _cfg(tmp_path, "old/champ.pt")
    assert _parse(["cand.pt", "--champion-config", str(cfg)])["mode"] == 1


def test_mode_2_is_threaded_through(tmp_path):
    cfg = _cfg(tmp_path, "old/champ.pt")
    assert _parse(["cand.pt", "--mode", "2", "--champion-config", str(cfg)])["mode"] == 2


def test_mode_4_is_rejected_by_argparse(tmp_path):
    with pytest.raises(SystemExit) as e:
        _parse(["cand.pt", "--mode", "4"])
    assert e.value.code == 2


def test_promote_if_pass_above_1v1_is_refused_before_any_compute(tmp_path, capsys):
    """The refusal is at PARSE time on purpose: a 2v2 gate is 20+ minutes of
    engine time, and champion_ck feeds the KL anchor, the league pool seed and
    the deploy candidate -- all 1v1 quantities. The pointer must not move."""
    cfg = _cfg(tmp_path, "old/champ.pt")
    with pytest.raises(SystemExit) as e:
        mg.main(["cand.pt", "--mode", "2", "--promote-if-pass",
                 "--champion-config", str(cfg)])
    assert e.value.code == 2
    assert "1v1-only" in capsys.readouterr().err
    assert mg.resolve_champion(cfg) == "old/champ.pt", "pointer must not move"


def test_promote_if_pass_at_1v1_still_promotes(tmp_path, monkeypatch, capsys):
    """The refusal must not have broken the real path."""
    cfg = _cfg(tmp_path, "old/champ.pt")
    # Keep league enrolment inside tmp_path -- promote() really does write the
    # registry, and a test must never touch the project's opponent pool.
    cfg.write_text(cfg.read_text().replace('registry = "league/r.jsonl"',
                                           f'registry = "{tmp_path / "pool.jsonl"}"'))
    monkeypatch.setattr(mg, "gate", lambda *a, **k: {
        "wins": 400, "draws": 40, "losses": 200, "n": 640, "share": 0.656,
        "se": 0.0198, "threshold": 0.55, "verdict": "PASS",
        "order1": (200, 20, 100), "order2": (200, 20, 100),
        "records": 640, "short_records": 3, "short_frac": 3 / 640})
    monkeypatch.setattr(mg, "_append_history", lambda *a, **k: None)
    monkeypatch.setattr(mg, "_ck_provenance", lambda ck: (1, 1, "construct_92_v1"))
    assert mg.main(["cand.pt", "--champion-config", str(cfg)]) == 0
    assert mg.resolve_champion(cfg) is not None
    mg.main(["cand.pt", "--promote-if-pass", "--champion-config", str(cfg)])
    assert mg.resolve_champion(cfg) == "cand.pt"


def test_history_row_carries_mode_and_the_blowup_census(tmp_path, monkeypatch):
    """One history file for every mode (the threshold is the same at every team
    size), so the row itself has to say which scale it is on -- and carry the
    short-record census, the only after-the-fact way to tell a blowup-contaminated
    run from a genuinely close one."""
    import argparse
    import json

    args = argparse.Namespace(candidate="c.pt", champion="ch.pt", arenas=32,
                              steps=45000, seed=11, mode=2)
    r = {"wins": 300, "draws": 100, "losses": 240, "n": 640, "share": 0.547,
         "se": 0.0198, "threshold": 0.55, "verdict": "FAIL", "promoted": False,
         "records": 640, "short_records": 71}
    monkeypatch.setattr(mg, "__file__", str(tmp_path / "scripts" / "matchwin_gate.py"))
    mg._append_history(args, r)
    row = json.loads((tmp_path / "logs" / "matchwin_history.jsonl").read_text())
    assert row["mode"] == 2
    assert row["records"] == 640 and row["short_records"] == 71


# --- the blowup census line --------------------------------------------------

def test_short_record_line_reports_the_count_and_fraction():
    line = mg._short_record_line({"records": 640, "short_records": 7, "share": 0.84})
    assert "7/640 (1.1%)" in line
    assert "heavy census" not in line, "1.1% is below the 2% annotation bar"


def test_short_record_line_reports_n_full_beside_n():
    """A blowup adds a RECORD without adding a match, so n (and therefore
    aggregate's se = sqrt(0.25/n)) is inflated by the census. n_full is what lets
    a reader see how much of the denominator is truncated matches."""
    line = mg._short_record_line({"records": 640, "short_records": 40, "share": 0.84})
    assert "n_full=600" in line


def test_short_record_line_claims_no_direction_for_the_contamination():
    """The old census said short records were 0-0 draws, so contamination could
    only pull the share toward 0.5 and "never cause a false promotion".
    episode.rs:779-807 zeroes only the blowup step's reward and returns before
    start_match(), so the record carries the partial score -- a blowup at 2-1 is
    a WIN. Neither a magnitude nor a direction is knowable; printing one was the
    bug. Both a strong candidate and a weak one must get the same unsigned flag."""
    strong = mg._short_record_line({"records": 100, "short_records": 22, "share": 0.84})
    weak = mg._short_record_line({"records": 100, "short_records": 60, "share": 0.31})
    for line in (strong, weak):
        assert "heavy census" in line and "idle box" in line
        assert "unknown direction" in line
        assert "0-0" in line and "not 0-0" in line
        assert "pulled toward 0.5" not in line
        assert "biased LOW" not in line and "biased HIGH" not in line


def test_short_record_line_needs_no_share_to_flag_a_heavy_census():
    """The flag is about the TAPE, not the verdict: it must still fire on a run
    whose share never got computed."""
    assert "heavy census" in mg._short_record_line(
        {"records": 100, "short_records": 22, "share": None})


def test_short_record_line_survives_no_records():
    assert "n/a" in mg._short_record_line({"records": 0, "short_records": 0,
                                           "share": None})
