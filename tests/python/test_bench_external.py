"""Tests for scripts/bench_external.py, the external-bot (Nexto/Necto)
benchmark harness.

NO NETWORK: fetch_external_bot() is the only network-touching function in
the module and is intentionally NOT exercised here (same policy as
test_h2h.py leaving MatchRunner/engine construction to a real-checkpoint
smoke, except here there's no offline equivalent of a real match at all --
see docs/external-bench.md). Everything below is pure-function coverage:
checksum verification (against the SAME real bytes the module's registry
was pinned from -- fetched once during development, embedded here so the
test has no network dependency of its own), match-config generation, and
match-result parsing."""
import hashlib
import json
import sys
from pathlib import Path

import pytest
import tomllib

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import bench_external  # noqa: E402
import h2h_eval  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

# The real bytes of VirxEC/NectoFamily's nexto/bot.toml at commit
# 0bdb6b49072f6f3829319e68bd6210a0ca4b24a2 -- exactly what
# bench_external.EXTERNAL_BOTS["nexto"]["files"]["bot.toml"]["sha256"] was
# computed from. Embedding the real content (not a fabricated fixture) lets
# test_verify_asset_accepts_the_real_pinned_bytes below prove the pinned
# checksum in the registry actually matches upstream, without hitting the
# network at test time. Upstream ships this file with CRLF line endings (a
# Windows-authored repo) -- the trailing .replace("\\n", "\\r\\n") below
# reproduces that; a plain LF encoding hashes to a different digest.
_REAL_NEXTO_BOT_TOML_LF = """#:schema https://rlbot.org/schemas/agent.json
[settings]
name = "Nexto"
loadout_file = "loadout.toml"
run_command = "uv run bot.py"
agent_id = "rlgym/nexto"

[details]
description = "Nexto is version 2 of Necto, the official RLGym community bot, trained using PPO with workers run by people all around the world."
fun_fact = "Nexto uses an attention mechanism, commonly used for text understanding, to support any number of players"
source_link = "https://github.com/Rolv-Arild/Necto"
developer = "Rolv, Soren, and several contributors"
language = "rlgym"
# ALL POSSIBLE TAGS: 1v1, teamplay, goalie, hoops, dropshot, snow-day, rumble, spike-rush, heatseeker, memebot
# NOTE: Only add the goalie tag if your bot only plays as a goalie; this directly contrasts with the teamplay tag!
# NOTE: Only add a tag for a special game mode if you bot properly supports it
tags = ["1v1", "teamplay", "rumble", "heatseeker"]
"""
_REAL_NEXTO_BOT_TOML_BYTES = _REAL_NEXTO_BOT_TOML_LF.replace("\n", "\r\n").encode()


# ---------------------------------------------------------------------------
# sha256_file / verify_asset
# ---------------------------------------------------------------------------

def test_sha256_file_matches_hashlib_directly(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"some bytes to hash \x00\x01\x02")
    assert bench_external.sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


def test_verify_asset_accepts_matching_checksum(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    digest = hashlib.sha256(b"hello").hexdigest()
    assert bench_external.verify_asset(p, digest) == digest


def test_verify_asset_rejects_mismatched_checksum(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello")
    wrong = hashlib.sha256(b"goodbye").hexdigest()
    with pytest.raises(bench_external.AssetChecksumError, match="checksum mismatch"):
        bench_external.verify_asset(p, wrong)


def test_verify_asset_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        bench_external.verify_asset(tmp_path / "nope.txt", "0" * 64)


def test_verify_asset_accepts_the_real_pinned_bytes(tmp_path):
    """The registry's pinned checksum for nexto/bot.toml actually matches
    the real upstream file content (embedded above, fetched 2026-07-19) --
    guards against a typo'd hex string in the registry."""
    p = tmp_path / "bot.toml"
    p.write_bytes(_REAL_NEXTO_BOT_TOML_BYTES)
    expected = bench_external.EXTERNAL_BOTS["nexto"]["files"]["bot.toml"]["sha256"]
    bench_external.verify_asset(p, expected)  # must not raise


def test_verify_asset_rejects_a_single_byte_tamper_of_real_bytes(tmp_path):
    tampered = _REAL_NEXTO_BOT_TOML_BYTES.replace(b'name = "Nexto"', b'name = "Nexto2"')
    p = tmp_path / "bot.toml"
    p.write_bytes(tampered)
    expected = bench_external.EXTERNAL_BOTS["nexto"]["files"]["bot.toml"]["sha256"]
    with pytest.raises(bench_external.AssetChecksumError):
        bench_external.verify_asset(p, expected)


# ---------------------------------------------------------------------------
# registry sanity (guards against silent drift/typos, same spirit as
# test_h2h.py's test_project_h2h_references_config_seeded_with_peak_562m)
# ---------------------------------------------------------------------------

def test_registry_has_nexto_and_necto():
    assert "nexto" in bench_external.EXTERNAL_BOTS
    assert "necto" in bench_external.EXTERNAL_BOTS


@pytest.mark.parametrize("bot_name", sorted(bench_external.EXTERNAL_BOTS))
def test_registry_checksums_are_well_formed(bot_name):
    bot = bench_external.EXTERNAL_BOTS[bot_name]
    all_files = {**bot["files"], **bench_external.SHARED_FILES}
    for fname, spec in all_files.items():
        digest = spec["sha256"]
        assert len(digest) == 64, f"{bot_name}/{fname}: sha256 not 64 hex chars ({digest!r})"
        int(digest, 16)  # raises ValueError if not valid hex
        assert spec["size"] > 0


@pytest.mark.parametrize("bot_name", sorted(bench_external.EXTERNAL_BOTS))
def test_registry_model_file_key_is_in_files(bot_name):
    bot = bench_external.EXTERNAL_BOTS[bot_name]
    assert bot["model_file"] in bot["files"]
    assert bot["model_file"].endswith(".pt")


def test_shared_files_include_license_and_requirements():
    assert "LICENSE" in bench_external.SHARED_FILES
    assert "requirements.txt" in bench_external.SHARED_FILES


# ---------------------------------------------------------------------------
# asset_plan
# ---------------------------------------------------------------------------

def test_asset_plan_unknown_bot_raises_keyerror():
    with pytest.raises(KeyError):
        bench_external.asset_plan("totally_not_a_bot")


def test_asset_plan_nexto_includes_model_and_shared_files():
    plan = bench_external.asset_plan("nexto")
    names = {item["dest"].name for item in plan}
    assert "nexto-model.pt" in names
    assert "bot.toml" in names
    assert "LICENSE" in names
    assert "requirements.txt" in names
    assert len(plan) == len(bench_external.EXTERNAL_BOTS["nexto"]["files"]) + len(bench_external.SHARED_FILES)


def test_asset_plan_default_dest_is_under_deploy_external_botname():
    plan = bench_external.asset_plan("nexto")
    for item in plan:
        assert item["dest"].parent == bench_external.EXTERNAL_DIR / "nexto"


def test_asset_plan_custom_dest_dir(tmp_path):
    plan = bench_external.asset_plan("necto", dest_dir=tmp_path / "custom")
    for item in plan:
        assert item["dest"].parent == tmp_path / "custom"


def test_asset_plan_every_item_has_url_dest_sha256_size():
    for item in bench_external.asset_plan("nexto"):
        assert item["url"].startswith("https://raw.githubusercontent.com/")
        assert isinstance(item["dest"], Path)
        assert len(item["sha256"]) == 64
        assert item["size"] > 0


# ---------------------------------------------------------------------------
# build_match_config / build_both_side_configs
# ---------------------------------------------------------------------------

def test_build_match_config_our_team_blue_exact_structure():
    cfg = bench_external.build_match_config(
        "deploy/bot.toml", "deploy/external/nexto/bot.toml", our_team="blue",
    )
    assert cfg == {
        "rlbot": {"launcher": "steam"},
        "match": {"game_mode": "Soccar", "game_map_upk": "Stadium_P"},
        "cars": [
            {"config_file": "deploy/bot.toml", "team": 0},
            {"config_file": "deploy/external/nexto/bot.toml", "team": 1},
        ],
        "mutators": {"match_length": "five_minutes"},
    }


def test_build_match_config_our_team_orange_swaps_car_order():
    cfg = bench_external.build_match_config(
        "deploy/bot.toml", "deploy/external/nexto/bot.toml", our_team="orange",
    )
    assert cfg["cars"] == [
        {"config_file": "deploy/external/nexto/bot.toml", "team": 0},
        {"config_file": "deploy/bot.toml", "team": 1},
    ]


def test_build_match_config_accepts_int_team():
    cfg = bench_external.build_match_config("a.toml", "b.toml", our_team=1)
    assert cfg["cars"][1]["config_file"] == "a.toml"


def test_build_match_config_rejects_bad_team_string():
    with pytest.raises(ValueError):
        bench_external.build_match_config("a.toml", "b.toml", our_team="purple")


def test_build_match_config_rejects_bad_team_int():
    with pytest.raises(ValueError):
        bench_external.build_match_config("a.toml", "b.toml", our_team=2)


def test_build_match_config_custom_match_length_and_map():
    cfg = bench_external.build_match_config(
        "a.toml", "b.toml", match_length="unlimited", game_map_upk="UtopiaStadium_P",
    )
    assert cfg["mutators"]["match_length"] == "unlimited"
    assert cfg["match"]["game_map_upk"] == "UtopiaStadium_P"


def test_build_both_side_configs_swaps_teams():
    cfg_blue, cfg_orange = bench_external.build_both_side_configs("a.toml", "b.toml")
    assert cfg_blue["cars"][0]["config_file"] == "a.toml"
    assert cfg_blue["cars"][0]["team"] == 0
    assert cfg_orange["cars"][1]["config_file"] == "a.toml"
    assert cfg_orange["cars"][1]["team"] == 1


# ---------------------------------------------------------------------------
# render_match_toml / write_match_toml
# ---------------------------------------------------------------------------

def test_render_match_toml_roundtrips_through_tomllib():
    cfg = bench_external.build_match_config("deploy/bot.toml", "deploy/external/nexto/bot.toml")
    text = bench_external.render_match_toml(cfg)
    parsed = tomllib.loads(text)
    assert parsed == cfg


def test_render_match_toml_exact_text_for_minimal_config():
    cfg = {
        "rlbot": {"launcher": "steam"},
        "match": {"game_mode": "Soccar", "game_map_upk": "Stadium_P"},
        "cars": [{"config_file": "bot.toml", "team": 0}],
        "mutators": {"match_length": "five_minutes"},
    }
    text = bench_external.render_match_toml(cfg)
    assert text == (
        '[rlbot]\n'
        'launcher = "steam"\n'
        '\n'
        '[match]\n'
        'game_mode = "Soccar"\n'
        'game_map_upk = "Stadium_P"\n'
        '\n'
        '[[cars]]\n'
        'config_file = "bot.toml"\n'
        'team = 0\n'
        '\n'
        '[mutators]\n'
        'match_length = "five_minutes"\n'
    )


def test_render_match_toml_matches_deploy_match_toml_shape():
    """The generated config parses to the same top-level shape as the real,
    working deploy/match.toml (verified by Elliot's existing setup) --
    [rlbot]/[match]/[[cars]]/[mutators] with the same key names."""
    real = tomllib.loads((REPO / "deploy" / "match.toml").read_text())
    cfg = bench_external.build_match_config("deploy/bot.toml", "deploy/external/nexto/bot.toml")
    generated = tomllib.loads(bench_external.render_match_toml(cfg))
    assert set(generated) == set(real)
    assert set(generated["rlbot"]) == set(real["rlbot"])
    assert set(generated["match"]) == set(real["match"])
    assert set(generated["mutators"]) == set(real["mutators"])
    assert set(generated["cars"][0]) >= {"config_file", "team"}


def test_toml_scalar_rejects_unsupported_type():
    with pytest.raises(TypeError):
        bench_external._toml_scalar(3.14)


def test_write_match_toml_writes_readable_file(tmp_path):
    cfg = bench_external.build_match_config("a.toml", "b.toml")
    out = bench_external.write_match_toml(cfg, tmp_path / "nested" / "match.toml")
    assert out.is_file()
    assert tomllib.loads(out.read_text()) == cfg


# ---------------------------------------------------------------------------
# approx_steps_from_duration
# ---------------------------------------------------------------------------

def test_approx_steps_from_duration_basic():
    assert bench_external.approx_steps_from_duration(300) == 36000  # 5 min * 120 Hz


def test_approx_steps_from_duration_none_is_zero():
    assert bench_external.approx_steps_from_duration(None) == 0


def test_approx_steps_from_duration_zero_is_zero():
    assert bench_external.approx_steps_from_duration(0) == 0


def test_approx_steps_from_duration_rounds():
    assert bench_external.approx_steps_from_duration(1.004) == 120  # 120.48 -> 120


# ---------------------------------------------------------------------------
# parse_match_result
# ---------------------------------------------------------------------------

def test_parse_match_result_our_team_blue():
    goals_ck, goals_ref, steps = bench_external.parse_match_result(
        {"blue_score": 4, "orange_score": 2, "match_length_s": 300}, our_team="blue",
    )
    assert (goals_ck, goals_ref, steps) == (4, 2, 36000)


def test_parse_match_result_our_team_orange_flips_scores():
    goals_ck, goals_ref, steps = bench_external.parse_match_result(
        {"blue_score": 4, "orange_score": 2, "match_length_s": 300}, our_team="orange",
    )
    assert (goals_ck, goals_ref, steps) == (2, 4, 36000)


def test_parse_match_result_missing_match_length_defaults_steps_zero():
    goals_ck, goals_ref, steps = bench_external.parse_match_result(
        {"blue_score": 1, "orange_score": 0}, our_team="blue",
    )
    assert steps == 0


def test_parse_match_result_missing_key_raises():
    with pytest.raises(bench_external.MatchResultError):
        bench_external.parse_match_result({"blue_score": 1}, our_team="blue")


def test_parse_match_result_negative_score_raises():
    with pytest.raises(bench_external.MatchResultError):
        bench_external.parse_match_result({"blue_score": -1, "orange_score": 0}, our_team="blue")


def test_parse_match_result_non_int_score_raises():
    with pytest.raises(bench_external.MatchResultError):
        bench_external.parse_match_result({"blue_score": "3", "orange_score": 0}, our_team="blue")


def test_parse_match_result_bool_score_rejected():
    # bool is a subclass of int in Python -- guard against True/False sneaking through as 1/0
    with pytest.raises(bench_external.MatchResultError):
        bench_external.parse_match_result({"blue_score": True, "orange_score": 0}, our_team="blue")


# ---------------------------------------------------------------------------
# combine_sides -- must agree with h2h_eval.aggregate_sides exactly
# ---------------------------------------------------------------------------

def test_combine_sides_matches_h2h_eval_aggregate_sides_directly():
    result_a = {"blue_score": 5, "orange_score": 3}   # our bot (ck) on blue
    result_b = {"blue_score": 4, "orange_score": 6}   # our bot (ck) on orange this time
    agg = bench_external.combine_sides(result_a, result_b, our_team_a="blue")
    expected = h2h_eval.aggregate_sides((5, 3), (6, 4))
    assert agg == expected


def test_combine_sides_totals_are_correct():
    agg = bench_external.combine_sides(
        {"blue_score": 5, "orange_score": 3}, {"blue_score": 4, "orange_score": 6},
        our_team_a="blue",
    )
    assert agg["a"] == 11 and agg["b"] == 7  # ck: 5+6=11, ref: 3+4=7


def test_combine_sides_flags_unstable_on_large_disagreement():
    agg = bench_external.combine_sides(
        {"blue_score": 9, "orange_score": 1}, {"blue_score": 9, "orange_score": 1},
        our_team_a="blue",
    )
    # side1 share (ck=9,ref=1)=90%; side2: our bot on orange, blue_score=9 is
    # the OPPONENT's goals this time -> (ck=1, ref=9) = 10% share -> 80% swing
    assert agg["unstable"] is True


# ---------------------------------------------------------------------------
# record_result_row -- schema parity with h2h_eval.append_h2h_history
# ---------------------------------------------------------------------------

def test_record_result_row_real_format_sample_produces_correct_jsonl_line(tmp_path):
    """A realistic result.json Elliot would hand-write after watching the
    scoreboard, parsed end-to-end into the exact jsonl line the dashboard's
    parse_h2h_history (scripts/dashboard.py) expects."""
    history = tmp_path / "h2h_history.jsonl"
    result = {"blue_score": 4, "orange_score": 2, "match_length_s": 300}

    row, warning = bench_external.record_result_row(
        history, "checkpoints_entity/ck_000909000000.pt", "nexto", result,
        our_team="blue", seed=11, ts=1784200000,
    )

    assert row == {
        "ts": 1784200000, "ck": "ck_000909000000.pt", "ref": "nexto-model.pt",
        "ref_label": "nexto-GC1", "goals_ck": 4, "goals_ref": 2,
        "share": pytest.approx(4 / 6), "steps": 36000, "seed": 11,
    }
    assert warning is not None and "side bias" in warning

    lines = history.read_text().splitlines()
    assert len(lines) == 1
    on_disk = json.loads(lines[0])
    assert on_disk == row


def test_record_result_row_matches_h2h_eval_append_directly(tmp_path):
    """Cross-check: calling record_result_row and calling
    h2h_eval.append_h2h_history by hand with the same resolved numbers must
    produce byte-identical rows -- proves "same schema" isn't just visually
    similar keys, it's the literal same function."""
    history_a = tmp_path / "a.jsonl"
    history_b = tmp_path / "b.jsonl"
    result = {"blue_score": 3, "orange_score": 1, "match_length_s": 100}

    row_a, _ = bench_external.record_result_row(
        history_a, "ck.pt", "nexto", result, our_team="blue", seed=5, ts=42,
    )
    row_b = h2h_eval.append_h2h_history(
        history_b, "ck.pt", str(bench_external.EXTERNAL_DIR / "nexto" / "nexto-model.pt"),
        "nexto-GC1", goals_ck=3, goals_ref=1, steps=12000, seed=5, ts=42,
    )
    assert row_a == row_b


def test_record_result_row_two_sided_combines_totals(tmp_path):
    history = tmp_path / "h2h_history.jsonl"
    result_blue = {"blue_score": 5, "orange_score": 3, "match_length_s": 300}
    result_orange = {"blue_score": 4, "orange_score": 6}  # our bot on orange -> goals_ck=6, goals_ref=4

    row, warning = bench_external.record_result_row(
        history, "ck.pt", "nexto", result_blue, our_team="blue",
        result_swapped=result_orange, seed=1, ts=1,
    )
    assert row["goals_ck"] == 11 and row["goals_ref"] == 7  # 5+6, 3+4
    assert row["steps"] == 36000  # from result_blue's match_length_s (the first-played side)
    assert warning is None  # side orders agree (both ~62-64%), not flagged unstable


def test_record_result_row_unknown_bot_raises_keyerror(tmp_path):
    with pytest.raises(KeyError):
        bench_external.record_result_row(
            tmp_path / "h.jsonl", "ck.pt", "not_a_real_bot",
            {"blue_score": 1, "orange_score": 0}, our_team="blue",
        )


def test_record_result_row_custom_label_override(tmp_path):
    row, _ = bench_external.record_result_row(
        tmp_path / "h.jsonl", "ck.pt", "nexto", {"blue_score": 1, "orange_score": 0},
        our_team="blue", ref_label="custom-label",
    )
    assert row["ref_label"] == "custom-label"


def test_record_result_row_ref_is_the_vendored_model_filename(tmp_path):
    row, _ = bench_external.record_result_row(
        tmp_path / "h.jsonl", "ck.pt", "necto", {"blue_score": 1, "orange_score": 0},
        our_team="blue",
    )
    assert row["ref"] == "necto-model.pt"


def test_record_result_row_appended_lines_are_valid_jsonl_and_dashboard_parseable(tmp_path):
    """Round-trip through scripts/dashboard.py's own parser (imported
    read-only -- this test never edits dashboard.py) to prove the external
    rows are indistinguishable in shape from internal h2h rows."""
    _SCRIPTS_DIR_LOCAL = REPO / "scripts"
    if str(_SCRIPTS_DIR_LOCAL) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS_DIR_LOCAL))
    import dashboard  # noqa: PLC0415

    history = tmp_path / "h2h_history.jsonl"
    bench_external.record_result_row(
        history, "ck.pt", "nexto", {"blue_score": 3, "orange_score": 1, "match_length_s": 60},
        our_team="blue", seed=2, ts=100,
    )
    rows = dashboard.parse_h2h_history(history.read_text())
    assert len(rows) == 1
    assert rows[0]["ref_label"] == "nexto-GC1"
    assert rows[0]["share"] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# CLI wiring (argparse only -- no subprocess/network)
# ---------------------------------------------------------------------------

def test_build_parser_has_all_subcommands():
    parser = bench_external.build_parser()
    args = parser.parse_args(["list"])
    assert args.func is bench_external._cmd_list


def test_build_parser_fetch_requires_known_bot():
    parser = bench_external.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["fetch", "not_a_bot"])


def test_build_parser_gen_match_defaults():
    parser = bench_external.build_parser()
    args = parser.parse_args(["gen-match", "nexto"])
    assert args.our_bot_toml == str(REPO / "deploy" / "bot.toml")
    assert args.match_length == "five_minutes"


def test_build_parser_record_result_requires_ck_and_result():
    parser = bench_external.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["record-result", "nexto"])
