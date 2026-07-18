"""Tests for scripts/ctl.py, the unified ops CLI.

No live process is ever signaled or queried here: every test exercises pure
functions (pattern helper, plan builders, log-tail parsers, ck-sweep logic
on tmp_path fixtures) or drives main()/subcommand handlers through --dry-run
with ctl.is_running monkeypatched out (real pgrep results would make tests
depend on whatever happens to be running on the box -- e.g. bc_train.py is
often actually running here, which would make an un-mocked test flaky)."""
import os
import re
import shlex
import sys
import time
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import ctl  # noqa: E402


# ---------------------------------------------------------------------------
# bracket_proof: self-match-proof pgrep/pkill pattern helper
# ---------------------------------------------------------------------------

NEEDLES = [
    "bc_train.py", "watch_loop.sh", "watch.py", "pull_ssl_duels.sh",
    "parse_v5_inplace.sh", "sync_remote.sh", "dashboard.py",
    "league_tick_loop.sh", "bc-export", "ck_000313794560.pt", "x",
]


@pytest.mark.parametrize("needle", NEEDLES)
def test_bracket_proof_still_matches_the_literal(needle):
    pattern = ctl.bracket_proof(needle)
    assert re.search(pattern, needle) is not None


@pytest.mark.parametrize("needle", [n for n in NEEDLES if len(n) > 1])
def test_bracket_proof_self_match_proof_property(needle):
    """The core property: the pattern must not match a string that contains
    the pattern itself as a literal substring -- e.g. the pkill invocation's
    own cmdline (`pkill -f <pattern>`) must not match <pattern>.

    Length-1 needles are excluded: a single character can't be hidden by
    bracketing it (the bracket must still display that exact character), so
    the property is unachievable by construction -- see
    test_bracket_proof_single_char_cannot_be_self_proof below. No real
    needle in this codebase is ever a single character."""
    pattern = ctl.bracket_proof(needle)
    assert re.search(pattern, pattern) is None
    # realistic containing strings from the SSH/pkill gotcha this codifies
    assert re.search(pattern, f"pkill -f {pattern}") is None
    assert re.search(pattern, f"ssh host 'pkill -f {pattern}'") is None


def test_bracket_proof_prefers_the_dot_when_present():
    # bracketing '.' both escapes the regex metachar and self-proofs
    pattern = ctl.bracket_proof("bc_train.py")
    assert pattern == "bc_train[.]py"


def test_bracket_proof_middle_char_when_no_dot():
    pattern = ctl.bracket_proof("bc-export")
    assert pattern.count("[") == 1 and pattern.count("]") == 1
    assert re.search(pattern, "bc-export")


def test_bracket_proof_single_char():
    assert ctl.bracket_proof("x") == "[x]"


def test_bracket_proof_single_char_cannot_be_self_proof():
    # documents the known, inherent limitation exercised above: [x] still
    # contains a literal 'x', so it trivially matches itself. Not a bug --
    # no real pgrep/pkill needle in this codebase is a single character.
    pattern = ctl.bracket_proof("x")
    assert re.search(pattern, pattern) is not None


def test_bracket_proof_empty_raises():
    with pytest.raises(ValueError):
        ctl.bracket_proof("")


def test_bracket_proof_avoids_unsafe_bracket_chars():
    # a lone '^' or ']' inside [] is invalid/ambiguous as a bracket
    # expression -- the helper must steer around them
    pattern = ctl.bracket_proof("a^b^c")
    bracketed_char = pattern[pattern.index("[") + 1]
    assert bracketed_char not in ("^", "]")


def test_venv_prefixed_path():
    assert ctl.venv_prefixed_path("/usr/bin") == f"{ctl.VENV_BIN}:/usr/bin"


# ---------------------------------------------------------------------------
# CONSTRUCT_VISER_ADDR resolution -- regression coverage for the live-found
# bug (viewer on streamed to localhost: CONSTRUCT_VISER_ADDR was resolved
# from a stale hardcoded constant, never actually detected at runtime)
# ---------------------------------------------------------------------------

def test_parse_default_gateway_real_format():
    # verbatim `ip route show default` output captured on the training box
    output = "default via 172.17.176.1 dev eth0 proto kernel \n"
    assert ctl.parse_default_gateway(output) == "172.17.176.1"


def test_parse_default_gateway_no_match():
    assert ctl.parse_default_gateway("") is None
    assert ctl.parse_default_gateway("no default route here") is None


def test_parse_default_gateway_multiline_route_table():
    output = (
        "default via 10.0.0.1 dev eth0 proto kernel \n"
        "172.17.176.0/20 dev eth0 proto kernel scope link src 172.17.179.238\n"
    )
    assert ctl.parse_default_gateway(output) == "10.0.0.1"


def test_resolve_viser_addr_prefers_explicit_env_override(monkeypatch):
    monkeypatch.setenv("CONSTRUCT_VISER_ADDR", "9.9.9.9:12345")

    def boom():
        raise AssertionError("must not shell out to `ip route` when an "
                              "explicit override is already set")
    monkeypatch.setattr(ctl, "detect_host_gateway_ip", boom)
    assert ctl.resolve_viser_addr() == "9.9.9.9:12345"


def test_resolve_viser_addr_uses_live_detected_gateway(monkeypatch):
    monkeypatch.delenv("CONSTRUCT_VISER_ADDR", raising=False)
    monkeypatch.setattr(ctl, "detect_host_gateway_ip", lambda timeout=5: "10.20.30.40")
    assert ctl.resolve_viser_addr() == "10.20.30.40:45250"


def test_resolve_viser_addr_custom_port_with_detected_gateway(monkeypatch):
    monkeypatch.delenv("CONSTRUCT_VISER_ADDR", raising=False)
    monkeypatch.setattr(ctl, "detect_host_gateway_ip", lambda timeout=5: "10.20.30.40")
    assert ctl.resolve_viser_addr(port=9999) == "10.20.30.40:9999"


def test_resolve_viser_addr_falls_back_when_detection_fails(monkeypatch):
    """This is the exact scenario the live bug hit: no env override, and
    (in the old code) no live detection was ever attempted at all -- the
    hardcoded fallback was used unconditionally instead of as a last
    resort. Now detection is always attempted first; only a genuine
    detection failure reaches the fallback."""
    monkeypatch.delenv("CONSTRUCT_VISER_ADDR", raising=False)
    monkeypatch.setattr(ctl, "detect_host_gateway_ip", lambda timeout=5: None)
    assert ctl.resolve_viser_addr() == ctl.VISER_ADDR_FALLBACK


def test_resolve_viser_addr_never_empty(monkeypatch):
    monkeypatch.delenv("CONSTRUCT_VISER_ADDR", raising=False)
    monkeypatch.setattr(ctl, "detect_host_gateway_ip", lambda timeout=5: None)
    assert ctl.resolve_viser_addr()  # truthy -- never "" or None


# ---------------------------------------------------------------------------
# ck-sweep: crash-recovery checkpoint quarantine (tmp_path only)
# ---------------------------------------------------------------------------

def _touch_with_mtime(path: Path, content: bytes, mtime: float):
    path.write_bytes(content)
    os.utime(path, (mtime, mtime))


def _make_valid_ck(path: Path):
    torch = pytest.importorskip("torch")
    torch.save({"model": {}, "config": {"net": {}}}, str(path))


def test_newest_checkpoints_orders_by_mtime(tmp_path):
    now = time.time()
    _touch_with_mtime(tmp_path / "ck_1.pt", b"a", now - 300)
    _touch_with_mtime(tmp_path / "ck_2.pt", b"a", now - 100)
    _touch_with_mtime(tmp_path / "ck_3.pt", b"a", now - 200)
    result = ctl.newest_checkpoints(tmp_path, n=2)
    assert [p.name for p in result] == ["ck_2.pt", "ck_3.pt"]


def test_newest_checkpoints_ignores_non_ck_and_non_pt(tmp_path):
    (tmp_path / "ck_1.pt").write_bytes(b"a")
    (tmp_path / "other.pt").write_bytes(b"a")
    (tmp_path / "ck_2.txt").write_bytes(b"a")
    (tmp_path / "ck_1.pt.corrupt").write_bytes(b"a")
    result = ctl.newest_checkpoints(tmp_path, n=10)
    assert [p.name for p in result] == ["ck_1.pt"]


def test_newest_checkpoints_missing_dir_returns_empty(tmp_path):
    assert ctl.newest_checkpoints(tmp_path / "does_not_exist") == []


def test_classify_checkpoint_zero_byte(tmp_path):
    p = tmp_path / "ck_zero.pt"
    p.write_bytes(b"")
    assert ctl.classify_checkpoint(p) == "zero_byte"


def test_classify_checkpoint_garbage_unloadable(tmp_path):
    pytest.importorskip("torch")
    p = tmp_path / "ck_garbage.pt"
    p.write_bytes(b"this is not a torch checkpoint file at all")
    assert ctl.classify_checkpoint(p) == "unloadable"


def test_classify_checkpoint_valid_ok(tmp_path):
    p = tmp_path / "ck_good.pt"
    _make_valid_ck(p)
    assert ctl.classify_checkpoint(p) == "ok"


def test_quarantine_renames_with_corrupt_suffix(tmp_path):
    p = tmp_path / "ck_bad.pt"
    p.write_bytes(b"")
    target = ctl.quarantine(p, dry_run=False)
    assert target == tmp_path / "ck_bad.pt.corrupt"
    assert target.exists()
    assert not p.exists()


def test_quarantine_dry_run_does_not_touch_disk(tmp_path):
    p = tmp_path / "ck_bad.pt"
    p.write_bytes(b"")
    target = ctl.quarantine(p, dry_run=True)
    assert target == tmp_path / "ck_bad.pt.corrupt"
    assert p.exists()
    assert not target.exists()


def test_sweep_dir_quarantines_zero_byte_and_garbage_keeps_valid(tmp_path):
    pytest.importorskip("torch")
    now = time.time()
    _touch_with_mtime(tmp_path / "ck_zero.pt", b"", now - 10)
    (tmp_path / "ck_garbage.pt").write_bytes(b"not a checkpoint")
    os.utime(tmp_path / "ck_garbage.pt", (now - 20, now - 20))
    _make_valid_ck(tmp_path / "ck_good.pt")
    os.utime(tmp_path / "ck_good.pt", (now - 30, now - 30))

    info = ctl.sweep_dir(tmp_path, n=3, dry_run=False)

    statuses = {r["path"].name: r["status"] for r in info["checked"]}
    assert statuses == {
        "ck_zero.pt": "zero_byte", "ck_garbage.pt": "unloadable", "ck_good.pt": "ok",
    }
    quarantined_names = {q["path"].name for q in info["quarantined"]}
    assert quarantined_names == {"ck_zero.pt", "ck_garbage.pt"}
    assert (tmp_path / "ck_zero.pt.corrupt").exists()
    assert (tmp_path / "ck_garbage.pt.corrupt").exists()
    assert not (tmp_path / "ck_zero.pt").exists()
    assert info["newest_intact"] == tmp_path / "ck_good.pt"


def test_sweep_dir_dry_run_reports_but_does_not_rename(tmp_path):
    (tmp_path / "ck_zero.pt").write_bytes(b"")
    info = ctl.sweep_dir(tmp_path, n=3, dry_run=True)
    assert len(info["quarantined"]) == 1
    assert (tmp_path / "ck_zero.pt").exists()
    assert not (tmp_path / "ck_zero.pt.corrupt").exists()


def test_sweep_dir_only_checks_newest_n(tmp_path):
    now = time.time()
    for i in range(5):
        _touch_with_mtime(tmp_path / f"ck_{i}.pt", b"", now - i * 100)
    info = ctl.sweep_dir(tmp_path, n=2, dry_run=True)
    assert len(info["checked"]) == 2
    assert {r["path"].name for r in info["checked"]} == {"ck_0.pt", "ck_1.pt"}


def test_sweep_dir_empty_dir(tmp_path):
    info = ctl.sweep_dir(tmp_path, n=3, dry_run=True)
    assert info["checked"] == []
    assert info["newest_intact"] is None


def test_ck_sweep_multiple_dirs(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    (d1 / "ck_x.pt").write_bytes(b"")
    report = ctl.ck_sweep([d1, d2], n=3, dry_run=True)
    assert set(report.keys()) == {str(d1), str(d2)}
    assert len(report[str(d1)]["checked"]) == 1
    assert report[str(d2)]["checked"] == []


# ---------------------------------------------------------------------------
# log-tail parsers -- fixture strings captured verbatim from real logs
# ---------------------------------------------------------------------------

def test_parse_bc_log_tail_batch_line():
    line = "bc epoch 1 batch 155350/170326 loss 1.7612 lr 1.43e-06 41390 samples/s"
    d = ctl.parse_bc_log_tail([line])
    assert d == {
        "kind": "batch", "epoch": 1, "batch": 155350, "total": 170326,
        "loss": 1.7612, "samples_s": 41390,
    }


def test_parse_bc_log_tail_done_line():
    line = ("bc epoch 1 done: train_loss 1.7500 val_loss 1.8100 top1 0.659 "
            "top3 0.843 recall_jump 0.412 recall_stall 0.301")
    d = ctl.parse_bc_log_tail([line])
    assert d["kind"] == "done"
    assert d["epoch"] == 1
    assert d["val_loss"] == 1.81
    assert d["top1"] == 0.659


def test_parse_bc_log_tail_banner_line():
    line = "bc: 62349 train / 5219 val shards, 170326 batches/epoch x 4 epochs"
    d = ctl.parse_bc_log_tail([line])
    assert d == {
        "kind": "banner", "train_shards": 62349, "val_shards": 5219,
        "batches_per_epoch": 170326, "epochs": 4,
    }


def test_parse_bc_log_tail_prefers_most_recent_recognizable_line():
    lines = [
        "bc: 62349 train / 5219 val shards, 170326 batches/epoch x 4 epochs",
        "bc epoch 0 batch 100/170326 loss 3.0 lr 3e-4 1000 samples/s",
        "some unrelated noise line",
    ]
    d = ctl.parse_bc_log_tail(lines)
    assert d["kind"] == "batch" and d["batch"] == 100


def test_parse_bc_log_tail_no_match_returns_none():
    assert ctl.parse_bc_log_tail(["nothing recognizable here"]) is None
    assert ctl.parse_bc_log_tail([]) is None


def test_parse_parse_v5_log_tail_batch_summary():
    line = "parsed=2697 skipped=0 failed=0 reset_states=43152"
    d = ctl.parse_parse_v5_log_tail([line])
    assert d == {"kind": "batch", "parsed": 2697, "skipped": 0, "failed": 0,
                 "reset_states": 43152}


def test_parse_parse_v5_log_tail_progress_line():
    line = "23:36:44 [batch_0009] (2697) — v5 re-parsing in place..."
    d = ctl.parse_parse_v5_log_tail([line])
    assert d == {"kind": "progress", "batch": "0009", "count": 2697}


def test_parse_parse_v5_log_tail_done_line():
    line = "00:31:43 v5 parse exit: 13/13 batches"
    d = ctl.parse_parse_v5_log_tail([line])
    assert d == {"kind": "done", "done": 13, "total": 13}


def test_parse_parse_v5_log_tail_prefers_most_recent():
    lines = [
        "23:36:44 [batch_0009] (2697) — v5 re-parsing in place...",
        "parsed=2697 skipped=0 failed=0 reset_states=43152",
        "23:48:42 [batch_0010] (4774) — v5 re-parsing in place...",
    ]
    d = ctl.parse_parse_v5_log_tail(lines)
    assert d["kind"] == "progress" and d["batch"] == "0010"


def test_parse_export_log_tail_summary():
    line = "exported=0 overwritten=67568 skipped_existing=0 failed=0 samples=734848733"
    d = ctl.parse_export_log_tail([line])
    assert d == {
        "kind": "summary", "exported": 0, "overwritten": 67568,
        "skipped_existing": 0, "failed": 0, "samples": 734848733,
    }


def test_parse_export_log_tail_no_summary_yet_means_running():
    assert ctl.parse_export_log_tail([]) is None
    assert ctl.parse_export_log_tail(["some other line"]) is None


def test_parse_ssl_log_tail_progress_with_host_free():
    line = ("07-18 18:07:27 progress: 4000 this run / 4000 total "
            "(183 deduped, 1 failed) — 186/h, host free 183.0G")
    d = ctl.parse_ssl_log_tail([line])
    assert d == {
        "kind": "progress", "ts": "07-18 18:07:27", "this_run": 4000,
        "total": 4000, "deduped": 183, "failed": 1, "rate_per_h": 186.0,
        "host_free_gb": 183.0,
    }


def test_parse_ssl_log_tail_page_fallback():
    line = ("07-18 22:44:16 page: 200 rows, 10000+ matching beyond cursor, "
            "oldest created 2025-01-22T22:55:58.591931Z")
    d = ctl.parse_ssl_log_tail([line])
    assert d == {
        "kind": "page", "ts": "07-18 22:44:16",
        "oldest_created": "2025-01-22T22:55:58.591931Z",
    }


def test_parse_ssl_log_tail_progress_preferred_over_page():
    lines = [
        "07-18 22:44:16 page: 200 rows, oldest created 2025-01-22T00:00:00Z",
        "07-18 23:00:00 progress: 10 this run / 20 total (1 deduped, 0 failed) — 5/h",
    ]
    d = ctl.parse_ssl_log_tail(lines)
    assert d["kind"] == "progress"


def test_parse_ssl_log_tail_no_match():
    assert ctl.parse_ssl_log_tail(["unrelated"]) is None


def test_last_nonempty_line():
    assert ctl.last_nonempty_line(["a", "", "  ", "b", ""]) == "b"
    assert ctl.last_nonempty_line([]) is None
    assert ctl.last_nonempty_line(["", "  "]) is None


# --- iter-line parsing (main trainer log) -----------------------------

KICK_LINE = ("iter 2 steps 333,824 sps 5,519 ep_rew 1.293 pi_loss 0.0047 v_loss 0.4045 "
             "ent 4.058 clip 0.261 kick_kl 1.0886 lambda_k 1.000")
PLAIN_LINE = ("iter 1117 steps 500,235,264 sps 8,137 ep_rew 4.756 pi_loss 0.0076 "
              "v_loss 0.8470 ent 3.594 clip 0.172")
KLPRI_LINE = ("iter 21 steps 564,976,128 sps 4,984 ep_rew -0.715 pi_loss 0.0016 "
              "v_loss 0.5251 ent 1.544 clip 0.056 kl_pri 1.1613 lambda_p 0.050")


def test_parse_iter_line_kickstart_era():
    d = ctl.parse_iter_line(KICK_LINE)
    assert d["steps"] == 333_824 and d["sps"] == 5519
    assert d["kick_kl"] == 1.0886 and d["lambda_k"] == 1.0
    assert "kl_pri" not in d


def test_parse_iter_line_plain_era():
    d = ctl.parse_iter_line(PLAIN_LINE)
    assert d["steps"] == 500_235_264 and d["sps"] == 8137
    assert "kick_kl" not in d and "kl_pri" not in d


def test_parse_iter_line_kl_prior_era():
    d = ctl.parse_iter_line(KLPRI_LINE)
    assert d["steps"] == 564_976_128 and d["kl_pri"] == 1.1613 and d["lambda_p"] == 0.05


def test_parse_iter_line_rejects_noise():
    assert ctl.parse_iter_line("this is not an iter line") is None


def test_last_iter_line_scans_from_the_end():
    lines = [KICK_LINE, "some noise", PLAIN_LINE, "resumed at 313,794,560 steps"]
    d = ctl.last_iter_line(lines)
    assert d["steps"] == 500_235_264


def test_last_iter_line_none_when_no_match():
    assert ctl.last_iter_line(["nothing here"]) is None


# --- remote: ps discovery + restart verification ------------------------

def test_parse_remote_ps_for_resume_ck_finds_positional_arg():
    ps_lines = [
        "48213 .venv/bin/python scripts/resume_train.py "
        "checkpoints_entity/ck_000313794560.pt --config configs/train_v1.toml "
        "--league --kl-prior checkpoints_bc/ck_bc_ep00.pt --kl-prior-lambda 0.05"
    ]
    assert (ctl.parse_remote_ps_for_resume_ck(ps_lines)
            == "checkpoints_entity/ck_000313794560.pt")


def test_parse_remote_ps_for_resume_ck_no_match():
    assert ctl.parse_remote_ps_for_resume_ck([]) is None
    assert ctl.parse_remote_ps_for_resume_ck(["48213 some other process"]) is None


def test_verify_restart_both_confirmed():
    lines = [
        "resumed at 313,794,560 steps | arenas=192 agents=384 device=cuda",
        "iter 1 steps 314,000,000 sps 5,000 ep_rew 1.0 pi_loss 0.0 v_loss 0.0 "
        "ent 1.0 clip 0.0 kl_pri 6.95 lambda_p 0.050",
    ]
    result = ctl.verify_restart(lines)
    assert result == {"resumed": True, "anchor_confirmed": True}


def test_verify_restart_neither_confirmed():
    result = ctl.verify_restart(["unrelated log noise"])
    assert result == {"resumed": False, "anchor_confirmed": False}


# ---------------------------------------------------------------------------
# dry-run plan builders -- exact argv/cwd/env assembly (viewer/bc/remote/etc)
# ---------------------------------------------------------------------------

def test_relay_check_plan():
    plan = ctl.relay_check_plan()
    assert plan["argv"] == [
        "powershell.exe", "-NoProfile", "-Command",
        "Get-NetUDPEndpoint -LocalPort 34254,45250 -ErrorAction SilentlyContinue",
    ]
    assert plan["cwd"] is None


def test_relay_start_plan():
    plan = ctl.relay_start_plan()
    assert plan["argv"] == [
        "cmd.exe", "/c", "start", "", "powershell.exe", "-NoProfile",
        "-ExecutionPolicy", "Bypass", "-File", ctl.RELAY_PS1,
    ]
    assert plan["cwd"] == "/mnt/c"


def test_rlviser_start_plan():
    plan = ctl.rlviser_start_plan()
    assert plan["argv"] == ["cmd.exe", "/c", "start", "", "/D", ctl.RLVISER_WIN_DIR, "rlviser.exe"]
    assert plan["cwd"] == "/mnt/c"


def test_overlay_start_plan():
    plan = ctl.overlay_start_plan()
    assert plan["argv"] == [
        "cmd.exe", "/c", "start", "", "powershell.exe", "-NoProfile",
        "-ExecutionPolicy", "Bypass", "-File", ctl.OVERLAY_PS1,
    ]
    assert plan["cwd"] == "/mnt/c"


@pytest.mark.parametrize("plan_fn", [ctl.relay_start_plan, ctl.rlviser_start_plan,
                                      ctl.overlay_start_plan])
def test_windows_interop_launches_are_marked_detached(plan_fn):
    """Regression test for the ~120s `viewer on` hang: these three are the
    only run_bg() callers invoked with log_path=None (no log file to
    redirect stdio to instead), so they're the only ones that were ever at
    risk of inheriting ctl.py's stdout pipe -- which WSL's interop layer
    keeps open until the spawned Windows process's console fully detaches.
    `detach=True` is what tells run_bg() to force DEVNULL instead."""
    assert plan_fn()["detach"] is True


def test_bc_start_plan_not_marked_detached():
    # contrast case: bc/ssl/parse/export/loop launches always get an
    # explicit log_path, so their stdio is redirected to a file regardless
    # -- they were never at risk and don't need the detach marker.
    assert ctl.bc_start_plan()["detach"] is False


def test_watch_loop_start_plan_runs_from_repo_root_not_mnt_c():
    plan = ctl.watch_loop_start_plan("172.17.176.1:45250", rotate_secs=300)
    assert plan["argv"] == ["setsid", "./scripts/watch_loop.sh", "300"]
    assert plan["cwd"] == str(ctl.REPO_ROOT)
    assert plan["cwd"] != "/mnt/c"  # the cwd trap this codifies
    assert plan["env"] == {"CONSTRUCT_VISER_ADDR": "172.17.176.1:45250"}


def test_watch_loop_start_plan_custom_rotate():
    plan = ctl.watch_loop_start_plan("1.2.3.4:45250", rotate_secs=60)
    assert plan["argv"] == ["setsid", "./scripts/watch_loop.sh", "60"]


def test_watch_loop_start_plan_rejects_empty_addr():
    """Regression test for the live-found bug: an empty/missing
    CONSTRUCT_VISER_ADDR silently makes the engine default to 127.0.0.1
    (engine/src/viser.rs) -- the stream never reaches the Windows-side
    viewer. The plan builder now refuses to build a plan for that case at
    all, rather than silently producing an env dict that omits the var."""
    with pytest.raises(ValueError):
        ctl.watch_loop_start_plan("")
    with pytest.raises(ValueError):
        ctl.watch_loop_start_plan(None)


def test_rlviser_stop_plan():
    plan = ctl.rlviser_stop_plan()
    assert plan["argv"] == [
        "powershell.exe", "-NoProfile", "-Command",
        "Stop-Process -Name rlviser -Force -ErrorAction SilentlyContinue",
    ]


def test_viewer_off_kill_patterns_are_bracket_proofed():
    patterns = ctl.viewer_off_kill_patterns()
    assert patterns == [ctl.bracket_proof("watch_loop.sh"), ctl.bracket_proof("watch.py")]
    for pat in patterns:
        assert re.search(pat, pat) is None  # self-match-proof


def test_bc_start_plan_no_epochs():
    plan = ctl.bc_start_plan()
    assert plan["argv"] == [
        "setsid", "nice", "-n", "10", str(ctl.VENV_PY), "scripts/bc_train.py",
        "--config", "configs/bc_v1.toml",
    ]
    assert plan["cwd"] == str(ctl.REPO_ROOT)


def test_bc_start_plan_with_epochs():
    plan = ctl.bc_start_plan(epochs=3)
    assert plan["argv"] == [
        "setsid", "nice", "-n", "10", str(ctl.VENV_PY), "scripts/bc_train.py",
        "--config", "configs/bc_v1.toml", "--epochs", "3",
    ]


def test_bc_stop_pattern():
    assert ctl.bc_stop_pattern() == ctl.bracket_proof("bc_train.py")


def test_ssl_start_plan():
    plan = ctl.ssl_start_plan()
    assert plan["argv"] == ["setsid", "nice", "-n", "15", "./scripts/pull_ssl_duels.sh"]
    assert plan["cwd"] == str(ctl.REPO_ROOT)


def test_ssl_stop_pattern():
    assert ctl.ssl_stop_pattern() == ctl.bracket_proof("pull_ssl_duels.sh")


def test_parse_v5_start_plan():
    plan = ctl.parse_v5_start_plan()
    assert plan["argv"] == ["setsid", "nice", "-n", "15", "./scripts/parse_v5_inplace.sh"]


def test_export_start_plan_no_force():
    plan = ctl.export_start_plan(force=False)
    assert plan["argv"] == [
        "setsid", "nice", "-n", "15", "./target/release/bc-export",
        "--shards", "data/shards_v4", "--out", "data/bc",
    ]


def test_export_start_plan_force():
    plan = ctl.export_start_plan(force=True)
    assert plan["argv"][-1] == "--force"
    assert "--force" in plan["argv"]


def test_sync_start_plan():
    plan = ctl.sync_start_plan()
    assert plan["argv"] == ["setsid", "./scripts/sync_remote.sh"]
    assert plan["env"] == {}  # sync_remote.sh never shells out to python


def test_dashboard_start_plan_default_port():
    plan = ctl.dashboard_start_plan()
    assert plan["argv"] == ["setsid", str(ctl.VENV_PY), "scripts/dashboard.py", "8420"]


def test_dashboard_start_plan_custom_port():
    plan = ctl.dashboard_start_plan(port=9000)
    assert plan["argv"] == ["setsid", str(ctl.VENV_PY), "scripts/dashboard.py", "9000"]


def test_league_loop_start_plan_prefixes_venv_path():
    plan = ctl.league_loop_start_plan()
    assert plan["argv"] == ["setsid", "./scripts/league_tick_loop.sh"]
    assert plan["env"]["PATH"].startswith(str(ctl.VENV_BIN))


def test_eval_plan_default_nice():
    plan = ctl.eval_plan("checkpoints_entity/ck_000313794560.pt")
    assert plan["argv"] == [
        "nice", "-n", "15", str(ctl.VENV_PY), "scripts/eval_metrics.py",
        "checkpoints_entity/ck_000313794560.pt",
    ]


def test_eval_plan_custom_nice():
    plan = ctl.eval_plan("ck.pt", nice=10)
    assert plan["argv"][:3] == ["nice", "-n", "10"]


# --- remote ------------------------------------------------------------

HOST = "elliot@192.168.86.117"
RDIR = "construct"


def test_remote_status_plans_exact():
    plans = ctl.remote_status_plans(HOST, RDIR)
    assert plans["trainer_log_tail"]["argv"] == [
        "ssh", HOST, "tail", "-n", "500", "construct/checkpoints_entity/train_v1.log",
    ]
    assert plans["league_loop_check"]["argv"] == ["ssh", HOST, "pgrep", "-af", "league_tick_loop.sh"]


def test_remote_discover_trainer_plan():
    plan = ctl.remote_discover_trainer_plan(HOST)
    assert plan["argv"] == ["ssh", HOST, "pgrep", "-af", "resume_train.py"]


def test_remote_kill_plan_uses_given_bracketed_pattern():
    pattern = ctl.bracket_proof("ck_000313794560.pt")
    plan = ctl.remote_kill_plan(HOST, pattern)
    assert plan["argv"] == ["ssh", HOST, "pkill", "-f", pattern]


def test_remote_launch_command_minimal():
    cmd = ctl.remote_launch_command(
        RDIR, "checkpoints_entity/ck_000313794560.pt", "configs/train_v1.toml",
        "checkpoints_entity/train_v1.log",
    )
    assert cmd == (
        "cd construct && nohup setsid .venv/bin/python scripts/resume_train.py "
        "checkpoints_entity/ck_000313794560.pt --config configs/train_v1.toml "
        ">> checkpoints_entity/train_v1.log 2>&1 < /dev/null & disown"
    )


def test_remote_launch_command_full_options():
    cmd = ctl.remote_launch_command(
        RDIR, "checkpoints_entity/ck_000313794560.pt", "configs/train_v1.toml",
        "checkpoints_entity/train_v1.log",
        reward_config="configs/reward_v3_1.toml", league=True,
        kl_prior="checkpoints_bc/ck_bc_ep00.pt", kl_prior_lambda=0.05,
    )
    assert "--reward-config configs/reward_v3_1.toml" in cmd
    assert "--league" in cmd
    assert "--kl-prior checkpoints_bc/ck_bc_ep00.pt" in cmd
    assert "--kl-prior-lambda 0.05" in cmd
    assert cmd.startswith("cd construct && nohup setsid .venv/bin/python scripts/resume_train.py")
    assert cmd.endswith("& disown")


def test_remote_launch_command_shell_quotes_dynamic_values():
    cmd = ctl.remote_launch_command(
        "some dir", "ck with space.pt", "configs/train_v1.toml", "log.log",
    )
    assert shlex.quote("some dir") in cmd
    assert shlex.quote("ck with space.pt") in cmd


def test_remote_launch_plan_wraps_in_ssh_argv():
    plan = ctl.remote_launch_plan(HOST, RDIR, "ck.pt", "configs/train_v1.toml")
    assert plan["argv"][0:2] == ["ssh", HOST]
    assert len(plan["argv"]) == 3  # ssh, host, single shell command string


def test_remote_verify_plan():
    plan = ctl.remote_verify_plan(HOST, RDIR)
    assert plan["argv"] == ["ssh", HOST, "tail", "-n", "20",
                             "construct/checkpoints_entity/train_v1.log"]


# ---------------------------------------------------------------------------
# render_plan / run_bg / run_fg / kill_pattern dry-run short-circuits
# ---------------------------------------------------------------------------

def test_render_plan_includes_cwd_and_env():
    plan = ctl._plan(["echo", "hi"], cwd="/some/dir", env={"FOO": "bar"})
    rendered = ctl.render_plan(plan)
    assert "echo hi" in rendered
    assert "/some/dir" in rendered
    assert "FOO=bar" in rendered


def test_run_bg_dry_run_never_touches_disk_or_subprocess(tmp_path, capsys, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("subprocess.Popen must not be called in dry-run")
    monkeypatch.setattr(ctl.subprocess, "Popen", boom)
    log_path = tmp_path / "sub" / "out.log"
    ctl.run_bg(ctl._plan(["true"]), log_path=log_path, dry_run=True)
    assert not log_path.exists()
    assert not log_path.parent.exists()
    out = capsys.readouterr().out
    assert "[dry-run]" in out


def test_run_bg_uses_devnull_not_inherited_stdio_when_no_log_path(monkeypatch):
    """Regression test for the live-found ~120s `viewer on` hang: launches
    with no log_path (relay/rlviser/overlay -- the `detach=True` plans) must
    get subprocess.DEVNULL for stdout/stderr, never None (which means
    "inherit from ctl.py's own stdio" and is what caused the hang under
    WSL's cmd.exe interop)."""
    captured = {}

    def fake_popen(argv, **kwargs):
        captured.update(kwargs)

        class FakeProc:
            pass
        return FakeProc()

    monkeypatch.setattr(ctl.subprocess, "Popen", fake_popen)
    ctl.run_bg(ctl.relay_start_plan(), log_path=None, dry_run=False)
    assert captured["stdout"] == ctl.subprocess.DEVNULL
    assert captured["stderr"] == ctl.subprocess.DEVNULL
    assert captured["stdin"] == ctl.subprocess.DEVNULL
    assert captured.get("start_new_session") is True


def test_run_bg_still_redirects_to_log_file_when_given(tmp_path, monkeypatch):
    captured = {}

    def fake_popen(argv, **kwargs):
        captured.update(kwargs)

        class FakeProc:
            pass
        return FakeProc()

    monkeypatch.setattr(ctl.subprocess, "Popen", fake_popen)
    log_path = tmp_path / "out.log"
    ctl.run_bg(ctl.bc_start_plan(), log_path=log_path, dry_run=False)
    assert captured["stdout"] is not ctl.subprocess.DEVNULL
    assert captured["stdout"] is not None  # a real open file handle
    assert log_path.exists()


def test_run_fg_dry_run_never_calls_subprocess(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("subprocess.run must not be called in dry-run")
    monkeypatch.setattr(ctl.subprocess, "run", boom)
    result = ctl.run_fg(ctl._plan(["true"]), dry_run=True)
    assert result is None
    assert "[dry-run]" in capsys.readouterr().out


def test_kill_pattern_dry_run_never_calls_subprocess(monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("subprocess.run must not be called in dry-run")
    monkeypatch.setattr(ctl.subprocess, "run", boom)
    ctl.kill_pattern("some[.]pattern", dry_run=True)
    assert "[dry-run]" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# sync freshness (reads file mtimes, no subprocess)
# ---------------------------------------------------------------------------

def test_sync_freshness_reports_age(tmp_path):
    ck_dir = tmp_path / "checkpoints_entity"
    ck_dir.mkdir()
    (ck_dir / "train_remote.log").write_text("x")
    result = ctl.sync_freshness(tmp_path)
    assert result is not None and "ago" in result


def test_sync_freshness_none_when_nothing_synced_yet(tmp_path):
    assert ctl.sync_freshness(tmp_path) is None


# ---------------------------------------------------------------------------
# key_stat_for -- per-service status-line formatting
# ---------------------------------------------------------------------------

def test_key_stat_for_bc():
    lines = ["bc epoch 1 batch 155350/170326 loss 1.7612 lr 1.43e-06 41390 samples/s"]
    stat = ctl.key_stat_for("bc-train", lines)
    assert "155350/170326" in stat and "41390" in stat


def test_key_stat_for_export_running():
    assert ctl.key_stat_for("export", []) == "running (no summary yet)"


def test_key_stat_for_export_done():
    lines = ["exported=0 overwritten=67568 skipped_existing=0 failed=0 samples=734848733"]
    stat = ctl.key_stat_for("export", lines)
    assert "overwritten=67568" in stat


def test_key_stat_for_generic_fallback_last_line():
    lines = ["first", "", "last real line"]
    assert ctl.key_stat_for("watch-loop", lines) == "last real line"


def test_key_stat_for_no_output():
    assert ctl.key_stat_for("watch-loop", []) == "no output yet"


# ---------------------------------------------------------------------------
# argparse wiring sanity
# ---------------------------------------------------------------------------

def test_build_parser_subcommands():
    parser = ctl.build_parser()
    sub_actions = [a for a in parser._subparsers._group_actions
                   if hasattr(a, "choices")]
    names = set(sub_actions[0].choices.keys())
    assert names == {
        "status", "viewer", "bc", "ssl", "parse-v5", "export", "loops",
        "eval", "ck-sweep", "recover", "remote",
    }


def test_viewer_bc_ssl_parse_export_loops_remote_actions():
    parser = ctl.build_parser()
    assert parser.parse_args(["viewer", "on"]).action == "on"
    assert parser.parse_args(["bc", "stop"]).action == "stop"
    assert parser.parse_args(["ssl", "status"]).action == "status"
    assert parser.parse_args(["parse-v5", "start"]).action == "start"
    assert parser.parse_args(["export", "start", "--force"]).force is True
    assert parser.parse_args(["loops", "up"]).action == "up"
    assert parser.parse_args(["remote", "status"]).action == "status"
    with pytest.raises(SystemExit):
        parser.parse_args(["parse-v5", "stop"])  # no stop action defined


def test_dry_run_flag_works_after_subcommand_not_before():
    parser = ctl.build_parser()
    # after the subcommand: works (this is the documented usage)
    args = parser.parse_args(["bc", "start", "--dry-run"])
    assert args.dry_run is True
    # bc has no top-level --dry-run at all now, so putting it first errors
    with pytest.raises(SystemExit):
        parser.parse_args(["--dry-run", "bc", "start"])


# ---------------------------------------------------------------------------
# CLI-level dry-run integration: main() end to end, is_running mocked out so
# tests never depend on (or query) real system process state
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=False)
def no_real_procs(monkeypatch):
    """Force every process-liveness check to report "not running" so start
    commands proceed down the dry-run path deterministically, regardless of
    what's actually running on the test box."""
    monkeypatch.setattr(ctl, "is_running", lambda pattern: False)
    yield


def test_main_bc_start_dry_run(capsys, no_real_procs):
    ctl.main(["bc", "start", "--dry-run", "--epochs", "2"])
    out = capsys.readouterr().out
    assert "[dry-run] would run:" in out
    assert "scripts/bc_train.py" in out
    assert "--epochs 2" in out


def test_main_bc_start_refuses_double_launch_when_not_dry_run(capsys, monkeypatch):
    monkeypatch.setattr(ctl, "is_running", lambda pattern: True)

    def boom(*a, **k):
        raise AssertionError("must not launch when already running")
    monkeypatch.setattr(ctl, "run_bg", boom)
    ctl.main(["bc", "start"])
    assert "refusing to double-launch" in capsys.readouterr().out


def test_main_viewer_on_dry_run_prints_all_four_steps(capsys, monkeypatch, no_real_procs):
    monkeypatch.setattr(ctl, "run_fg", lambda plan, dry_run=False, timeout=60:
                         None if dry_run else pytest.fail("run_fg must be dry-run"))
    monkeypatch.setattr(ctl, "resolve_viser_addr", lambda port=45250: "10.0.0.99:45250")
    ctl.main(["viewer", "on", "--dry-run"])
    out = capsys.readouterr().out
    assert out.count("[dry-run]") >= 4  # relay check + relay/rlviser/overlay/watch_loop
    # Regression coverage for the live-found bug: argv-only assertions (the
    # count above) let a missing/wrong CONSTRUCT_VISER_ADDR slip through
    # silently, since the watch_loop argv itself doesn't carry the addr --
    # it's env-only. Must assert the actual resolved value made it into the
    # printed plan for the watch_loop.sh launch specifically.
    watch_loop_line = next(ln for ln in out.splitlines() if "watch_loop.sh" in ln)
    assert "CONSTRUCT_VISER_ADDR=10.0.0.99:45250" in watch_loop_line


def test_main_viewer_on_uses_resolve_viser_addr_not_raw_environ(capsys, monkeypatch, no_real_procs):
    """Guards against regressing back to the old
    os.environ.get("CONSTRUCT_VISER_ADDR", VISER_ADDR_FALLBACK) call, which
    never attempted live `ip route` detection at all."""
    monkeypatch.setattr(ctl, "run_fg", lambda plan, dry_run=False, timeout=60: None)
    calls = []
    monkeypatch.setattr(ctl, "resolve_viser_addr", lambda port=45250: calls.append(1) or "1.2.3.4:45250")
    ctl.main(["viewer", "on", "--dry-run"])
    assert calls, "resolve_viser_addr() must be called by `viewer on`"
    out = capsys.readouterr().out
    assert "CONSTRUCT_VISER_ADDR=1.2.3.4:45250" in out


def test_main_ck_sweep_dry_run_on_tmp_dirs(tmp_path, capsys, monkeypatch):
    (tmp_path / "ck_zero.pt").write_bytes(b"")
    monkeypatch.setattr(ctl, "REPO_ROOT", tmp_path)
    ctl.main(["ck-sweep", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "would quarantine" in out
    assert (tmp_path / "ck_zero.pt").exists()  # untouched


def test_main_export_start_dry_run(capsys, no_real_procs):
    ctl.main(["export", "start", "--force", "--dry-run"])
    out = capsys.readouterr().out
    assert "--force" in out
    assert "bc-export" in out


def test_main_remote_restart_trainer_requires_yes(capsys, monkeypatch):
    fake_ps = ("48213 .venv/bin/python scripts/resume_train.py "
               "checkpoints_entity/ck_000313794560.pt --config configs/train_v1.toml")

    class FakeCompleted:
        returncode = 0
        stdout = fake_ps

    monkeypatch.setattr(ctl, "run_fg", lambda plan, dry_run=False, timeout=60: FakeCompleted())
    with pytest.raises(SystemExit):
        ctl.main(["remote", "restart-trainer", "--host", HOST])
    err_or_out = capsys.readouterr()
    combined = err_or_out.out + err_or_out.err
    assert "restart-trainer plan" in combined
    assert "refusing to execute without --yes" in combined


def test_main_remote_restart_trainer_dry_run_does_not_execute(capsys, monkeypatch):
    fake_ps = ("48213 .venv/bin/python scripts/resume_train.py "
               "checkpoints_entity/ck_000313794560.pt --config configs/train_v1.toml")

    class FakeCompleted:
        returncode = 0
        stdout = fake_ps

    calls = []
    monkeypatch.setattr(ctl, "run_fg", lambda plan, dry_run=False, timeout=60:
                         calls.append(plan) or FakeCompleted())
    ctl.main(["remote", "restart-trainer", "--host", HOST, "--dry-run"])
    out = capsys.readouterr().out
    assert "[dry-run] not executing" in out
    assert "checkpoints_entity/ck_000313794560.pt" in out
