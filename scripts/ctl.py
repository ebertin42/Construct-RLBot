#!/usr/bin/env python3
"""Unified operations CLI for Construct-RLBot.

Codifies procedures that used to live only in shell history / journal
entries / Elliot's memory: local-proc status, the RLViser viewer start/stop
order, BC/SSL/parse-v5/export wrappers, crash-recovery checkpoint sweeps,
and remote (trainer-box) status + restart. Stdlib-only so it runs under
either the repo venv or a bare system python3:

    .venv/bin/python scripts/ctl.py status
    ./scripts/ctl.py status              (chmod +x'd, shebang-friendly)

Always run from the repo root (paths below are repo-root-relative).

Design: every *mutating* action is built as a pure "plan" (argv list + cwd +
env overlay, or a pgrep/pkill pattern) by a small builder function, so it can
be asserted on exactly in tests without ever touching a real process. The
`--dry-run` flag (present on every mutating subcommand, e.g. `bc start
--dry-run`) prints the plan instead of executing it. It's defined per-
subcommand rather than once at the top level: argparse's subparsers build a
fresh sub-namespace and blit it onto the parent, so a `--dry-run` given
*before* the subcommand name would silently be clobbered by the subparser's
own default -- put it after the subcommand. Process self-match-proofing follows the
pattern learned the hard way on this project: a pgrep/pkill -f pattern that
appears LITERALLY in the ssh/shell command line matches its own invocation
(see docs/training-journal.md, physics-nan-containment memory) -- so every
pattern used to find/kill a process is passed through bracket_proof() first.

Sources of truth read while writing this tool (kept in sync by hand, not by
import, since several are shell scripts): scripts/sync_remote.sh,
scripts/league_tick_loop.sh, scripts/parse_v5_inplace.sh,
scripts/pull_ssl_duels.sh, scripts/watch_loop.sh, scripts/dashboard.py,
scripts/bc_train.py, scripts/resume_train.py, scripts/eval_metrics.py,
docs/training-journal.md, and the operator's memory files (two-box-training,
rlviser-viewer-setup, local-relaunch-checklist, wsl-disk-space,
physics-nan-containment, ask-before-deploy, ballchasing-token).
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = REPO_ROOT / "logs"
VENV_BIN = REPO_ROOT / ".venv" / "bin"
VENV_PY = VENV_BIN / "python"

HOST_DEFAULT = "elliot@192.168.86.117"
RDIR_DEFAULT = "construct"
REMOTE_TRAIN_LOG = "checkpoints_entity/train_v1.log"

RLVISER_WIN_DIR = r"C:\Users\Elliot\AppData\Local\Construct"
RELAY_PS1 = RLVISER_WIN_DIR + r"\windows_viser_relay.ps1"
OVERLAY_PS1 = RLVISER_WIN_DIR + r"\windows_stream_overlay.ps1"
VISER_ADDR_FALLBACK = "172.17.176.1:45250"  # last-known host IP (memory: rlviser-viewer-setup)

CK_SWEEP_DEFAULT_DIRS = ["checkpoints_entity", "checkpoints_bc", "checkpoints_b"]


# ---------------------------------------------------------------------------
# self-match-proof pgrep/pkill pattern helper (pure, tested)
# ---------------------------------------------------------------------------

def bracket_proof(needle: str) -> str:
    """Turn a literal string into a pgrep/pkill -f (extended-regex) pattern
    that still matches the literal, but does NOT appear as a literal
    substring of itself -- so a `pkill -f <pattern>` invocation whose own
    cmdline contains <pattern> never matches its own ssh/shell process.

    Brackets exactly one character: prefers a '.' if present (this both
    escapes the regex metachar AND self-proofs, e.g. "bc_train.py" ->
    "bc_train[.]py"), else the middle character (e.g. "bc_train.py" minus
    its dot -> brackets a letter, e.g. "bc_trai[n].py"). A single character
    inside [] is always a literal match in POSIX extended regex, except '^'
    (negation) and ']' (needs special positioning) -- both are avoided.
    """
    if not needle:
        raise ValueError("bracket_proof: empty needle")
    if len(needle) == 1:
        return f"[{needle}]"

    idx = needle.find(".")
    if idx == -1:
        idx = len(needle) // 2

    # avoid bracketing a char that's ambiguous/invalid as a lone [x] class
    unsafe = {"^", "]"}
    if needle[idx] in unsafe:
        for cand in range(idx + 1, len(needle)):
            if needle[cand] not in unsafe:
                idx = cand
                break
        else:
            for cand in range(idx - 1, -1, -1):
                if needle[cand] not in unsafe:
                    idx = cand
                    break

    return needle[:idx] + "[" + needle[idx] + "]" + needle[idx + 1:]


def venv_prefixed_path(base_path: str) -> str:
    """PATH with the repo venv's bin/ prepended -- shell loops (league_tick_
    loop.sh, watch_loop.sh) that shell out to `python` need this on remote/
    bare-python boxes (commit d517ee0)."""
    return f"{VENV_BIN}:{base_path}"


# ---------------------------------------------------------------------------
# process discovery / log tailing (real subprocess calls -- not unit tested
# directly; only the pure functions above/below are)
# ---------------------------------------------------------------------------

def pgrep_lines(pattern: str) -> list[str]:
    """`pgrep -af <pattern>` locally. Empty list if nothing matches or pgrep
    itself is unavailable."""
    try:
        out = subprocess.run(
            ["pgrep", "-af", pattern], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode not in (0, 1):
        return []
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def is_running(pattern: str) -> bool:
    return bool(pgrep_lines(pattern))


def tail_lines(path: Path, n: int = 200) -> list[str]:
    """Cheap tail: fine for the log sizes these tools produce (few MB)."""
    if not path.is_file():
        return []
    try:
        with path.open("r", errors="replace") as f:
            lines = f.readlines()
    except OSError:
        return []
    return [ln.rstrip("\n") for ln in lines[-n:]]


def last_nonempty_line(lines: list[str]) -> str | None:
    for ln in reversed(lines):
        if ln.strip():
            return ln.strip()
    return None


# ---------------------------------------------------------------------------
# log-tail parsers (pure -- tested against real captured log fixtures)
# ---------------------------------------------------------------------------

BC_BATCH_RE = re.compile(
    r"bc epoch (\d+) batch (\d+)/(\d+) loss (\S+) lr (\S+) (\d+) samples/s"
)
BC_DONE_RE = re.compile(
    r"bc epoch (\d+) done: train_loss (\S+) val_loss (\S+) top1 (\S+) top3 (\S+) "
    r"recall_jump (\S+) recall_stall (\S+)"
)
BC_BANNER_RE = re.compile(
    r"bc: (\d+) train / (\d+) val shards, (\d+) batches/epoch x (\d+) epochs"
)


def parse_bc_log_tail(lines: list[str]) -> dict | None:
    """Most-recent-first scan for the richest recognizable bc_train.py line."""
    for ln in reversed(lines):
        m = BC_DONE_RE.search(ln)
        if m:
            return {
                "kind": "done", "epoch": int(m.group(1)),
                "train_loss": float(m.group(2)), "val_loss": float(m.group(3)),
                "top1": float(m.group(4)), "top3": float(m.group(5)),
            }
        m = BC_BATCH_RE.search(ln)
        if m:
            return {
                "kind": "batch", "epoch": int(m.group(1)),
                "batch": int(m.group(2)), "total": int(m.group(3)),
                "loss": float(m.group(4)), "samples_s": int(m.group(6)),
            }
        m = BC_BANNER_RE.search(ln)
        if m:
            return {
                "kind": "banner", "train_shards": int(m.group(1)),
                "val_shards": int(m.group(2)), "batches_per_epoch": int(m.group(3)),
                "epochs": int(m.group(4)),
            }
    return None


PARSE_V5_DONE_RE = re.compile(r"v5 parse exit: (\d+)/(\d+) batches")
PARSE_V5_BATCH_RE = re.compile(
    r"parsed=(\d+) skipped=(\d+) failed=(\d+) reset_states=(\d+)"
)
PARSE_V5_PROGRESS_RE = re.compile(
    r"\[batch_(\d+)\] \((\d+)\) — v5 re-parsing in place"
)


def parse_parse_v5_log_tail(lines: list[str]) -> dict | None:
    for ln in reversed(lines):
        m = PARSE_V5_DONE_RE.search(ln)
        if m:
            return {"kind": "done", "done": int(m.group(1)), "total": int(m.group(2))}
        m = PARSE_V5_BATCH_RE.search(ln)
        if m:
            return {
                "kind": "batch", "parsed": int(m.group(1)), "skipped": int(m.group(2)),
                "failed": int(m.group(3)), "reset_states": int(m.group(4)),
            }
        m = PARSE_V5_PROGRESS_RE.search(ln)
        if m:
            return {"kind": "progress", "batch": m.group(1), "count": int(m.group(2))}
    return None


EXPORT_SUMMARY_RE = re.compile(
    r"exported=(\d+) overwritten=(\d+) skipped_existing=(\d+) failed=(\d+) samples=(\d+)"
)


def parse_export_log_tail(lines: list[str]) -> dict | None:
    """bc-export only prints one line, at the very end -- no summary line yet
    means the run is still in progress (it's not a streaming logger)."""
    for ln in reversed(lines):
        m = EXPORT_SUMMARY_RE.search(ln)
        if m:
            return {
                "kind": "summary", "exported": int(m.group(1)),
                "overwritten": int(m.group(2)), "skipped_existing": int(m.group(3)),
                "failed": int(m.group(4)), "samples": int(m.group(5)),
            }
    return None


SSL_PROGRESS_RE = re.compile(
    r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}) progress: (\d+) this run / (\d+) total "
    r"\((\d+) deduped, (\d+) failed\).*?([\d.]+)/h(?:, host free ([\d.]+)G)?"
)
SSL_PAGE_RE = re.compile(
    r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}) page: .* oldest created (\S+)"
)


def parse_ssl_log_tail(lines: list[str]) -> dict | None:
    for ln in reversed(lines):
        m = SSL_PROGRESS_RE.search(ln)
        if m:
            d = {
                "kind": "progress", "ts": m.group(1), "this_run": int(m.group(2)),
                "total": int(m.group(3)), "deduped": int(m.group(4)),
                "failed": int(m.group(5)), "rate_per_h": float(m.group(6)),
            }
            if m.group(7) is not None:
                d["host_free_gb"] = float(m.group(7))
            return d
    for ln in reversed(lines):
        m = SSL_PAGE_RE.search(ln)
        if m:
            return {"kind": "page", "ts": m.group(1), "oldest_created": m.group(2)}
    return None


ITER_LINE_RE = re.compile(
    r"iter (\d+) steps ([\d,]+) sps ([\d,]+) ep_rew ([-\d.]+) "
    r"pi_loss ([-\d.]+) v_loss ([-\d.]+) ent ([-\d.]+) clip ([-\d.]+)"
    r"(?: kick_kl ([-\d.]+) lambda_k ([-\d.]+))?"
    r"(?: kl_pri ([-\d.]+) lambda_p ([-\d.]+))?"
)
RESUMED_AT_RE = re.compile(r"resumed at ([\d,]+) steps")


def parse_iter_line(line: str) -> dict | None:
    m = ITER_LINE_RE.search(line)
    if not m:
        return None
    d = {
        "iter": int(m.group(1)), "steps": int(m.group(2).replace(",", "")),
        "sps": int(m.group(3).replace(",", "")), "ep_rew": float(m.group(4)),
        "ent": float(m.group(7)),
    }
    if m.group(9) is not None:
        d["kick_kl"] = float(m.group(9))
        d["lambda_k"] = float(m.group(10))
    if m.group(11) is not None:
        d["kl_pri"] = float(m.group(11))
        d["lambda_p"] = float(m.group(12))
    return d


def last_iter_line(lines: list[str]) -> dict | None:
    for ln in reversed(lines):
        d = parse_iter_line(ln)
        if d:
            return d
    return None


def parse_remote_ps_for_resume_ck(ps_lines: list[str]) -> str | None:
    """Given `pgrep -af resume_train.py` output lines, extract the checkpoint
    literal (first .pt-suffixed token) from the first matching line."""
    for ln in ps_lines:
        if "resume_train.py" not in ln:
            continue
        for tok in ln.split():
            if tok.endswith(".pt"):
                return tok
    return None


def verify_restart(lines: list[str]) -> dict:
    """Post-launch check: did the log show a resume line, and is the KL-prior
    anchor confirmed live (a kl_pri field on a subsequent iter line)?"""
    resumed = any(RESUMED_AT_RE.search(ln) for ln in lines)
    anchor_confirmed = any("kl_pri" in ln for ln in lines)
    return {"resumed": resumed, "anchor_confirmed": anchor_confirmed}


# ---------------------------------------------------------------------------
# ck-sweep: crash-recovery step zero (pure logic; real torch.load lazily)
# ---------------------------------------------------------------------------

def newest_checkpoints(dir_path: Path, n: int = 3) -> list[Path]:
    if not dir_path.is_dir():
        return []
    cks = [
        p for p in dir_path.iterdir()
        if p.is_file() and p.name.startswith("ck_") and p.name.endswith(".pt")
    ]
    cks.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cks[:n]


def classify_checkpoint(path: Path) -> str:
    """Returns "ok" | "zero_byte" | "unloadable". Lazily imports torch so
    this module stays importable (and the CLI's non-torch subcommands stay
    usable) without torch installed; if torch truly isn't available, only
    the 0-byte check runs."""
    if path.stat().st_size == 0:
        return "zero_byte"
    try:
        import torch
    except ImportError:
        return "ok"
    try:
        torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception:
        return "unloadable"
    return "ok"


def quarantine(path: Path, dry_run: bool = False) -> Path:
    target = path.with_name(path.name + ".corrupt")
    if not dry_run:
        path.rename(target)
    return target


def sweep_dir(dir_path: Path, n: int = 3, dry_run: bool = False) -> dict:
    checked = newest_checkpoints(dir_path, n)
    results = []
    quarantined = []
    for p in checked:
        status = classify_checkpoint(p)
        results.append({"path": p, "status": status})
        if status != "ok":
            q = quarantine(p, dry_run=dry_run)
            quarantined.append({"path": p, "quarantined_to": q, "status": status})
    intact = [r["path"] for r in results if r["status"] == "ok"]
    return {
        "dir": dir_path, "checked": results, "quarantined": quarantined,
        "newest_intact": intact[0] if intact else None,
    }


def ck_sweep(dirs: list[Path], n: int = 3, dry_run: bool = False) -> dict:
    return {str(d): sweep_dir(d, n=n, dry_run=dry_run) for d in dirs}


# ---------------------------------------------------------------------------
# mutating-action plan builders (pure -- argv/cwd/env only, no execution)
# ---------------------------------------------------------------------------
# Every builder returns {"argv": [...], "cwd": str|None, "env": {...}}. `env`
# is an OVERLAY merged onto a copy of the caller's environment at run time,
# never the full environment itself (keeps builders pure/small to assert on).

def _plan(argv: list[str], cwd: str | None = None, env: dict | None = None) -> dict:
    return {"argv": list(argv), "cwd": cwd, "env": dict(env) if env else {}}


# --- viewer ------------------------------------------------------------

def relay_check_plan() -> dict:
    return _plan([
        "powershell.exe", "-NoProfile", "-Command",
        "Get-NetUDPEndpoint -LocalPort 34254,45250 -ErrorAction SilentlyContinue",
    ])


def relay_start_plan() -> dict:
    return _plan([
        "cmd.exe", "/c", "start", "", "powershell.exe", "-NoProfile",
        "-ExecutionPolicy", "Bypass", "-File", RELAY_PS1,
    ], cwd="/mnt/c")


def rlviser_start_plan() -> dict:
    return _plan(["cmd.exe", "/c", "start", "", "/D", RLVISER_WIN_DIR, "rlviser.exe"],
                 cwd="/mnt/c")


def overlay_start_plan() -> dict:
    return _plan([
        "cmd.exe", "/c", "start", "", "powershell.exe", "-NoProfile",
        "-ExecutionPolicy", "Bypass", "-File", OVERLAY_PS1,
    ], cwd="/mnt/c")


def watch_loop_start_plan(viser_addr: str, rotate_secs: int = 300) -> dict:
    """Must run from the repo root -- the classic cwd trap is launching this
    from a shell that `cd /mnt/c`'d for the relay/rlviser/overlay steps."""
    return _plan(
        ["setsid", "./scripts/watch_loop.sh", str(rotate_secs)],
        cwd=str(REPO_ROOT),
        env={"CONSTRUCT_VISER_ADDR": viser_addr},
    )


def rlviser_stop_plan() -> dict:
    return _plan([
        "powershell.exe", "-NoProfile", "-Command",
        "Stop-Process -Name rlviser -Force -ErrorAction SilentlyContinue",
    ])


def viewer_off_kill_patterns() -> list[str]:
    """Bracket-proof pgrep -f patterns for the two local processes `viewer
    off` tears down; relay and overlay stay up (cheap, GPU-free)."""
    return [bracket_proof("watch_loop.sh"), bracket_proof("watch.py")]


# --- bc ------------------------------------------------------------------

def bc_start_plan(epochs: int | None = None) -> dict:
    argv = ["setsid", "nice", "-n", "10", str(VENV_PY), "scripts/bc_train.py",
            "--config", "configs/bc_v1.toml"]
    if epochs is not None:
        argv += ["--epochs", str(epochs)]
    return _plan(argv, cwd=str(REPO_ROOT))


def bc_stop_pattern() -> str:
    return bracket_proof("bc_train.py")


# --- ssl -------------------------------------------------------------------

def ssl_start_plan() -> dict:
    return _plan(["setsid", "nice", "-n", "15", "./scripts/pull_ssl_duels.sh"],
                 cwd=str(REPO_ROOT))


def ssl_stop_pattern() -> str:
    return bracket_proof("pull_ssl_duels.sh")


# --- parse-v5 ----------------------------------------------------------

def parse_v5_start_plan() -> dict:
    return _plan(["setsid", "nice", "-n", "15", "./scripts/parse_v5_inplace.sh"],
                 cwd=str(REPO_ROOT))


# --- export (bc-export) -------------------------------------------------

def export_start_plan(force: bool = False) -> dict:
    argv = ["setsid", "nice", "-n", "15", "./target/release/bc-export",
            "--shards", "data/shards_v4", "--out", "data/bc"]
    if force:
        argv.append("--force")
    return _plan(argv, cwd=str(REPO_ROOT))


# --- loops -----------------------------------------------------------------

def sync_start_plan() -> dict:
    return _plan(["setsid", "./scripts/sync_remote.sh"], cwd=str(REPO_ROOT))


def dashboard_start_plan(port: int = 8420) -> dict:
    return _plan(["setsid", str(VENV_PY), "scripts/dashboard.py", str(port)],
                 cwd=str(REPO_ROOT))


def league_loop_start_plan() -> dict:
    return _plan(
        ["setsid", "./scripts/league_tick_loop.sh"],
        cwd=str(REPO_ROOT),
        env={"PATH": venv_prefixed_path(os.environ.get("PATH", ""))},
    )


LOOP_SPECS = {
    "sync": ("sync_remote.sh", sync_start_plan, "sync.log"),
    "dashboard": ("dashboard.py", dashboard_start_plan, "dashboard.log"),
    "league-loop": ("league_tick_loop.sh", league_loop_start_plan, "league_tick.log"),
}


# --- eval --------------------------------------------------------------

def eval_plan(ck: str, nice: int = 15) -> dict:
    return _plan(
        ["nice", "-n", str(nice), str(VENV_PY), "scripts/eval_metrics.py", ck],
        cwd=str(REPO_ROOT),
    )


# --- remote ------------------------------------------------------------

def remote_ssh(host: str, *remote_argv: str) -> dict:
    return _plan(["ssh", host, *remote_argv])


def remote_status_plans(host: str, rdir: str) -> dict:
    return {
        "trainer_log_tail": remote_ssh(host, "tail", "-n", "500", f"{rdir}/{REMOTE_TRAIN_LOG}"),
        "league_loop_check": remote_ssh(host, "pgrep", "-af", "league_tick_loop.sh"),
    }


def remote_discover_trainer_plan(host: str) -> dict:
    return remote_ssh(host, "pgrep", "-af", "resume_train.py")


def remote_kill_plan(host: str, bracketed_pattern: str) -> dict:
    return remote_ssh(host, "pkill", "-f", bracketed_pattern)


def remote_launch_command(
    rdir: str, resume_ck: str, config: str, log_path: str,
    reward_config: str | None = None, league: bool = False,
    kl_prior: str | None = None, kl_prior_lambda: float | None = None,
) -> str:
    """Builds the single shell command string ssh runs on the remote box.
    A single string (not argv) because nohup/setsid/redirection/disown are
    shell syntax -- ssh's remote command is inherently a shell one-liner
    (SSH RULES: minimal, single-purpose per call; this IS the one launch
    call). Every dynamic value is shlex.quote'd."""
    parts = [
        "nohup", "setsid", ".venv/bin/python", "scripts/resume_train.py",
        resume_ck, "--config", config,
    ]
    if reward_config:
        parts += ["--reward-config", reward_config]
    if league:
        parts.append("--league")
    if kl_prior:
        parts += ["--kl-prior", kl_prior]
    if kl_prior_lambda is not None:
        parts += ["--kl-prior-lambda", str(kl_prior_lambda)]
    quoted = " ".join(shlex.quote(p) for p in parts)
    return (
        f"cd {shlex.quote(rdir)} && {quoted} "
        f">> {shlex.quote(log_path)} 2>&1 < /dev/null & disown"
    )


def remote_launch_plan(host: str, rdir: str, resume_ck: str, config: str,
                        reward_config: str | None = None, league: bool = False,
                        kl_prior: str | None = None,
                        kl_prior_lambda: float | None = None) -> dict:
    cmd = remote_launch_command(
        rdir, resume_ck, config, REMOTE_TRAIN_LOG,
        reward_config=reward_config, league=league,
        kl_prior=kl_prior, kl_prior_lambda=kl_prior_lambda,
    )
    return remote_ssh(host, cmd)


def remote_verify_plan(host: str, rdir: str) -> dict:
    return remote_ssh(host, "tail", "-n", "20", f"{rdir}/{REMOTE_TRAIN_LOG}")


# ---------------------------------------------------------------------------
# plan execution (real subprocess -- never exercised by tests)
# ---------------------------------------------------------------------------

def render_plan(plan: dict) -> str:
    s = shlex.join(plan["argv"])
    if plan.get("cwd"):
        s = f"(cwd={plan['cwd']}) {s}"
    if plan.get("env"):
        envs = " ".join(f"{k}={v}" for k, v in plan["env"].items())
        s = f"{envs} {s}"
    return s


def run_bg(plan: dict, log_path: Path | None, dry_run: bool = False) -> None:
    """Detached background launch (setsid-style): stdout/stderr appended to
    log_path if given."""
    if dry_run:
        dest = f" >> {log_path} 2>&1" if log_path else ""
        print(f"[dry-run] would run: {render_plan(plan)}{dest}")
        return
    env = dict(os.environ)
    env.update(plan.get("env", {}))
    logf = None
    stdout = stderr = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "ab")
        stdout = logf
        stderr = subprocess.STDOUT
    try:
        subprocess.Popen(
            plan["argv"], cwd=plan.get("cwd"), env=env,
            stdout=stdout, stderr=stderr, stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    finally:
        if logf is not None:
            logf.close()


def run_fg(plan: dict, dry_run: bool = False, timeout: float | None = 60) -> subprocess.CompletedProcess | None:
    """Foreground call, output captured (status/ssh/powershell reads)."""
    if dry_run:
        print(f"[dry-run] would run: {render_plan(plan)}")
        return None
    env = dict(os.environ)
    env.update(plan.get("env", {}))
    try:
        return subprocess.run(
            plan["argv"], cwd=plan.get("cwd"), env=env,
            capture_output=True, text=True, timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"warning: {shlex.join(plan['argv'])} failed: {e}", file=sys.stderr)
        return None


def kill_pattern(pattern: str, dry_run: bool = False) -> None:
    plan = _plan(["pkill", "-f", pattern])
    if dry_run:
        print(f"[dry-run] would run: {render_plan(plan)}")
        return
    subprocess.run(plan["argv"])


def disk_free_host_gb() -> float | None:
    """Windows-host C: free, in GB. WSL df LIES (vhdx nominal size, not real
    host free) -- never report that; this is the only trustworthy source."""
    plan = _plan([
        "powershell.exe", "-NoProfile", "-Command",
        "[math]::Round((Get-PSDrive C).Free/1GB,1)",
    ], cwd="/mnt/c")
    out = run_fg(plan, dry_run=False, timeout=60)
    if out is None or out.returncode != 0:
        return None
    try:
        return float(out.stdout.strip())
    except ValueError:
        return None


def count_ssl_replays(ssl_dir: Path) -> int:
    """Recursive walk, never a shell glob -- data/replays/ssl holds 100k+
    files and `ls *.replay` blows ARG_MAX."""
    n = 0
    for _, _, files in os.walk(ssl_dir):
        n += sum(1 for f in files if f.endswith(".replay"))
    return n


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

LOCAL_PROC_SPECS = [
    ("bc-train", "bc_train.py", "bc_train.log"),
    ("parse", "parse_v5_inplace.sh", "parse_v5.log"),
    ("export", "bc-export", "bc_export.log"),
    ("ssl", "pull_ssl_duels.sh", "ssl_pull.log"),
    ("sync", "sync_remote.sh", "sync.log"),
    ("dashboard", "dashboard.py", "dashboard.log"),
    ("league-loop", "league_tick_loop.sh", "league_tick.log"),
    ("watch-loop", "watch_loop.sh", "watch_loop.log"),
]


def key_stat_for(name: str, lines: list[str]) -> str:
    if name == "bc-train":
        d = parse_bc_log_tail(lines)
        if not d:
            return "no output yet"
        if d["kind"] == "batch":
            return (f"epoch {d['epoch']} batch {d['batch']}/{d['total']} "
                     f"loss {d['loss']} {d['samples_s']} samples/s")
        if d["kind"] == "done":
            return f"epoch {d['epoch']} done: val_loss {d['val_loss']} top1 {d['top1']}"
        return f"{d['train_shards']} train shards, {d['epochs']} epochs"
    if name == "parse":
        d = parse_parse_v5_log_tail(lines)
        if not d:
            return "no output yet"
        if d["kind"] == "done":
            return f"{d['done']}/{d['total']} batches (ALL V5-PARSED)" if d["done"] >= d["total"] else f"{d['done']}/{d['total']} batches"
        if d["kind"] == "batch":
            return f"parsed={d['parsed']} skipped={d['skipped']} failed={d['failed']}"
        return f"batch_{d['batch']} ({d['count']}) parsing..."
    if name == "export":
        d = parse_export_log_tail(lines)
        if not d:
            return "running (no summary yet)"
        return (f"exported={d['exported']} overwritten={d['overwritten']} "
                f"skipped={d['skipped_existing']} failed={d['failed']}")
    if name == "ssl":
        d = parse_ssl_log_tail(lines)
        if not d:
            return "no output yet"
        if d["kind"] == "progress":
            return f"{d['this_run']}/{d['total']} this run, {d['rate_per_h']}/h"
        return f"paging (oldest created {d['oldest_created']})"
    ln = last_nonempty_line(lines)
    return ln if ln else "no output yet"


def sync_freshness(repo_root: Path) -> str | None:
    """sync_remote.sh prints nothing on success (silent rsync loop) -- the
    only honest freshness signal is the mtime of what it actually writes."""
    candidates = [
        repo_root / "checkpoints_entity" / "train_remote.log",
        repo_root / "league" / "registry_remote.jsonl",
    ]
    mtimes = [p.stat().st_mtime for p in candidates if p.is_file()]
    if not mtimes:
        return None
    age = time.time() - max(mtimes)
    return f"last synced {int(age)}s ago"


def local_status(repo_root: Path = REPO_ROOT) -> list[dict]:
    rows = []
    for name, pattern, log_name in LOCAL_PROC_SPECS:
        up = is_running(bracket_proof(pattern))
        lines = tail_lines(LOGS_DIR / log_name)
        if name == "sync" and up:
            stat = sync_freshness(repo_root) or "up, no sync yet"
        else:
            stat = key_stat_for(name, lines)
        rows.append({"name": name, "up": up, "stat": stat})
    return rows


def newest_ck_report(dirs: list[str], repo_root: Path = REPO_ROOT) -> dict:
    out = {}
    for d in dirs:
        cks = newest_checkpoints(repo_root / d, n=1)
        out[d] = cks[0].name if cks else None
    return out


def print_status(args) -> None:
    print("=== local procs ===")
    for row in local_status():
        state = "UP  " if row["up"] else "DOWN"
        print(f"  {state}  {row['name']:<12} {row['stat']}")

    print("\n=== remote ===")
    plans = remote_status_plans(args.host, args.remote_dir)
    tail = run_fg(plans["trainer_log_tail"], dry_run=False, timeout=30)
    if tail and tail.returncode == 0:
        iter_line = last_iter_line(tail.stdout.splitlines())
        print(f"  trainer: {iter_line if iter_line else '(no iter line found)'}")
    else:
        print("  trainer: unreachable (ssh failed)")
    league = run_fg(plans["league_loop_check"], dry_run=False, timeout=30)
    league_up = bool(league and league.returncode == 0 and league.stdout.strip())
    print(f"  league loop: {'UP' if league_up else 'DOWN'}")

    print("\n=== disk (Windows host C:, real free — WSL df lies) ===")
    free = disk_free_host_gb()
    print(f"  {free} GB free" if free is not None else "  unavailable")

    print("\n=== SSL corpus ===")
    n = count_ssl_replays(REPO_ROOT / "data" / "replays" / "ssl")
    print(f"  {n} replays on disk")

    print("\n=== newest checkpoints ===")
    for d, ck in newest_ck_report(CK_SWEEP_DEFAULT_DIRS + ["checkpoints"]).items():
        print(f"  {d:<20} {ck if ck else '(none)'}")


# ---------------------------------------------------------------------------
# subcommand handlers
# ---------------------------------------------------------------------------

def cmd_viewer(args) -> None:
    if args.action == "status":
        relay_bound = None
        out = run_fg(relay_check_plan(), dry_run=False, timeout=15)
        if out is not None and out.returncode == 0:
            relay_bound = bool(out.stdout.strip())
        print(f"relay bound: {relay_bound}")
        print(f"watch_loop.sh: {'UP' if is_running(bracket_proof('watch_loop.sh')) else 'DOWN'}")
        print(f"watch.py: {'UP' if is_running(bracket_proof('watch.py')) else 'DOWN'}")
        return

    if args.action == "on":
        out = run_fg(relay_check_plan(), dry_run=args.dry_run, timeout=15)
        bound = bool(out and out.returncode == 0 and out.stdout.strip())
        if not bound:
            run_bg(relay_start_plan(), log_path=None, dry_run=args.dry_run)
        run_bg(rlviser_start_plan(), log_path=None, dry_run=args.dry_run)
        run_bg(overlay_start_plan(), log_path=None, dry_run=args.dry_run)
        if is_running(bracket_proof("watch_loop.sh")) and not args.dry_run:
            print("refusing to double-launch: watch_loop.sh already running")
            return
        addr = os.environ.get("CONSTRUCT_VISER_ADDR", VISER_ADDR_FALLBACK)
        run_bg(watch_loop_start_plan(addr), log_path=LOGS_DIR / "watch_loop.log",
               dry_run=args.dry_run)
        return

    if args.action == "off":
        for pat in viewer_off_kill_patterns():
            kill_pattern(pat, dry_run=args.dry_run)
        run_fg(rlviser_stop_plan(), dry_run=args.dry_run, timeout=15)
        print("relay/overlay left running (cheap, no GPU cost)")


def _start_stop_status(kind: str, action: str, start_plan_fn, stop_pattern_fn,
                        log_name: str, dry_run: bool, extra_start_kwargs=None) -> None:
    # pgrep -f accepts the bracket-proofed pattern directly (it's still a
    # valid regex matching the same literal cmdline) -- no need to strip it.
    pattern = stop_pattern_fn() if stop_pattern_fn else None
    if action == "status":
        up = is_running(pattern) if pattern else False
        lines = tail_lines(LOGS_DIR / log_name)
        print(f"{kind}: {'UP' if up else 'DOWN'}")
        print(f"  {key_stat_for(kind, lines)}")
        return
    if action == "start":
        if pattern and is_running(pattern) and not dry_run:
            print(f"refusing to double-launch {kind}: already running ({pattern})")
            return
        plan = start_plan_fn(**(extra_start_kwargs or {}))
        run_bg(plan, log_path=LOGS_DIR / log_name, dry_run=dry_run)
        return
    if action == "stop":
        kill_pattern(stop_pattern_fn(), dry_run=dry_run)
        return
    raise ValueError(action)


def cmd_bc(args) -> None:
    _start_stop_status(
        "bc-train", args.action, bc_start_plan, bc_stop_pattern, "bc_train.log",
        args.dry_run, extra_start_kwargs={"epochs": args.epochs},
    )


def cmd_ssl(args) -> None:
    _start_stop_status("ssl", args.action, ssl_start_plan, ssl_stop_pattern,
                        "ssl_pull.log", args.dry_run)


def cmd_parse_v5(args) -> None:
    pattern = bracket_proof("parse_v5_inplace.sh")
    if args.action == "status":
        up = is_running(pattern)
        lines = tail_lines(LOGS_DIR / "parse_v5.log")
        print(f"parse: {'UP' if up else 'DOWN'}")
        print(f"  {key_stat_for('parse', lines)}")
        return
    if is_running(pattern) and not args.dry_run:
        print("refusing to double-launch parse-v5: already running")
        return
    run_bg(parse_v5_start_plan(), log_path=LOGS_DIR / "parse_v5.log", dry_run=args.dry_run)


def cmd_export(args) -> None:
    pattern = bracket_proof("bc-export")
    if args.action == "status":
        up = is_running(pattern)
        lines = tail_lines(LOGS_DIR / "bc_export.log")
        print(f"export: {'UP' if up else 'DOWN'}")
        print(f"  {key_stat_for('export', lines)}")
        return
    if is_running(pattern) and not args.dry_run:
        print("refusing to double-launch export: already running")
        return
    run_bg(export_start_plan(force=args.force), log_path=LOGS_DIR / "bc_export.log",
           dry_run=args.dry_run)


def cmd_loops(args) -> None:
    assert args.action == "up"
    for name, (pattern, plan_fn, log_name) in LOOP_SPECS.items():
        if is_running(bracket_proof(pattern)):
            print(f"{name}: already up")
            continue
        print(f"{name}: starting")
        run_bg(plan_fn(), log_path=LOGS_DIR / log_name, dry_run=args.dry_run)


def cmd_eval(args) -> None:
    plan = eval_plan(args.checkpoint, nice=args.nice)
    if args.dry_run:
        print(f"[dry-run] would run: {render_plan(plan)}")
        return
    proc = subprocess.run(plan["argv"], cwd=plan.get("cwd"))
    sys.exit(proc.returncode)


def cmd_ck_sweep(args) -> None:
    dirs = [REPO_ROOT / d for d in (args.dirs or CK_SWEEP_DEFAULT_DIRS)]
    report = ck_sweep(dirs, n=args.n, dry_run=args.dry_run)
    for d, info in report.items():
        print(f"{d}:")
        if not info["checked"]:
            print("  (no checkpoints found)")
            continue
        for r in info["checked"]:
            print(f"  {r['path'].name}: {r['status']}")
        for q in info["quarantined"]:
            verb = "would quarantine" if args.dry_run else "quarantined"
            print(f"  {verb} {q['path'].name} -> {q['quarantined_to'].name}")
        ni = info["newest_intact"]
        print(f"  newest intact: {ni.name if ni else '(none!)'}")


def cmd_recover(args) -> None:
    print("=== step 1: ck-sweep ===")
    dirs = [REPO_ROOT / d for d in CK_SWEEP_DEFAULT_DIRS]
    report = ck_sweep(dirs, n=3, dry_run=args.dry_run)
    for d, info in report.items():
        for q in info["quarantined"]:
            verb = "would quarantine" if args.dry_run else "quarantined"
            print(f"  {verb} {q['path'].name} in {d}")

    print("\n=== step 2: loops up ===")
    for name, (pattern, plan_fn, log_name) in LOOP_SPECS.items():
        if is_running(bracket_proof(pattern)):
            print(f"  {name}: already up")
            continue
        print(f"  {name}: starting")
        run_bg(plan_fn(), log_path=LOGS_DIR / log_name, dry_run=args.dry_run)

    print("\n=== step 3: bc / parse / ssl ===")
    offers = [
        ("bc-train", "bc_train.py", bc_start_plan, "bc_train.log"),
        ("parse-v5", "parse_v5_inplace.sh", parse_v5_start_plan, "parse_v5.log"),
        ("ssl", "pull_ssl_duels.sh", ssl_start_plan, "ssl_pull.log"),
    ]
    for name, pattern, plan_fn, log_name in offers:
        if is_running(bracket_proof(pattern)):
            print(f"  {name}: already up")
            continue
        do_it = args.all
        if not do_it and not args.dry_run and sys.stdin.isatty():
            ans = input(f"  restart {name}? [y/N] ").strip().lower()
            do_it = ans == "y"
        if do_it or args.dry_run:
            print(f"  {name}: starting")
            run_bg(plan_fn(), log_path=LOGS_DIR / log_name, dry_run=args.dry_run)
        else:
            print(f"  {name}: skipped")


def cmd_remote(args) -> None:
    if args.action == "status":
        plans = remote_status_plans(args.host, args.remote_dir)
        if args.dry_run:
            for name, plan in plans.items():
                print(f"[dry-run] {name}: would run: {render_plan(plan)}")
            return
        tail = run_fg(plans["trainer_log_tail"], dry_run=False, timeout=30)
        if tail and tail.returncode == 0:
            iter_line = last_iter_line(tail.stdout.splitlines())
            print(f"trainer: {iter_line if iter_line else '(no iter line found)'}")
        else:
            print("trainer: unreachable (ssh failed)")
        league = run_fg(plans["league_loop_check"], dry_run=False, timeout=30)
        league_up = bool(league and league.returncode == 0 and league.stdout.strip())
        print(f"league loop: {'UP' if league_up else 'DOWN'}")
        return

    assert args.action == "restart-trainer"
    discover = remote_discover_trainer_plan(args.host)
    out = run_fg(discover, dry_run=False, timeout=30)
    ps_lines = out.stdout.splitlines() if out and out.returncode == 0 else []
    current_ck = parse_remote_ps_for_resume_ck(ps_lines)
    if current_ck is None and not args.resume_ck:
        print("could not discover the running checkpoint via ssh pgrep, and "
              "no --resume-ck given -- aborting", file=sys.stderr)
        sys.exit(1)

    resume_ck = args.resume_ck or current_ck
    kill_pattern_str = bracket_proof(current_ck) if current_ck else None

    kill_plan = remote_kill_plan(args.host, kill_pattern_str) if kill_pattern_str else None
    launch_plan = remote_launch_plan(
        args.host, args.remote_dir, resume_ck, args.config,
        reward_config=args.reward_config, league=args.league,
        kl_prior=args.kl_prior, kl_prior_lambda=args.kl_prior_lambda,
    )
    verify_plan = remote_verify_plan(args.host, args.remote_dir)

    print("=== restart-trainer plan ===")
    print(f"  discovered running ck: {current_ck}")
    if kill_plan:
        print(f"  1. kill:   {render_plan(kill_plan)}")
    else:
        print("  1. kill:   (nothing running to kill)")
    print(f"  2. launch: {render_plan(launch_plan)}")
    print(f"  3. verify: {render_plan(verify_plan)}")

    if args.dry_run:
        print("[dry-run] not executing")
        return
    if not args.yes:
        print("refusing to execute without --yes (plan printed above)", file=sys.stderr)
        sys.exit(1)

    if kill_plan:
        run_fg(kill_plan, dry_run=False, timeout=30)
        time.sleep(2)
    run_fg(launch_plan, dry_run=False, timeout=30)
    time.sleep(5)
    verify = run_fg(verify_plan, dry_run=False, timeout=30)
    if verify and verify.returncode == 0:
        result = verify_restart(verify.stdout.splitlines())
        print(f"verify: resumed={result['resumed']} anchor_confirmed={result['anchor_confirmed']}")
    else:
        print("verify: ssh failed", file=sys.stderr)


# ---------------------------------------------------------------------------
# argparse wiring
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ctl.py", description=__doc__.splitlines()[0])
    # NOTE: --dry-run is defined per-subcommand below, not here -- see the
    # module docstring for why a top-level one would silently misbehave.
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="one-screen operational overview")

    sp = sub.add_parser("viewer", help="RLViser stream viewer stack")
    sp.add_argument("action", choices=["on", "off", "status"])
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("bc", help="BC pretrain (scripts/bc_train.py)")
    sp.add_argument("action", choices=["start", "stop", "status"])
    sp.add_argument("--epochs", type=int, default=None)
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("ssl", help="ballchasing SSL corpus pull")
    sp.add_argument("action", choices=["start", "stop", "status"])
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("parse-v5", help="in-place v5 shard re-parse")
    sp.add_argument("action", choices=["start", "status"])
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("export", help="bc-export (v4/v5 shards -> bc tensors)")
    sp.add_argument("action", choices=["start", "status"])
    sp.add_argument("--force", action="store_true")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("loops", help="sync/dashboard/league-loop supervisor")
    sp.add_argument("action", choices=["up"])
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("eval", help="run scripts/eval_metrics.py on a checkpoint")
    sp.add_argument("checkpoint")
    sp.add_argument("--nice", type=int, default=15)
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("ck-sweep", help="find/quarantine corrupt newest checkpoints")
    sp.add_argument("dirs", nargs="*", default=None)
    sp.add_argument("-n", type=int, default=3, help="newest N per dir to check")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("recover", help="full crash-recovery runbook")
    sp.add_argument("--all", action="store_true", help="restart every offered service, no prompt")
    sp.add_argument("--dry-run", action="store_true")

    sp = sub.add_parser("remote", help="remote trainer box (ssh)")
    sp.add_argument("action", choices=["status", "restart-trainer"])
    sp.add_argument("--host", default=HOST_DEFAULT)
    sp.add_argument("--remote-dir", default=RDIR_DEFAULT)
    sp.add_argument("--resume-ck", default=None)
    sp.add_argument("--config", default="configs/train_v1.toml")
    sp.add_argument("--reward-config", default=None)
    sp.add_argument("--league", action="store_true")
    sp.add_argument("--kl-prior", default=None)
    sp.add_argument("--kl-prior-lambda", type=float, default=None)
    sp.add_argument("--yes", action="store_true")
    sp.add_argument("--dry-run", action="store_true")

    return p


HANDLERS = {
    "status": print_status,
    "viewer": cmd_viewer,
    "bc": cmd_bc,
    "ssl": cmd_ssl,
    "parse-v5": cmd_parse_v5,
    "export": cmd_export,
    "loops": cmd_loops,
    "eval": cmd_eval,
    "ck-sweep": cmd_ck_sweep,
    "recover": cmd_recover,
    "remote": cmd_remote,
}


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "status":
        args.host = HOST_DEFAULT
        args.remote_dir = RDIR_DEFAULT
    HANDLERS[args.command](args)


if __name__ == "__main__":
    main()
