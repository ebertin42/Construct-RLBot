#!/usr/bin/env python3
"""Unattended GATED HILL-CLIMB: PPO as a mutation operator that cannot regress.

WHY THIS EXISTS (docs/training-journal.md, 2026-07-19 ~17:40 -> 2026-07-20
~04:30). Head-to-head measurement -- both side orders, against the frozen
champion -- proved this project's PPO self-play DEGRADES the policy. Controlled
145-iteration arms, all from ck_000320471040, all measured identically:

    arm A  ent .01,  no anchor, reward v4.1        22.0-25.5%
    arm B  ent .001, no anchor, reward v4.1        10.4-11.4%
    arm C  ent .003, league 0.5, clean pool        27.1%
    arm E  ent .01,  no anchor, reward v3          29.6%
    arm F  self-anchor lambda_p 0.2                32.8%
    arm G  self-anchor lambda_p 0.5                46.4%
    arm D  v0-teacher kickstart, lambda_k 0.36     49.2%
    null control (ck_000320471040 vs ITSELF)       51.1%, rerun variance ~3%

Two things fall out of that table. A competent ANCHOR is the dominant variable
(~20 points, holding reward fixed); reward choice is a minor term (~5 points).
And -- stated plainly, because it is the premise of this whole script -- NOTHING
TESTED RELIABLY IMPROVES THE POLICY. Arm D holds parity. Holding is not
progress.

So this script stops trusting training. It repeatedly attempts a BOUNDED run
from the current champion, GATES the result against that champion, promotes the
rare winner and discards the rest. Expected behavior is MOSTLY FAILURES: a run
of rejects is the design working, not a bug. What makes that acceptable is that
a loss costs nothing (the champion pointer never moves) while a win compounds
(the champion is simultaneously the gate reference, the KL anchor, and the
league seed -- so a promotion advances all three at once).

DIVISION OF LABOR -- this script does NOT implement any of the following, it
CALLS them, so there is exactly one implementation of each in the repo:
  * gating, the mandatory both-side-orders match, promotion, the league append,
    and the champion-history audit trail -> scripts/champion_gate.py
  * self-match-proof pgrep/pkill patterns                  -> scripts/ctl.py
  * bounded anchored training                              -> scripts/resume_train.py

SSH RULES, learned the hard way this session (every one of these is a bug that
actually bit):
  * MINIMAL, SINGLE-PURPOSE ssh calls. A compound command that combines a kill
    with a launch dies with exit 255. Nothing here combines two jobs; this
    script never kills a remote process at all.
  * Launches use `nohup setsid ... & disown` so the trainer outlives the ssh.
  * Every pgrep pattern goes through ctl.bracket_proof, because a pattern that
    appears LITERALLY in the ssh command line matches its own invocation.
  * An ssh transport failure (255) is NOT evidence that training finished.
    Tool/process timeouts have silently killed detached jobs mid-run this
    session, so completion is confirmed by the CHECKPOINTS ON DISK, never by
    "the poll stopped seeing a process".

SAFETY: attempt checkpoints are written to their own remote directory under
checkpoints_hc/ and pulled into a local scratch dir. checkpoints_entity/ is
never written, never deleted, never used as an attempt directory (enforced in
code by _assert_safe_attempt_dir, not by convention).

    scripts/hillclimb.py --dry-run --max-attempts 3     # print the plan, run nothing
    scripts/hillclimb.py --max-attempts 10              # ten gated attempts
    scripts/hillclimb.py --status                       # summarize the log
    touch logs/hillclimb.stop                           # end the loop cleanly
"""
from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# scripts/ isn't a package; import its siblings the way champion_gate.py and
# the tests do. IMPORT, never copy -- see DIVISION OF LABOR above.
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import champion_gate  # noqa: E402
import ctl  # noqa: E402

HOST_DEFAULT = "elliot@192.168.86.117"
RDIR_DEFAULT = "construct"

# The band, and why it is a band rather than a number (journal ~04:30). Arm G
# measured lambda_p 0.5 at 46.4% -- the best self-anchor result, and monotone
# in trust-region strength from arm F's 0.2 -> 32.8%. But lambda is a brake,
# not a booster: as lambda_p -> infinity the student cannot move away from the
# anchor at all, so share -> 50% with ZERO progress. High lambda buys retention
# by FORBIDDING CHANGE, and a policy that cannot change can never beat the
# champion it is anchored to -- it would fail this gate forever. The useful
# setting is therefore the LARGEST lambda that still permits improvement, which
# is a quantity nobody has measured. So we sample the band above the measured-
# good point and below the freeze, one draw per attempt, and let the gate
# adjudicate. A wrong guess costs one attempt; that is exactly what the gate is
# for.
LAMBDA_MIN_DEFAULT = 0.5
LAMBDA_MAX_DEFAULT = 0.7

# 145 iterations ~ 20M steps: the bound every arm in the journal used, so gate
# results here are directly comparable to that table.
ITERS_DEFAULT = 145
# Trainer default (configs/train_v1.toml run.save_every_iters). NOTE the
# consequence, which is load-bearing below: with save_every=20 a 145-iteration
# run's LAST checkpoint lands at iteration 140, not 145. There is no
# "ck at exactly max-iterations" to wait for -- the newest file is the answer.
SAVE_EVERY_DEFAULT = 20

# reward v3 measured ~5 points better than v4.1 with the anchor held fixed
# (journal ~01:20), and is the regime arm D held parity under.
REWARD_DEFAULT = "configs/reward_v3.toml"
# entropy_coef moves I(S;A) but NOT skill (arms A vs B: the sharper arm lost
# worse). 0.01 is arm D/G's value; changing it is not a lever, so it stays.
ENTROPY_COEF_DEFAULT = 0.01

TRAIN_CONFIG_DEFAULT = "configs/train_v1.toml"
REMOTE_CK_ROOT = "checkpoints_hc"      # remote, per-attempt subdirs
LOCAL_SCRATCH_DEFAULT = "checkpoints_hillclimb"
LOG_DEFAULT = "logs/hillclimb.jsonl"
STOP_FILE_DEFAULT = "logs/hillclimb.stop"

# A directory we refuse to write attempt checkpoints into, ever.
PROTECTED_DIRS = ("checkpoints_entity",)

MAX_CONSECUTIVE_FAILURES_DEFAULT = 20
POLL_SECONDS_DEFAULT = 120
ATTEMPT_TIMEOUT_DEFAULT = 6 * 3600     # a 145-iter run is ~2-4h; 6h is a hang
SSH_TIMEOUT_DEFAULT = 120
# Consecutive ssh transport failures tolerated while polling before the attempt
# is abandoned. Below this we RETRY: a flaky link must never be misread as
# "training finished".
SSH_RETRY_LIMIT = 5

# verdict/outcome strings for our own log (champion_gate owns PASS/FAIL)
PASS = champion_gate.PASS
FAIL = champion_gate.FAIL
ERROR = "ERROR"          # the attempt never produced a gradeable checkpoint


class HillclimbAbort(RuntimeError):
    """Loud, loop-ending condition: a trainer is already running, the attempt
    directory is unsafe, or too many attempts failed in a row. Never caught
    inside the loop -- these mean a human must look."""


# ---------------------------------------------------------------------------
# pure helpers (no subprocess, no filesystem beyond the log)
# ---------------------------------------------------------------------------

def make_seed(seed_base: int, attempt: int) -> int:
    """Distinct seed per attempt, and distinct ACROSS RESTARTS of the loop:
    `attempt` is derived from the log (next_attempt_number), not from a
    counter that resets to 0 when the process restarts. Two attempts sharing a
    seed would be the same mutation drawn twice -- wasted GPU-hours."""
    return int(seed_base) + int(attempt)


def sample_lambda(seed: int, lo: float, hi: float) -> float:
    """One lambda draw, DETERMINED BY THE SEED. Deliberately not global
    random: it makes --dry-run honest (it prints the lambda the real run will
    use) and makes any attempt reproducible from its logged seed alone."""
    if hi < lo:
        raise ValueError(f"lambda band inverted: min={lo} > max={hi}")
    return round(random.Random(seed).uniform(lo, hi), 4)


def attempt_tag(attempt: int, seed: int) -> str:
    """Identifier that is unique per attempt and appears in the remote
    cmdline, so polling can find exactly this attempt's trainer."""
    return f"hc_a{int(attempt):04d}_s{int(seed)}"


def remote_attempt_dir(tag: str, root: str = REMOTE_CK_ROOT) -> str:
    return f"{root}/{tag}"


def _assert_safe_attempt_dir(path: str) -> str:
    """checkpoints_entity/ holds the champion lineage. An attempt must never
    be able to write there, so this is a hard check on the built path rather
    than a rule someone has to remember."""
    parts = Path(path).parts
    for protected in PROTECTED_DIRS:
        if protected in parts:
            raise HillclimbAbort(
                f"refusing to use {path!r} as an attempt checkpoint dir: it is inside "
                f"{protected}/, which holds the champion lineage and is never written "
                f"by the hill-climb")
    return path


def pgrep_pattern(tag: str) -> str:
    """Self-match-proof pattern for THIS attempt's trainer. The tag appears in
    the remote cmdline (as --checkpoint-dir .../<tag>), and bracketing one
    character means the pattern is not a literal substring of itself -- so the
    `ssh host pgrep -f <pattern>` invocation cannot match its own cmdline."""
    return ctl.bracket_proof(tag)


def trainer_pattern() -> str:
    """Self-match-proof pattern for ANY resume_train.py on the box."""
    return ctl.bracket_proof("resume_train.py")


def launch_command(rdir: str, champion: str, attempt_ck_dir: str, log_path: str,
                   seed: int, lam: float, iters: int, reward: str,
                   entropy_coef: float, config: str = TRAIN_CONFIG_DEFAULT) -> str:
    """The single shell one-liner ssh runs to start one bounded attempt.

    A string rather than an argv because nohup/setsid/redirection/disown are
    shell syntax. Every dynamic value is shlex.quote'd.

    --kl-prior points at the CHAMPION ITSELF (self-distillation). KL starts at
    exactly 0 because student == anchor, so lambda_p is a pure trust region on
    drift rather than a pull toward someone else's policy -- which is why the
    failed BC-prior experiment does not apply: that anchor was incompetent,
    this one is the strongest policy we own.

    Not reusing ctl.remote_launch_command: it has no slot for --max-iterations,
    --checkpoint-dir, --seed or --entropy-coef, all four of which are exactly
    what makes an attempt bounded, isolated, distinct and comparable. Modifying
    ctl.py is out of scope for this tool.
    """
    _assert_safe_attempt_dir(attempt_ck_dir)
    parts = [
        "nohup", "setsid", ".venv/bin/python", "scripts/resume_train.py",
        champion,
        "--config", config,
        "--reward-config", reward,
        "--kl-prior", champion,
        "--kl-prior-lambda", str(lam),
        "--entropy-coef", str(entropy_coef),
        "--max-iterations", str(int(iters)),
        "--checkpoint-dir", attempt_ck_dir,
        "--seed", str(int(seed)),
    ]
    quoted = " ".join(shlex.quote(p) for p in parts)
    return (f"cd {shlex.quote(rdir)} && {quoted} "
            f">> {shlex.quote(log_path)} 2>&1 < /dev/null & disown")


def expected_saves(iters: int, save_every: int) -> int:
    """How many checkpoints a COMPLETE attempt must have written. 145 iters at
    save_every=20 -> 7 (iters 20..140). Fewer than this means the run was cut
    short -- the silent-kill failure mode -- and its newest checkpoint is not
    the thing we meant to measure."""
    if save_every <= 0:
        return 1
    return max(1, int(iters) // int(save_every))


def newest_checkpoint(names) -> str | None:
    """Newest ck_*.pt from a bare `ls -1` listing. Names are zero-padded step
    counts (ck_{steps:012d}.pt), so lexical max IS numeric max -- no stat call
    and no mtime, both of which lie after an rsync."""
    cks = sorted(n.strip() for n in names if n.strip().startswith("ck_")
                 and n.strip().endswith(".pt"))
    return cks[-1] if cks else None


def hillclimb_row(attempt, seed, lam, champion_before, candidate, share,
                  verdict, promoted, wall_seconds, reason="", ts=None) -> dict:
    """One attempt. The required keys are the contract with --status and with
    anything that later reads this file; `reason` is the audit trail for the
    attempts that never reached a verdict."""
    return {
        "ts": int(ts if ts is not None else time.time()),
        "attempt": int(attempt),
        "seed": int(seed),
        "lambda": float(lam),
        "champion_before": str(champion_before),
        "candidate": str(candidate) if candidate else None,
        "share": float(share) if isinstance(share, (int, float)) else None,
        "verdict": str(verdict),
        "promoted": bool(promoted),
        "wall_seconds": int(wall_seconds),
        "reason": str(reason),
    }


def read_rows(path) -> list:
    """Rows in file order. A truncated tail (crash mid-append) is skipped
    rather than fatal -- the loop must survive its own log being ugly."""
    path = Path(path)
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict) and "attempt" in row:
            rows.append(row)
    return rows


def append_row(path, row) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


def next_attempt_number(rows) -> int:
    """Continue numbering across restarts, so seeds never repeat after a
    crash or a deliberate stop."""
    return max((int(r.get("attempt", 0)) for r in rows), default=0) + 1


def consecutive_failures(rows) -> int:
    """Trailing non-promotions. A FAIL and an ERROR both count: 20 attempts
    that never produced a checkpoint is just as broken as 20 that lost."""
    n = 0
    for row in reversed(rows):
        if row.get("promoted"):
            break
        n += 1
    return n


def stop_requested(stop_file) -> bool:
    return Path(stop_file).exists()


def plan_attempt(attempt, seed, lam, champion, host, rdir, iters, reward,
                 entropy_coef, config=TRAIN_CONFIG_DEFAULT,
                 local_dir=LOCAL_SCRATCH_DEFAULT, save_every=SAVE_EVERY_DEFAULT) -> dict:
    """Everything about one attempt, as data: the exact commands, the seed, the
    lambda, the paths. Execution and --dry-run BOTH read this, which is what
    makes the dry-run trustworthy -- there is no second code path that could
    print one thing and run another."""
    tag = attempt_tag(attempt, seed)
    ck_dir = _assert_safe_attempt_dir(remote_attempt_dir(tag))
    log_path = f"{ck_dir}.log"
    cmd = launch_command(rdir, champion, ck_dir, log_path, seed, lam, iters,
                         reward, entropy_coef, config=config)
    return {
        "attempt": int(attempt),
        "seed": int(seed),
        "lambda": float(lam),
        "tag": tag,
        "champion": str(champion),
        "remote_ck_dir": ck_dir,
        "remote_log": log_path,
        "local_dir": str(local_dir),
        "expected_saves": expected_saves(iters, save_every),
        "pattern": pgrep_pattern(tag),
        "busy_cmd": ["ssh", host, "pgrep", "-af", trainer_pattern()],
        "mkdir_cmd": ["ssh", host, "mkdir", "-p", f"{rdir}/{ck_dir}"],
        "launch_cmd": ["ssh", host, cmd],
        "poll_cmd": ["ssh", host, "pgrep", "-f", pgrep_pattern(tag)],
        "list_cmd": ["ssh", host, "ls", "-1", f"{rdir}/{ck_dir}"],
    }


def scp_command(host: str, remote_path: str, local_path: str) -> list:
    return ["scp", "-q", f"{host}:{remote_path}", str(local_path)]


def format_plan(plan: dict) -> str:
    lines = [
        f"--- attempt {plan['attempt']}  seed={plan['seed']}  lambda_p={plan['lambda']} ---",
        f"    champion (re-read from champion.toml at attempt start): {plan['champion']}",
        f"    remote ck dir : {plan['remote_ck_dir']}   (expects >= {plan['expected_saves']} checkpoints)",
        f"    pgrep pattern : {plan['pattern']}   (self-match-proof)",
    ]
    for key in ("busy_cmd", "mkdir_cmd", "launch_cmd", "poll_cmd", "list_cmd"):
        lines.append(f"    {key:<10} : {shlex.join(plan[key])}")
    lines.append(f"    {'scp_cmd':<10} : "
                 f"{shlex.join(scp_command('HOST', plan['remote_ck_dir'] + '/ck_<newest>.pt', plan['local_dir'] + '/' + plan['tag'] + '_ck_<newest>.pt'))}")
    lines.append(f"    {'gate':<10} : champion_gate.gate_one(candidate, promote_if_pass=True)"
                 f"  [both side orders, threshold from champion.toml]")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# process glue (the only code here that touches the network; injected in tests)
# ---------------------------------------------------------------------------

def subprocess_runner(argv, timeout=SSH_TIMEOUT_DEFAULT):
    """Run argv, capture output. Never raises: a transport failure is a
    RESULT (returncode 255) that callers interpret, not an exception that
    unwinds an overnight loop."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"  warning: {shlex.join(str(a) for a in argv)} failed: {e}", file=sys.stderr)
        return subprocess.CompletedProcess(argv, 255, "", str(e))


def _ok(res) -> bool:
    return res is not None and res.returncode == 0


def _ssh_broken(res) -> bool:
    """255 is ssh's own transport failure. pgrep never returns it, so this
    cleanly separates 'the link died' from 'no such process'."""
    return res is None or res.returncode == 255


def trainer_busy(runner, host) -> str | None:
    """Returns the offending pgrep line if a trainer is already running.
    Checked before EVERY attempt: a live arm on the box is the normal state of
    this project, and launching on top of one would fight it for the GPU and
    corrupt both measurements."""
    res = runner(["ssh", host, "pgrep", "-af", trainer_pattern()])
    if _ssh_broken(res):
        raise HillclimbAbort(
            f"cannot reach {host} to check for a running trainer -- refusing to launch "
            f"blind (an unseen live arm would be fought for the GPU)")
    out = (res.stdout or "").strip()
    return out or None


def wait_for_idle(runner, host, sleep, poll_seconds, timeout) -> None:
    waited = 0
    while True:
        busy = trainer_busy(runner, host)
        if not busy:
            return
        if waited >= timeout:
            raise HillclimbAbort(
                f"a trainer has been running on {host} for {waited}s and --wait-for-idle "
                f"timed out. Not launching on top of it.\n  {busy}")
        print(f"  trainer busy on {host}, waiting {poll_seconds}s "
              f"({waited}s so far):\n    {busy}", flush=True)
        sleep(poll_seconds)
        waited += poll_seconds


def poll_until_done(runner, plan, sleep, poll_seconds, timeout) -> tuple[bool, str]:
    """Wait for THIS attempt's trainer to leave the process table.

    Returns (finished, reason). Note what this does NOT do: conclude anything
    about success. Completion is confirmed by counting checkpoints afterwards,
    because a detached job that gets silently killed mid-run also 'leaves the
    process table'.
    """
    waited = 0
    ssh_failures = 0
    while True:
        res = runner(plan["poll_cmd"])
        if _ssh_broken(res):
            ssh_failures += 1
            if ssh_failures > SSH_RETRY_LIMIT:
                return False, (f"lost ssh to the trainer box for {ssh_failures} consecutive "
                               f"polls; abandoning this attempt (training may still be running)")
            print(f"  poll: ssh failed ({ssh_failures}/{SSH_RETRY_LIMIT}), retrying", flush=True)
        else:
            ssh_failures = 0
            if res.returncode != 0:          # pgrep found nothing -> process gone
                return True, "trainer process exited"
        if waited >= timeout:
            return False, (f"attempt still running after {waited}s (timeout {timeout}s); "
                           f"abandoning it rather than blocking the loop")
        sleep(poll_seconds)
        waited += poll_seconds


def fetch_checkpoint(runner, plan, host, rdir, local_dir) -> tuple[str | None, str]:
    """List the attempt's remote dir, verify it really finished, scp the newest
    checkpoint down. Returns (local_path, reason); local_path is None for every
    'this attempt produced nothing gradeable' case -- all of which are logged
    failures, never crashes."""
    res = runner(plan["list_cmd"])
    if _ssh_broken(res):
        return None, "could not list the attempt checkpoint dir (ssh failed)"
    if not _ok(res):
        return None, (f"attempt checkpoint dir {plan['remote_ck_dir']} does not exist -- "
                      f"the trainer never wrote a checkpoint (launch failed, or it died "
                      f"before the first save)")
    names = [n for n in (res.stdout or "").splitlines() if n.strip()]
    cks = [n.strip() for n in names if n.strip().startswith("ck_") and n.strip().endswith(".pt")]
    newest = newest_checkpoint(names)
    if newest is None:
        return None, (f"no ck_*.pt in {plan['remote_ck_dir']} -- attempt produced no "
                      f"checkpoint at all")
    if len(cks) < plan["expected_saves"]:
        # The silent-kill signature: some checkpoints, but not a full run's
        # worth. Gating this would measure a truncated experiment and
        # (worse) could promote one.
        return None, (f"attempt truncated: {len(cks)} checkpoints in "
                      f"{plan['remote_ck_dir']}, expected >= {plan['expected_saves']} "
                      f"for a complete run -- not gating a run that did not finish")

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    local_path = local_dir / f"{plan['tag']}_{newest}"
    res = runner(scp_command(host, f"{rdir}/{plan['remote_ck_dir']}/{newest}", str(local_path)))
    if not _ok(res):
        return None, f"scp of {newest} failed: {(res.stderr or '').strip()[:200]}"
    if not local_path.is_file():
        return None, f"scp reported success but {local_path} is missing"
    return str(local_path), f"fetched {newest}"


def default_gate(cfg, candidate, n_confirm=2):
    """Gate via champion_gate, opting IN to promotion. gate_one moves the
    pointer, appends to the league pool, and writes the champion-history row --
    all three advance together, which is the point of the hill-climb.

    n_confirm is LOAD-BEARING for an unattended loop: the promote threshold
    (0.52) sits inside the null control's band (unchanged policy = 51.1% +/-
    ~3%), so a single gate promotes noise about one run in three. Over a night
    of attempts that reliably corrupts the champion -- and the champion is also
    the anchor and league seed, so the error compounds. Confirmation gates on
    independent seeds cut it to ~(1/3)^n. Do not set this to 0 for unattended
    runs."""
    return champion_gate.gate_one(cfg, candidate, promote_if_pass=True,
                                  n_confirm=n_confirm)


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------

def run_attempt(args, plan, cfg, runner, gate_fn, sleep, started) -> dict:
    """One attempt, start to logged row. Returns the row; raises only
    HillclimbAbort (conditions a human must see)."""
    def _row(share, verdict, promoted, reason, candidate=None):
        return hillclimb_row(
            plan["attempt"], plan["seed"], plan["lambda"], plan["champion"],
            candidate, share, verdict, promoted, time.time() - started, reason)

    runner(plan["mkdir_cmd"])
    res = runner(plan["launch_cmd"])
    if _ssh_broken(res):
        return _row(None, ERROR, False, "ssh failed while launching the attempt")

    print(f"  launched; polling every {args.poll_seconds}s "
          f"(timeout {args.attempt_timeout}s)", flush=True)
    finished, reason = poll_until_done(runner, plan, sleep, args.poll_seconds,
                                       args.attempt_timeout)
    if not finished:
        return _row(None, ERROR, False, reason)

    candidate, reason = fetch_checkpoint(runner, plan, args.host, args.remote_dir,
                                         plan["local_dir"])
    if candidate is None:
        print(f"  attempt produced nothing gradeable: {reason}", flush=True)
        return _row(None, ERROR, False, reason)

    print(f"  gating {candidate} vs champion {Path(plan['champion']).name}", flush=True)
    try:
        gate_row = gate_fn(cfg, candidate)
    except champion_gate.SchemaMismatchError as e:
        return _row(None, ERROR, False, f"gate refused (schema mismatch): {e}", candidate)
    except Exception as e:  # noqa: BLE001 -- one bad match must not end the night
        return _row(None, ERROR, False, f"gate raised {type(e).__name__}: {e}", candidate)

    return _row(gate_row.get("share"), gate_row.get("verdict", FAIL),
                bool(gate_row.get("promoted")), gate_row.get("reason", ""), candidate)


def run_loop(args, runner=subprocess_runner, gate_fn=default_gate, sleep=time.sleep) -> int:
    """Attempt -> gate -> promote-or-discard, until a stop file, --max-attempts,
    or too many consecutive failures."""
    log_path = Path(args.log)
    done = 0
    while args.max_attempts is None or done < args.max_attempts:
        if stop_requested(args.stop_file):
            print(f"stop file {args.stop_file} present -- exiting cleanly after "
                  f"{done} attempt(s) this session.")
            return 0

        # RE-READ the champion every attempt, never cache it. A promotion in the
        # previous attempt moved this pointer, and the next attempt must anchor
        # to (and be measured against) the NEW champion. This is the single
        # correctness property the whole hill-climb rests on: caching it would
        # silently keep climbing from a checkpoint we have already beaten.
        cfg = champion_gate.load_config(args.champion_config)
        champion = cfg["champion_ck"]

        rows = read_rows(log_path)
        streak = consecutive_failures(rows)
        if streak >= args.max_consecutive_failures:
            raise HillclimbAbort(
                f"\n!! {streak} consecutive attempts without a promotion (limit "
                f"{args.max_consecutive_failures}).\n"
                f"!! Mostly-failing is EXPECTED for this loop, but {streak} in a row "
                f"suggests something is broken rather than unlucky: check the remote "
                f"trainer log, the gate history ({cfg['history']}), and whether the "
                f"attempts are producing checkpoints at all.\n"
                f"!! Champion left untouched at {champion}")

        attempt = next_attempt_number(rows)
        seed = make_seed(args.seed_base, attempt)
        lam = sample_lambda(seed, args.lambda_min, args.lambda_max)
        plan = plan_attempt(attempt, seed, lam, champion, args.host, args.remote_dir,
                            args.iters, args.reward, args.entropy_coef,
                            config=args.train_config, local_dir=args.local_dir,
                            save_every=args.save_every)

        print(f"\n=== attempt {attempt}  seed={seed}  lambda_p={lam}  "
              f"champion={Path(champion).name} ===", flush=True)

        busy = trainer_busy(runner, args.host)
        if busy:
            if not args.wait_for_idle:
                raise HillclimbAbort(
                    f"a trainer is ALREADY RUNNING on {args.host}; refusing to launch on "
                    f"top of it (it would fight for the GPU and corrupt both runs). "
                    f"Pass --wait-for-idle to queue behind it instead.\n  {busy}")
            wait_for_idle(runner, args.host, sleep, args.poll_seconds, args.attempt_timeout)

        started = time.time()
        row = run_attempt(args, plan, cfg, runner, gate_fn, sleep, started)
        append_row(log_path, row)
        done += 1

        share = row["share"]
        share_s = f"{share * 100:.1f}%" if isinstance(share, (int, float)) else "n/a"
        mark = "PROMOTED" if row["promoted"] else row["verdict"]
        print(f"  -> {mark}  share={share_s}  ({row['reason']})", flush=True)
        if row["promoted"]:
            new_cfg = champion_gate.load_config(args.champion_config)
            print(f"  *** champion advanced -> {new_cfg['champion_ck']}", flush=True)

    print(f"\nreached --max-attempts ({args.max_attempts}); stopping.")
    return 0


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_dry_run(args) -> int:
    """Print the exact plan for every attempt. Touches NOTHING: no ssh, no scp,
    no gate, no champion read beyond the pointer it would anchor to."""
    cfg = champion_gate.load_config(args.champion_config)
    champion = cfg["champion_ck"]
    rows = read_rows(args.log)
    start = next_attempt_number(rows)
    n = args.max_attempts if args.max_attempts is not None else 3

    print(f"=== DRY RUN: {n} attempt(s), starting at attempt {start} ===")
    print(f"champion now      : {champion}")
    print(f"  (re-read at the start of EVERY attempt -- a promotion mid-loop means")
    print(f"   the next attempt anchors to and is measured against the NEW champion,")
    print(f"   so the plans below are only accurate while no promotion happens)")
    print(f"host / remote dir : {args.host} : {args.remote_dir}")
    print(f"bound             : {args.iters} iters (~20M steps), reward {args.reward}, "
          f"entropy_coef {args.entropy_coef}")
    print(f"lambda band       : [{args.lambda_min}, {args.lambda_max}] "
          f"(0.5 measured 46.4%; >=1.0 risks freezing the policy so it can never improve)")
    print(f"gate              : champion_gate, threshold {cfg['promote_threshold'] * 100:.1f}%, "
          f"{cfg['steps']} steps/side x {cfg['arenas']} arenas, BOTH side orders")
    print(f"log / stop file   : {args.log} / {args.stop_file}")
    print(f"abort after       : {args.max_consecutive_failures} consecutive failures\n")

    for i in range(n):
        attempt = start + i
        seed = make_seed(args.seed_base, attempt)
        lam = sample_lambda(seed, args.lambda_min, args.lambda_max)
        plan = plan_attempt(attempt, seed, lam, champion, args.host, args.remote_dir,
                            args.iters, args.reward, args.entropy_coef,
                            config=args.train_config, local_dir=args.local_dir,
                            save_every=args.save_every)
        print(format_plan(plan))
        print()
    print("nothing was executed.")
    return 0


def cmd_status(args) -> int:
    rows = read_rows(args.log)
    try:
        cfg = champion_gate.load_config(args.champion_config)
        champion = cfg["champion_ck"]
    except champion_gate.ChampionConfigError as e:
        champion = f"(unreadable: {e})"

    proms = [r for r in rows if r.get("promoted")]
    shares = [r["share"] for r in rows if isinstance(r.get("share"), (int, float))]
    errors = [r for r in rows if r.get("verdict") == ERROR]

    print("=== hillclimb status ===")
    print(f"  log            : {args.log}")
    print(f"  current champion: {champion}")
    print(f"  attempts        : {len(rows)}  "
          f"({len(proms)} promoted, {len(rows) - len(proms) - len(errors)} gated-and-failed, "
          f"{len(errors)} never reached a verdict)")
    if shares:
        best = max(shares)
        best_row = next(r for r in rows if r.get("share") == best)
        print(f"  best share      : {best * 100:.1f}%  "
              f"(attempt {best_row['attempt']}, seed {best_row['seed']}, "
              f"lambda {best_row['lambda']})")
    else:
        print("  best share      : n/a (no attempt has been gated yet)")
    print(f"  current streak  : {consecutive_failures(rows)} consecutive without a promotion")

    if proms:
        print(f"\n=== promotions ({len(proms)}) ===")
        for r in proms:
            stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
            print(f"  {stamp}  attempt {r['attempt']:4d}  lambda {r['lambda']}  "
                  f"share {r['share'] * 100:.1f}%  -> {r['candidate']}")

    print(f"\n=== recent attempts ===")
    for r in rows[-12:]:
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["ts"]))
        share = r.get("share")
        share_s = f"{share * 100:5.1f}%" if isinstance(share, (int, float)) else "   n/a"
        flag = " PROMOTED" if r.get("promoted") else ""
        print(f"  {stamp}  a{r['attempt']:04d}  lam {r['lambda']:.3f}  "
              f"{r['verdict']:5s} {share_s}  {int(r['wall_seconds']) // 60}m{flag}")
    if not rows:
        print("  (none)")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        prog="hillclimb.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            "Unattended gated hill-climb: repeatedly attempt a bounded anchored run from "
            "the champion, gate each result head-to-head, promote only winners.\n\n"
            "PPO self-play degrades this policy on average (journal 2026-07-19 ~17:40 -> "
            "2026-07-20 ~04:30), so MOSTLY-FAILING IS THE EXPECTED BEHAVIOR. A failed "
            "attempt costs nothing because the champion pointer never moves; a rare win "
            "advances the champion, the KL anchor and the league seed together."),
        epilog=(
            "control:\n"
            "  touch logs/hillclimb.stop     end the loop cleanly between attempts\n"
            "  --dry-run                     print the plan for every attempt, run nothing\n"
            "  --status                      summarize logs/hillclimb.jsonl\n"))

    ap.add_argument("--attempts", "--max-attempts", dest="max_attempts", type=int, default=None,
                    metavar="N",
                    help="stop after N attempts (default: run forever until a stop file "
                         "or --max-consecutive-failures)")
    ap.add_argument("--iters", type=int, default=ITERS_DEFAULT,
                    help=f"iterations per attempt (default {ITERS_DEFAULT} ~ 20M steps, the "
                         f"bound every arm in the journal used)")
    ap.add_argument("--lambda-min", type=float, default=LAMBDA_MIN_DEFAULT,
                    help=f"low end of the kl_prior lambda band (default {LAMBDA_MIN_DEFAULT}; "
                         f"0.5 measured 46.4%%, 0.2 measured 32.8%%)")
    ap.add_argument("--lambda-max", type=float, default=LAMBDA_MAX_DEFAULT,
                    help=f"high end of the band (default {LAMBDA_MAX_DEFAULT}; lambda 1.0 risks "
                         f"freezing the policy so it CANNOT improve -- the useful band is the "
                         f"largest lambda that still permits change)")
    ap.add_argument("--reward", default=REWARD_DEFAULT,
                    help=f"reward config (default {REWARD_DEFAULT}; measured ~5 points better "
                         f"than v4.1 with the anchor held fixed)")
    ap.add_argument("--entropy-coef", type=float, default=ENTROPY_COEF_DEFAULT,
                    help=f"PPO entropy_coef (default {ENTROPY_COEF_DEFAULT}; moves I(S;A) but "
                         f"not skill, so it is not a lever)")
    ap.add_argument("--train-config", default=TRAIN_CONFIG_DEFAULT,
                    help=f"trainer config passed to resume_train.py (default {TRAIN_CONFIG_DEFAULT})")
    ap.add_argument("--host", default=HOST_DEFAULT, help=f"trainer box (default {HOST_DEFAULT})")
    ap.add_argument("--remote-dir", default=RDIR_DEFAULT,
                    help=f"repo dir on the trainer box (default {RDIR_DEFAULT})")
    ap.add_argument("--local-dir", default=LOCAL_SCRATCH_DEFAULT,
                    help=f"local scratch for fetched candidates (default {LOCAL_SCRATCH_DEFAULT}; "
                         f"checkpoints_entity/ is never written)")
    ap.add_argument("--champion-config", default=str(champion_gate.DEFAULT_CONFIG),
                    help="champion pointer, RE-READ before every attempt")
    ap.add_argument("--log", default=LOG_DEFAULT, help=f"jsonl attempt log (default {LOG_DEFAULT})")
    ap.add_argument("--stop-file", default=STOP_FILE_DEFAULT,
                    help=f"touch this to end the loop between attempts (default {STOP_FILE_DEFAULT})")
    ap.add_argument("--seed-base", type=int, default=20260720,
                    help="seeds are seed_base + attempt number, and attempt numbers continue "
                         "across restarts, so no two attempts share a seed")
    ap.add_argument("--save-every", type=int, default=SAVE_EVERY_DEFAULT,
                    help=f"trainer's save_every_iters (default {SAVE_EVERY_DEFAULT}); used to "
                         f"tell a COMPLETE attempt from one that was silently killed mid-run")
    ap.add_argument("--poll-seconds", type=int, default=POLL_SECONDS_DEFAULT,
                    help=f"how often to poll the remote trainer (default {POLL_SECONDS_DEFAULT})")
    ap.add_argument("--attempt-timeout", type=int, default=ATTEMPT_TIMEOUT_DEFAULT,
                    help=f"give up on one attempt after this many seconds "
                         f"(default {ATTEMPT_TIMEOUT_DEFAULT})")
    ap.add_argument("--max-consecutive-failures", type=int,
                    default=MAX_CONSECUTIVE_FAILURES_DEFAULT,
                    help=f"abort loudly after this many attempts without a promotion "
                         f"(default {MAX_CONSECUTIVE_FAILURES_DEFAULT}; failures are EXPECTED, "
                         f"but this many in a row means something is broken)")
    ap.add_argument("--wait-for-idle", action="store_true",
                    help="if a trainer is already running on the host, wait for it instead of "
                         "aborting (default: abort with a clear message)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact plan (commands, seeds, lambdas) for every attempt "
                         "and execute nothing")
    ap.add_argument("--status", action="store_true",
                    help="summarize the attempt log and exit")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.status:
        return cmd_status(args)
    try:
        if args.dry_run:
            return cmd_dry_run(args)
        return run_loop(args)
    except champion_gate.ChampionConfigError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    except HillclimbAbort as e:
        print(f"ABORT: {e}", file=sys.stderr)
        return 3
    except KeyboardInterrupt:
        print("\ninterrupted -- the champion pointer is untouched.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
