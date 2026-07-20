"""Tests for scripts/hillclimb.py, the gated hill-climb orchestrator.

Everything here is PURE. No ssh, no scp, no engine, no training, no network:
the process runner is a fake that answers argv patterns from a canned script,
and the gate is a stub that returns champion_gate-shaped rows. The one thing
the fake runner really does to the filesystem is create the file a mocked
`scp` claims to have copied -- because fetch_checkpoint verifies the local
file exists, and a test that skipped that would not be testing the real path.

The numbers used as fixtures are the measured ones from
docs/training-journal.md (arms F/G at 32.8%/46.4%, the 51.1% null control),
so the pass/fail cases are anchored to reality rather than invented data.
"""
import json
import re
import sys
from pathlib import Path

import pytest

# scripts/ isn't a package (same pattern as tests/python/test_champion_gate.py).
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
import champion_gate  # noqa: E402
import hillclimb  # noqa: E402

REPO = Path(__file__).resolve().parents[2]

CHAMPION = "checkpoints_entity/ck_000320471040.pt"
PROMOTED_CK = "checkpoints_hillclimb/hc_a0002_s20260722_ck_000343000000.pt"


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------

class FakeResult:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class FakeRemote:
    """Answers hillclimb's ssh/scp calls without a network.

    Defaults describe the happy path: no trainer running, the launch succeeds,
    the attempt's process APPEARS on the first poll and is gone on the second
    (a real detached trainer needs ~20-30s to reach the process table, so
    hillclimb waits for it to appear before waiting for it to leave -- see
    STARTUP_GRACE; on 2026-07-20 the missing appear-phase raced the loop
    through 12 attempts in 60s and left 13 concurrent trainers running), and
    the checkpoint dir holds a COMPLETE run's worth of checkpoints (7 saves for
    145 iters at save_every=20 -- the last at iteration 140, not 145).
    """

    def __init__(self, ck_names=None, list_rc=0, busy_line="", launch_rc=0, scp_rc=0):
        self.calls = []
        self.ck_names = ck_names if ck_names is not None else [
            f"ck_{320471040 + i * 3200000:012d}.pt" for i in range(1, 8)
        ]
        self.list_rc = list_rc
        self.busy_line = busy_line
        self.launch_rc = launch_rc
        self.scp_rc = scp_rc
        # per-attempt-dir poll counter: first poll = "running", then "gone"
        self._polls = {}
        self.never_appears = False

    def __call__(self, argv, timeout=None):
        argv = [str(a) for a in argv]
        self.calls.append(argv)
        joined = " ".join(argv)

        if argv[0] == "scp":
            if self.scp_rc == 0:
                dest = Path(argv[-1])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(b"fake checkpoint")
            return FakeResult(self.scp_rc)
        if "pgrep" in argv and "-af" in argv:          # busy check
            return FakeResult(0 if self.busy_line else 1, self.busy_line)
        if "pgrep" in argv:                            # per-attempt poll
            if self.never_appears:
                return FakeResult(1, "")               # launch failed: never visible
            key = joined
            self._polls[key] = self._polls.get(key, 0) + 1
            if self._polls[key] == 1:
                return FakeResult(0, "12345")          # appeared (phase 1 satisfied)
            return FakeResult(1, "")                   # then finished (phase 2)
        if "mkdir" in argv:
            return FakeResult(0)
        if "ls" in argv:
            if self.list_rc != 0:
                return FakeResult(self.list_rc, "", "ls: No such file or directory")
            return FakeResult(0, "\n".join(self.ck_names) + "\n")
        if "resume_train.py" in joined:                # the launch one-liner
            return FakeResult(self.launch_rc)
        return FakeResult(0)

    def launch_commands(self):
        return [" ".join(c) for c in self.calls if "resume_train.py" in " ".join(c)]


def make_gate(shares, promote_on=(), champion_config=None, new_ck=PROMOTED_CK, record=None):
    """A champion_gate.gate_one stand-in. `shares` is consumed one per call.
    For attempts listed in `promote_on` it really rewrites the champion pointer
    (via champion_gate.write_champion_ck) so the loop's re-read is exercised
    against a genuinely changed file, not a mock."""
    state = {"n": 0}

    def gate(cfg, candidate):
        state["n"] += 1
        n = state["n"]
        share = shares[min(n - 1, len(shares) - 1)]
        if record is not None:
            record.append({"call": n, "candidate": candidate,
                           "champion": cfg["champion_ck"]})
        promoted = n in promote_on
        if promoted and champion_config is not None:
            champion_gate.write_champion_ck(champion_config, new_ck)
        return {
            "share": share,
            "verdict": champion_gate.PASS if promoted else champion_gate.FAIL,
            "promoted": promoted,
            "reason": f"share {share * 100:.1f}%",
        }

    return gate


@pytest.fixture
def champion_config(tmp_path):
    """A writable copy of the REAL configs/champion.toml -- the tests exercise
    the committed file's actual shape but never mutate it."""
    dst = tmp_path / "champion.toml"
    dst.write_text((REPO / "configs" / "champion.toml").read_text())
    return dst


@pytest.fixture
def args(tmp_path, champion_config):
    def build(*extra):
        return hillclimb.build_parser().parse_args([
            "--champion-config", str(champion_config),
            "--log", str(tmp_path / "hillclimb.jsonl"),
            "--stop-file", str(tmp_path / "hillclimb.stop"),
            "--local-dir", str(tmp_path / "scratch"),
            "--poll-seconds", "1",
            *extra,
        ])
    return build


def _noop_sleep(_seconds):
    """Any real sleep in this suite is a bug -- the fakes never make the loop
    wait, so a call here means the loop took a path it should not have."""
    raise AssertionError("run_loop slept; the fakes should never require waiting")


# ---------------------------------------------------------------------------
# lambda sampling: in band, deterministic from the seed, recorded per attempt
# ---------------------------------------------------------------------------

def test_sample_lambda_stays_inside_the_band():
    for attempt in range(200):
        seed = hillclimb.make_seed(20260720, attempt)
        lam = hillclimb.sample_lambda(seed, 0.5, 0.7)
        assert 0.5 <= lam <= 0.7


def test_sample_lambda_is_determined_by_the_seed():
    # the property --dry-run depends on: the printed lambda IS the run's lambda
    assert hillclimb.sample_lambda(1234, 0.5, 0.7) == hillclimb.sample_lambda(1234, 0.5, 0.7)


def test_sample_lambda_actually_varies_across_attempts():
    lams = {hillclimb.sample_lambda(hillclimb.make_seed(20260720, a), 0.5, 0.7)
            for a in range(30)}
    assert len(lams) > 20, "sampling should explore the band, not return one value"


def test_sample_lambda_rejects_an_inverted_band():
    with pytest.raises(ValueError):
        hillclimb.sample_lambda(1, 0.7, 0.5)


def test_custom_band_is_honored():
    for a in range(50):
        lam = hillclimb.sample_lambda(hillclimb.make_seed(7, a), 0.55, 0.6)
        assert 0.55 <= lam <= 0.6


def test_lambda_is_recorded_per_attempt(args, tmp_path, champion_config):
    a = args("--max-attempts", "3")
    remote = FakeRemote()
    hillclimb.run_loop(a, runner=remote, gate_fn=make_gate([0.328]), sleep=_noop_sleep)

    rows = hillclimb.read_rows(a.log)
    assert len(rows) == 3
    for row in rows:
        assert 0.5 <= row["lambda"] <= 0.7
        # the logged lambda is the one the trainer was actually launched with
        assert hillclimb.sample_lambda(row["seed"], 0.5, 0.7) == row["lambda"]
    launches = remote.launch_commands()
    assert len(launches) == 3
    for row, cmd in zip(rows, launches):
        assert f"--kl-prior-lambda {row['lambda']}" in cmd


# ---------------------------------------------------------------------------
# seeds: distinct per attempt, and distinct across loop RESTARTS
# ---------------------------------------------------------------------------

def test_seeds_are_distinct_across_attempts():
    seeds = [hillclimb.make_seed(20260720, a) for a in range(1, 101)]
    assert len(set(seeds)) == 100


def test_seeds_are_distinct_across_attempts_in_a_real_loop(args):
    a = args("--max-attempts", "4")
    hillclimb.run_loop(a, runner=FakeRemote(), gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    rows = hillclimb.read_rows(a.log)
    assert len({r["seed"] for r in rows}) == 4
    assert [r["attempt"] for r in rows] == [1, 2, 3, 4]


def test_seeds_do_not_repeat_after_a_restart(args):
    """The failure this guards: a counter that resets to 0 on restart would
    re-run the same mutation and waste hours of GPU time."""
    a = args("--max-attempts", "2")
    hillclimb.run_loop(a, runner=FakeRemote(), gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    first = {r["seed"] for r in hillclimb.read_rows(a.log)}

    b = args("--max-attempts", "2")          # fresh process, same log
    hillclimb.run_loop(b, runner=FakeRemote(), gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    rows = hillclimb.read_rows(b.log)
    assert len(rows) == 4
    assert [r["attempt"] for r in rows] == [1, 2, 3, 4]
    assert not (first & {r["seed"] for r in rows[2:]})


def test_next_attempt_number_continues_from_the_log():
    assert hillclimb.next_attempt_number([]) == 1
    assert hillclimb.next_attempt_number([{"attempt": 7}, {"attempt": 3}]) == 8


# ---------------------------------------------------------------------------
# THE CORE PROPERTY: the champion is re-read between attempts
# ---------------------------------------------------------------------------

def test_champion_is_reread_after_a_midloop_promotion(args, champion_config):
    """Simulate a promotion on attempt 2 and assert attempt 3 anchors to the
    NEW champion. This is the property the whole hill-climb rests on: a cached
    champion would keep mutating a checkpoint we have already beaten, and
    would gate new candidates against a stale reference."""
    a = args("--max-attempts", "3")
    remote = FakeRemote()
    seen = []
    gate = make_gate([0.328, 0.556, 0.44], promote_on=(2,),
                     champion_config=champion_config, record=seen)

    hillclimb.run_loop(a, runner=remote, gate_fn=gate, sleep=_noop_sleep)

    # what the GATE compared against, per attempt
    assert [s["champion"] for s in seen] == [CHAMPION, CHAMPION, PROMOTED_CK]

    # what the TRAINER was anchored to (--kl-prior) and resumed from, per attempt
    launches = remote.launch_commands()
    assert len(launches) == 3
    for cmd in launches[:2]:
        assert f"--kl-prior {CHAMPION}" in cmd
        assert CHAMPION in cmd.split("resume_train.py")[1].split("--config")[0]
    assert f"--kl-prior {PROMOTED_CK}" in launches[2]
    assert CHAMPION not in launches[2]

    # and what was logged
    rows = hillclimb.read_rows(a.log)
    assert [r["champion_before"] for r in rows] == [CHAMPION, CHAMPION, PROMOTED_CK]
    assert [r["promoted"] for r in rows] == [False, True, False]

    # the pointer on disk really moved (the loop did not fake it)
    assert champion_gate.load_config(champion_config)["champion_ck"] == PROMOTED_CK


def test_champion_hand_edit_between_attempts_is_picked_up(args, champion_config):
    """A human hand-editing configs/champion.toml mid-loop (an expected,
    documented workflow) must take effect on the very next attempt."""
    hand_picked = "checkpoints_entity/ck_000440000000.pt"
    a = args("--max-attempts", "2")
    remote = FakeRemote()
    seen = []

    calls = {"n": 0}

    def gate(cfg, candidate):
        calls["n"] += 1
        seen.append(cfg["champion_ck"])
        if calls["n"] == 1:                      # human edits the pointer after attempt 1
            champion_gate.write_champion_ck(champion_config, hand_picked)
        return {"share": 0.30, "verdict": champion_gate.FAIL, "promoted": False,
                "reason": "share 30.0%"}

    hillclimb.run_loop(a, runner=remote, gate_fn=gate, sleep=_noop_sleep)
    assert seen == [CHAMPION, hand_picked]
    assert f"--kl-prior {hand_picked}" in remote.launch_commands()[1]


# ---------------------------------------------------------------------------
# jsonl row schema round-trip
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"ts", "attempt", "seed", "lambda", "champion_before", "candidate",
                 "share", "verdict", "promoted", "wall_seconds"}


def test_row_has_every_required_key():
    row = hillclimb.hillclimb_row(3, 20260723, 0.62, CHAMPION, "scratch/c.pt",
                                  0.464, champion_gate.FAIL, False, 9312, "share 46.4%")
    assert REQUIRED_KEYS <= set(row)


def test_row_round_trips_through_jsonl(tmp_path):
    path = tmp_path / "hillclimb.jsonl"
    row = hillclimb.hillclimb_row(3, 20260723, 0.62, CHAMPION, "scratch/c.pt",
                                  0.464, champion_gate.FAIL, False, 9312, "share 46.4%")
    hillclimb.append_row(path, row)
    back = hillclimb.read_rows(path)
    assert back == [row]
    assert json.loads(path.read_text().strip()) == row


def test_row_types_are_json_native():
    row = hillclimb.hillclimb_row(1, 2, 0.5, CHAMPION, None, None,
                                  hillclimb.ERROR, False, 0.9)
    assert row["share"] is None and row["candidate"] is None
    assert isinstance(row["ts"], int) and isinstance(row["wall_seconds"], int)
    assert isinstance(row["promoted"], bool) and isinstance(row["lambda"], float)
    json.dumps(row)  # must not raise


def test_read_rows_survives_a_truncated_tail(tmp_path):
    """A crash mid-append must not make the log unreadable -- the loop reads it
    to compute the next attempt number on every pass."""
    path = tmp_path / "hillclimb.jsonl"
    good = hillclimb.hillclimb_row(1, 5, 0.5, CHAMPION, "c.pt", 0.3,
                                   champion_gate.FAIL, False, 10)
    hillclimb.append_row(path, good)
    with path.open("a") as f:
        f.write('{"attempt": 2, "seed": tru')      # torn write
    assert hillclimb.read_rows(path) == [good]
    assert hillclimb.next_attempt_number(hillclimb.read_rows(path)) == 2


def test_read_rows_on_missing_file(tmp_path):
    assert hillclimb.read_rows(tmp_path / "nope.jsonl") == []


def test_logged_rows_match_the_gate_verdicts(args):
    a = args("--max-attempts", "2")
    hillclimb.run_loop(a, runner=FakeRemote(),
                       gate_fn=make_gate([0.328, 0.464]), sleep=_noop_sleep)
    rows = hillclimb.read_rows(a.log)
    assert [r["share"] for r in rows] == [0.328, 0.464]
    assert all(r["verdict"] == champion_gate.FAIL for r in rows)
    assert all(REQUIRED_KEYS <= set(r) for r in rows)


# ---------------------------------------------------------------------------
# stop file
# ---------------------------------------------------------------------------

def test_stop_file_ends_the_loop_before_any_attempt(args, tmp_path):
    a = args("--max-attempts", "5")
    Path(a.stop_file).write_text("")
    remote = FakeRemote()
    assert hillclimb.run_loop(a, runner=remote, gate_fn=make_gate([0.3]),
                              sleep=_noop_sleep) == 0
    assert remote.calls == []
    assert hillclimb.read_rows(a.log) == []


def test_stop_file_is_honored_between_attempts(args):
    """Mid-flight attempts are never abandoned: the stop file is checked
    between attempts, so the loop finishes what it started and then exits."""
    a = args("--max-attempts", "5")
    stop = Path(a.stop_file)

    calls = {"n": 0}

    def gate(cfg, candidate):
        calls["n"] += 1
        if calls["n"] == 2:
            stop.write_text("")                  # human touches it during attempt 2
        return {"share": 0.3, "verdict": champion_gate.FAIL, "promoted": False,
                "reason": "share 30.0%"}

    assert hillclimb.run_loop(a, runner=FakeRemote(), gate_fn=gate, sleep=_noop_sleep) == 0
    rows = hillclimb.read_rows(a.log)
    assert len(rows) == 2, "attempt 2 should complete and be logged, then the loop stops"


def test_stop_requested_helper(tmp_path):
    p = tmp_path / "hillclimb.stop"
    assert not hillclimb.stop_requested(p)
    p.write_text("")
    assert hillclimb.stop_requested(p)


# ---------------------------------------------------------------------------
# missing / truncated checkpoint -> logged failure, never a crash
# ---------------------------------------------------------------------------

def test_missing_checkpoint_dir_is_a_logged_failure_not_a_crash(args):
    a = args("--max-attempts", "2")
    remote = FakeRemote(list_rc=2)               # ls: no such directory

    def gate(cfg, candidate):
        raise AssertionError("must not gate an attempt that produced no checkpoint")

    assert hillclimb.run_loop(a, runner=remote, gate_fn=gate, sleep=_noop_sleep) == 0
    rows = hillclimb.read_rows(a.log)
    assert len(rows) == 2, "the loop keeps going after a barren attempt"
    for row in rows:
        assert row["verdict"] == hillclimb.ERROR
        assert row["promoted"] is False
        assert row["share"] is None
        assert row["candidate"] is None
        assert "does not exist" in row["reason"]


def test_empty_checkpoint_dir_is_a_logged_failure(args):
    a = args("--max-attempts", "1")
    remote = FakeRemote(ck_names=[])
    hillclimb.run_loop(a, runner=remote, gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    row = hillclimb.read_rows(a.log)[0]
    assert row["verdict"] == hillclimb.ERROR
    assert "no checkpoint at all" in row["reason"]


def test_truncated_run_is_not_gated(args):
    """The silent-kill signature seen this session: a detached job dies partway,
    leaving SOME checkpoints. Gating that would measure an experiment that never
    finished -- and could promote it."""
    a = args("--max-attempts", "1")
    remote = FakeRemote(ck_names=["ck_000323671040.pt", "ck_000326871040.pt"])  # 2 of 7

    def gate(cfg, candidate):
        raise AssertionError("must not gate a truncated attempt")

    hillclimb.run_loop(a, runner=remote, gate_fn=gate, sleep=_noop_sleep)
    row = hillclimb.read_rows(a.log)[0]
    assert row["verdict"] == hillclimb.ERROR
    assert "truncated" in row["reason"]
    assert "expected >= 7" in row["reason"]


def test_failed_scp_is_a_logged_failure(args):
    a = args("--max-attempts", "1")
    remote = FakeRemote(scp_rc=1)
    hillclimb.run_loop(a, runner=remote, gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    row = hillclimb.read_rows(a.log)[0]
    assert row["verdict"] == hillclimb.ERROR
    assert "scp" in row["reason"]


def test_a_raising_gate_does_not_kill_the_loop(args):
    """One bad match (schema mismatch, engine hiccup) must cost one attempt,
    not the whole night."""
    a = args("--max-attempts", "2")

    def gate(cfg, candidate):
        raise RuntimeError("engine exploded")

    assert hillclimb.run_loop(a, runner=FakeRemote(), gate_fn=gate, sleep=_noop_sleep) == 0
    rows = hillclimb.read_rows(a.log)
    assert len(rows) == 2
    assert all(r["verdict"] == hillclimb.ERROR for r in rows)
    assert "engine exploded" in rows[0]["reason"]
    # the candidate is still recorded: it is on disk and worth keeping
    assert rows[0]["candidate"] is not None


def test_expected_saves_matches_the_trainer_cadence():
    assert hillclimb.expected_saves(145, 20) == 7      # saves at iters 20..140
    assert hillclimb.expected_saves(20, 20) == 1
    assert hillclimb.expected_saves(5, 20) == 1        # never demands zero
    assert hillclimb.expected_saves(145, 0) == 1


def test_newest_checkpoint_picks_the_highest_step_count():
    names = ["ck_000320471040.pt", "ck_000343671040.pt", "ck_000326871040.pt",
             "train_v1.log", "", "notes.txt"]
    assert hillclimb.newest_checkpoint(names) == "ck_000343671040.pt"
    assert hillclimb.newest_checkpoint(["train_v1.log"]) is None
    assert hillclimb.newest_checkpoint([]) is None


# ---------------------------------------------------------------------------
# consecutive-failure abort
# ---------------------------------------------------------------------------

def test_consecutive_failures_counts_the_trailing_streak():
    rows = [{"promoted": True}, {"promoted": False}, {"promoted": False}]
    assert hillclimb.consecutive_failures(rows) == 2
    assert hillclimb.consecutive_failures([{"promoted": True}]) == 0
    assert hillclimb.consecutive_failures([]) == 0


def test_loop_aborts_after_max_consecutive_failures(args):
    a = args("--max-attempts", "10", "--max-consecutive-failures", "4")
    with pytest.raises(hillclimb.HillclimbAbort) as excinfo:
        hillclimb.run_loop(a, runner=FakeRemote(), gate_fn=make_gate([0.25]),
                           sleep=_noop_sleep)
    assert "4 consecutive" in str(excinfo.value)
    assert len(hillclimb.read_rows(a.log)) == 4


def test_a_promotion_resets_the_failure_streak(args, champion_config):
    """Mostly-failing is the EXPECTED behavior, so the abort must only fire on
    an unbroken streak -- a win in the middle means the loop is working."""
    a = args("--max-attempts", "5", "--max-consecutive-failures", "3")
    gate = make_gate([0.3, 0.3, 0.556, 0.3, 0.3], promote_on=(3,),
                     champion_config=champion_config)
    assert hillclimb.run_loop(a, runner=FakeRemote(), gate_fn=gate,
                              sleep=_noop_sleep) == 0
    rows = hillclimb.read_rows(a.log)
    assert len(rows) == 5
    assert [r["promoted"] for r in rows] == [False, False, True, False, False]


def test_abort_streak_counts_errors_too(args):
    a = args("--max-attempts", "10", "--max-consecutive-failures", "3")
    with pytest.raises(hillclimb.HillclimbAbort):
        hillclimb.run_loop(a, runner=FakeRemote(list_rc=2), gate_fn=make_gate([0.3]),
                           sleep=_noop_sleep)
    assert len(hillclimb.read_rows(a.log)) == 3


def test_default_max_consecutive_failures_is_20():
    assert hillclimb.build_parser().parse_args([]).max_consecutive_failures == 20


# ---------------------------------------------------------------------------
# a trainer already running on the box
# ---------------------------------------------------------------------------

def test_busy_trainer_aborts_before_launching(args):
    """A live arm on the remote is the normal state of this project. Launching
    on top of one would fight it for the GPU and corrupt both measurements."""
    a = args("--max-attempts", "1")
    remote = FakeRemote(busy_line="4242 .venv/bin/python scripts/resume_train.py ck_x.pt")
    with pytest.raises(hillclimb.HillclimbAbort) as excinfo:
        hillclimb.run_loop(a, runner=remote, gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    assert "ALREADY RUNNING" in str(excinfo.value)
    assert remote.launch_commands() == [], "nothing may be launched on a busy box"


def test_unreachable_host_aborts_rather_than_launching_blind(args):
    a = args("--max-attempts", "1")

    def dead(argv, timeout=None):
        return FakeResult(255, "", "ssh: connect to host ... port 22: No route to host")

    with pytest.raises(hillclimb.HillclimbAbort) as excinfo:
        hillclimb.run_loop(a, runner=dead, gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    assert "cannot reach" in str(excinfo.value)


# ---------------------------------------------------------------------------
# --dry-run executes NOTHING
# ---------------------------------------------------------------------------

def test_dry_run_executes_nothing(args, monkeypatch, capsys):
    """The strongest form of the assertion: the module's real runner is
    replaced by a tripwire, and subprocess.run itself is poisoned."""
    import subprocess as _sp

    def tripwire(*a, **k):
        raise AssertionError("--dry-run executed a command")

    monkeypatch.setattr(hillclimb, "subprocess_runner", tripwire)
    monkeypatch.setattr(_sp, "run", tripwire)
    monkeypatch.setattr(hillclimb, "default_gate", tripwire)

    a = args("--dry-run", "--max-attempts", "3")
    assert hillclimb.cmd_dry_run(a) == 0
    out = capsys.readouterr().out
    assert "nothing was executed." in out
    assert hillclimb.read_rows(a.log) == [], "--dry-run must not write the log"


def test_dry_run_shows_distinct_seeds_and_in_band_lambdas(args, capsys):
    a = args("--dry-run", "--max-attempts", "3")
    hillclimb.cmd_dry_run(a)
    out = capsys.readouterr().out

    seeds = re.findall(r"seed=(\d+)", out)
    assert len(seeds) == 3 and len(set(seeds)) == 3

    lams = [float(x) for x in re.findall(r"lambda_p=([0-9.]+)", out)]
    assert len(lams) == 3
    assert all(0.5 <= v <= 0.7 for v in lams)

    assert out.count("--- attempt ") == 3
    assert "ssh " in out and "scp " in out          # the plan is concrete


def test_dry_run_prints_the_lambdas_the_real_run_would_use(args, capsys):
    """A dry-run that printed different numbers than the loop uses would be
    worse than no dry-run at all."""
    a = args("--dry-run", "--max-attempts", "2")
    hillclimb.cmd_dry_run(a)
    printed = [float(x) for x in re.findall(r"lambda_p=([0-9.]+)", capsys.readouterr().out)]

    b = args("--max-attempts", "2")
    hillclimb.run_loop(b, runner=FakeRemote(), gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    actual = [r["lambda"] for r in hillclimb.read_rows(b.log)]
    assert printed == actual


def test_main_dry_run_touches_no_remote(args, monkeypatch, champion_config, tmp_path):
    def tripwire(*a, **k):
        raise AssertionError("--dry-run executed a command")

    monkeypatch.setattr(hillclimb, "subprocess_runner", tripwire)
    assert hillclimb.main([
        "--dry-run", "--max-attempts", "2",
        "--champion-config", str(champion_config),
        "--log", str(tmp_path / "h.jsonl"),
    ]) == 0


# ---------------------------------------------------------------------------
# self-match-proof pgrep patterns
# ---------------------------------------------------------------------------

def test_attempt_pattern_matches_its_own_tag():
    for attempt, seed in ((1, 20260721), (42, 999), (9999, 1)):
        tag = hillclimb.attempt_tag(attempt, seed)
        assert re.search(hillclimb.pgrep_pattern(tag), tag) is not None


def test_attempt_pattern_is_not_a_substring_of_itself():
    """The bug this prevents: `ssh host pgrep -f <pattern>` whose own cmdline
    contains <pattern> matches its own invocation, so the poll never reports
    the trainer as finished and the attempt hangs until timeout."""
    for attempt in range(1, 25):
        pattern = hillclimb.pgrep_pattern(hillclimb.attempt_tag(attempt, 20260720 + attempt))
        assert re.search(pattern, pattern) is None


def test_trainer_pattern_is_self_match_proof():
    pattern = hillclimb.trainer_pattern()
    assert re.search(pattern, "resume_train.py") is not None
    assert re.search(pattern, pattern) is None


def test_poll_and_busy_commands_carry_the_proofed_pattern(args):
    a = args()
    plan = hillclimb.plan_attempt(1, 20260721, 0.6, CHAMPION, a.host, a.remote_dir,
                                  a.iters, a.reward, a.entropy_coef)
    poll = " ".join(plan["poll_cmd"])
    assert "[" in plan["pattern"]
    assert plan["pattern"] in poll
    assert re.search(plan["pattern"], poll) is None, "the poll command must not self-match"
    busy = " ".join(plan["busy_cmd"])
    assert re.search(hillclimb.trainer_pattern(), busy) is None


def test_tags_are_unique_per_attempt():
    tags = {hillclimb.attempt_tag(a, hillclimb.make_seed(20260720, a)) for a in range(1, 51)}
    assert len(tags) == 50


# ---------------------------------------------------------------------------
# launch command shape + safety
# ---------------------------------------------------------------------------

def test_launch_command_is_detached_and_bounded():
    cmd = hillclimb.launch_command("construct", CHAMPION, "checkpoints_hc/hc_a0001_s1",
                                   "checkpoints_hc/hc_a0001_s1.log", 1, 0.6, 145,
                                   "configs/reward_v3.toml", 0.01)
    assert "nohup setsid" in cmd and "& disown" in cmd
    assert "< /dev/null" in cmd
    assert "--max-iterations 145" in cmd
    assert "--checkpoint-dir checkpoints_hc/hc_a0001_s1" in cmd
    assert "--seed 1" in cmd
    assert f"--kl-prior {CHAMPION}" in cmd and "--kl-prior-lambda 0.6" in cmd
    assert "--reward-config configs/reward_v3.toml" in cmd
    assert "--entropy-coef 0.01" in cmd
    assert "pkill" not in cmd and "kill" not in cmd, "launches never combine with a kill"


def test_launch_refuses_to_write_into_checkpoints_entity():
    """checkpoints_entity/ holds the champion lineage -- a hard check, not a
    convention someone has to remember."""
    with pytest.raises(hillclimb.HillclimbAbort):
        hillclimb.launch_command("construct", CHAMPION, "checkpoints_entity/hc_a0001",
                                 "x.log", 1, 0.6, 145, "configs/reward_v3.toml", 0.01)
    with pytest.raises(hillclimb.HillclimbAbort):
        hillclimb._assert_safe_attempt_dir("checkpoints_entity")


def test_attempt_dirs_are_unique_and_outside_protected_dirs(args):
    a = args()
    dirs = set()
    for attempt in range(1, 20):
        seed = hillclimb.make_seed(a.seed_base, attempt)
        plan = hillclimb.plan_attempt(attempt, seed, 0.6, CHAMPION, a.host, a.remote_dir,
                                      a.iters, a.reward, a.entropy_coef)
        assert not plan["remote_ck_dir"].startswith("checkpoints_entity")
        dirs.add(plan["remote_ck_dir"])
    assert len(dirs) == 19


def test_ssh_calls_are_single_purpose(args):
    """Compound remote commands (notably kill + launch) die with exit 255 --
    every ssh this tool makes does exactly one job."""
    a = args("--max-attempts", "1")
    remote = FakeRemote()
    hillclimb.run_loop(a, runner=remote, gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    for call in remote.calls:
        if call[0] != "ssh":
            continue
        body = " ".join(call[2:])
        if "resume_train.py" in body:      # the one launch; && cd is part of it
            assert body.count("&&") == 1
            continue
        assert "&&" not in body and ";" not in body and "|" not in body


def test_candidates_land_in_the_local_scratch_dir(args, tmp_path):
    a = args("--max-attempts", "1")
    hillclimb.run_loop(a, runner=FakeRemote(), gate_fn=make_gate([0.3]), sleep=_noop_sleep)
    row = hillclimb.read_rows(a.log)[0]
    assert row["candidate"].startswith(str(tmp_path / "scratch"))
    # a FAILED candidate is KEPT on disk, never deleted
    assert Path(row["candidate"]).is_file()


# ---------------------------------------------------------------------------
# polling robustness
# ---------------------------------------------------------------------------

def test_poll_retries_transient_ssh_failures(args):
    """An ssh transport failure is NOT evidence that training finished."""
    a = args()
    plan = hillclimb.plan_attempt(1, 1, 0.6, CHAMPION, a.host, a.remote_dir,
                                  a.iters, a.reward, a.entropy_coef)
    seq = [FakeResult(255), FakeResult(255), FakeResult(0, "4242"), FakeResult(1, "")]
    calls = {"n": 0}

    def runner(argv, timeout=None):
        r = seq[min(calls["n"], len(seq) - 1)]
        calls["n"] += 1
        return r

    finished, reason = hillclimb.poll_until_done(runner, plan, lambda s: None, 1, 100)
    assert finished is True
    assert calls["n"] == 4, "must not stop at the first ssh failure"


def test_poll_gives_up_after_persistent_ssh_failure(args):
    a = args()
    plan = hillclimb.plan_attempt(1, 1, 0.6, CHAMPION, a.host, a.remote_dir,
                                  a.iters, a.reward, a.entropy_coef)
    finished, reason = hillclimb.poll_until_done(
        lambda argv, timeout=None: FakeResult(255), plan, lambda s: None, 1, 100)
    assert finished is False and "lost ssh" in reason


def test_poll_times_out_rather_than_blocking_forever(args):
    a = args()
    plan = hillclimb.plan_attempt(1, 1, 0.6, CHAMPION, a.host, a.remote_dir,
                                  a.iters, a.reward, a.entropy_coef)
    finished, reason = hillclimb.poll_until_done(
        lambda argv, timeout=None: FakeResult(0, "4242 python resume_train.py"),
        plan, lambda s: None, poll_seconds=10, timeout=30)
    assert finished is False and "timeout" in reason


# ---------------------------------------------------------------------------
# --status
# ---------------------------------------------------------------------------

def test_status_summarizes_the_log(args, capsys):
    a = args("--max-attempts", "3")
    hillclimb.run_loop(a, runner=FakeRemote(),
                       gate_fn=make_gate([0.328, 0.464, 0.20]), sleep=_noop_sleep)
    assert hillclimb.cmd_status(a) == 0
    out = capsys.readouterr().out
    assert "attempts        : 3" in out
    assert "46.4%" in out, "best share should be reported"
    assert CHAMPION in out


def test_status_on_an_empty_log(args, capsys):
    assert hillclimb.cmd_status(args()) == 0
    out = capsys.readouterr().out
    assert "attempts        : 0" in out
    assert "n/a" in out


def test_status_never_runs_anything(args, monkeypatch, capsys):
    def tripwire(*a, **k):
        raise AssertionError("--status executed a command")

    monkeypatch.setattr(hillclimb, "subprocess_runner", tripwire)
    assert hillclimb.cmd_status(args("--status")) == 0


# ---------------------------------------------------------------------------
# CLI defaults documented in --help
# ---------------------------------------------------------------------------

def test_defaults_match_the_measured_evidence():
    a = hillclimb.build_parser().parse_args([])
    assert (a.lambda_min, a.lambda_max) == (0.5, 0.7)
    assert a.iters == 145
    assert a.reward == "configs/reward_v3.toml"
    assert a.entropy_coef == 0.01
    assert a.host == "elliot@192.168.86.117"
    assert a.remote_dir == "construct"
    assert a.max_attempts is None, "default is to run forever"
    assert a.stop_file.endswith("hillclimb.stop")
    assert a.log.endswith("hillclimb.jsonl")


def test_attempts_and_max_attempts_are_the_same_flag():
    assert hillclimb.build_parser().parse_args(["--attempts", "5"]).max_attempts == 5
    assert hillclimb.build_parser().parse_args(["--max-attempts", "5"]).max_attempts == 5


def test_help_documents_the_defaults_and_the_evidence(capsys):
    with pytest.raises(SystemExit):
        hillclimb.build_parser().parse_args(["--help"])
    out = capsys.readouterr().out
    assert "46.4" in out, "--help should justify the lambda band with the measurement"
    assert "0.5" in out and "0.7" in out
    assert "hillclimb.stop" in out
    assert "145" in out


def test_never_concludes_finished_from_a_process_it_never_saw(args):
    """THE 2026-07-20 REGRESSION: polling immediately after a detached launch
    saw no process and reported 'trainer exited', so the loop raced ahead and
    left 13 concurrent trainers thrashing the remote. A process that never
    appears must be a FAILED launch, never a completed run."""
    a = args()
    plan = hillclimb.plan_attempt(1, 1, 0.6, CHAMPION, a.host, a.remote_dir,
                                  a.iters, a.reward, a.entropy_coef)
    finished, reason = hillclimb.poll_until_done(
        lambda argv, timeout=None: FakeResult(1, ""),  # pgrep: never running
        plan, lambda s: None, 1, 100)
    assert finished is False
    assert "never appeared" in reason and "launch failed" in reason


def test_full_loop_does_not_stampede_when_launches_never_start(args, tmp_path):
    """A broken launch must not let the loop fire off attempt after attempt."""
    a = args("--max-attempts", "3")
    remote = FakeRemote()
    remote.never_appears = True
    # this path DOES sleep -- it is the startup grace period elapsing, which is
    # exactly the wait that stops the loop from stampeding
    waited = []
    hillclimb.run_loop(a, runner=remote, gate_fn=make_gate([0.3] * 3),
                       sleep=waited.append)
    assert sum(waited) >= hillclimb.STARTUP_GRACE, (
        "each failed attempt must burn the full startup grace, not spin")
    rows = hillclimb.read_rows(a.log)
    assert all(r["verdict"] == "ERROR" for r in rows)
    assert all("never appeared" in r["reason"] for r in rows)


def _ssh_cmds(plan):
    return [v for k, v in plan.items() if isinstance(v, list) and v[:1] == ["ssh"]]


def test_every_glob_char_crossing_ssh_is_quoted_for_the_remote_shell(args):
    """THE 2026-07-20 SILENT-POLL BUG: ssh does not deliver argv -- it joins the
    arguments and the remote LOGIN SHELL re-parses them. The trainer box runs
    zsh, which ABORTS on a glob that matches nothing, so an unquoted
    bracket_proof pattern made pgrep never run and rc=1 read as "no such
    process". The poll and the busy check both reported an idle box while a
    trainer held the GPU at 99%, and a second trainer was launched on top."""
    a = args()
    plan = hillclimb.plan_attempt(1, 1, 0.6, CHAMPION, a.host, a.remote_dir,
                                  a.iters, a.reward, a.entropy_coef)
    checked = 0
    for cmd in _ssh_cmds(plan):
        for token in cmd[2:]:
            if any(c in token for c in "[]*?"):
                checked += 1
                assert token.startswith("'") and token.endswith("'"), (
                    f"{token!r} carries a glob char across ssh unquoted; "
                    f"remote zsh will abort the command instead of running it")
    assert checked, "expected at least one bracket_proof pattern in the plan"


def test_busy_check_pattern_is_quoted(args):
    """--wait-for-idle shares the bug: an unquoted pattern makes the busy check
    report 'idle' unconditionally, which is exactly how the second trainer got
    launched on top of the first."""
    a = args()
    seen = {}

    def runner(argv, timeout=None):
        seen["argv"] = argv
        return FakeResult(1, "")

    hillclimb.trainer_busy(runner, a.host)
    pattern = [t for t in seen["argv"] if "[" in t]
    assert pattern and pattern[0].startswith("'") and pattern[0].endswith("'")


# --- reset-pool preflight ---------------------------------------------------

def _write_configs(tmp_path, replay_weight, pool_path="data/reset_pool_v5.jsonl"):
    curriculum = tmp_path / "curr.toml"
    body = f"replay_weight = {replay_weight}\nkickoff_weight = 0.1\nrandom_weight = 0.2\n"
    if pool_path is not None:
        body += f'\n[replay_pool]\npath = "{pool_path}"\n'
    curriculum.write_text(body)
    train = tmp_path / "train.toml"
    train.write_text(f'curriculum_config_path = "{curriculum}"\n')
    return str(train)


def test_no_pool_requirement_when_replay_weight_is_zero(tmp_path):
    assert hillclimb.replay_pool_requirement(_write_configs(tmp_path, 0.0)) is None


def test_pool_requirement_read_through_the_curriculum(tmp_path):
    train = _write_configs(tmp_path, 0.7)
    assert hillclimb.replay_pool_requirement(train) == "data/reset_pool_v5.jsonl"


def test_preflight_aborts_when_the_pool_is_missing_on_the_box(tmp_path):
    """A missing pool does not fail the run -- the engine warns and falls back
    to kickoff/random. The arm would then measure a train_v1 rerun and report
    parity, and we would wrongly conclude replay resets don't help."""
    train = _write_configs(tmp_path, 0.7)
    with pytest.raises(hillclimb.HillclimbAbort) as e:
        hillclimb.preflight_replay_pool(
            lambda argv, timeout=None: FakeResult(1, ""), "h", "construct", train)
    assert "missing or empty" in str(e.value) and "scp" in str(e.value)


def test_preflight_aborts_rather_than_guessing_when_ssh_is_down(tmp_path):
    train = _write_configs(tmp_path, 0.7)
    with pytest.raises(hillclimb.HillclimbAbort) as e:
        hillclimb.preflight_replay_pool(
            lambda argv, timeout=None: FakeResult(255), "h", "construct", train)
    assert "ssh failed" in str(e.value)


def test_preflight_passes_when_the_pool_is_present(tmp_path):
    train = _write_configs(tmp_path, 0.7)
    hillclimb.preflight_replay_pool(
        lambda argv, timeout=None: FakeResult(0, ""), "h", "construct", train)


def test_preflight_is_a_noop_for_the_default_train_config(args):
    """train_v1 uses curriculum_v1 (no replay branch) -- the running loop must
    not be gated on a pool it does not use."""
    a = args()
    calls = []

    def runner(argv, timeout=None):
        calls.append(argv)
        return FakeResult(1, "")

    hillclimb.preflight_replay_pool(runner, a.host, a.remote_dir, "configs/train_v1.toml")
    assert calls == []


def test_quarantined_rows_do_not_count_toward_the_abort_tripwire():
    """Rows whose verdict came from a harness bug are evidence about the
    harness, not the search -- counting them trips the abort early (2026-07-20:
    13 phantom ERRORs from a blind poll)."""
    rows = [{"promoted": False, "verdict": "ERROR", "quarantined": True} for _ in range(13)]
    rows += [{"promoted": False, "verdict": "FAIL"}] * 2
    assert hillclimb.consecutive_failures(rows) == 2


def test_quarantine_does_not_hide_a_real_promotion():
    rows = [{"promoted": True, "verdict": "PASS"},
            {"promoted": False, "verdict": "ERROR", "quarantined": True},
            {"promoted": False, "verdict": "FAIL"}]
    assert hillclimb.consecutive_failures(rows) == 1
