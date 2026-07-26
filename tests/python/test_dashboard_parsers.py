"""Pure-parser tests for scripts/dashboard.py — every log format the dashboard
reads: all three main-run iter-line eras, the per-bot auto-curriculum lines
(the live difficulty ladder), the gate-history jsonl, and ssl_pull lines
(cursor, counters, trailing-hour rate)."""
import sys
import time
from pathlib import Path

import pytest

# scripts/ isn't a package, so import it by adding it to sys.path.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from dashboard import (  # noqa: E402
    _ssl_epoch,
    downsample,
    estimate_train_times,
    parse_curriculum,
    parse_gate_history,
    parse_iter_line,
    parse_ssl_log,
    parse_train_log,
    ssl_last_hour,
)

KICK = ("iter 2 steps 333,824 sps 5,519 ep_rew 1.293 pi_loss 0.0047 v_loss 0.4045 "
        "ent 4.058 clip 0.261 kick_kl 1.0886 lambda_k 1.000")
PLAIN = ("iter 1117 steps 500,235,264 sps 8,137 ep_rew 4.756 pi_loss 0.0076 "
         "v_loss 0.8470 ent 3.594 clip 0.172")
KLPRI = ("iter 21 steps 564,976,128 sps 4,984 ep_rew -0.715 pi_loss 0.0016 "
         "v_loss 0.5251 ent 1.544 clip 0.056 kl_pri 1.1613 lambda_p 0.050")


# --- main-run iter lines ----------------------------------------------------

def test_iter_line_kickstart_era():
    r = parse_iter_line(KICK)
    assert r["iter"] == 2 and r["steps"] == 333_824 and r["sps"] == 5519
    assert r["ep_rew"] == 1.293 and r["pi_loss"] == 0.0047 and r["v_loss"] == 0.4045
    assert r["ent"] == 4.058 and r["clip"] == 0.261
    assert r["kick_kl"] == 1.0886 and r["lambda_k"] == 1.0
    assert "kl_pri" not in r and "lambda_p" not in r


def test_iter_line_plain_between_eras():
    r = parse_iter_line(PLAIN)
    assert r["steps"] == 500_235_264 and r["sps"] == 8137 and r["clip"] == 0.172
    assert "kick_kl" not in r and "kl_pri" not in r


def test_iter_line_kl_prior_era():
    r = parse_iter_line(KLPRI)
    assert r["steps"] == 564_976_128 and r["ep_rew"] == -0.715
    assert r["kl_pri"] == 1.1613 and r["lambda_p"] == 0.05
    assert "kick_kl" not in r


def test_iter_line_rejects_noise():
    assert parse_iter_line("") is None
    assert parse_iter_line("[construct-engine] physics blowup contained (tick 80): "
                           "episode terminated, arena rebuilt") is None
    assert parse_iter_line("league: opponents ['ck_a.pt']") is None


def test_parse_train_log_mixed_eras():
    text = "\n".join([
        KICK,
        "resumed at 400,000,000 steps | arenas=192 agents=652 device=cuda",
        PLAIN,
        KLPRI,
        "[construct-engine] physics blowup contained (tick 80): episode terminated",
        "[construct-engine] physics blowup contained (tick 40): episode terminated",
    ])
    out = parse_train_log(text)
    assert [r["steps"] for r in out["rows"]] == [333_824, 500_235_264, 564_976_128]
    assert out["restarts"] == [400_000_000]
    assert out["containment"] == 2


def test_parse_train_log_resume_prunes_abandoned_branch():
    # K4 rolled the run back: iters ran past the resume point, then training
    # resumed from an earlier checkpoint — the higher-step rows are stale.
    text = "\n".join([
        KICK,                                            # 333,824
        PLAIN,                                           # 500,235,264 (abandoned)
        KLPRI,                                           # 564,976,128 (abandoned)
        "resumed at 400,000,000 steps | arenas=192 agents=652 device=cuda",
        KLPRI.replace("564,976,128", "400,100,000"),     # current branch
    ])
    out = parse_train_log(text)
    assert [r["steps"] for r in out["rows"]] == [333_824, 400_100_000]


def test_downsample():
    rows = list(range(1000))
    out = downsample(rows, 500)
    assert len(out) == 500 and out[0] == 0 and out[-1] == 999
    assert downsample(rows[:5], 500) == rows[:5]


def test_downsample_dense_tail():
    rows = list(range(1000))
    out = downsample(rows, 500, tail=150)
    assert len(out) == 500 and out[0] == 0
    assert out[-150:] == rows[-150:]  # live end stays dense


# --- bc log -----------------------------------------------------------------

def _row(steps, sps):
    return {"steps": steps, "sps": sps}


def test_estimate_train_times_anchor_and_monotonic():
    rows = [_row(100_000, 5000), _row(200_000, 5000), _row(300_000, 4000)]
    out = estimate_train_times(rows, anchor_ts=1_000_000.0)
    assert out[-1] == (1_000_000.0, False)  # last row anchored exactly at mtime
    ts = [t for t, _ in out]
    assert ts == sorted(ts)  # monotone non-decreasing
    assert ts[2] - ts[1] == 100_000 / 4000  # steps_delta / later row's sps
    assert ts[1] - ts[0] == 100_000 / 5000
    assert not any(rough for _, rough in out)  # no restarts -> nothing rough


def test_estimate_train_times_resume_boundary_marks_rough():
    rows = [_row(100_000, 5000), _row(200_000, 5000),
            _row(410_000, 5000), _row(500_000, 5000)]
    out = estimate_train_times(rows, 1_000_000.0, restarts=[400_000])
    # the 200k -> 410k gap contains the resume: everything at or before it is
    # shifted by unknowable downtime
    assert [rough for _, rough in out] == [True, True, False, False]
    ts = [t for t, _ in out]
    assert ts == sorted(ts) and ts[-1] == 1_000_000.0


def test_estimate_train_times_edge_cases():
    assert estimate_train_times([], 5.0) == []
    assert estimate_train_times([_row(1, 100)], 5.0) == [(5.0, False)]
    # zero sps must not divide by zero; ts stays monotone
    out = estimate_train_times([_row(100, 0), _row(200, 0)], 5.0)
    assert [t for t, _ in out] == [5.0, 5.0]


def _epoch(y, mo, d, h, mi, s):
    return time.mktime((y, mo, d, h, mi, s, 0, 0, -1))


SSL_TEXT = """07-18 18:00:00 start: 0 replay(s) on disk, filling batch_0001 (165/10000), cursor created-before=<none: from newest>
07-18 18:00:01 disk guard: host C: free 187.6G (min 130G)
07-18 18:00:02 page: 200 rows, 10000+ matching beyond cursor, oldest created 2025-02-13T21:27:24.847011Z
07-18 18:30:00 landed aaaa-bbbb -> batch_0001 (50 this run)
07-18 19:30:00 progress: 150 this run / 4150 total (10 deduped, 1 failed) — 100/h, host free 183.0G
"""


def test_parse_ssl_log():
    now = _epoch(2026, 7, 18, 20, 0, 0)
    p = parse_ssl_log(SSL_TEXT, now)
    assert p["cursor_oldest"] == "2025-02-13T21:27:24.847011Z"
    assert p["this_run"] == 150 and p["total"] == 4150
    assert p["deduped"] == 10 and p["failed"] == 1 and p["logged_rate_h"] == 100.0
    assert p["batch"] == "batch_0001" and p["batch_fill"] == 165
    assert p["batch_target"] == 10000
    assert p["run_start_ts"] == _epoch(2026, 7, 18, 18, 0, 0)
    assert p["last_ts"] == _epoch(2026, 7, 18, 19, 30, 0)
    assert p["samples"] == [(_epoch(2026, 7, 18, 18, 30, 0), 50),
                            (_epoch(2026, 7, 18, 19, 30, 0), 150)]


def test_ssl_start_resets_counters():
    text = SSL_TEXT + "07-18 19:45:00 start: 4150 replay(s) on disk, filling batch_0001 (4150/10000), cursor created-before=x\n"
    p = parse_ssl_log(text, _epoch(2026, 7, 18, 20, 0, 0))
    assert p["this_run"] == 0 and p["samples"] == [] and p["batch_fill"] == 4150


def test_ssl_last_hour_interpolates():
    now = _epoch(2026, 7, 18, 20, 0, 0)
    p = parse_ssl_log(SSL_TEXT, now)
    # counter at 19:00 interpolates 50@18:30 .. 150@19:30 -> 100; 150-100 = 50
    assert ssl_last_hour(p["samples"], p["run_start_ts"], now) == 50
    # run started (18:00, counter 0) inside the window: counts from zero up to
    # the counter interpolated at `now` (18:45 sits at 75 between 50@18:30 and
    # 150@19:30)
    assert ssl_last_hour(p["samples"], p["run_start_ts"],
                         _epoch(2026, 7, 18, 18, 45, 0)) == 75
    assert ssl_last_hour([], None, now) is None


def test_ssl_epoch_year_wrap():
    now = _epoch(2026, 1, 1, 12, 0, 0)
    t = _ssl_epoch("12-31 23:00:00", now)
    assert t is not None and t < now and time.localtime(t).tm_year == 2025
    # same-year timestamp keeps the current year
    t2 = _ssl_epoch("07-17 16:13:19", _epoch(2026, 7, 18, 0, 0, 0))
    assert time.localtime(t2).tm_year == 2026


# --- eval history jsonl -----------------------------------------------------


# --- per-bot auto-curriculum (the live difficulty ladder) -------------------

CURR_TEXT = """foreign: ['element', 'immortal'] periods=[4, 3] on 77/192 arenas (frac=0.4)
auto-curriculum: match-mode eval engine built (24 arenas, 14000 steps)
auto-curriculum ON: hold win rate in [0.35,0.65], period in [1,12], adjust every 20 iters
iter 1 steps 1,000 sps 8,000 ep_rew 1.0 pi_loss 0.001 v_loss 0.1 ent 2.9 clip 0.1
auto-curriculum: element wr0.69/ema0.69 p4->3 | immortal wr0.20/ema0.20 p3->4
auto-curriculum: element wr0.55/ema0.58 p3 dwell1/2 | immortal wr0.50/ema0.50 p4->4
"""

# v9's train.py, non-ladder config: the banner dropped the word "in" and the
# arrow target gained its "p" ("p4->p3", not "p4->3"). The second one is the
# dangerous kind of drift -- the OLD regex still MATCHED it and silently
# returned an empty arrow group, so the panel reported the pre-move period
# forever with no error anywhere.
CURR_V9_PLAIN = """foreign: [('element', 1), ('immortal', 1)] periods=[4, 3] cars=[1, 1] \
on 77/192 arenas (frac=0.4 PER BLOCK)
auto-curriculum ON: hold win rate in [0.35,0.65], period [1,12], adjust every 20 iters
auto-curriculum: element wr0.69/ema0.69 p4->p3 | immortal wr0.20/ema0.20 p3->p4
"""

# v9's ladder config: rungs are (cars, period) pairs, the roster carries the team
# size, necto/nexto appear at THREE team sizes each, and the staggered eval
# measures one team size per cycle so most slots read `n/a`.
CURR_V9_LADDER = """foreign: [('element', 1), ('necto', 1), ('necto', 2), ('necto', 3)] \
periods=[12, 12, 12, 12] cars=[1, 1, 1, 1] on 48/144 arenas (frac=0.3333 PER BLOCK)
auto-curriculum: match-mode eval engines built {1: 24, 2: 12, 3: 12} arenas, 14000 steps, \
ROUND-ROBIN one mode per cycle
auto-curriculum ON: hold win rate in [0.35,0.65], ladder [1, 2, 3, 4, 6, 12, 20, 24], \
adjust every 20 iters
auto-curriculum: element wr0.10/ema0.10 1c/p12->1c/p20 | necto wr0.06/ema0.06 1c/p12->1c/p24 \
| necto n/a 1c/p12 | necto n/a 1c/p1
auto-curriculum: element n/a 1c/p20 | necto n/a 1c/p24 | necto wr0.90/ema0.90 1c/p12->1c/p4 \
| necto wr0.95/ema0.95 1c/p1->2c/p24
"""


def test_parse_curriculum_band_roster_and_current_state():
    c = parse_curriculum(CURR_TEXT)
    assert c["band"] == [0.35, 0.65]
    assert (c["period_min"], c["period_max"]) == (1, 12)
    assert c["roster"] == ["element", "immortal"]
    assert c["arenas"] == "77/192"
    # the LAST decision is the current state
    el, im = c["bots"]
    assert el["name"] == "element" and el["wr"] == 0.55 and el["ema"] == 0.58
    assert (el["period"], el["dwell"], el["dwell_of"]) == (3, 1, 2), "held slot keeps p3"
    assert im["period"] == 4 and im["dwell"] is None


def test_parse_curriculum_moved_slot_reports_the_NEW_rung():
    """'p4->3' means the controller just moved it to 3; showing 4 would lag a
    whole eval behind and mislabel the RLViser/overlay rung too."""
    c = parse_curriculum(CURR_TEXT)
    first = c["history"][0]
    assert first["element"] == 3 and first["immortal"] == 4
    assert first["element_wr"] == 0.69


def test_parse_curriculum_only_reports_the_current_run():
    """The log is appended across restarts; rows before the last banner belong to
    a dead process and would draw phantom period jumps."""
    older = CURR_TEXT.replace("[0.35,0.65]", "[0.45,0.55]")
    c = parse_curriculum(older + CURR_TEXT)
    assert c["band"] == [0.35, 0.65], "last banner wins"
    assert len(c["history"]) == 2, "only the decisions after that banner"


def test_parse_curriculum_without_a_banner_is_empty_not_a_crash():
    c = parse_curriculum("iter 1 steps 5 sps 5 ep_rew 0 pi_loss 0 v_loss 0 ent 0 clip 0\n")
    assert c["band"] is None and c["bots"] == [] and c["history"] == []


def test_parse_curriculum_v9_plain_format():
    """v9 dropped 'in' from the banner, tuple-ised the roster, inserted cars=,
    and changed the arrow target from '3' to 'p3'. The last one is why this test
    exists: it still MATCHED the pre-v9 regex and silently reported the OLD
    period, so the ladder panel froze with nothing in any log to say so."""
    c = parse_curriculum(CURR_V9_PLAIN)
    assert c["band"] == [0.35, 0.65]
    assert (c["period_min"], c["period_max"]) == (1, 12)
    assert c["roster"] == ["element", "immortal"], "tuple-ised roster still yields names"
    assert c["arenas"] == "77/192", "cars=[...] sits between periods= and the arena count"
    el, im = c["bots"]
    assert (el["period"], im["period"]) == (3, 4), "'p4->p3' must report the NEW rung"


def test_parse_curriculum_v9_ladder_rungs_and_repeated_bots():
    """The v9 rung is a (cars, period) PAIR, and the same bot runs at three team
    sizes -- three independent slots that would otherwise collapse onto one dict
    key and show whichever segment came last."""
    c = parse_curriculum(CURR_V9_LADDER)
    assert (c["period_min"], c["period_max"]) == (1, 24), "bounds come from the ladder"
    assert c["arenas"] == "48/144"
    names = [b["name"] for b in c["bots"]]
    assert names == ["element", "necto·1s", "necto·2s", "necto·3s"], \
        "a bot at several team sizes needs several labels"
    by = {b["name"]: b for b in c["bots"]}
    assert by["necto·2s"]["rung"] == "1c/p4" and by["necto·2s"]["period"] == 4
    assert by["necto·3s"]["rung"] == "2c/p24" and by["necto·3s"]["cars"] == 2, \
        "a car-count change is a rung change too"
    # history keeps all four slots apart
    assert c["history"][0]["necto·1s"] == 24 and c["history"][0]["necto·3s"] == 1


def test_parse_curriculum_staggered_eval_carries_the_last_measurement():
    """Under the staggered eval only ONE team size is measured per cycle, so most
    slots print `n/a`. Their last real win rate is carried forward: rendering
    them as 0.0 would read as 'losing every match' on a slot that simply was not
    measured this cycle."""
    c = parse_curriculum(CURR_V9_LADDER)
    by = {b["name"]: b for b in c["bots"]}
    assert by["element"]["wr"] == 0.10 and by["element"]["measured"] is False, \
        "carried forward from the previous cycle, and flagged as stale"
    assert by["necto·2s"]["measured"] is True and by["necto·2s"]["wr"] == 0.90
    # ...but the history row for an unmeasured slot stays None, so the per-bot
    # win-rate chart never draws a fabricated point.
    assert c["history"][1]["element_wr"] is None


# --- gate history ----------------------------------------------------------

def test_parse_gate_history_sorted_and_skips_junk():
    text = (
        '{"ts": 200, "win_share": 0.84, "wins": 510, "draws": 51, "losses": 79, "n": 640, "passed": true}\n'
        "not json\n"
        '{"ts": 100, "win_share": 0.03, "wins": 1, "draws": 34, "losses": 608, "n": 643, "passed": false}\n'
        '{"ts": 300, "no_share_key": 1}\n'
    )
    rows = parse_gate_history(text)
    assert [r["ts"] for r in rows] == [100, 200], "sorted by ts, rows without win_share dropped"
    assert rows[-1]["passed"] is True


def test_parse_gate_history_defaults_mode_to_1v1():
    """Every row written before the team gate existed was 1v1 and carries no
    `mode`. Normalise on READ -- backfilling published records would be worse --
    so the renderer can label a 2v2 run without hiding the old ones."""
    rows = parse_gate_history(
        '{"ts": 100, "win_share": 0.84, "wins": 510, "draws": 51, "losses": 79, '
        '"n": 640, "passed": true}\n')
    assert rows[0]["mode"] == 1


def test_parse_gate_history_keeps_an_explicit_team_mode():
    """A 2v2 win share is a DIFFERENT quantity from the 1v1 ruler, so the row has
    to stay distinguishable all the way to the tile."""
    rows = parse_gate_history(
        '{"ts": 100, "win_share": 0.60, "wins": 300, "draws": 100, "losses": 240, '
        '"n": 640, "passed": true, "mode": 2, "records": 640, "short_records": 71}\n')
    assert rows[0]["mode"] == 2 and rows[0]["short_records"] == 71
