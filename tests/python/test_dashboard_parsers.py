"""Pure-parser tests for scripts/dashboard.py — every log format the dashboard
reads: all three main-run iter-line eras, bc banner/batch/epoch-done lines,
league registry rows, ssl_pull lines (cursor, counters, trailing-hour rate),
and both eval-history jsonl schemas."""
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
    bc_summary,
    downsample,
    estimate_bc_times,
    estimate_train_times,
    parse_bc_log,
    parse_eval_history,
    parse_h2h_history,
    parse_iter_line,
    parse_registry,
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

BC_TEXT = """EntityPolicyNet: 488,390 params (d_model=128 layers=2 heads=4 ff=512 aux=False action_table=92)
bc: 64140 train / 3428 val shards, 176938 batches/epoch x 2 epochs (device cuda)
bc epoch 0 batch 50/176938 loss 4.4770 lr 3.00e-04 13293 samples/s
bc epoch 0 done: train_loss 1.9484 val_loss 1.5999 top1 0.659 top3 0.844 recall_jump 0.458 recall_stall 0.000
bc: class-count cache data/bc/bc_class_counts.json is stale, recomputing
bc: 64140 train / 3428 val shards, 170326 batches/epoch x 2 epochs (device cuda)
bc epoch 0 batch 100/170326 loss 3.1000 lr 2.00e-04 20000 samples/s
bc epoch 0 done: train_loss 1.9669 val_loss 1.6498 top1 0.627 top3 0.826 recall_jump 0.459 recall_stall nan
bc epoch 1 batch 8800/170326 loss 1.8241 lr 1.38e-04 47243 samples/s
"""


def test_parse_bc_log_splits_runs_on_banner():
    runs = parse_bc_log(BC_TEXT)
    assert len(runs) == 2  # the "class-count cache" bc: line is NOT a banner
    assert runs[0]["train_shards"] == 64140 and runs[0]["val_shards"] == 3428
    assert runs[0]["batches_per_epoch"] == 176938 and runs[0]["epochs"] == 2
    assert len(runs[0]["batches"]) == 1 and runs[0]["batches"][0]["loss"] == 4.477
    assert runs[1]["batches_per_epoch"] == 170326
    assert [b["batch"] for b in runs[1]["batches"]] == [100, 8800]
    assert runs[1]["batches"][1]["samples_s"] == 47243


def test_parse_bc_log_epoch_done_and_nan():
    runs = parse_bc_log(BC_TEXT)
    d0 = runs[0]["epochs_done"][0]
    assert d0["train_loss"] == 1.9484 and d0["val_loss"] == 1.5999
    assert d0["top1"] == 0.659 and d0["top3"] == 0.844
    assert d0["recall_jump"] == 0.458 and d0["recall_stall"] == 0.0
    d1 = runs[1]["epochs_done"][0]
    assert d1["recall_stall"] is None  # nan must not reach json.dumps


def test_bc_summary_progress_and_history():
    s = bc_summary(parse_bc_log(BC_TEXT))
    assert s["runs"] == 2
    assert [(h["run"], h["epoch"]) for h in s["history"]] == [(1, 0), (2, 0)]
    cur = s["current"]
    assert cur["progress"]["epoch"] == 1 and cur["progress"]["batch"] == 8800
    expect = (1 * 170326 + 8800) / (2 * 170326)
    assert abs(cur["frac"] - expect) < 1e-9
    assert cur["last_done"]["val_loss"] == 1.6498
    assert [pt["gb"] for pt in cur["loss"]] == [100, 170326 + 8800]


def test_bc_summary_empty():
    assert bc_summary([]) == {"runs": 0, "history": [], "current": None}


# --- estimated wall times (logs carry no timestamps) ------------------------

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


def test_estimate_bc_times():
    b = [{"epoch": 0, "batch": 100, "total": 1000, "samples_s": 8192, "loss": 1.0},
         {"epoch": 0, "batch": 200, "total": 1000, "samples_s": 8192, "loss": 1.0},
         {"epoch": 1, "batch": 50, "total": 1000, "samples_s": 4096, "loss": 1.0}]
    ts = estimate_bc_times(b, 10_000.0)
    assert ts[-1] == 10_000.0
    # gb 200 -> 1050 across the epoch rollover: 850 batches * 4096 / 4096 s/s
    assert ts[2] - ts[1] == 850 * 4096 / 4096
    assert ts[1] - ts[0] == 100 * 4096 / 8192
    assert ts == sorted(ts)
    assert estimate_bc_times([], 1.0) == []


def test_bc_summary_attaches_ts_est():
    s = bc_summary(parse_bc_log(BC_TEXT), anchor_ts=50_000.0)
    pts = s["current"]["loss"]
    assert pts[-1]["ts_est"] == 50_000.0  # anchored at the log mtime
    assert [p["ts_est"] for p in pts] == sorted(p["ts_est"] for p in pts)
    # without an anchor the key is absent (payload stays lean)
    s2 = bc_summary(parse_bc_log(BC_TEXT))
    assert "ts_est" not in s2["current"]["loss"][0]


# --- league registry --------------------------------------------------------

REG_TEXT = """{"ck": "checkpoints/ck_003238789120.pt", "steps": 3238789120, "run": "main", "reward_config": "configs/reward_v1.toml", "added_ts": 1784074329, "mu": 25.544151785977643, "sigma": 4.8474433028836215, "games": 5, "schema_version": 0}
{"ck": "checkpoints_entity/ck_000163573760.pt", "steps": 163573760, "run": "v1", "mu": 30.1, "sigma": 6.2, "games": 3, "schema_version": 1}
{"ck": "old_no_sv.pt", "steps": 1, "run": "b", "mu": 10.0, "sigma": 8.0, "games": 0}
not json at all
"""


def test_parse_registry():
    rows = parse_registry(REG_TEXT, src="local")
    assert len(rows) == 3  # garbage line skipped
    assert rows[0]["ck"] == "ck_003238789120.pt"  # basename only
    assert rows[0]["schema_version"] == 0 and abs(rows[0]["mu"] - 25.5441517) < 1e-6
    assert rows[1]["schema_version"] == 1 and rows[1]["run"] == "v1"
    assert rows[2]["schema_version"] == 0  # missing key defaults to 0
    assert all(r["src"] == "local" for r in rows)


# --- ssl pull log -----------------------------------------------------------

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

EVAL_TEXT = """{"ts": 1784200000, "ck": "ck_000163573760.pt", "goals_min": 0.4, "touches_min": 9.1, "dist": 2100.5}
{"ts": 1784163285, "steps": 4917688320, "touches_per_min": 18.48, "dist_uu": 1542.0, "goals_per_min": 1.69}
garbage
{"no_ts": true}
"""


def test_parse_eval_history_both_schemas():
    rows = parse_eval_history(EVAL_TEXT)
    assert len(rows) == 2
    assert [r["ts"] for r in rows] == [1784163285, 1784200000]  # sorted by ts
    legacy, new = rows
    assert legacy["ck"] == "ck_4917688320"  # derived from steps
    assert legacy["goals_min"] == 1.69 and legacy["touches_min"] == 18.48
    assert legacy["dist"] == 1542.0
    assert new["ck"] == "ck_000163573760.pt" and new["goals_min"] == 0.4
    assert new["dist"] == 2100.5


# --- h2h history jsonl -------------------------------------------------------

H2H_TEXT = """{"ts": 1784200000, "ck": "ck_001382942720.pt", "ref": "ck_000562083840.pt", "ref_label": "peak-562M", "goals_ck": 35, "goals_ref": 119, "share": 0.2273, "steps": 5400, "seed": 11}
{"ts": 1784163285, "ck": "ck_000909000000.pt", "ref": "ck_000562083840.pt", "ref_label": "peak-562M", "goals_ck": 22, "goals_ref": 77, "share": 0.2222, "steps": 5400, "seed": 11}
garbage
{"no_ts": true}
"""


def test_parse_h2h_history_sorted_by_ts():
    rows = parse_h2h_history(H2H_TEXT)
    assert len(rows) == 2
    assert [r["ts"] for r in rows] == [1784163285, 1784200000]
    first, second = rows
    assert first["ck"] == "ck_000909000000.pt"
    assert first["ref_label"] == "peak-562M"
    assert first["goals_ck"] == 22 and first["goals_ref"] == 77
    assert first["share"] == pytest.approx(0.2222)
    assert second["ck"] == "ck_001382942720.pt" and second["steps"] == 5400 and second["seed"] == 11


def test_parse_h2h_history_recomputes_missing_share():
    text = '{"ts": 1, "ck": "a.pt", "ref": "b.pt", "ref_label": "L", "goals_ck": 3, "goals_ref": 1, "steps": 10, "seed": 0}\n'
    rows = parse_h2h_history(text)
    assert rows[0]["share"] == pytest.approx(0.75)


def test_parse_h2h_history_zero_zero_share_is_none():
    text = '{"ts": 1, "ck": "a.pt", "ref": "b.pt", "ref_label": "L", "goals_ck": 0, "goals_ref": 0, "steps": 10, "seed": 0}\n'
    rows = parse_h2h_history(text)
    assert rows[0]["share"] is None
