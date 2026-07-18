"""Multi-run training dashboard. One dark page, a panel per workstream:

  main    remote KL-PPO entity run   checkpoints_entity/train_remote.log (synced)
  bc      local BC pre-training      logs/bc_train.log (multiple runs, banner-split)
  league  ladder registries          league/registry.jsonl + league/registry_remote.jsonl
  ssl     SSL replay pull            logs/ssl_pull.log + data/replays/ssl/**/*.replay
  evals   skill evals over time      logs/eval_history.jsonl (+ legacy checkpoints/eval_history.jsonl)
  system  CPU / RAM / GPU samplers   + Windows-host C: free (powershell, cached)

Parsing lives in pure functions at the top of this file, tested in
tests/python/test_dashboard_parsers.py. Slow host queries (powershell C: free,
the replay walk) run in background sampler threads and are cached — the request
path never blocks on them. Log parses are cached on (mtime, size).

Usage: python scripts/dashboard.py [port] [--eval-every MINUTES]
       (default port 8420, eval every 30 min; --no-eval disables)
From Windows: http://localhost:<port> (WSL2 forwards localhost TCP).
Stdlib only; the page works offline (no CDN, hand-rolled SVG charts).
"""
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAIN_LOG = REPO / "checkpoints_entity" / "train_remote.log"   # synced remote train_v1.log
CKPT_DIR = REPO / "checkpoints_entity"
BC_LOG = REPO / "logs" / "bc_train.log"
LEAGUE_LOCAL = REPO / "league" / "registry.jsonl"
LEAGUE_REMOTE = REPO / "league" / "registry_remote.jsonl"     # synced by sync_remote.sh
SSL_LOG = REPO / "logs" / "ssl_pull.log"
SSL_DIR = REPO / "data" / "replays" / "ssl"
EVAL_HISTORY = REPO / "logs" / "eval_history.jsonl"           # appended by eval_metrics.py
EVAL_HISTORY_LEGACY = REPO / "checkpoints" / "eval_history.jsonl"
MAX_POINTS = 500

# ---------------------------------------------------------------------------
# pure parsers (tested in tests/python/test_dashboard_parsers.py)
# ---------------------------------------------------------------------------

# Three iter-line eras share a prefix: kickstart appends "kick_kl X lambda_k Y",
# post-K4 appends "kl_pri X lambda_p Y", and the span between eras is plain.
ITER_LINE = re.compile(
    r"iter (\d+) steps ([\d,]+) sps ([\d,]+) ep_rew ([-\d.]+) "
    r"pi_loss ([-\d.]+) v_loss ([-\d.]+) ent ([-\d.]+) clip ([-\d.]+)"
    r"(?: kick_kl ([-\d.]+) lambda_k ([-\d.]+))?"
    r"(?: kl_pri ([-\d.]+) lambda_p ([-\d.]+))?"
)
RESUME = re.compile(r"resumed at ([\d,]+) steps")
CONTAINMENT = "physics blowup contained"

BC_BANNER = re.compile(r"bc: (\d+) train / (\d+) val shards, (\d+) batches/epoch x (\d+) epochs")
BC_BATCH = re.compile(r"bc epoch (\d+) batch (\d+)/(\d+) loss (\S+) lr (\S+) (\d+) samples/s")
BC_DONE = re.compile(
    r"bc epoch (\d+) done: train_loss (\S+) val_loss (\S+) top1 (\S+) top3 (\S+) "
    r"recall_jump (\S+) recall_stall (\S+)"
)

SSL_TS = r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
SSL_START = re.compile(SSL_TS + r" start: (\d+) replay\(s\) on disk, filling (\S+) \((\d+)/(\d+)\)")
SSL_PAGE = re.compile(SSL_TS + r" page: .* oldest created (\S+)")
SSL_LANDED = re.compile(SSL_TS + r" landed \S+ -> (\S+) \((\d+) this run\)")
SSL_PROGRESS = re.compile(
    SSL_TS + r" progress: (\d+) this run / (\d+) total \((\d+) deduped, (\d+) failed\).*?([\d.]+)/h"
)
SSL_ANYTS = re.compile(r"^" + SSL_TS + " ")


def _f(s):
    """float() that maps unparseable / non-finite (nan, inf) to None — NaN must
    never reach json.dumps (browsers reject bare NaN tokens)."""
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def parse_iter_line(line):
    """One training-log iter line -> dict, or None. Handles all three eras;
    kick_kl/lambda_k and kl_pri/lambda_p keys are present only when logged."""
    m = ITER_LINE.search(line)
    if not m:
        return None
    row = {
        "iter": int(m.group(1)),
        "steps": int(m.group(2).replace(",", "")),
        "sps": int(m.group(3).replace(",", "")),
        "ep_rew": float(m.group(4)),
        "pi_loss": float(m.group(5)),
        "v_loss": float(m.group(6)),
        "ent": float(m.group(7)),
        "clip": float(m.group(8)),
    }
    if m.group(9) is not None:
        row["kick_kl"], row["lambda_k"] = float(m.group(9)), float(m.group(10))
    if m.group(11) is not None:
        row["kl_pri"], row["lambda_p"] = float(m.group(11)), float(m.group(12))
    return row


def parse_train_log(text):
    """Whole main-run log -> {rows (sorted by steps), restarts, containment}.

    A 'resumed at S steps' marker prunes earlier rows with steps > S: those
    belong to an abandoned branch (e.g. K4 rolled back from 1.38B to the 562M
    checkpoint to add the BC prior), and would otherwise pollute the charts and
    masquerade as the latest iteration."""
    rows, restarts = [], []
    for line in text.splitlines():
        m = RESUME.search(line)
        if m:
            s = int(m.group(1).replace(",", ""))
            restarts.append(s)
            rows = [r for r in rows if r["steps"] <= s]
            continue
        row = parse_iter_line(line)
        if row:
            rows.append(row)
    rows.sort(key=lambda r: r["steps"])
    return {"rows": rows, "restarts": restarts, "containment": text.count(CONTAINMENT)}


def downsample(rows, n, tail=0):
    """Evenly thin a list to at most n entries, keeping first and last. With
    tail > 0 the last `tail` entries are kept dense (the live end of a log is
    what's being watched) and only the head is thinned."""
    if len(rows) <= n:
        return rows
    if tail:
        tail = min(tail, n - 2)
        return downsample(rows[:-tail], n - tail) + rows[-tail:]
    stride = len(rows) / n
    return [rows[int(i * stride)] for i in range(n - 1)] + [rows[-1]]


def parse_bc_log(text):
    """BC log -> list of runs (split on banner lines; anything before the first
    banner is ignored). Each run: banner fields + batch lines + epoch-done rows.
    'bc: class-count cache … stale' lines are not banners."""
    runs = []
    cur = None
    for line in text.splitlines():
        m = BC_BANNER.search(line)
        if m:
            cur = {
                "train_shards": int(m.group(1)), "val_shards": int(m.group(2)),
                "batches_per_epoch": int(m.group(3)), "epochs": int(m.group(4)),
                "batches": [], "epochs_done": [],
            }
            runs.append(cur)
            continue
        if cur is None:
            continue
        m = BC_BATCH.search(line)
        if m:
            cur["batches"].append({
                "epoch": int(m.group(1)), "batch": int(m.group(2)), "total": int(m.group(3)),
                "loss": _f(m.group(4)), "lr": _f(m.group(5)), "samples_s": int(m.group(6)),
            })
            continue
        m = BC_DONE.search(line)
        if m:
            cur["epochs_done"].append({
                "epoch": int(m.group(1)), "train_loss": _f(m.group(2)), "val_loss": _f(m.group(3)),
                "top1": _f(m.group(4)), "top3": _f(m.group(5)),
                "recall_jump": _f(m.group(6)), "recall_stall": _f(m.group(7)),
            })
    return runs


def bc_summary(runs, max_points=300):
    """Shape parse_bc_log() output for the page: current-run progress + loss
    series, and epoch-done history across all runs labeled by banner index."""
    history = [{"run": i + 1, **d} for i, run in enumerate(runs) for d in run["epochs_done"]]
    out = {"runs": len(runs), "history": history, "current": None}
    if runs:
        cur = runs[-1]
        prog = cur["batches"][-1] if cur["batches"] else None
        frac = None
        if prog and cur["epochs"] and prog["total"]:
            frac = (prog["epoch"] * prog["total"] + prog["batch"]) / (cur["epochs"] * prog["total"])
        series = [
            {"gb": b["epoch"] * b["total"] + b["batch"], "loss": b["loss"], "sps": b["samples_s"]}
            for b in cur["batches"] if b["loss"] is not None
        ]
        out["current"] = {
            "banner": {k: cur[k] for k in ("train_shards", "val_shards", "batches_per_epoch", "epochs")},
            "progress": prog, "frac": frac, "loss": downsample(series, max_points),
            "last_done": cur["epochs_done"][-1] if cur["epochs_done"] else None,
        }
    return out


def parse_registry(text, src=""):
    """League registry jsonl -> rows for the ladder table. Bad lines skipped;
    schema_version defaults to 0 (pre-v1 entries)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict) or "ck" not in e:
            continue
        rows.append({
            "ck": str(e["ck"]).rsplit("/", 1)[-1],
            "steps": int(e.get("steps") or 0),
            "run": str(e.get("run", "?")),
            "schema_version": int(e.get("schema_version") or 0),
            "mu": _f(e.get("mu")),
            "sigma": _f(e.get("sigma")),
            "games": int(e.get("games") or 0),
            "src": src,
        })
    return rows


def _ssl_epoch(mmdd_hms, now):
    """'07-17 16:13:19' -> epoch seconds. The log carries no year: assume now's
    year, roll back one if that lands in the future (Dec->Jan wrap)."""
    yr = time.localtime(now).tm_year
    for y in (yr, yr - 1):
        try:
            t = time.mktime(time.strptime(f"{y}-{mmdd_hms}", "%Y-%m-%d %H:%M:%S"))
        except ValueError:
            continue
        if t <= now + 2 * 86400:
            return t
    return None


def parse_ssl_log(text, now):
    """SSL pull log -> cursor position, run counters, (ts, cumulative) samples
    for the trailing-hour rate. Counters reset at each 'start:' line."""
    out = {"cursor_oldest": None, "this_run": None, "total": None, "deduped": None,
           "failed": None, "logged_rate_h": None, "batch": None, "batch_fill": None,
           "batch_target": None, "run_start_ts": None, "last_ts": None, "samples": []}

    def _seen(ts):
        if ts is not None:
            out["last_ts"] = ts

    for line in text.splitlines():
        m = SSL_START.search(line)
        if m:
            ts = _ssl_epoch(m.group(1), now)
            out.update(run_start_ts=ts, samples=[], this_run=0, batch=m.group(3),
                       batch_fill=int(m.group(4)), batch_target=int(m.group(5)))
            _seen(ts)
            continue
        m = SSL_PAGE.search(line)
        if m:
            out["cursor_oldest"] = m.group(2)
            _seen(_ssl_epoch(m.group(1), now))
            continue
        m = SSL_LANDED.search(line)
        if m:
            ts, n = _ssl_epoch(m.group(1), now), int(m.group(3))
            out["this_run"], out["batch"] = n, m.group(2)
            if ts is not None:
                out["samples"].append((ts, n))
            _seen(ts)
            continue
        m = SSL_PROGRESS.search(line)
        if m:
            ts = _ssl_epoch(m.group(1), now)
            out.update(this_run=int(m.group(2)), total=int(m.group(3)), deduped=int(m.group(4)),
                       failed=int(m.group(5)), logged_rate_h=_f(m.group(6)))
            if ts is not None:
                out["samples"].append((ts, int(m.group(2))))
            _seen(ts)
            continue
        m = SSL_ANYTS.match(line)
        if m:
            _seen(_ssl_epoch(m.group(1), now))
    return out


def _counter_at(pts, t):
    """Linear interpolation of a cumulative counter at time t (pts sorted)."""
    if t <= pts[0][0]:
        return pts[0][1]
    for (t1, n1), (t2, n2) in zip(pts, pts[1:]):
        if t1 <= t <= t2:
            return n2 if t2 == t1 else n1 + (n2 - n1) * (t - t1) / (t2 - t1)
    return pts[-1][1]


def ssl_last_hour(samples, run_start_ts, now):
    """Replays landed in the trailing hour, interpolated from the sparse
    (ts, cumulative-this-run) samples. None when there are no samples."""
    if not samples:
        return None
    pts = list(samples)
    if run_start_ts is not None and run_start_ts < pts[0][0]:
        pts.insert(0, (run_start_ts, 0))
    return max(0, round(_counter_at(pts, now) - _counter_at(pts, now - 3600)))


def parse_eval_history(text):
    """Eval-history jsonl -> normalized rows sorted by ts. Accepts both the new
    eval_metrics.py schema {ts, ck, goals_min, touches_min, dist} and the legacy
    EvalRunner schema {ts, steps, goals_per_min, touches_per_min, dist_uu}."""
    rows = []
    for line in text.splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if not isinstance(e, dict) or "ts" not in e:
            continue
        ck = e.get("ck") or (f"ck_{e['steps']}" if "steps" in e else "?")
        rows.append({
            "ts": int(e["ts"]), "ck": str(ck),
            "goals_min": _f(e.get("goals_min", e.get("goals_per_min"))),
            "touches_min": _f(e.get("touches_min", e.get("touches_per_min"))),
            "dist": _f(e.get("dist", e.get("dist_uu"))),
        })
    rows.sort(key=lambda r: r["ts"])
    return rows


# ---------------------------------------------------------------------------
# IO + caching (parses re-run only when the file's mtime/size changes)
# ---------------------------------------------------------------------------

_CACHE = {}


def _read(path):
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def cached_parse(path, fn, tag=None):
    key = (str(path), tag or fn.__name__)
    try:
        st = path.stat()
        sig = (st.st_mtime_ns, st.st_size)
    except OSError:
        sig = None
    hit = _CACHE.get(key)
    if hit and hit[0] == sig:
        return hit[1]
    val = fn(_read(path))
    _CACHE[key] = (sig, val)
    return val


def checkpoint_info():
    cks = sorted(CKPT_DIR.glob("ck_*.pt"))
    if not cks:
        return {}
    total = sum(f.stat().st_size for f in cks)
    oldest_mtime = min(f.stat().st_mtime for f in cks)
    # by mtime, not name: after a rollback (see parse_train_log) the current
    # branch writes lower-numbered checkpoints than the abandoned one
    latest = max(cks, key=lambda f: f.stat().st_mtime)
    return {
        "count": len(cks),
        "total_gb": round(total / 1e9, 2),
        "latest": latest.name,
        "latest_steps": int(latest.stem.split("_")[1]),
        "runtime_s": int(time.time() - oldest_mtime),
    }


# ---------------------------------------------------------------------------
# background samplers — everything slow is cached off the request path
# ---------------------------------------------------------------------------

class SysSampler(threading.Thread):
    """Samples GPU (nvidia-smi) + CPU + RAM every 5s into a rolling window."""

    def __init__(self):
        super().__init__(daemon=True)
        self.history = deque(maxlen=720)  # 1h at 5s
        self._prev_cpu = None

    def _cpu_pct(self):
        parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
        vals = list(map(int, parts))
        idle, total = vals[3] + vals[4], sum(vals)
        if self._prev_cpu is None:
            self._prev_cpu = (idle, total)
            return None
        pi, pt = self._prev_cpu
        self._prev_cpu = (idle, total)
        dt = total - pt
        return round(100 * (1 - (idle - pi) / dt), 1) if dt > 0 else None

    def _ram_pct(self):
        info = dict(
            line.split(":") for line in Path("/proc/meminfo").read_text().splitlines()
        )
        total = int(info["MemTotal"].split()[0])
        avail = int(info["MemAvailable"].split()[0])
        return round(100 * (1 - avail / total), 1)

    def _gpu(self):
        try:
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            ).stdout.strip().split(", ")
            return int(out[0]), int(out[1]), int(out[2]), int(out[3])
        except Exception:
            return None

    def run(self):
        while True:
            sample = {"ts": int(time.time())}
            cpu = self._cpu_pct()
            if cpu is not None:
                sample["cpu"] = cpu
            sample["ram"] = self._ram_pct()
            gpu = self._gpu()
            if gpu:
                sample["gpu"], sample["vram_used"], sample["vram_total"], sample["gpu_temp"] = gpu
            if "cpu" in sample:
                self.history.append(sample)
            time.sleep(5)


class SlowSampler(threading.Thread):
    """Caches the two slow host queries: Windows-host C: free GB (powershell
    interop, >1s — every 5 min max) and the on-disk SSL replay count (directory
    walk, not a shell glob — ARG_MAX — every 60s). The WSL df number is a lie;
    host free is the real one."""

    def __init__(self):
        super().__init__(daemon=True)
        self.host_free_gb = None
        self.ssl_count = None
        self._last_host = 0.0

    def _host_free(self):
        try:
            out = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command",
                 "[math]::Round((Get-PSDrive C).Free/1GB,1)"],
                capture_output=True, text=True, timeout=60, cwd="/mnt/c",
            )
            return float(out.stdout.strip())
        except Exception:
            return None

    def _count_ssl(self):
        n = 0
        for _, _, files in os.walk(SSL_DIR):
            n += sum(1 for f in files if f.endswith(".replay"))
        return n

    def run(self):
        while True:
            now = time.time()
            if now - self._last_host >= 300:
                self._last_host = now  # even on failure — don't hammer broken interop
                free = self._host_free()
                if free is not None:
                    self.host_free_gb = free
            try:
                self.ssl_count = self._count_ssl()
            except OSError:
                pass
            time.sleep(60)


class EvalRunner(threading.Thread):
    """Every N minutes, evals the newest main-run checkpoint it hasn't seen
    (nice -n 15). eval_metrics.py appends the result to logs/eval_history.jsonl
    itself; this thread only schedules and dedupes by checkpoint name."""

    def __init__(self, every_min):
        super().__init__(daemon=True)
        self.every = every_min * 60
        self.status = "idle"

    def _evaluated(self):
        return {row["ck"] for row in parse_eval_history(_read(EVAL_HISTORY))}

    def run(self):
        while True:
            cks = sorted(CKPT_DIR.glob("ck_*.pt"))
            if cks and cks[-1].name not in self._evaluated():
                latest = cks[-1]
                self.status = f"evaluating {latest.name}…"
                try:
                    proc = subprocess.run(
                        ["nice", "-n", "15", sys.executable,
                         str(REPO / "scripts" / "eval_metrics.py"), str(latest)],
                        capture_output=True, text=True, cwd=REPO, timeout=1800,
                    )
                    self.status = "idle" if proc.returncode == 0 else \
                        f"eval failed (rc {proc.returncode})"
                except Exception as e:
                    self.status = f"eval failed: {e}"
            time.sleep(self.every)


SAMPLER = SysSampler()
SLOW = SlowSampler()
EVALER = None


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------

def _main_payload(now):
    train = cached_parse(MAIN_LOG, parse_train_log)
    rows = downsample(train["rows"], MAX_POINTS, tail=150)
    last = rows[-1] if rows else None
    eta = None
    if last and last["sps"] > 0:
        nxt = (last["steps"] // 100_000_000 + 1) * 100_000_000
        eta = int((nxt - last["steps"]) / last["sps"])
    try:
        age = int(now - MAIN_LOG.stat().st_mtime)
    except OSError:
        age = None
    return {"rows": rows, "restarts": train["restarts"], "containment": train["containment"],
            "ckpt": checkpoint_info(), "eta_s": eta, "log_age_s": age}


def _ssl_payload(now):
    p = cached_parse(SSL_LOG, lambda t: parse_ssl_log(t, time.time()), tag="ssl")
    out = {k: p[k] for k in ("cursor_oldest", "this_run", "total", "deduped", "failed",
                             "logged_rate_h", "batch", "batch_fill", "batch_target")}
    out["last_hour"] = ssl_last_hour(p["samples"], p["run_start_ts"], now)
    out["last_age_s"] = int(now - p["last_ts"]) if p["last_ts"] else None
    out["disk_count"] = SLOW.ssl_count
    return out


def payload():
    now = time.time()
    league = []
    for path, src in ((LEAGUE_LOCAL, "local"), (LEAGUE_REMOTE, "remote")):
        league += cached_parse(path, lambda t, s=src: parse_registry(t, s), tag="reg_" + src)
    league.sort(key=lambda e: -(e["mu"] if e["mu"] is not None else float("-inf")))
    evals = sorted(
        cached_parse(EVAL_HISTORY_LEGACY, parse_eval_history, tag="ev_legacy")
        + cached_parse(EVAL_HISTORY, parse_eval_history, tag="ev_new"),
        key=lambda r: r["ts"],
    )
    return {
        "main": _main_payload(now),
        "bc": bc_summary(cached_parse(BC_LOG, parse_bc_log)),
        "league": league,
        "ssl": _ssl_payload(now),
        "evals": evals,
        "eval_status": EVALER.status if EVALER else "disabled",
        "sys": list(SAMPLER.history),
        "host_free_gb": SLOW.host_free_gb,
    }


# ---------------------------------------------------------------------------
# page
# ---------------------------------------------------------------------------

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Construct training</title>
<style>
:root { --surface:#131312; --panel:#1b1b19; --card:#232320; --ink:#f4f3ec;
        --ink2:#c3c2b7; --muted:#8a897e; --grid:#33322f; --border:#33322f;
        --series:#3987e5; --accent:rgba(57,135,229,.16) }
* { box-sizing:border-box; margin:0 }
body { background:var(--surface); color:var(--ink);
       font:14px/1.45 system-ui,-apple-system,sans-serif; padding:18px; }
h1 { font-size:17px; font-weight:650 }
.sub { color:var(--ink2); font-size:12.5px; margin:2px 0 14px }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:10px;
         padding:14px 16px 12px; margin-bottom:14px }
.panel.main { border-color:#2c5e9e }
.panel h2 { font-size:13px; font-weight:650; color:var(--ink2); margin-bottom:10px;
            text-transform:uppercase; letter-spacing:.05em }
.panel h2 .meta { font-weight:400; text-transform:none; letter-spacing:0;
                  color:var(--muted); margin-left:8px }
.cols { display:grid; grid-template-columns:repeat(auto-fit,minmax(430px,1fr)); gap:14px;
        margin-bottom:14px }
.cols .panel { margin-bottom:0 }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:10px; margin-bottom:12px }
.tile { background:var(--card); border:1px solid var(--border); border-radius:8px;
        padding:9px 12px }
.tile .k { font-size:11px; color:var(--ink2); text-transform:uppercase;
           letter-spacing:.04em }
.tile .v { font-size:20px; font-weight:650; font-variant-numeric:tabular-nums }
.tile .d { font-size:11.5px; color:var(--muted) }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px }
.card { background:var(--card); border:1px solid var(--border); border-radius:8px;
        padding:12px 12px 8px }
.card h3 { font-size:12.5px; font-weight:600; color:var(--ink2); margin-bottom:2px }
.card .why { font-size:11.5px; color:var(--muted); margin-top:4px }
svg.chart { display:block; width:100%; height:150px }
.spark { display:inline-block; width:90px; height:22px; vertical-align:middle;
         margin-left:6px }
.spark polyline { fill:none; stroke:var(--series); stroke-width:1.5 }
.axis { font-size:10px; fill:var(--muted); font-variant-numeric:tabular-nums }
.gridline { stroke:var(--grid); stroke-width:1 }
.line { stroke:var(--series); stroke-width:2; fill:none;
        stroke-linejoin:round; stroke-linecap:round }
.mark { fill:var(--series); stroke:var(--card); stroke-width:2 }
.restart { stroke:var(--muted); stroke-width:1; stroke-dasharray:3 3 }
.cross { stroke:var(--muted); stroke-width:1; stroke-dasharray:2 3 }
.dot { fill:var(--series); stroke:var(--card); stroke-width:2 }
.tip { position:fixed; pointer-events:none; background:var(--card); color:var(--ink);
       border:1px solid var(--border); border-radius:6px; padding:5px 8px;
       font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.4); display:none;
       font-variant-numeric:tabular-nums; z-index:5 }
.pbar { position:relative; height:22px; background:var(--surface);
        border:1px solid var(--border); border-radius:6px; overflow:hidden;
        margin:0 0 12px }
.pfill { position:absolute; top:0; bottom:0; left:0; background:var(--series);
         opacity:.45 }
.pbar span { position:relative; display:block; text-align:center; font-size:11.5px;
             line-height:20px; color:var(--ink); font-variant-numeric:tabular-nums }
details { margin-top:10px }
summary { cursor:pointer; color:var(--ink2); font-size:13px }
table { border-collapse:collapse; margin-top:8px; font-variant-numeric:tabular-nums;
        font-size:12.5px; width:100% }
th,td { text-align:right; padding:4px 10px; border-bottom:1px solid var(--border) }
th { color:var(--ink2); font-weight:600 }
tr.v1 td { background:var(--accent) }
tr.v1 td:first-child { border-left:3px solid var(--series) }
tr.curr td { background:var(--accent) }
.wrap { overflow-x:auto }
</style></head><body data-palette="#3987e5">
<h1>Construct — training</h1>
<div class="sub" id="status">loading…</div>

<section class="panel main">
  <h2>Main run · remote KL-PPO entity<span class="meta" id="main-meta"></span></h2>
  <div class="tiles" id="main-tiles"></div>
  <div class="grid" id="main-charts"></div>
  <details><summary>Recent iterations</summary>
    <div class="wrap"><table id="tbl"></table></div></details>
</section>

<div class="cols">
  <section class="panel">
    <h2>BC training · local<span class="meta" id="bc-meta"></span></h2>
    <div class="pbar" id="bc-bar"><span>—</span></div>
    <div class="tiles" id="bc-tiles"></div>
    <div class="grid" id="bc-charts"></div>
    <div class="wrap"><table id="bc-hist"></table></div>
  </section>
  <section class="panel">
    <h2>League ladder<span class="meta" id="lg-meta"></span></h2>
    <div class="wrap"><table id="lg-tbl"></table></div>
  </section>
  <section class="panel">
    <h2>SSL replay pull<span class="meta" id="ssl-meta"></span></h2>
    <div class="pbar" id="ssl-bar"><span>—</span></div>
    <div class="tiles" id="ssl-tiles"></div>
  </section>
  <section class="panel">
    <h2>Skill evals<span class="meta" id="evalstatus"></span></h2>
    <div class="grid" id="evalcharts"></div>
  </section>
</div>

<section class="panel">
  <h2>System</h2>
  <div class="tiles" id="systiles"></div>
  <div class="grid" id="syscharts"></div>
</section>
<div class="tip" id="tip"></div>
<script>
const MAIN_METRICS = [
  {key:"sps", title:"Throughput (steps/sec)", fmt:v=>v.toLocaleString(),
   why:"How fast experience is collected and learned. Dips = thermal throttle, evals competing for CPU, or checkpoint writes."},
  {key:"ep_rew", title:"Reward per completed episode", fmt:v=>v.toFixed(2), cap:.98,
   why:"Total reward per finished episode. Rising = scoring more / conceding less (y capped at p98 for readability)."},
  {key:"ent", title:"Policy entropy", fmt:v=>v.toFixed(3),
   why:"Action randomness. Falls as the policy commits. Falling too fast = premature convergence; flat at max = not learning."},
  {key:"kl_pri", title:"KL to BC prior (kl_pri)", fmt:v=>v.toFixed(3),
   why:"Post-K4: divergence from the frozen BC prior, penalized at lambda_p. Only iterations from the kl-prior era plot here; the kickstart era logged kick_kl instead."},
];
const BC_METRICS = [
  {key:"loss", title:"Train loss (current run)", fmt:v=>v.toFixed(3),
   why:"Running cross-entropy over the current run's batches. x = global batch."},
  {key:"sps", title:"Samples / sec", fmt:v=>v.toLocaleString(),
   why:"BC dataloader + GPU throughput. Sustained dips = shard IO or the GPU busy elsewhere."},
];
const SYS_METRICS = [
  {key:"gpu", title:"GPU utilization (%)", fmt:v=>Math.round(v)+"%",
   why:"BC training + any local eval. Bursty is normal."},
  {key:"gpu_temp", title:"GPU temperature (°C)", fmt:v=>Math.round(v)+"°C",
   why:"Laptop GPUs throttle around ~87°C."},
  {key:"cpu", title:"CPU utilization (%)", fmt:v=>Math.round(v)+"%",
   why:"Dataloaders, SSL pull, sync loops, RocketSim evals."},
];
const EVAL_METRICS = [
  {key:"goals_min", title:"Goals / min / match", fmt:v=>v.toFixed(2), marks:true,
   why:"Goals per match-minute in a headless eval — the objective skill metric, plotted over wall time across checkpoints."},
  {key:"touches_min", title:"Ball touches / min / agent", fmt:v=>v.toFixed(1), marks:true,
   why:"How often the bot contacts the ball. Random baseline 0.0."},
  {key:"dist", title:"Mean dist to ball (uu)", fmt:v=>v.toFixed(0), marks:true,
   why:"Average car-to-ball distance. Random baseline 3769 uu; lower = involved in the play."},
];
const tip = document.getElementById("tip");
const fmtSteps = v => v >= 1e9 ? (v/1e9).toFixed(2)+"B" : v >= 1e6 ? (v/1e6).toFixed(0)+"M" : v.toLocaleString();
const fmtDur = s => { const h = Math.floor(s/3600), m = Math.floor(s%3600/60);
                      return h ? `${h}h ${m}m` : `${m}m`; };
const fmtAgo = s => s == null ? "—" : s < 90 ? "just now" :
                    s < 5400 ? Math.round(s/60)+"m ago" : (s/3600).toFixed(1)+"h ago";
const fmtClock = v => new Date(v*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
const fmtDay = v => { const d = new Date(v*1000);
  return `${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")} ${fmtClock(v)}`; };
const mkcard = () => Object.assign(document.createElement("div"), {className:"card"});
const num = (v,f) => v == null ? "—" : f(v);

function spark(vals) {
  if (!vals || vals.length < 2) return "";
  const w = 90, h = 22, lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1;
  const pts = vals.map((v,i) =>
    `${(i/(vals.length-1)*w).toFixed(1)},${(h-2-(v-lo)/span*(h-4)).toFixed(1)}`).join(" ");
  return `<svg class="spark" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none"><polyline points="${pts}"/></svg>`;
}

function chart(el, rows, m, xKey, xFmt, restarts) {
  const W = el.clientWidth || 320, H = 150, L = 48, R = 8, T = 8, B = 20;
  const xs = rows.map(r => r[xKey]), ys = rows.map(r => r[m.key]);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (m.cap) { const s = [...ys].sort((a,b)=>a-b); hi = s[Math.floor((s.length-1)*m.cap)]; }
  if (lo === hi) { lo -= 1; hi += 1; }
  const x0 = xs[0], x1 = xs[xs.length-1] || 1;
  const X = v => L + (v - x0) / (x1 - x0 || 1) * (W - L - R);
  const Y = v => T + (1 - (Math.min(v, hi) - lo) / (hi - lo)) * (H - T - B);
  let g = "";
  for (let i = 0; i <= 3; i++) {
    const yv = lo + (hi - lo) * i / 3, y = Y(yv);
    g += `<line class="gridline" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"/>` +
         `<text class="axis" x="${L-5}" y="${y+3}" text-anchor="end">${m.fmt(yv)}</text>`;
  }
  [x0, (x0+x1)/2, x1].forEach(v => {
    g += `<text class="axis" x="${X(v)}" y="${H-5}" text-anchor="middle">${xFmt(v)}</text>`;
  });
  (restarts||[]).forEach(rv => {
    if (rv > x0 && rv < x1)
      g += `<line class="restart" x1="${X(rv)}" x2="${X(rv)}" y1="${T}" y2="${H-B}"><title>training resumed here</title></line>`;
  });
  const path = rows.map((r,i)=>`${i?"L":"M"}${X(r[xKey]).toFixed(1)},${Y(r[m.key]).toFixed(1)}`).join("");
  const marks = m.marks ? rows.map(r=>`<circle class="mark" r="4" cx="${X(r[xKey]).toFixed(1)}" cy="${Y(r[m.key]).toFixed(1)}"/>`).join("") : "";
  el.innerHTML = `<h3>${m.title}</h3>
    <svg class="chart" viewBox="0 0 ${W} ${H}">${g}<path class="line" d="${path}"/>${marks}
      <line class="cross" y1="${T}" y2="${H-B}" x1="-9" x2="-9"/>
      <circle class="dot" r="4" cx="-9" cy="-9"/></svg>
    <div class="why">${m.why}</div>`;
  const svg = el.querySelector("svg"), cross = el.querySelector(".cross"),
        dot = el.querySelector(".dot");
  svg.addEventListener("mousemove", e => {
    const box = svg.getBoundingClientRect();
    const mx = (e.clientX - box.left) * (W / box.width);
    let best = 0, bd = 1e18;
    rows.forEach((r,i) => { const d = Math.abs(X(r[xKey])-mx); if (d < bd) { bd = d; best = i; } });
    const r = rows[best], px = X(r[xKey]), py = Y(r[m.key]);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px);
    dot.setAttribute("cx", px); dot.setAttribute("cy", py);
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 190) + "px";
    tip.style.top = (e.clientY + 14) + "px";
    tip.innerHTML = `<b>${m.fmt(r[m.key])}</b><br><span style="color:var(--ink2)">at ${xFmt(r[xKey])}</span>`;
  });
  svg.addEventListener("mouseleave", () => {
    tip.style.display = "none";
    cross.setAttribute("x1", -9); cross.setAttribute("x2", -9);
    dot.setAttribute("cx", -9); dot.setAttribute("cy", -9);
  });
}

function tiles(el, defs) {
  el.innerHTML = defs.map(([k,v,d]) =>
    `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
}

function grids(id, metrics) {
  const g = document.getElementById(id);
  if (!g.children.length) metrics.forEach(() => g.appendChild(mkcard()));
  return g;
}

function renderMain(md) {
  const meta = document.getElementById("main-meta");
  meta.textContent = md.log_age_s != null ? `log synced ${fmtAgo(md.log_age_s)}` : "log missing";
  const rows = md.rows;
  if (!rows.length) return;
  const last = rows[rows.length-1];
  const klRows = rows.filter(r => r.kl_pri != null);
  const klLast = klRows.length ? klRows[klRows.length-1] : null;
  tiles(document.getElementById("main-tiles"), [
    ["Total steps", fmtSteps(last.steps), "experience consumed"],
    ["Steps / sec", last.sps.toLocaleString(), "sim + learning"],
    ["Ep. reward", last.ep_rew.toFixed(2), "latest iteration"],
    ["Entropy", last.ent.toFixed(3), "policy randomness"],
    ["KL to prior", klLast ? klLast.kl_pri.toFixed(3) + spark(klRows.slice(-60).map(r=>r.kl_pri)) : "—",
     klLast ? "λ_p " + klLast.lambda_p.toFixed(3) : "no kl-prior iters yet"],
    ["Blowups contained", md.containment, "physics NaN events, engine-side"],
    ["Latest ck", md.ckpt.latest_steps ? fmtSteps(md.ckpt.latest_steps) : "—",
     md.ckpt.count ? `${md.ckpt.count} on disk · ${md.ckpt.total_gb} GB` : "none synced"],
    ["Next 100M in", md.eta_s ? fmtDur(md.eta_s) : "—", "at current throughput"],
  ]);
  const grid = grids("main-charts", MAIN_METRICS);
  MAIN_METRICS.forEach((m,i) => {
    const rs = m.key === "kl_pri" ? klRows : rows;
    if (rs.length > 1) chart(grid.children[i], rs, m, "steps", fmtSteps, md.restarts);
    else grid.children[i].innerHTML = `<h3>${m.title}</h3><div class="why">${m.why}</div>`;
  });
  const cols = ["steps","sps","ep_rew","pi_loss","v_loss","ent","clip","kl_pri","lambda_p"];
  document.getElementById("tbl").innerHTML =
    `<tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>` +
    rows.slice(-15).reverse().map(r =>
      `<tr>${cols.map(c=>`<td>${r[c]==null ? "—" :
        (typeof r[c]==="number" && !Number.isInteger(r[c]) ? r[c].toFixed(4) : r[c].toLocaleString())}</td>`).join("")}</tr>`).join("");
}

function renderBC(bc) {
  const meta = document.getElementById("bc-meta"), cur = bc.current;
  if (!cur) { meta.textContent = "no bc_train.log"; return; }
  meta.textContent = `run ${bc.runs} · ${cur.banner.train_shards.toLocaleString()} train / ${cur.banner.val_shards.toLocaleString()} val shards`;
  const p = cur.progress;
  if (p) {
    const pct = cur.frac != null ? Math.min(100, cur.frac * 100) : 0;
    document.getElementById("bc-bar").innerHTML =
      `<div class="pfill" style="width:${pct.toFixed(1)}%"></div>` +
      `<span>epoch ${p.epoch+1}/${cur.banner.epochs} · batch ${p.batch.toLocaleString()}/${p.total.toLocaleString()} · ${pct.toFixed(1)}%</span>`;
  }
  const ld = cur.last_done;
  tiles(document.getElementById("bc-tiles"), [
    ["Loss", p && p.loss != null ? p.loss.toFixed(4) : "—", "running train CE"],
    ["Samples / s", p ? p.samples_s.toLocaleString() : "—", "current throughput"],
    ["Val top-1", ld ? num(ld.top1, v=>(v*100).toFixed(1)+"%") : "—",
     ld ? `after epoch ${ld.epoch}` : "no epoch done yet"],
    ["Val top-3", ld ? num(ld.top3, v=>(v*100).toFixed(1)+"%") : "—",
     ld && ld.val_loss != null ? "val_loss " + ld.val_loss.toFixed(3) : ""],
  ]);
  const grid = grids("bc-charts", BC_METRICS);
  const kfmt = v => v >= 1000 ? (v/1000).toFixed(0)+"k" : String(Math.round(v));
  BC_METRICS.forEach((m,i) => {
    if (cur.loss.length > 1) chart(grid.children[i], cur.loss, m, "gb", kfmt);
    else grid.children[i].innerHTML = `<h3>${m.title}</h3><div class="why">no batches yet</div>`;
  });
  const h = bc.history, keys = ["train_loss","val_loss","top1","top3","recall_jump","recall_stall"];
  document.getElementById("bc-hist").innerHTML = h.length ?
    `<tr><th>run</th><th>epoch</th><th>train</th><th>val</th><th>top1</th><th>top3</th><th>r_jump</th><th>r_stall</th></tr>` +
    h.slice(-10).reverse().map(e =>
      `<tr${e.run===bc.runs ? ' class="curr"' : ''}><td>#${e.run}</td><td>${e.epoch}</td>` +
      keys.map(k=>`<td>${num(e[k], v=>v.toFixed(3))}</td>`).join("") + `</tr>`).join("") : "";
}

function renderLeague(rows) {
  const meta = document.getElementById("lg-meta");
  if (!rows.length) { meta.textContent = "no registry"; return; }
  const v1 = rows.filter(r=>r.schema_version===1).length;
  meta.textContent = `${rows.length} entries · ${v1} in v1 pool (highlighted) · by μ`;
  document.getElementById("lg-tbl").innerHTML =
    `<tr><th style="text-align:left">checkpoint</th><th>sv</th><th>μ</th><th>σ</th><th>games</th><th>src</th></tr>` +
    rows.slice(0, 24).map(e =>
      `<tr${e.schema_version===1 ? ' class="v1"' : ''}>` +
      `<td style="text-align:left" title="${e.ck}">${e.run} ${fmtSteps(e.steps)}</td>` +
      `<td>v${e.schema_version}</td><td>${num(e.mu, v=>v.toFixed(1))}</td>` +
      `<td>${num(e.sigma, v=>v.toFixed(1))}</td><td>${e.games}</td><td>${e.src}</td></tr>`).join("");
}

function renderSSL(s) {
  const meta = document.getElementById("ssl-meta");
  meta.textContent = s.last_age_s != null ? `last activity ${fmtAgo(s.last_age_s)}` : "no ssl_pull.log";
  if (s.batch_target) {
    const fill = Math.min(s.batch_target, (s.batch_fill||0) + (s.this_run||0));
    const pct = 100 * fill / s.batch_target;
    document.getElementById("ssl-bar").innerHTML =
      `<div class="pfill" style="width:${pct.toFixed(1)}%"></div>` +
      `<span>${s.batch||"batch"} · ~${fill.toLocaleString()}/${s.batch_target.toLocaleString()}</span>`;
  }
  tiles(document.getElementById("ssl-tiles"), [
    ["On disk", s.disk_count != null ? s.disk_count.toLocaleString() : "—", ".replay files (walked)"],
    ["Pulled", s.this_run != null ? `${s.this_run.toLocaleString()} / ${s.total != null ? s.total.toLocaleString() : "?"}` : "—",
     s.deduped != null ? `${s.deduped} deduped · ${s.failed} failed` : "this run / total"],
    ["Last hour", s.last_hour != null ? s.last_hour.toLocaleString() : "—",
     s.logged_rate_h != null ? `log says ${s.logged_rate_h}/h` : "from log timestamps"],
    ["Cursor at", s.cursor_oldest ? s.cursor_oldest.slice(0, 10) : "—", "oldest created, walking back"],
  ]);
}

function renderEvals(d) {
  document.getElementById("evalstatus").textContent = "· " + d.eval_status;
  const grid = grids("evalcharts", EVAL_METRICS);
  EVAL_METRICS.forEach((m,i) => {
    const rows = d.evals.filter(e => e[m.key] != null);
    if (!rows.length) {
      grid.children[i].innerHTML = `<h3>${m.title}</h3>
        <div class="d">no data yet — appears after the next eval</div>
        <div class="why">${m.why}</div>`;
    } else if (rows.length === 1) {
      const e0 = rows[0];
      grid.children[i].innerHTML = `<h3>${m.title}</h3>
        <div class="tile" style="border:none;padding:6px 0"><div class="v">${m.fmt(e0[m.key])}</div>
        <div class="d">single eval (${e0.ck}) — chart appears after the next one</div></div>
        <div class="why">${m.why}</div>`;
    } else chart(grid.children[i], rows, m, "ts", fmtDay);
  });
}

function renderSys(d) {
  const sys = d.sys;
  if (!sys.length) return;
  const s = sys[sys.length-1];
  tiles(document.getElementById("systiles"), [
    ["GPU", (s.gpu ?? "—") + "%", "BC training + evals"],
    ["VRAM", s.vram_used ? `${(s.vram_used/1024).toFixed(1)} / ${(s.vram_total/1024).toFixed(1)} GB` : "—", "model + batches"],
    ["GPU temp", (s.gpu_temp ?? "—") + "°C", "throttles near ~87°C"],
    ["CPU", s.cpu + "%", "loaders, pulls, syncs"],
    ["RAM", s.ram + "%", "of WSL allocation"],
    ["Host C: free", d.host_free_gb != null ? d.host_free_gb + " GB" : "—",
     "real free space (WSL df lies)"],
  ]);
  const grid = grids("syscharts", SYS_METRICS);
  SYS_METRICS.forEach((m,i) => {
    const have = sys.filter(r => r[m.key] !== undefined);
    if (have.length > 1) chart(grid.children[i], have, m, "ts", fmtClock);
    else grid.children[i].innerHTML = `<h3>${m.title}</h3><div class="why">collecting samples…</div>`;
  });
}

let LAST = null;
function renderAll(d) {
  renderMain(d.main);
  renderBC(d.bc);
  renderLeague(d.league);
  renderSSL(d.ssl);
  renderEvals(d);
  renderSys(d);
}
async function refresh() {
  let d;
  try { d = await (await fetch("/data")).json(); }
  catch { document.getElementById("status").textContent = "server unreachable"; return; }
  document.getElementById("status").textContent =
    `updated ${new Date().toLocaleTimeString()} · auto-refresh 5s · dashed lines = training restarts`;
  LAST = d;
  renderAll(d);
}
refresh(); setInterval(refresh, 5000);
addEventListener("resize", () => {
  ["main-charts","bc-charts","syscharts","evalcharts"].forEach(id =>
    document.getElementById(id).innerHTML = "");
  if (LAST) renderAll(LAST);  // synchronous — no blank flash while refetching
});
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(payload()).encode()
            ctype = "application/json"
        elif self.path == "/":
            body = PAGE.encode()
            ctype = "text/html; charset=utf-8"
        else:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    args = sys.argv[1:]
    port = int(args[0]) if args and args[0].isdigit() else 8420
    eval_every = 30
    if "--eval-every" in args:
        eval_every = int(args[args.index("--eval-every") + 1])
    SAMPLER.start()
    SLOW.start()
    if "--no-eval" not in args:
        EVALER = EvalRunner(eval_every)
        EVALER.start()
    print(f"dashboard: http://localhost:{port}  (main log: {MAIN_LOG}, eval every {eval_every}m)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
