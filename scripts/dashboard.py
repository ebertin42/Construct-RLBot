"""Training dashboard for the live runs. One dark page, A TAB PER RUNNING ARM,
plus a panel per workstream.

TABS (2026-07-31, three concurrent arms — see RUNS below). The per-run panels
(live run + curriculum) show whichever tab is selected; gate / SSL / system are
global and identical under every tab, so they are not duplicated. The selected
tab is stored in localStorage, so a page left open on run C stays on run C.
Each tab carries a live dot — green advancing, amber log stale >5 min, grey not
synced — because a training log outlives the process that writes it, and a dead
arm must look dead here rather than merely stop moving.

  main        the live run             checkpoints_scratch/train_remote.log (synced)
  curriculum  per-bot difficulty       the same log's auto-curriculum lines --
              each external opponent's rung + its measured win rate against the
              target band. THIS is the live skill signal: the bots are fixed
              external references, so beating a harder rung is real progress.
  gate        vs the frozen champion   logs/matchwin_history.jsonl (appended by
              scripts/matchwin_gate.py) -- the absolute ruler, run on demand
  ssl         SSL replay pull          logs/ssl_pull.log + data/replays/ssl/**/*.replay
  system      CPU / RAM / GPU samplers + Windows-host C: free (powershell, cached)

Removed 2026-07-25 (dead weight, see the git history for the panels): BC
pre-training (a human-replay BC net loses 639-0 to the champion, so BC-as-init is
abandoned), the v0 league ladder (the from-scratch run does not use it), and the
self-play behaviour panel plus its EvalRunner thread -- self-play goals/min is
NOT a skill metric, it once hid an 800M-step regression, and the thread was
burning laptop CPU every 30 min to keep computing it. The old h2h panel went too:
its references are from the retired lineage; the curriculum + gate panels are the
current rulers.

Parsing lives in pure functions at the top of this file, tested in
tests/python/test_dashboard_parsers.py. Slow host queries (powershell C: free,
the replay walk) run in background sampler threads and are cached — the request
path never blocks on them. Log parses are cached on (mtime, size).

Usage: python scripts/dashboard.py [port]     (default 8420)
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
# Env-overridable so the dashboard can follow whichever run is live without an
# edit. Defaults now point at the live from-scratch run, so no env vars are needed
# for the common case; set CONSTRUCT_DASH_MAIN_LOG / CONSTRUCT_DASH_CKPT_DIR to
# follow a different lineage.
import os as _os
# v9 is the live run since 2026-07-26 (fresh net, 104-row air action table, equal
# 1s/2s/3s). The v8 from-scratch lineage in checkpoints_scratch was retired at
# 1.589B; point the env vars at it to inspect that history.
MAIN_LOG = Path(_os.environ.get(
    "CONSTRUCT_DASH_MAIN_LOG", REPO / "checkpoints_v9" / "train_remote.log"))
CKPT_DIR = Path(_os.environ.get(
    "CONSTRUCT_DASH_CKPT_DIR", REPO / "checkpoints_v9"))

# --- the live arms, one dashboard tab each -------------------------------------
# THREE runs since 2026-07-31. Each has its own log and checkpoint dir, and the
# per-run panels (live run + curriculum) render whichever tab is selected; the
# gate / SSL / system panels are global and shown under every tab.
#
# `role` is the sentence that says what a tab MEANS, because the whole point of
# three arms is that they are not interchangeable: reading B's curve as if it
# were A's is exactly the confusion that made B-vs-A uninterpretable for a week.
#
# MAIN_LOG / CKPT_DIR above still drive the FIRST entry, so the env overrides keep
# working for inspecting a retired lineage.
RUNS = [
    {"id": "mt", "label": "multi-teacher \u00b7 LIVE",
     "log": REPO / "checkpoints_mt" / "train_remote.log",
     "ckpt": REPO / "checkpoints_mt",
     "role": "Distillation from a MIXTURE: nexto 0.75 / immortal 0.25, one teacher sampled "
             "per iteration. Forked from distill at 4,070,379,520 on 2026-08-09; every "
             "other setting is identical, so the mixture is the only variable. Read fd_ce "
             "PER TEACHER \u2014 the fd_t field names who labelled that iteration, and the two "
             "series are on completely different scales (nexto ~1.7 against a 3.7 marginal, "
             "immortal ~4.6 against 2.6). fd_lab is ALSO per teacher: 1.000 for nexto, "
             "~0.80-0.88 for immortal, whose 126-row action table only partly overlaps "
             "ours. A rising fd_ce is EXPECTED here (a hard-label mixture raises the "
             "irreducible floor) and is not a fault. The verdict is goals_against vs "
             "distill over a matched window; the falsifier is in launch_multiteacher.sh."},
    {"id": "ppo", "label": "PPO-from-distilled \u00b7 retired",
     "log": REPO / "checkpoints_ppo_distilled" / "train_remote.log",
     "ckpt": REPO / "checkpoints_ppo_distilled",
     "role": "PPO on top of the nexto-distilled policy with a 0.1 distillation anchor. It "
             "won the project's first wins at p1 (30W/12D/278L vs element, against a "
             "history of 0W/0D/320L) but RETIRED 2026-08-05: against pure distillation over "
             "a matched window it is measurably WORSE, t=+3.63 on goals_against. The "
             "goal-only variant then failed the same way at t=+4.90, which is what closed "
             "the reward-resolution explanation."},
    {"id": "distill", "label": "distill \u00b7 LIVE (control)",
     "log": REPO / "checkpoints_distill" / "train_remote.log",
     "ckpt": REPO / "checkpoints_distill",
     "role": "Pure distillation from nexto (policy_coef=0, so the CE is the ONLY gradient). "
             "RESUMED 2026-08-04 from ck_003300874240 after being stopped 28M steps early: "
             "fd_ce had plateaued at 43% but goals_against was still falling -0.061/Mstep "
             "(r=-0.907). This arm tests whether that keeps paying. fd_pct is the clone "
             "completeness; it is NOT the stopping signal \u2014 goals_against is, and that "
             "costs a bench to read."},
    {"id": "a", "label": "A \u00b7 control (retired)", "log": MAIN_LOG, "ckpt": CKPT_DIR,
     "role": "v9 control lineage, reward_v9_aerial.toml. RETIRED 2026-08-04 at "
             "ck_003674219520 (1919 checkpoints retained). Beaten on every channel by both "
             "distillation arms; it was holding ~6 cores the live experiments could use. "
             "This is the BASELINE every distillation number is quoted against."},
    {"id": "d", "label": "D \u00b7 aux heads (retired)",
     "log": REPO / "checkpoints_v12" / "train_remote.log",
     "ckpt": REPO / "checkpoints_v12",
     "role": "Forked from A at ck_002780933120; differed only by the aux losses. RETIRED "
             "2026-08-03 at 342M divergence: mechanism flat (aux_ev 0.497-0.526 across "
             "260M steps), effect worth ~32M steps, settling it would have cost ~387M."},
]
SSL_LOG = REPO / "logs" / "ssl_pull.log"
SSL_DIR = REPO / "data" / "replays" / "ssl"
GATE_HISTORY = REPO / "logs" / "matchwin_history.jsonl"        # appended by matchwin_gate.py
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
    # FOURTH ERA (2026-08-01, run D): the aux heads append raw, unweighted
    # aux_rec/aux_rew. Optional like the two above, so A's line -- which has no aux --
    # still parses. These are the numbers that say whether the heads are LIVE: both must
    # be nonzero and falling, or the run is training dead scaffolding, which is exactly
    # the state the feature shipped in for weeks.
    r"(?: aux_rec ([-\d.eE+]+) aux_rew ([-\d.eE+]+))?"
    # FIFTH ERA (2026-08-04, the distillation arms): fd_ce / fd_marg / fd_lab. NOTE these
    # are printed BEFORE the aux block in train.py, but the optional groups above are
    # order-fixed and `.search()` does not anchor to end-of-line -- so without this the line
    # still MATCHED and the three fields were silently dropped. A silent omission, which is
    # the same failure the aux era exists to prevent, one layer down.
    r"(?:.*? fd_ce ([-\d.eE+]+) fd_marg ([-\d.eE+]+) fd_lab ([-\d.eE+]+))?"
)
RESUME = re.compile(r"resumed at ([\d,]+) steps")
CONTAINMENT = "physics blowup contained"

# --- per-bot auto-curriculum (the live difficulty ladder) ------------------
# THREE ERAS OF WIRE FORMAT, all of which must parse, because the log is
# appended across restarts and the live remote run is still emitting the oldest
# one. These regexes are the ONLY consumer of train.py's print()s -- there is no
# schema between them, so a format drift shows up as an empty panel at best and
# a silently STALE panel at worst. That is not hypothetical: the v9 change from
# "p4->3" to "p4->p3" still MATCHED the old AC_SEG but returned an empty arrow
# group, so `int(p_to) if p_to else int(p_from)` reported the pre-move period
# forever, with no error anywhere. Hence the `test_dashboard_parsers` cases
# below covering all three eras.
#
# Banner  pre-v9:  auto-curriculum ON: hold win rate in [0.35,0.65], period in [1,12], ...
#         v9:      ... , period [1,12], ...          (non-ladder configs)
#         v9:      ... , ladder [1, 2, 3, 4, 6, 12], ...
# Roster  pre-v9:  foreign: ['element', 'immortal'] periods=[4, 3] on 77/192 arenas (frac=0.4)
#         v9:      foreign: [('element', 1), ('necto', 3)] periods=[4, 3] cars=[1, 2]
#                           on 48/144 arenas (frac=0.3333 PER BLOCK)
# Decision pre-v9: auto-curriculum: element wr0.69/ema0.69 p4->3 | immortal ... p3->2
#         v9:      auto-curriculum: element wr0.69/ema0.69 p4->p3 | ...
#         v9 ladder: auto-curriculum: nexto wr0.50/ema0.50 1c/p12->1c/p20 | ...
#           a slot inside its post-change dwell reads "... p3 dwell1/2" instead,
#           and a slot not measured this cycle (the staggered eval measures ONE
#           team size per cycle) reads "... n/a 1c/p12".
AC_BANNER = re.compile(
    r"auto-curriculum ON: hold win rate in \[([\d.]+),([\d.]+)\], "
    r"(?:period(?: in)? \[(\d+),(\d+)\]|ladder \[([\d,\s]+)\])")
AC_ROSTER = re.compile(
    r"foreign: \[([^\]]*)\] periods=\[([^\]]*)\](?: cars=\[([^\]]*)\])? "
    r"on (\d+)/(\d+) arenas")
AC_DECISION = re.compile(r"auto-curriculum: (\w+ (?:wr[\d.]+|n/a).*)")
# rung is `p12`, `1c/p12` (ladder), and the arrow target is `3` (pre-v9),
# `p3` or `1c/p20`. Unmeasured slots print `n/a` where wr/ema would be; they are
# matched deliberately so the segments stay POSITIONALLY aligned with the roster,
# which is how a repeated bot name (necto at 1s, 2s and 3s) is told apart.
AC_SEG = re.compile(
    r"(\w+) (?:wr([\d.]+)/ema([\d.]+)|n/a) "
    r"(?:(\d+)c/)?p(\d+)"
    r"(?:->(?:(\d+)c/)?p?(\d+))?"
    r"(?: dwell(\d+)/(\d+))?")

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
    if m.group(13) is not None:
        row["aux_rec"], row["aux_rew"] = float(m.group(13)), float(m.group(14))
    if m.group(15) is not None:
        # fd_ce alone is uninterpretable -- a nats figure means nothing without the
        # state-independent baseline it has to beat -- so the gap is derived here rather
        # than left for whoever reads the tile. fd_lab must sit at 1.000; a drift downward
        # means the teacher's controls stopped matching the action table, which would
        # silently shrink the training set instead of erroring.
        row["fd_ce"], row["fd_marg"] = float(m.group(15)), float(m.group(16))
        row["fd_lab"] = float(m.group(17))
        row["fd_pct"] = (1.0 - row["fd_ce"] / row["fd_marg"]) * 100.0 if row["fd_marg"] else 0.0
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


def parse_curriculum(text):
    """Train log -> the live per-bot difficulty ladder.

    Only the CURRENT run is reported: everything before the last
    'auto-curriculum ON:' banner belongs to an earlier process (the log is
    appended across restarts) and mixing eras would draw phantom period jumps.

    Returns {"band": [lo, hi], "period_min", "period_max", "roster", "arenas",
    "bots": [{name, wr, ema, period, cars, rung, dwell, dwell_of}],
    "history": [...]}, where each history row is one eval:
    {"i": n, "<label>": period, "<label>_wr": wr}.

    `<label>` is the bot name, EXCEPT when v9's roster runs the same bot at
    several team sizes (necto at 1s/2s/3s are three independent slots with three
    independent rungs); then it is "necto·2s". Without that, three slots would
    collapse onto one dict key and the panel would show whichever came last.

    Under the staggered eval only ONE team size is measured per cycle, so most
    segments read `n/a`. Their last real wr/ema is CARRIED FORWARD rather than
    shown as 0 -- a slot measured two cycles ago is stale, not zero.
    """
    lines = text.splitlines()
    start = 0
    band = pmin = pmax = None
    roster, modes, arenas = [], [], None
    for i, line in enumerate(lines):
        m = AC_BANNER.search(line)
        if m:
            start = i
            band = [_f(m.group(1)), _f(m.group(2))]
            if m.group(5):                       # ladder [1, 2, 3, ...]
                rungs = [int(x) for x in m.group(5).replace(" ", "").split(",") if x]
                pmin, pmax = min(rungs), max(rungs)
            else:
                pmin, pmax = int(m.group(3)), int(m.group(4))
        m = AC_ROSTER.search(line)
        if m:
            # 'element' (pre-v9) and ('element', 1) (v9) both yield the quoted
            # name; the tuple form additionally yields the team size.
            roster = re.findall(r"'([^']+)'", m.group(1))
            modes = [int(x) for x in re.findall(r"'[^']+'\s*,\s*(\d+)", m.group(1))]
            arenas = f"{m.group(4)}/{m.group(5)}"
    # one label per slot: bare name unless that name is used by several slots
    dupes = {n for n in roster if roster.count(n) > 1}
    labels = [f"{n}·{modes[i]}s" if n in dupes and i < len(modes) else n
              for i, n in enumerate(roster)]
    out = {"band": band, "period_min": pmin, "period_max": pmax,
           "roster": roster, "arenas": arenas, "bots": [], "history": []}
    if band is None:
        return out
    last = {}                                    # label -> last real {wr, ema}
    for line in lines[start:]:
        m = AC_DECISION.search(line)
        if not m:
            continue
        segs = AC_SEG.findall(m.group(1))
        snap, row = [], {"i": len(out["history"]) + 1}
        for i, (name, wr, ema, c_from, p_from, c_to, p_to, dwell, dwell_of) in enumerate(segs):
            # a moved slot reports <from>-><to>; a held/dwelling one just <rung>
            period = int(p_to) if p_to else int(p_from)
            cars = int(c_to) if c_to else (int(c_from) if c_from else None)
            # positional against the roster, which is why `n/a` segments are
            # matched too -- they are the majority under the staggered eval
            label = labels[i] if len(segs) == len(labels) else name
            prev = last.get(label, {})
            w = _f(wr) if wr else prev.get("wr")
            e = _f(ema) if ema else prev.get("ema")
            if wr:
                last[label] = {"wr": w, "ema": e}
            snap.append({"name": label, "wr": w, "ema": e, "period": period,
                         "cars": cars, "rung": f"{cars}c/p{period}" if cars else f"p{period}",
                         "measured": bool(wr),
                         "dwell": int(dwell) if dwell else None,
                         "dwell_of": int(dwell_of) if dwell_of else None})
            row[label] = period
            row[label + "_wr"] = _f(wr) if wr else None
        if snap:
            out["bots"] = snap                  # last decision wins = current state
            out["history"].append(row)
    return out


def parse_gate_history(text):
    """logs/matchwin_history.jsonl -> gate rows, newest last. Written by
    scripts/matchwin_gate.py: the absolute ruler (candidate vs frozen champion,
    both side orders, full 300s matches).

    Rows predate the `mode` key (1v1 was the only option), so it is normalised to
    1 on READ rather than backfilled into the file -- rewriting published records
    is worse than defaulting them. The renderer LABELS the mode instead of
    filtering by it: filtering hides runs, whereas a label makes it impossible to
    misread a 2v2 share as the 1v1 ruler."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if "win_share" in d:
            d.setdefault("mode", 1)
            rows.append(d)
    rows.sort(key=lambda r: r.get("ts", 0))
    return rows


def estimate_train_times(rows, anchor_ts, restarts=()):
    """Iter lines carry no timestamps. Estimate per-row wall time by anchoring
    the LAST row at the log's mtime and walking backwards: each gap costs
    steps_delta / sps seconds (sps of the later row — its iteration rate).
    Crossing a 'resumed at S' boundary makes every earlier estimate 'rough':
    the downtime at the seam is unknowable, so everything before it is shifted
    by an unknown amount. Returns [(ts, rough), ...] aligned to rows."""
    n = len(rows)
    if not n:
        return []
    ts, rough = [0.0] * n, [False] * n
    ts[-1] = anchor_ts
    rs = sorted(restarts)
    beyond_boundary = False
    for i in range(n - 1, 0, -1):
        prev, cur = rows[i - 1], rows[i]
        delta = max(0, cur["steps"] - prev["steps"])
        sps = cur.get("sps") or prev.get("sps") or 0
        ts[i - 1] = ts[i] - (delta / sps if sps > 0 else 0.0)
        if any(prev["steps"] <= s < cur["steps"] for s in rs):
            beyond_boundary = True
        rough[i - 1] = beyond_boundary
    return list(zip(ts, rough))


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


def checkpoint_info(ckpt_dir=None):
    cks = sorted((ckpt_dir or CKPT_DIR).glob("ck_*.pt"))
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


SAMPLER = SysSampler()
SLOW = SlowSampler()


# ---------------------------------------------------------------------------
# payload
# ---------------------------------------------------------------------------

def _make_parse_main(log_path):
    """parse_train_log + estimated wall times (anchored at the log's mtime —
    consistent with the cache key, which includes mtime). Runs once per file
    change, so the O(rows) walk stays off the steady-state request path.

    Closes over the log path so each run anchors on ITS OWN mtime; using a single
    global here would date every arm's wall-clock estimates to whichever log was
    written last."""
    def _parse(text):
        out = parse_train_log(text)
        try:
            anchor = log_path.stat().st_mtime
        except OSError:
            anchor = time.time()
        for r, (t, rough) in zip(out["rows"],
                                 estimate_train_times(out["rows"], anchor, out["restarts"])):
            r["ts_est"] = round(t, 1)
            if rough:
                r["ts_rough"] = True
        return out
    return _parse


def _main_payload(now, log_path=None, ckpt_dir=None):
    log_path = log_path or MAIN_LOG
    train = cached_parse(log_path, _make_parse_main(log_path), tag="main")
    rows = downsample(train["rows"], MAX_POINTS, tail=150)
    last = rows[-1] if rows else None
    eta = None
    if last and last["sps"] > 0:
        nxt = (last["steps"] // 100_000_000 + 1) * 100_000_000
        eta = int((nxt - last["steps"]) / last["sps"])
    try:
        age = int(now - log_path.stat().st_mtime)
    except OSError:
        age = None
    return {"rows": rows, "restarts": train["restarts"], "containment": train["containment"],
            "ckpt": checkpoint_info(ckpt_dir), "eta_s": eta, "log_age_s": age}


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
    runs = {}
    for r in RUNS:
        log, ckpt = Path(r["log"]), Path(r["ckpt"])
        # A run whose log has not synced yet (a freshly launched arm) must render as
        # an EMPTY tab, not vanish and not crash the whole page -- cached_parse and
        # checkpoint_info both already degrade to {} / [] on a missing path.
        runs[r["id"]] = {
            "label": r["label"], "role": r["role"], "log": str(log.relative_to(REPO)),
            "main": _main_payload(now, log, ckpt),
            "curriculum": cached_parse(log, parse_curriculum, tag="curr"),
        }
    return {
        "runs": runs,
        "run_order": [r["id"] for r in RUNS],
        # global panels -- identical under every tab, so they are NOT duplicated per run
        "gate": cached_parse(GATE_HISTORY, parse_gate_history, tag="gate"),
        "ssl": _ssl_payload(now),
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
        --series:#3987e5; --accent:rgba(57,135,229,.16);
        /* the amber accent marks whichever panel is the REAL skill ruler; that is
           now the curriculum (fixed external opponents) and the champion gate */
        --rule:#e0a730; --rule-series:#e0a730; --rule-accent:rgba(224,167,48,.18) }
* { box-sizing:border-box; margin:0 }
body { background:var(--surface); color:var(--ink);
       font:14px/1.45 system-ui,-apple-system,sans-serif; padding:18px; }
h1 { font-size:17px; font-weight:650 }
.sub { color:var(--ink2); font-size:12.5px; margin:2px 0 14px }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:10px;
         padding:14px 16px 12px; margin-bottom:14px }
.panel.main { border-color:#2c5e9e }
.panel.rule { border-color:var(--rule); border-width:2px; box-shadow:0 0 0 1px rgba(224,167,48,.25) }
.panel.rule h2 { color:var(--rule) }
svg.chart.rule .line { stroke:var(--rule-series) }
svg.chart.rule .mark, svg.chart.rule .dot { fill:var(--rule-series) }
/* --- curriculum: one row per opponent, win rate against the target band ---
   The gauge IS the control law: shaded region = the band the controller holds
   each bot in, thick mark = its smoothed win rate (what it acts on), thin mark =
   the latest single measurement (noisy, hence the smoothing). */
.bot { display:grid; grid-template-columns:96px 92px 1fr 132px; gap:12px;
       align-items:center; padding:9px 2px; border-top:1px solid var(--border) }
.bot:first-child { border-top:0 }
.bot .name { font-weight:600 }
.bot .rung { font-size:12px; color:var(--ink2); font-variant-numeric:tabular-nums }
.bot .rung b { color:var(--ink); font-weight:650 }
.gauge { position:relative; height:24px; background:var(--card); border-radius:5px;
         border:1px solid var(--border) }
.gauge .band { position:absolute; top:0; bottom:0; background:var(--accent);
               border-left:1px solid rgba(57,135,229,.55);
               border-right:1px solid rgba(57,135,229,.55) }
.gauge .ema { position:absolute; top:3px; bottom:3px; width:3px; border-radius:2px;
              background:var(--ink) }
.gauge .raw { position:absolute; top:8px; bottom:8px; width:2px; background:var(--muted) }
.bot .verdict { font-size:11.5px; text-align:right; color:var(--ink2) }
.bot.advancing .verdict { color:var(--series) }
.bot.easing .verdict { color:var(--rule) }
.bot .verdict .sub2 { display:block; color:var(--muted); font-size:10.5px }
.parity { stroke:var(--muted); stroke-width:1; stroke-dasharray:4 3 }
/* --- run tabs: one per live arm -------------------------------------------
   The arms are NOT interchangeable (A control / B planar flag / C null control),
   so each tab states its role and carries a live dot: green = advancing, amber =
   log stale, grey = nothing synced yet. A dead arm must look dead here, because
   the log file outlives the process that writes it. */
.tabs { display:flex; gap:8px; margin:0 0 14px; flex-wrap:wrap }
.tab { background:var(--panel); border:1px solid var(--border); border-radius:9px;
       padding:8px 13px; cursor:pointer; color:var(--ink2); text-align:left;
       font:inherit; line-height:1.25; min-width:150px }
.tab:hover { border-color:var(--muted) }
.tab.on { background:var(--card); border-color:var(--series); color:var(--ink) }
.tab .t { font-weight:650; font-size:13px; display:flex; align-items:center; gap:6px }
.tab .s { font-size:11px; color:var(--muted); font-variant-numeric:tabular-nums }
.tab.on .s { color:var(--ink2) }
.dot { width:7px; height:7px; border-radius:50%; background:var(--muted); flex:none }
.dot.live { background:#4bbf73 }
.dot.stale { background:var(--rule) }
.runrole { color:var(--muted); font-size:11.5px; margin:-6px 0 12px }
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

<div class="tabs" id="tabs"></div>
<div class="runrole" id="runrole"></div>

<section class="panel main">
  <h2>Live run · from-scratch vs external bots<span class="meta" id="main-meta"></span></h2>
  <div class="tiles" id="main-tiles"></div>
  <div class="grid" id="main-charts"></div>
  <details><summary>Recent iterations</summary>
    <div class="wrap"><table id="tbl"></table></div></details>
</section>

<section class="panel rule">
  <h2>Curriculum · external opponents<span class="meta" id="curr-meta"></span></h2>
  <div id="curr-rows"></div>
  <div class="grid" id="curr-charts"></div>
</section>

<div class="cols">
  <section class="panel rule">
    <h2>Gate vs frozen champion<span class="meta" id="gate-meta"></span></h2>
    <div class="tiles" id="gate-tiles"></div>
    <div class="wrap"><table id="gate-tbl"></table></div>
  </section>
  <section class="panel">
    <h2>SSL replay pull<span class="meta" id="ssl-meta"></span></h2>
    <div class="pbar" id="ssl-bar"><span>—</span></div>
    <div class="tiles" id="ssl-tiles"></div>
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
const SYS_METRICS = [
  {key:"gpu", title:"GPU utilization (%)", fmt:v=>Math.round(v)+"%",
   why:"BC training + any local eval. Bursty is normal."},
  {key:"gpu_temp", title:"GPU temperature (°C)", fmt:v=>Math.round(v)+"°C",
   why:"Laptop GPUs throttle around ~87°C."},
  {key:"cpu", title:"CPU utilization (%)", fmt:v=>Math.round(v)+"%",
   why:"Dataloaders, SSL pull, sync loops, RocketSim evals."},
];
const tip = document.getElementById("tip");
const fmtSteps = v => v >= 1e9 ? (v/1e9).toFixed(2)+"B" : v >= 1e6 ? (v/1e6).toFixed(0)+"M" : v.toLocaleString();
const fmtDur = s => { const h = Math.floor(s/3600), m = Math.floor(s%3600/60);
                      return h ? `${h}h ${m}m` : `${m}m`; };
const fmtAgo = s => s == null ? "—" : s < 90 ? "just now" :
                    s < 5400 ? Math.round(s/60)+"m ago" : (s/3600).toFixed(1)+"h ago";
const fmtClock = v => new Date(v*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit",hour12:false});
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

function chart(el, rows, m, xKey, xFmt, restarts, extraClass) {
  const W = el.clientWidth || 320, H = 150, L = 48, R = 8, T = 8, B = 20;
  const xs = rows.map(r => r[xKey]), ys = rows.map(r => r[m.key]);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (m.cap) { const s = [...ys].sort((a,b)=>a-b); hi = s[Math.floor((s.length-1)*m.cap)]; }
  if (m.yMin != null) lo = Math.min(lo, m.yMin);
  if (m.yMax != null) hi = Math.max(hi, m.yMax);
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
  if (m.refLine != null && m.refLine >= lo && m.refLine <= hi) {
    const y = Y(m.refLine);
    g += `<line class="parity" x1="${L}" x2="${W-R}" y1="${y}" y2="${y}"><title>${m.refLineLabel||""}</title></line>`;
  }
  const path = rows.map((r,i)=>`${i?"L":"M"}${X(r[xKey]).toFixed(1)},${Y(r[m.key]).toFixed(1)}`).join("");
  const marks = m.marks ? rows.map(r=>`<circle class="mark" r="4" cx="${X(r[xKey]).toFixed(1)}" cy="${Y(r[m.key]).toFixed(1)}"/>`).join("") : "";
  el.innerHTML = `<h3>${m.title}</h3>
    <svg class="chart${extraClass ? " " + extraClass : ""}" viewBox="0 0 ${W} ${H}">${g}<path class="line" d="${path}"/>${marks}
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
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 210) + "px";
    tip.style.top = (e.clientY + 14) + "px";
    // datetime: real ts when the row has one; otherwise the estimate
    // reconstructed from the log mtime ('~', '~~' once past a resume boundary
    // where the downtime is unknowable)
    const dt = r.ts_est != null ? (r.ts_rough ? "~~" : "~") + fmtDay(r.ts_est)
             : r.ts != null ? fmtDay(r.ts) : null;
    const xtxt = xKey === "ts" ? null : "at " + xFmt(r[xKey]);
    tip.innerHTML = `<b>${m.fmt(r[m.key])}</b> <span style="color:var(--ink2)">${m.title}</span><br>` +
      `<span style="color:var(--ink2)">${[xtxt, dt].filter(Boolean).join(" · ")}</span>`;
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

// Aux-head liveness. The heads sat DEAD for weeks -- declared, never wired -- so
// "the run was launched with --aux" is not evidence. Both raw losses must be nonzero
// AND FALLING. Compares the last 20 aux iters against the 20 before them.
function auxTile(rows) {
  const a = rows.filter(r => r.aux_rec != null);
  if (!a.length) return {v: "off", why: "no aux terms on this run's iter line"};
  const last = a[a.length-1];
  const mean = xs => xs.reduce((p,c)=>p+c,0) / Math.max(1, xs.length);
  const recent = a.slice(-20).map(r=>r.aux_rec);
  const prior  = a.slice(-40,-20).map(r=>r.aux_rec);
  let trend = "";
  if (prior.length >= 5) {
    const d = mean(recent) - mean(prior);
    trend = d < 0 ? " falling" : (d > 0 ? " RISING" : " flat");
  }
  const bad = !(last.aux_rec > 0) || !(last.aux_rew > 0) || last.aux_rec > 1e6;
  return {
    v: (bad ? "CHECK " : "live") + " rec " + last.aux_rec.toExponential(2) +
       " · rew " + last.aux_rew.toFixed(2),
    why: bad ? "a zero loss means inert heads; a huge one means the head is blowing up"
             : "raw, unweighted" + trend + spark(recent),
  };
}

function distillTile(rows) {
  const a = rows.filter(r => r.fd_ce != null);
  if (!a.length) return {v: "off", why: "no distillation term on this run's iter line"};
  const last = a[a.length-1];
  const mean = xs => xs.reduce((p,c)=>p+c,0) / Math.max(1, xs.length);
  const recent = a.slice(-20).map(r=>r.fd_pct);
  const prior  = a.slice(-60,-20).map(r=>r.fd_pct);
  let trend = "";
  if (prior.length >= 10) {
    const d = mean(recent) - mean(prior);
    trend = d > 0.2 ? " climbing" : (d < -0.2 ? " FALLING" : " flat");
  }
  // fd_lab is the tripwire: it must sit at 1.000. Below that the teacher's controls have
  // stopped matching the action table and those frames are dropped -- which shrinks the
  // training set silently rather than erroring.
  const labBad = last.fd_lab < 0.999;
  return {
    v: (labBad ? "CHECK " : "") + last.fd_pct.toFixed(1) + "% below marginal",
    why: labBad
      ? "fd_lab " + last.fd_lab.toFixed(3) + " — labels are being DROPPED, training set is shrinking"
      : "ce " + last.fd_ce.toFixed(3) + " vs marg " + last.fd_marg.toFixed(3) +
        trend + " — clone completeness, NOT the stopping signal" + spark(recent),
  };
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
    ["Aux heads", auxTile(rows).v, auxTile(rows).why],
    ["Distillation", distillTile(rows).v, distillTile(rows).why],
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
  const cols = rows.some(r => r.aux_rec != null)
    ? ["steps","sps","ep_rew","pi_loss","v_loss","ent","clip","aux_rec","aux_rew"]
    : ["steps","sps","ep_rew","pi_loss","v_loss","ent","clip","kl_pri","lambda_p"];
  document.getElementById("tbl").innerHTML =
    `<tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>` +
    rows.slice(-15).reverse().map(r =>
      `<tr>${cols.map(c=>`<td>${r[c]==null ? "—" :
        (typeof r[c]==="number" && !Number.isInteger(r[c]) ? r[c].toFixed(4) : r[c].toLocaleString())}</td>`).join("")}</tr>`).join("");
}

function renderCurriculum(c) {
  const meta = document.getElementById("curr-meta");
  const rows = document.getElementById("curr-rows");
  if (!c || !c.band || !c.bots.length) {
    meta.textContent = "no auto-curriculum lines in the live log yet";
    rows.innerHTML = ""; return;
  }
  const [lo, hi] = c.band;
  meta.textContent = `${c.bots.length} bots on ${c.arenas||"?"} arenas · `
    + `hold ${lo}–${hi} · rungs ${c.period_max}(easy)→${c.period_min}(full strength) · `
    + `${c.history.length} evals this run`;
  rows.innerHTML = c.bots.map(b => {
    // above the band -> we beat it comfortably, so it advances a rung next;
    // below -> it backs off. In band = a fair fight, which is the goal.
    // ema is null until a slot has been measured at least once -- under the
    // staggered eval only one team size moves per cycle, so that is normal and
    // must NOT render as 0% ("losing every match").
    const seen = b.ema != null;
    const state = !seen ? "" : b.ema > hi ? "advancing" : b.ema < lo ? "easing" : "";
    const verdict = !seen ? "not measured yet"
      : b.dwell ? `settling ${b.dwell}/${b.dwell_of}`
      : b.ema > hi ? "→ harder next" : b.ema < lo ? "→ easier next" : "fair fight";
    const pct = v => (100 * Math.max(0, Math.min(1, v || 0))).toFixed(1) + "%";
    return `<div class="bot ${state}">
      <div class="name">${b.name}</div>
      <div class="rung">rung <b>${b.rung || ("p" + b.period)}</b></div>
      <div class="gauge" title="win rate 0 → 1; shaded = target band ${lo}–${hi}">
        <div class="band" style="left:${pct(lo)};width:${pct(hi-lo)}"></div>
        ${seen ? `<div class="raw" style="left:${pct(b.wr)}"></div>
        <div class="ema" style="left:${pct(b.ema)}"></div>` : ""}
      </div>
      <div class="verdict">${seen ? (b.ema*100).toFixed(0) + "% smoothed" : "—"}
        <span class="sub2">${verdict}${b.measured === false && seen ? " · stale" : ""}</span></div>
    </div>`;
  }).join("");

  // one chart per bot: the rung it is being held at over this run's evals.
  // Falling = the opponent got harder = the policy improved.
  const metrics = c.bots.map(b => ({
    key: b.name, title: `${b.name} · rung over time`, fmt: v => "p" + v,
    why: "Lower is harder. The controller drops a rung once this bot's smoothed "
       + "win rate clears the top of the band, so a falling line is real progress "
       + "against a fixed external opponent.",
  }));
  const grid = document.getElementById("curr-charts");
  while (grid.children.length > metrics.length) grid.lastChild.remove();
  while (grid.children.length < metrics.length) grid.appendChild(mkcard());
  metrics.forEach((m, i) => {
    const rs = c.history.filter(r => r[m.key] != null);
    if (rs.length > 1) chart(grid.children[i], rs, m, "i", v => "eval " + v, [], "rule");
    else grid.children[i].innerHTML = `<h3>${m.title}</h3><div class="why">${m.why}</div>`;
  });
}

function renderGate(rows) {
  const meta = document.getElementById("gate-meta");
  const tbl = document.getElementById("gate-tbl");
  if (!rows || !rows.length) {
    meta.textContent = "no gates yet — run scripts/matchwin_gate.py CK";
    tiles(document.getElementById("gate-tiles"), []);
    tbl.innerHTML = ""; return;
  }
  const last = rows[rows.length - 1];
  meta.textContent = `${rows.length} gate${rows.length>1?"s":""} · latest ${fmtDay(last.ts)}`;
  // Team size is LABELLED, never filtered. A 2v2 win share is a different
  // quantity from the 1v1 ruler (the net has never had a teammate; goals/match
  // falls 7.1 -> 4.5), so the tile and every row have to say which one they are.
  const lm = last.mode || 1;
  tiles(document.getElementById("gate-tiles"), [
    ["Win share", (last.win_share*100).toFixed(1) + "%",
     `${last.wins}W/${last.draws}D/${last.losses}L over ${last.n} matches · ${lm}v${lm}`],
    ["Verdict", last.passed ? "PASS" : "FAIL",
     `threshold ${last.threshold != null ? last.threshold : "0.55"}`],
    ["Candidate", (last.candidate||"?").split("/").pop(), "vs the frozen champion"],
  ]);
  tbl.innerHTML = "<tr><th>when</th><th>candidate</th><th>mode</th><th>W/D/L</th>"
    + "<th>win share</th><th>verdict</th></tr>"
    + rows.slice().reverse().map(r => `<tr><td>${fmtDay(r.ts)}</td>`
      + `<td>${(r.candidate||"?").split("/").pop()}</td>`
      + `<td>${r.mode||1}v${r.mode||1}</td>`
      + `<td>${r.wins}/${r.draws}/${r.losses}</td>`
      + `<td>${(r.win_share*100).toFixed(1)}%</td>`
      + `<td>${r.passed ? "PASS" : "FAIL"}</td></tr>`).join("");
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
// Selected arm. Survives refreshes and reloads, so a tab you left open on run C
// does not silently snap back to A every 5 seconds.
let RUN = localStorage.getItem("construct.run") || "a";

function renderTabs(d) {
  const bar = document.getElementById("tabs");
  bar.innerHTML = "";
  d.run_order.forEach(id => {
    const r = d.runs[id], rows = r.main.rows, last = rows[rows.length-1];
    const age = r.main.log_age_s;
    // grey = never synced; amber = log has not moved in 5 min (an eval iteration
    // can stretch a normal gap past a minute, so 60s would cry wolf); green = live.
    const state = last == null ? "" : (age != null && age > 300 ? "stale" : "live");
    const b = document.createElement("button");
    b.className = "tab" + (id === RUN ? " on" : "");
    b.innerHTML = `<div class="t"><span class="dot ${state}"></span>${r.label}</div>` +
                  `<div class="s">${last ? fmtSteps(last.steps) + " · " + last.sps.toLocaleString() + " sps"
                                         : "no data synced"}</div>`;
    b.onclick = () => { RUN = id; localStorage.setItem("construct.run", id);
                        if (LAST) renderAll(LAST); };
    bar.appendChild(b);
  });
}

function renderAll(d) {
  // A stored id can name a run that no longer exists (RUNS edited between
  // sessions); fall back rather than render a blank page.
  if (!d.runs[RUN]) RUN = d.run_order[0];
  const r = d.runs[RUN];
  renderTabs(d);
  document.getElementById("runrole").textContent = r.role + "  ·  " + r.log;
  renderMain(r.main);
  renderCurriculum(r.curriculum);
  renderGate(d.gate);
  renderSSL(d.ssl);
  renderSys(d);
}
async function refresh() {
  let d;
  try { d = await (await fetch("/data")).json(); }
  catch { document.getElementById("status").textContent = "server unreachable"; return; }
  document.getElementById("status").textContent =
    `updated ${new Date().toLocaleTimeString()} · auto-refresh 5s · dashed lines = training restarts · ~time = estimated from log mtime (~~ past a restart)`;
  LAST = d;
  renderAll(d);
}
refresh(); setInterval(refresh, 5000);
addEventListener("resize", () => {
  ["main-charts","curr-charts","syscharts"].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.innerHTML = "";           // charts are sized on build, so rebuild
  });
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
    SAMPLER.start()
    SLOW.start()
    print(f"dashboard: http://localhost:{port}  (main log: {MAIN_LOG})")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
