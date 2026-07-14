"""Live training dashboard. Parses checkpoints/train_v0.log, samples system
stats, runs periodic skill evals, serves auto-refreshing charts.

Usage: python scripts/dashboard.py [port] [--eval-every MINUTES]
       (default port 8420, eval every 30 min; --no-eval disables)
From Windows: http://localhost:<port> (WSL2 forwards localhost TCP).
Stdlib only.
"""
import json
import re
import subprocess
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LOG = REPO / "checkpoints" / "train_v0.log"
EVAL_HISTORY = REPO / "checkpoints" / "eval_history.jsonl"
LINE = re.compile(
    r"iter (\d+) steps ([\d,]+) sps ([\d,]+) ep_rew ([-\d.]+) "
    r"pi_loss ([-\d.]+) v_loss ([-\d.]+) ent ([-\d.]+) clip ([-\d.]+)"
)
RESUME = re.compile(r"resumed at ([\d,]+) steps")
MAX_POINTS = 500


def parse_log():
    rows, restarts = [], []
    try:
        text = LOG.read_text(errors="replace")
    except FileNotFoundError:
        return rows, restarts
    for m in RESUME.finditer(text):
        restarts.append(int(m.group(1).replace(",", "")))
    for m in LINE.finditer(text):
        rows.append({
            "steps": int(m.group(2).replace(",", "")),
            "sps": int(m.group(3).replace(",", "")),
            "ep_rew": float(m.group(4)),
            "pi_loss": float(m.group(5)),
            "v_loss": float(m.group(6)),
            "ent": float(m.group(7)),
            "clip": float(m.group(8)),
        })
    rows.sort(key=lambda r: r["steps"])
    if len(rows) > MAX_POINTS:
        stride = len(rows) / MAX_POINTS
        rows = [rows[int(i * stride)] for i in range(MAX_POINTS - 1)] + [rows[-1]]
    return rows, restarts


def checkpoint_info():
    cks = sorted(REPO.glob("checkpoints/ck_*.pt"))
    if not cks:
        return {}
    total = sum(f.stat().st_size for f in cks)
    oldest_mtime = min(f.stat().st_mtime for f in cks)
    latest = cks[-1]
    return {
        "count": len(cks),
        "total_gb": round(total / 1e9, 2),
        "latest": latest.name,
        "latest_steps": int(latest.stem.split("_")[1]),
        "runtime_s": int(time.time() - oldest_mtime),
    }


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


class EvalRunner(threading.Thread):
    """Every N minutes, runs eval_metrics.py on the newest checkpoint (low
    priority) and appends the result to eval_history.jsonl."""

    def __init__(self, every_min):
        super().__init__(daemon=True)
        self.every = every_min * 60
        self.status = "idle"

    def _evaluated_steps(self):
        done = set()
        if EVAL_HISTORY.exists():
            for line in EVAL_HISTORY.read_text().splitlines():
                try:
                    done.add(json.loads(line)["steps"])
                except Exception:
                    pass
        return done

    def run(self):
        while True:
            cks = sorted(REPO.glob("checkpoints/ck_*.pt"))
            if cks:
                latest = cks[-1]
                steps = int(latest.stem.split("_")[1])
                if steps not in self._evaluated_steps():
                    self.status = f"evaluating {latest.name}…"
                    try:
                        out = subprocess.run(
                            ["nice", "-n", "15", sys.executable,
                             str(REPO / "scripts" / "eval_metrics.py"), str(latest)],
                            capture_output=True, text=True, cwd=REPO, timeout=1800,
                        ).stdout
                        touches = float(re.search(r"touches/min/agent: ([\d.]+)", out).group(1))
                        dist = float(re.search(r"mean dist to ball: (\d+)", out).group(1))
                        with EVAL_HISTORY.open("a") as f:
                            f.write(json.dumps({
                                "ts": int(time.time()), "steps": steps,
                                "touches_per_min": touches, "dist_uu": dist,
                            }) + "\n")
                        self.status = "idle"
                    except Exception as e:
                        self.status = f"eval failed: {e}"
            time.sleep(self.every)


SAMPLER = SysSampler()
EVALER = None


def payload():
    rows, restarts = parse_log()
    last = rows[-1] if rows else None
    eta = None
    if last and last["sps"] > 0:
        nxt = (last["steps"] // 100_000_000 + 1) * 100_000_000
        eta = int((nxt - last["steps"]) / last["sps"])
    evals = []
    if EVAL_HISTORY.exists():
        for line in EVAL_HISTORY.read_text().splitlines():
            try:
                evals.append(json.loads(line))
            except Exception:
                pass
        evals.sort(key=lambda e: e["steps"])
    return {
        "rows": rows,
        "restarts": restarts,
        "ckpt": checkpoint_info(),
        "eta_s": eta,
        "evals": evals,
        "sys": list(SAMPLER.history),
        "eval_status": EVALER.status if EVALER else "disabled",
    }


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Construct training</title>
<style>
:root { --surface:#fbfaf7; --ink:#1a1a19; --ink2:#5f5e56; --muted:#8a897e;
        --grid:#e5e4dc; --series:#2a78d6; --card:#ffffff; --border:#e5e4dc; }
@media (prefers-color-scheme: dark) {
  :root { --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#8a897e;
          --grid:#33322f; --series:#3987e5; --card:#232320; --border:#33322f; }
}
* { box-sizing:border-box; margin:0 }
body { background:var(--surface); color:var(--ink);
       font:14px/1.45 system-ui,-apple-system,sans-serif; padding:20px; }
h1 { font-size:17px; font-weight:650 }
h2.sec { font-size:13px; font-weight:650; color:var(--ink2); margin:22px 0 8px;
         text-transform:uppercase; letter-spacing:.05em }
.sub { color:var(--ink2); font-size:12.5px; margin:2px 0 4px }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
         gap:10px; margin-bottom:12px }
.tile { background:var(--card); border:1px solid var(--border); border-radius:8px;
        padding:10px 12px }
.tile .k { font-size:11.5px; color:var(--ink2); text-transform:uppercase;
           letter-spacing:.04em }
.tile .v { font-size:22px; font-weight:650; font-variant-numeric:tabular-nums }
.tile .d { font-size:11.5px; color:var(--muted) }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:12px }
.card { background:var(--card); border:1px solid var(--border); border-radius:8px;
        padding:12px 12px 8px }
.card h3 { font-size:12.5px; font-weight:600; color:var(--ink2); margin-bottom:2px }
.card .why { font-size:11.5px; color:var(--muted); margin-top:4px }
svg { display:block; width:100%; height:150px }
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
       font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.12); display:none;
       font-variant-numeric:tabular-nums; z-index:5 }
details { margin-top:16px }
summary { cursor:pointer; color:var(--ink2); font-size:13px }
table { border-collapse:collapse; margin-top:8px; font-variant-numeric:tabular-nums;
        font-size:12.5px; width:100%; max-width:860px }
th,td { text-align:right; padding:4px 10px; border-bottom:1px solid var(--border) }
th { color:var(--ink2); font-weight:600 }
.wrap { overflow-x:auto }
</style></head><body data-palette="#2a78d6">
<h1>Construct — training run</h1>
<div class="sub" id="status">loading…</div>

<h2 class="sec">Run</h2>
<div class="tiles" id="runtiles"></div>

<h2 class="sec">Training metrics</h2>
<div class="tiles" id="tiles"></div>
<div class="grid" id="charts"></div>

<h2 class="sec">System</h2>
<div class="tiles" id="systiles"></div>
<div class="grid" id="syscharts"></div>

<h2 class="sec">Skill evals <span id="evalstatus" style="font-weight:400;text-transform:none"></span></h2>
<div class="sub">Headless self-play evals of saved checkpoints — the only charts here measuring
actual skill rather than optimizer internals. Random-policy baseline: 0.0 touches/min, 3769 uu.</div>
<div class="grid" id="evalcharts"></div>

<details><summary>Recent iterations (table)</summary>
  <div class="wrap"><table id="tbl"></table></div></details>
<div class="tip" id="tip"></div>
<script>
const METRICS = [
  {key:"sps", title:"Throughput (steps/sec)", fmt:v=>v.toLocaleString(),
   why:"How fast experience is collected and learned. Dips = thermal throttle, the viewer/evals competing for CPU, or checkpoint writes. Higher is strictly better."},
  {key:"ep_rew", title:"Reward per completed episode", fmt:v=>v.toFixed(2), cap:.98,
   why:"Total reward per finished episode. Rising = scoring more / conceding less. Late-run spikes are a denominator artifact: fewer episodes finish as both cars defend better (y capped at p98 for readability)."},
  {key:"ent", title:"Policy entropy", fmt:v=>v.toFixed(3),
   why:"How random action choices are. Starts near uniform over 90 actions (~4.5), falls as the policy commits to what works. Falling too fast = premature convergence; flat at max = not learning."},
  {key:"pi_loss", title:"Policy loss", fmt:v=>v.toFixed(4),
   why:"PPO's clipped surrogate objective. Hovers near zero BY DESIGN — its magnitude is not skill. Only sustained large swings matter (instability)."},
  {key:"v_loss", title:"Value loss", fmt:v=>v.toFixed(3),
   why:"How wrong the critic's future-reward predictions are. Falls as it understands the game; bumps when the policy discovers new behavior the critic hasn't priced in yet."},
  {key:"clip", title:"PPO clip fraction", fmt:v=>v.toFixed(3),
   why:"Share of gradient updates hitting PPO's trust-region ceiling. 0.05–0.2 healthy. Sustained >0.3 = updates too aggressive (learning rate hot or stale data)."},
];
const SYSMETRICS = [
  {key:"gpu", title:"GPU utilization (%)", fmt:v=>v+"%",
   why:"Batched inference + PPO updates. Bursty by design: collect (low) then learn (spike). Sustained 100% = learner-bound; near 0 = collection-bound."},
  {key:"gpu_temp", title:"GPU temperature (°C)", fmt:v=>v+"°C",
   why:"Laptop GPUs throttle around ~87°C — when this climbs, watch throughput sag in the chart above."},
  {key:"cpu", title:"CPU utilization (%)", fmt:v=>v+"%",
   why:"RocketSim arena workers live here. Pegged CPU is expected and good — physics simulation is the throughput bottleneck."},
];
const EVALMETRICS = [
  {key:"touches_per_min", title:"Ball touches / min / agent", fmt:v=>v.toFixed(1), marks:true,
   why:"From a 5-game-minute headless eval of each checkpoint: how often the bot contacts the ball. Random baseline 0.0; P0 exit gate was 3x baseline."},
  {key:"dist_uu", title:"Mean distance to ball (uu)", fmt:v=>v.toFixed(0), marks:true, lowerBetter:true,
   why:"Average car-to-ball distance during eval. Random baseline 3769 uu. Lower = stays involved in the play. Field is ~10,000 uu long."},
];
const tip = document.getElementById("tip");
const fmtSteps = v => v >= 1e9 ? (v/1e9).toFixed(2)+"B" : v >= 1e6 ? (v/1e6).toFixed(0)+"M" : v.toLocaleString();
const fmtDur = s => { const h = Math.floor(s/3600), m = Math.floor(s%3600/60);
                      return h ? `${h}h ${m}m` : `${m}m`; };

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
    <svg viewBox="0 0 ${W} ${H}">${g}<path class="line" d="${path}"/>${marks}
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

async function refresh() {
  let d;
  try { d = await (await fetch("/data")).json(); }
  catch { document.getElementById("status").textContent = "server unreachable"; return; }
  const rows = d.rows;
  if (!rows.length) { document.getElementById("status").textContent = "no data yet"; return; }
  const last = rows[rows.length-1];
  document.getElementById("status").textContent =
    `updated ${new Date().toLocaleTimeString()} \\u00b7 auto-refresh 5s \\u00b7 ${rows.length} points \\u00b7 dashed lines = training restarts`;

  tiles(document.getElementById("runtiles"), [
    ["Runtime", d.ckpt.runtime_s ? fmtDur(d.ckpt.runtime_s) : "—", "since first checkpoint"],
    ["Checkpoints", d.ckpt.count ? `${d.ckpt.count} · ${d.ckpt.total_gb} GB` : "—", "on disk, resumable"],
    ["Latest", d.ckpt.latest_steps ? fmtSteps(d.ckpt.latest_steps) : "—", "newest saved policy"],
    ["Next 100M in", d.eta_s ? fmtDur(d.eta_s) : "—", "at current throughput"],
  ]);
  tiles(document.getElementById("tiles"), [
    ["Total steps", fmtSteps(last.steps), "experience consumed so far"],
    ["Steps / sec", last.sps.toLocaleString(), "sim + learning combined"],
    ["Ep. reward", last.ep_rew.toFixed(1), "latest iteration (proxy metric)"],
    ["Entropy", last.ent.toFixed(3), "lower = more decisive policy"],
  ]);
  const grid = document.getElementById("charts");
  if (!grid.children.length)
    METRICS.forEach(() => grid.appendChild(Object.assign(document.createElement("div"), {className:"card"})));
  METRICS.forEach((m,i) => chart(grid.children[i], rows, m, "steps", fmtSteps, d.restarts));

  const sys = d.sys;
  if (sys.length) {
    const s = sys[sys.length-1];
    tiles(document.getElementById("systiles"), [
      ["GPU", (s.gpu ?? "—") + "%", "inference + PPO updates"],
      ["VRAM", s.vram_used ? `${(s.vram_used/1024).toFixed(1)} / ${(s.vram_total/1024).toFixed(1)} GB` : "—", "model + batches"],
      ["GPU temp", (s.gpu_temp ?? "—") + "°C", "throttles near ~87°C"],
      ["CPU", s.cpu + "%", "RocketSim workers"],
      ["RAM", s.ram + "%", "of 19 GB WSL allocation"],
    ]);
    const sgrid = document.getElementById("syscharts");
    if (!sgrid.children.length)
      SYSMETRICS.forEach(() => sgrid.appendChild(Object.assign(document.createElement("div"), {className:"card"})));
    const tFmt = v => new Date(v*1000).toLocaleTimeString([], {hour:"2-digit",minute:"2-digit"});
    SYSMETRICS.forEach((m,i) => {
      const have = sys.filter(r => r[m.key] !== undefined);
      if (have.length > 1) chart(sgrid.children[i], have, m, "ts", tFmt);
    });
  }

  document.getElementById("evalstatus").textContent = "· " + d.eval_status;
  if (d.evals.length) {
    const egrid = document.getElementById("evalcharts");
    if (!egrid.children.length)
      EVALMETRICS.forEach(() => egrid.appendChild(Object.assign(document.createElement("div"), {className:"card"})));
    EVALMETRICS.forEach((m,i) => {
      if (d.evals.length === 1) {
        const e0 = d.evals[0];
        egrid.children[i].innerHTML = `<h3>${m.title}</h3>
          <div class="tile" style="border:none;padding:6px 0"><div class="v">${m.fmt(e0[m.key])}</div>
          <div class="d">single eval at ${fmtSteps(e0.steps)} steps — chart appears after the next one</div></div>
          <div class="why">${m.why}</div>`;
      } else chart(egrid.children[i], d.evals, m, "steps", fmtSteps);
    });
  }

  const cols = ["steps","sps","ep_rew","pi_loss","v_loss","ent","clip"];
  document.getElementById("tbl").innerHTML =
    `<tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>` +
    rows.slice(-15).reverse().map(r =>
      `<tr>${cols.map(c=>`<td>${typeof r[c]==="number"&&!Number.isInteger(r[c])?r[c].toFixed(4):r[c].toLocaleString()}</td>`).join("")}</tr>`).join("");
}
refresh(); setInterval(refresh, 5000);
addEventListener("resize", () => {
  ["charts","syscharts","evalcharts"].forEach(id => document.getElementById(id).innerHTML = "");
  refresh();
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
    if "--no-eval" not in args:
        EVALER = EvalRunner(eval_every)
        EVALER.start()
    print(f"dashboard: http://localhost:{port}  (log: {LOG}, eval every {eval_every}m)")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
