"""Live training dashboard. Parses checkpoints/train_v0.log, serves charts.

Usage: python scripts/dashboard.py [port]   (default 8420)
From Windows: http://localhost:<port> (WSL2 forwards localhost TCP).
Stdlib only.
"""
import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

LOG = Path(__file__).resolve().parent.parent / "checkpoints" / "train_v0.log"
LINE = re.compile(
    r"iter (\d+) steps ([\d,]+) sps ([\d,]+) ep_rew ([-\d.]+) "
    r"pi_loss ([-\d.]+) v_loss ([-\d.]+) ent ([-\d.]+) clip ([-\d.]+)"
)
MAX_POINTS = 500


def parse_log():
    rows = []
    try:
        text = LOG.read_text(errors="replace")
    except FileNotFoundError:
        return rows
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
    # training restarts reset iter but steps keep growing; sort + dedupe on steps
    rows.sort(key=lambda r: r["steps"])
    if len(rows) > MAX_POINTS:
        stride = len(rows) / MAX_POINTS
        rows = [rows[int(i * stride)] for i in range(MAX_POINTS - 1)] + [rows[-1]]
    return rows


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
h1 { font-size:17px; font-weight:650; }
.sub { color:var(--ink2); font-size:12.5px; margin:2px 0 16px; }
.tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:10px; margin-bottom:16px; }
.tile { background:var(--card); border:1px solid var(--border); border-radius:8px;
        padding:10px 12px; }
.tile .k { font-size:11.5px; color:var(--ink2); text-transform:uppercase;
           letter-spacing:.04em; }
.tile .v { font-size:22px; font-weight:650; font-variant-numeric:tabular-nums; }
.tile .d { font-size:11.5px; color:var(--muted); }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
        gap:12px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:8px;
        padding:12px 12px 6px; }
.card h2 { font-size:12.5px; font-weight:600; color:var(--ink2); margin-bottom:4px; }
.card .note { font-size:11px; color:var(--muted); }
svg { display:block; width:100%; height:150px; }
.axis { font-size:10px; fill:var(--muted); font-variant-numeric:tabular-nums; }
.gridline { stroke:var(--grid); stroke-width:1 }
.line { stroke:var(--series); stroke-width:2; fill:none;
        stroke-linejoin:round; stroke-linecap:round }
.cross { stroke:var(--muted); stroke-width:1; stroke-dasharray:2 3 }
.dot { fill:var(--series); stroke:var(--card); stroke-width:2 }
.tip { position:fixed; pointer-events:none; background:var(--card); color:var(--ink);
       border:1px solid var(--border); border-radius:6px; padding:5px 8px;
       font-size:12px; box-shadow:0 2px 8px rgba(0,0,0,.12); display:none;
       font-variant-numeric:tabular-nums; z-index:5 }
details { margin-top:16px; }
summary { cursor:pointer; color:var(--ink2); font-size:13px; }
table { border-collapse:collapse; margin-top:8px; font-variant-numeric:tabular-nums;
        font-size:12.5px; width:100%; max-width:820px; }
th,td { text-align:right; padding:4px 10px; border-bottom:1px solid var(--border); }
th { color:var(--ink2); font-weight:600; }
.wrap { overflow-x:auto; }
</style></head><body data-palette="#2a78d6">
<h1>Construct — training run</h1>
<div class="sub" id="status">loading…</div>
<div class="tiles" id="tiles"></div>
<div class="grid" id="charts"></div>
<details><summary>Recent iterations (table)</summary>
  <div class="wrap"><table id="tbl"></table></div></details>
<div class="tip" id="tip"></div>
<script>
const METRICS = [
  {key:"sps",    title:"Throughput (steps/sec)",     fmt:v=>v.toLocaleString()},
  {key:"ep_rew", title:"Reward per completed episode", fmt:v=>v.toFixed(2), cap:.98,
   note:"y capped at p98 \\u2014 completed-episode denominator makes late values spike"},
  {key:"ent",    title:"Policy entropy",             fmt:v=>v.toFixed(3)},
  {key:"pi_loss",title:"Policy loss",                fmt:v=>v.toFixed(4)},
  {key:"v_loss", title:"Value loss",                 fmt:v=>v.toFixed(3)},
  {key:"clip",   title:"PPO clip fraction",          fmt:v=>v.toFixed(3)},
];
const tip = document.getElementById("tip");
const fmtSteps = v => v >= 1e9 ? (v/1e9).toFixed(2)+"B" : v >= 1e6 ? (v/1e6).toFixed(0)+"M" : v.toLocaleString();

function chart(el, rows, m) {
  const W = el.clientWidth || 320, H = 150, L = 46, R = 8, T = 8, B = 20;
  const xs = rows.map(r => r.steps), ys = rows.map(r => r[m.key]);
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
    g += `<text class="axis" x="${X(v)}" y="${H-5}" text-anchor="middle">${fmtSteps(v)}</text>`;
  });
  const path = rows.map((r,i)=>`${i?"L":"M"}${X(r.steps).toFixed(1)},${Y(r[m.key]).toFixed(1)}`).join("");
  el.innerHTML = `<h2>${m.title}</h2>
    <svg viewBox="0 0 ${W} ${H}">${g}<path class="line" d="${path}"/>
      <line class="cross" y1="${T}" y2="${H-B}" x1="-9" x2="-9"/>
      <circle class="dot" r="4" cx="-9" cy="-9"/></svg>
    ${m.note?`<div class="note">${m.note}</div>`:""}`;
  const svg = el.querySelector("svg"), cross = el.querySelector(".cross"),
        dot = el.querySelector(".dot");
  svg.addEventListener("mousemove", e => {
    const box = svg.getBoundingClientRect();
    const mx = (e.clientX - box.left) * (W / box.width);
    let best = 0, bd = 1e18;
    rows.forEach((r,i) => { const d = Math.abs(X(r.steps)-mx); if (d < bd) { bd = d; best = i; } });
    const r = rows[best], px = X(r.steps), py = Y(r[m.key]);
    cross.setAttribute("x1", px); cross.setAttribute("x2", px);
    dot.setAttribute("cx", px); dot.setAttribute("cy", py);
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 170) + "px";
    tip.style.top = (e.clientY + 14) + "px";
    tip.innerHTML = `<b>${m.fmt(r[m.key])}</b> ${m.title.split("(")[0]}<br>` +
                    `<span style="color:var(--ink2)">at ${fmtSteps(r.steps)} steps</span>`;
  });
  svg.addEventListener("mouseleave", () => {
    tip.style.display = "none";
    cross.setAttribute("x1", -9); cross.setAttribute("x2", -9);
    dot.setAttribute("cx", -9); dot.setAttribute("cy", -9);
  });
}

async function refresh() {
  let rows;
  try { rows = await (await fetch("/data")).json(); }
  catch { document.getElementById("status").textContent = "log unreachable"; return; }
  if (!rows.length) { document.getElementById("status").textContent = "no data yet"; return; }
  const last = rows[rows.length-1];
  document.getElementById("status").textContent =
    `updated ${new Date().toLocaleTimeString()} \\u00b7 auto-refresh 5s \\u00b7 ${rows.length} points shown`;
  document.getElementById("tiles").innerHTML = [
    ["Total steps", fmtSteps(last.steps), "target-free run"],
    ["Steps / sec", last.sps.toLocaleString(), "collection + learning"],
    ["Ep. reward", last.ep_rew.toFixed(1), "latest iteration"],
    ["Entropy", last.ent.toFixed(3), "lower = more decisive"],
  ].map(([k,v,d]) => `<div class="tile"><div class="k">${k}</div><div class="v">${v}</div><div class="d">${d}</div></div>`).join("");
  const grid = document.getElementById("charts");
  if (!grid.children.length)
    METRICS.forEach(() => grid.appendChild(Object.assign(document.createElement("div"), {className:"card"})));
  METRICS.forEach((m,i) => chart(grid.children[i], rows, m));
  const cols = ["steps","sps","ep_rew","pi_loss","v_loss","ent","clip"];
  document.getElementById("tbl").innerHTML =
    `<tr>${cols.map(c=>`<th>${c}</th>`).join("")}</tr>` +
    rows.slice(-15).reverse().map(r =>
      `<tr>${cols.map(c=>`<td>${typeof r[c]==="number"&&!Number.isInteger(r[c])?r[c].toFixed(4):r[c].toLocaleString()}</td>`).join("")}</tr>`).join("");
}
refresh(); setInterval(refresh, 5000);
addEventListener("resize", () => { document.getElementById("charts").innerHTML = ""; refresh(); });
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/data":
            body = json.dumps(parse_log()).encode()
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
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8420
    print(f"dashboard: http://localhost:{port}  (log: {LOG})")
    ThreadingHTTPServer(("0.0.0.0", port), Handler).serve_forever()
