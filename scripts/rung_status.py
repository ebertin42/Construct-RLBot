#!/usr/bin/env python3
"""All TWELVE foreign-slot rungs, plus the per-mode win-rate trend.

EXISTS BECAUSE OF A REPORTING BUG, not because the data was missing. Every rung
report on 2026-07-27 was produced with `... | head -4` against a 12-slot roster.
Slots 0-3 are exactly the four 1v1 slots, so the eight 2v2/3v3 slots -- which
have never left the ladder floor -- were invisible in all of them, and the run
was repeatedly described as "converged" on the strength of a quarter of it.
A truncating shell pipeline is not a status command; this is.

The staggered eval measures ONE TEAM SIZE per cycle, so on any given line 8 of
the 12 slots read `n/a` and carry their previous rung. That is normal and is why
the trend below pools over decision lines rather than reading the last one.

Usage:
  scripts/rung_status.py [log]            # default: the local synced mirror
  scripts/rung_status.py --remote         # read the live log over ssh
"""
import argparse
import re
import subprocess
import sys

# The [foreign].slots order in configs/train_v9_fromscratch.toml. The log prints
# segments in this order and nothing in the line itself says which mode a
# segment belongs to -- the mapping IS this list, so it must track the config.
SLOTS = [
    ("element", 1), ("immortal", 1), ("necto", 1), ("nexto", 1),
    ("necto", 2), ("nexto", 2), ("element", 2), ("immortal", 2),
    ("necto", 3), ("nexto", 3), ("element", 3), ("immortal", 3),
]
LADDER = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24]   # index 0 = hardest
BAND = (0.35, 0.65)

WR = re.compile(r"wr([0-9.]+)")
EMA = re.compile(r"ema([0-9.]+)")
RUNG = re.compile(r"(\d+)c/p(\d+)")
REMOTE = "elliot@192.168.86.117"
REMOTE_LOG = "/home/elliot/construct/checkpoints_v9/v9_s20260726.log"


def parse(lines):
    """-> list of 12-element records. Segments that were not measured this
    cycle read wr/ema None but still carry their rung."""
    rows = []
    for ln in lines:
        if "auto-curriculum:" not in ln or "eval engines" in ln:
            continue
        segs = [s.strip() for s in ln.split("auto-curriculum:", 1)[1].split("|")]
        if len(segs) != len(SLOTS):
            continue                    # a banner or an older roster; skip
        rec = []
        for seg in segs:
            wr, ema = WR.search(seg), EMA.search(seg)
            # LAST rung on the segment: "1c/p20->1c/p16" is a MOVE, and the
            # slot's rung is where it landed. Reading the first one reports the
            # rung the controller just left.
            rungs = RUNG.findall(seg)
            rec.append({
                "wr": float(wr.group(1)) if wr else None,
                "ema": float(ema.group(1)) if ema else None,
                "cars": int(rungs[-1][0]) if rungs else None,
                "p": int(rungs[-1][1]) if rungs else None,
            })
        rows.append(rec)
    return rows


def harder(a, b):
    """True if rung a is strictly harder than b. Lexicographic (cars, then
    period index), matching the controller's own ordering."""
    ka = (-a[0], LADDER.index(a[1]) if a[1] in LADDER else 99)
    kb = (-b[0], LADDER.index(b[1]) if b[1] in LADDER else 99)
    return ka < kb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", default="checkpoints_v9/train_remote.log")
    ap.add_argument("--remote", action="store_true", help="read the live log over ssh")
    ap.add_argument("--buckets", type=int, default=6)
    args = ap.parse_args()

    if args.remote:
        out = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", REMOTE,
             f"grep -a 'auto-curriculum:' {REMOTE_LOG}"],
            capture_output=True, text=True, timeout=180)
        lines = out.stdout.splitlines()
    else:
        lines = open(args.log, errors="replace").read().splitlines()

    rows = parse(lines)
    if not rows:
        print("no 12-slot curriculum lines found", file=sys.stderr)
        return 1
    print(f"{len(rows)} decision lines\n")

    # --- per-mode trend. Pooled over slots AND lines, because any single line
    # measured only one mode.
    nb = args.buckets
    bs = max(1, len(rows) // nb)
    print("MEASURED WIN RATE BY MODE (band 0.35-0.65)")
    print(f"{'bucket':>7}{'1v1':>9}{'2v2':>9}{'3v3':>9}")
    for b in range(nb):
        chunk = rows[b * bs:] if b == nb - 1 else rows[b * bs:(b + 1) * bs]
        if not chunk:
            continue
        cells = []
        for m in (1, 2, 3):
            idxs = [i for i, (_, mm) in enumerate(SLOTS) if mm == m]
            v = [r[i]["wr"] for r in chunk for i in idxs if r[i]["wr"] is not None]
            cells.append(sum(v) / len(v) if v else float("nan"))
        print(f"{b + 1:>7}" + "".join(f"{c:>9.3f}" for c in cells))

    # --- current rung for every slot, and whether it has ever moved.
    last = rows[-1]
    print("\nRUNG NOW  (p24 = ladder FLOOR = easiest; 1c = one foreign car)")
    for i, (kind, m) in enumerate(SLOTS):
        d = last[i]
        seen = [(r[i]["cars"], r[i]["p"]) for r in rows if r[i]["p"] is not None]
        now = (d["cars"], d["p"])
        best = now
        for cp in seen:
            if harder(cp, best):
                best = cp
        stuck = "  never moved" if best == now and len(set(seen)) == 1 else ""
        wr = f"{d['wr']:.2f}" if d["wr"] is not None else " n/a"
        ema = f"{d['ema']:.2f}" if d["ema"] is not None else " n/a"
        print(f"  {kind:>8}.{m}s  {d['cars']}c/p{d['p']:<3} wr {wr} ema {ema}"
              f"   best {best[0]}c/p{best[1]}{stuck}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
