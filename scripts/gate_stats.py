#!/usr/bin/env python3
"""Read logs/champion_history.jsonl and report gate results WITH their error bars.

Why this exists: on 2026-07-20 I read the lambda ladder off raw shares and
wrote "monotone, saturating at parity" into the journal. With the goal counts
in hand that claim evaporated -- every point from lambda 0.5 upward overlapped
every other one. The shares were real; the ORDERING was noise. A gate is ~180
goals, so SE ~3.7% and the 95% band is ~+/-7%, which is wider than most of the
differences anyone is tempted to interpret.

The share alone invites that mistake, so this tool never prints one without its
interval, and `compare` states plainly when a difference is unresolvable rather
than leaving a suggestive gap for the reader to fill in.

    scripts/gate_stats.py list                 # every gate, newest last
    scripts/gate_stats.py list --grep hc_a     # only hill-climb attempts
    scripts/gate_stats.py pool --grep hc_a0016,hc_a0017   # OR: pool these gates
    scripts/gate_stats.py compare armG hc_a    # two-proportion z-test

Stdlib only; reads the same history file champion_gate.py writes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DEFAULT = REPO_ROOT / "logs" / "champion_history.jsonl"

# A gate is ~180 goals. Differences below roughly this are not resolvable by a
# single gate and must not be reported as an ordering.
UNRESOLVABLE_HINT = 0.07


def load_rows(path) -> list:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def name_of(row) -> str:
    return str(row.get("candidate", "?")).split("/")[-1]


def select(rows, pattern) -> list:
    """Substring match, OR-ing over a comma-separated list.

    Comma-separated rather than regex on purpose: the natural thing to reach
    for is a bracket glob like `hc_a001[45]`, and that dies in zsh before the
    tool ever sees it (2026-07-20). A comma list has no shell-active characters
    and needs no quoting.
    """
    if not pattern:
        return list(rows)
    needles = [p.strip() for p in str(pattern).split(",") if p.strip()]
    if not needles:
        return list(rows)
    return [r for r in rows if any(n in name_of(r) for n in needles)]


def counts(row) -> tuple:
    """(goals_for_candidate, goals_for_champion). Falls back to reconstructing
    from the share when a legacy row lacks the raw counts -- and says so, since
    a reconstructed count carries no independent information about n."""
    gc, gh = row.get("goals_c"), row.get("goals_champ")
    if gc is None or gh is None:
        return None, None
    return int(gc), int(gh)


def wilson(k, n, z=1.96) -> tuple:
    """Wilson score interval. Preferred over the normal approximation because
    gate shares sit near the tails often enough (32.8%, 49.2%) that the naive
    interval misbehaves, and it stays inside [0,1] by construction."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)


def summarize(gc, gh) -> dict:
    n = gc + gh
    p = gc / n if n else 0.0
    se = math.sqrt(p * (1 - p) / n) if n else 0.0
    lo, hi = wilson(gc, n)
    return {"goals_c": gc, "goals_champ": gh, "n": n, "share": p,
            "se": se, "lo": lo, "hi": hi}


def fmt(name, s) -> str:
    return (f"{name[:40]:40s} {s['goals_c']:4d}-{s['goals_champ']:4d}  n={s['n']:4d}  "
            f"share={s['share']:.3f}  SE={s['se']:.3f}  95%CI=[{s['lo']:.3f},{s['hi']:.3f}]")


def pool(rows) -> dict:
    gc = gh = 0
    used = 0
    for r in rows:
        a, b = counts(r)
        if a is None:
            continue
        gc += a
        gh += b
        used += 1
    if not used:
        return {}
    out = summarize(gc, gh)
    out["gates"] = used
    return out


def two_proportion_z(a, b) -> dict:
    """Unpooled two-proportion z-test. `a` and `b` are summarize() dicts."""
    d = a["share"] - b["share"]
    se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
    z = d / se if se else 0.0
    # two-sided normal p-value without scipy
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"diff": d, "se_diff": se, "z": z, "p": p}


def cmd_list(args) -> int:
    rows = select(load_rows(args.history), args.grep)
    if not rows:
        print("no matching gates")
        return 1
    skipped = 0
    for r in rows:
        gc, gh = counts(r)
        if gc is None:
            skipped += 1
            continue
        print(fmt(name_of(r), summarize(gc, gh)))
    if skipped:
        print(f"\n({skipped} row(s) skipped: no raw goal counts, so no interval "
              f"can be computed -- a bare share is not reportable here)")
    return 0


def cmd_pool(args) -> int:
    rows = select(load_rows(args.history), args.grep)
    s = pool(rows)
    if not s:
        print("no matching gates with goal counts")
        return 1
    print(fmt(f"POOLED ({s['gates']} gates)", s))
    return 0


def cmd_compare(args) -> int:
    all_rows = load_rows(args.history)
    a_rows, b_rows = select(all_rows, args.a), select(all_rows, args.b)
    a, b = pool(a_rows), pool(b_rows)
    if not a or not b:
        print("one or both selectors matched no gates with goal counts")
        return 1
    print(fmt(f"A {args.a} ({a['gates']} gates)", a))
    print(fmt(f"B {args.b} ({b['gates']} gates)", b))
    t = two_proportion_z(a, b)
    print(f"\nA - B = {t['diff']:+.3f}   SE_diff={t['se_diff']:.3f}   "
          f"z={t['z']:+.2f}   p={t['p']:.3f}")
    if t["p"] >= 0.05:
        print(f"\nNOT RESOLVED. These two are statistically indistinguishable at "
              f"this sample size.\nDo not report an ordering between them. To resolve "
              f"a difference of {abs(t['diff']):.3f} you would need roughly "
              f"{needed_gates(abs(t['diff']) or 0.01):.0f} gates per arm.")
    else:
        print(f"\nRESOLVED at p={t['p']:.3f}: A is "
              f"{'better' if t['diff'] > 0 else 'worse'} than B.")
    return 0


def needed_gates(diff, per_gate_n=180, power_z=2.8) -> float:
    """Gates per arm to resolve `diff` at ~80% power, 5% two-sided.
    power_z = z_{0.975} + z_{0.80} = 1.96 + 0.84."""
    if diff <= 0:
        return float("inf")
    n_per_arm = 2 * 0.25 * (power_z / diff) ** 2
    return n_per_arm / per_gate_n


def step_of(row) -> int:
    """Training-step count parsed from the candidate filename (ck_<steps>.pt).

    Used to ORDER a trajectory's rungs. Returns -1 when absent so unparseable
    names sort first and are visible rather than silently interleaved."""
    import re
    m = re.search(r"ck_(\d+)", name_of(row))
    return int(m.group(1)) if m else -1


def cochran_armitage(groups) -> dict:
    """Trend test for proportions across ORDERED groups.

    `groups` is [(score, successes, total), ...] with `score` the ordering
    variable (here: training steps). This is the right test for a trajectory:
    each rung carries a ~+/-7% band so no single pair of rungs is resolvable,
    but a CONSISTENT DIRECTION across many independent gates is evidence that
    no pairwise comparison contains. Testing only the endpoints would throw
    away every rung in between.

    Returns z and a two-sided p. Positive z = share rises with the score.
    """
    n = sum(t for _, _, t in groups)
    if n == 0:
        return {"z": 0.0, "p": 1.0, "slope_sign": 0}
    successes = sum(s for _, s, _ in groups)
    p_bar = successes / n
    s_mean = sum(sc * t for sc, _, t in groups) / n
    # T = sum over groups of score * successes, centred under H0
    t_stat = sum(sc * s for sc, s, _ in groups)
    e_t = p_bar * sum(sc * tot for sc, _, tot in groups)
    var_t = p_bar * (1 - p_bar) * sum(tot * (sc - s_mean) ** 2 for sc, _, tot in groups)
    if var_t <= 0:
        return {"z": 0.0, "p": 1.0, "slope_sign": 0}
    z = (t_stat - e_t) / math.sqrt(var_t)
    return {"z": z, "p": math.erfc(abs(z) / math.sqrt(2)),
            "slope_sign": 1 if z > 0 else (-1 if z < 0 else 0)}


def cmd_trend(args) -> int:
    rows = select(load_rows(args.history), args.grep)
    rows = [r for r in rows if counts(r)[0] is not None]
    if len(rows) < 3:
        print(f"trend needs at least 3 gates with goal counts; got {len(rows)}")
        return 1
    rows.sort(key=step_of)
    groups = []
    for r in rows:
        gc, gh = counts(r)
        print(fmt(f"{step_of(r):>12d}  {name_of(r)[:24]}", summarize(gc, gh)))
        groups.append((step_of(r), gc, gc + gh))
    # rescale steps to millions so the arithmetic stays well-conditioned
    groups = [(sc / 1e6, s, t) for sc, s, t in groups]
    t = cochran_armitage(groups)
    print(f"\nCochran-Armitage trend across {len(groups)} rungs: "
          f"z={t['z']:+.2f}  p={t['p']:.4f}")
    if t["p"] < 0.05:
        direction = "RISES" if t["z"] > 0 else "FALLS"
        print(f"TREND RESOLVED: goal share {direction} with training steps.")
    else:
        print("NO RESOLVED TREND. The rungs are consistent with a flat line at "
              "this sample size;\ndo not read a slope off the point estimates.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--history", default=str(HISTORY_DEFAULT))
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("list", help="every gate with its interval")
    pl.add_argument("--grep", default="",
                    help="substring(s) of the candidate name; comma-separated = OR")
    pl.set_defaults(func=cmd_list)

    pp = sub.add_parser("pool", help="pool several gates into one estimate")
    pp.add_argument("--grep", default="",
                    help="substring(s) of the candidate name; comma-separated = OR")
    pp.set_defaults(func=cmd_pool)

    pt = sub.add_parser("trend", help="Cochran-Armitage trend across ordered rungs")
    pt.add_argument("--grep", default="",
                    help="substring(s) of the candidate name; comma-separated = OR")
    pt.set_defaults(func=cmd_trend)

    pc = sub.add_parser("compare", help="two-proportion z-test between two selections")
    pc.add_argument("a")
    pc.add_argument("b")
    pc.set_defaults(func=cmd_compare)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
