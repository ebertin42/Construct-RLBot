"""Full version-level round-robin: every (bot, handicap-period) "version" plays
every other, ranked by mean win-share vs the whole field. Gives complete
visibility of the difficulty ladder -- handicapped and full versions on one axis,
decided by bot-vs-bot outcomes (not our champion).

This is the spine the ladder-index auto-curriculum walks. Output: a sorted TSV
(logs/version_ladder.tsv) + a printed ranking.
"""
import argparse
import itertools
import os

import numpy as np


def winshare(rewards, terminated):
    from construct.league.matches import match_record, split_matches
    rec = match_record(split_matches(np.asarray(rewards), np.asarray(terminated)))
    n = rec["wins"] + rec["draws"] + rec["losses"]
    return ((rec["wins"] + 0.5 * rec["draws"]) / n, n) if n else (float("nan"), 0)


def main():
    from construct._engine import BotMatch

    W = {b: f"~/.cache/construct/{b}_weights.npz"
         for b in ("element", "immortal", "necto", "nexto")}
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", default="element,immortal,necto,nexto")
    ap.add_argument("--periods", default="1,3,6,12")
    ap.add_argument("--arenas", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--out", default="logs/version_ladder.tsv")
    args = ap.parse_args()

    bots = [b for b in args.bots.split(",") if os.path.exists(os.path.expanduser(W[b]))]
    periods = [int(p) for p in args.periods.split(",")]
    wt = {b: {k: v.astype(np.float32) for k, v in np.load(os.path.expanduser(W[b])).items()}
          for b in bots}
    versions = [(b, p) for b in bots for p in periods]
    name = lambda v: f"{v[0]}:p{v[1]}"

    # accumulate win_share and count per version (averaged over its matches)
    ws_sum = {name(v): 0.0 for v in versions}
    cnt = {name(v): 0 for v in versions}
    total = len(versions) * (len(versions) - 1) // 2
    done = 0
    for va, vb in itertools.combinations(versions, 2):
        bm = BotMatch(va[0], wt[va[0]], va[1], vb[0], wt[vb[0]], vb[1], arenas=args.arenas)
        r, t = bm.run(args.steps)
        ws, n = winshare(r, t)
        if ws == ws:  # not nan
            ws_sum[name(va)] += ws
            cnt[name(va)] += 1
            ws_sum[name(vb)] += 1.0 - ws
            cnt[name(vb)] += 1
        done += 1
        print(f"  [{done}/{total}] {name(va):12s} vs {name(vb):12s}: {ws:.3f} (n={n})",
              flush=True)

    ladder = []
    for v in versions:
        nm = name(v)
        mean = ws_sum[nm] / cnt[nm] if cnt[nm] else float("nan")
        ladder.append((nm, v[0], v[1], mean))
    ladder.sort(key=lambda x: -x[3])  # strongest (highest mean win-share) first

    print("\n=== FULL VERSION LADDER (mean win_share vs field, STRONGEST first) ===")
    with open(args.out, "w") as f:
        f.write("version\tbot\tperiod\tmean_win_share\n")
        for nm, b, p, m in ladder:
            print(f"    {nm:14s}  {m:.3f}")
            f.write(f"{nm}\t{b}\t{p}\t{m:.4f}\n")
    print(f"\nwrote {args.out} ({len(ladder)} versions)")


if __name__ == "__main__":
    main()
