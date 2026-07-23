"""Bot-vs-bot round-robin: rank the ported community bots against EACH OTHER
(and across handicap periods), decided by the bots themselves rather than by our
champion (which loses 0.0 to every full bot and can't discriminate them).

Uses the in-engine BotMatch (both cars foreign-driven). Prints a win-share matrix
and a single strength ranking (mean win-share vs the field), plus a period sweep
so handicapped versions sit on the same scale for the ladder.
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

    W = {
        "element": "~/.cache/construct/element_weights.npz",
        "immortal": "~/.cache/construct/immortal_weights.npz",
        "necto": "~/.cache/construct/necto_weights.npz",
        "nexto": "~/.cache/construct/nexto_weights.npz",
    }
    ap = argparse.ArgumentParser()
    ap.add_argument("--bots", default="element,immortal,necto,nexto")
    ap.add_argument("--arenas", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20000)
    args = ap.parse_args()

    bots = [b for b in args.bots.split(",") if os.path.exists(os.path.expanduser(W[b]))]
    wt = {b: {k: v.astype(np.float32) for k, v in np.load(os.path.expanduser(W[b])).items()}
          for b in bots}

    # --- full-strength round robin (period 1 both sides) ---
    print("=== FULL-STRENGTH ROUND ROBIN (row beats col, win_share) ===")
    mat = {a: {} for a in bots}
    for a, b in itertools.combinations(bots, 2):
        bm = BotMatch(a, wt[a], 1, b, wt[b], 1, arenas=args.arenas)
        r, t = bm.run(args.steps)
        ws, n = winshare(r, t)
        mat[a][b] = ws
        mat[b][a] = 1.0 - ws if ws == ws else float("nan")
        print(f"  {a:9s} vs {b:9s}: {ws:.3f}  (n={n})", flush=True)
    for a in bots:
        mat[a][a] = 0.5
    print("\n  matrix:")
    hdr = "           " + "".join(f"{b[:6]:>8s}" for b in bots)
    print(hdr)
    for a in bots:
        print(f"  {a:9s}" + "".join(f"{mat[a].get(b, float('nan')):8.3f}" for b in bots))
    print("\n  STRENGTH RANKING (mean win_share vs field, strongest first):")
    strength = sorted(bots, key=lambda a: -np.nanmean([mat[a][b] for b in bots if b != a]))
    for a in strength:
        m = np.nanmean([mat[a][b] for b in bots if b != a])
        print(f"    {a:9s}  {m:.3f}")

    # --- period sweep: strongest bot handicapped vs weakest at full strength ---
    if len(strength) >= 2:
        strong, weak = strength[0], strength[-1]
        print(f"\n=== HANDICAP SCALE: {strong} at period P  vs  {weak} full (p1) ===")
        for pd in [1, 2, 3, 4, 6, 8, 12]:
            bm = BotMatch(strong, wt[strong], pd, weak, wt[weak], 1, arenas=args.arenas)
            r, t = bm.run(args.steps)
            ws, n = winshare(r, t)
            print(f"  {strong} p{pd:<2d} vs {weak} full: {ws:.3f}  (n={n})", flush=True)


if __name__ == "__main__":
    main()
