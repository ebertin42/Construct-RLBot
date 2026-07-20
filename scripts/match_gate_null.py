#!/usr/bin/env python3
"""Measure the match-win gate's null distribution: a policy against ITSELF.

Why this must exist before any promotion is trusted: on 2026-07-20 a pure
random perturbation of the champion PASSED the 52% goal-share gate at 56.3%.
That gate had SE ~3.7%. The match-win gate is far noisier per unit compute
(~19 matches in today's budget, SE ~11.5%), so a threshold picked by intuition
would promote noise routinely.

A policy played against itself must centre on 0.5. The SPREAD across seeds is
the number that matters -- it sets the smallest win-share difference this gate
can resolve, and therefore where a defensible threshold sits.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys


def null_summary(shares):
    """Mean and spread of self-play win shares. sd/lo/hi are None below n=2,
    where a spread is undefined rather than zero."""
    shares = [float(s) for s in shares]
    n = len(shares)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(shares)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "lo": None, "hi": None}
    sd = statistics.stdev(shares)
    half = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", default="checkpoints_entity/ck_000320471040.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 31)))
    ap.add_argument("--arenas", type=int, default=32)
    ap.add_argument("--steps", type=int, default=45000,
                    help="engine steps per seed; a match is ~4500 steps")
    args = ap.parse_args(argv)

    from construct.league.matches import MatchRunner, load_sd, match_record, split_matches

    sd = load_sd(args.champion)
    shares = []
    for seed in args.seeds:
        mr = MatchRunner(num_arenas=args.arenas, seed=seed, mode=1,
                         schema_version=1, net_heads=4,
                         reward_config="configs/reward_v0.toml")
        mr.eng.set_weights(sd)
        mr.eng.set_opponents([sd])
        out = mr.eng.collect(args.steps, arena_opponents=mr.assignment)
        rec = match_record(split_matches(out["rewards"], out["terminated"]))
        if rec["win_share"] is not None:
            shares.append(rec["win_share"])
            print(f"  seed {seed:5d}: {rec['wins']}W/{rec['draws']}D/{rec['losses']}L "
                  f"share={rec['win_share']:.3f}", flush=True)

    s = null_summary(shares)
    if s["mean"] is None:
        print("\nNULL: no completed matches across any seed "
              "(match_mode not producing match outcomes -- see G1/G2 deployment gaps).")
        return 0
    print(f"\nNULL over {s['n']} seeds: mean={s['mean']:.4f}")
    if s["sd"] is not None:
        print(f"  sd={s['sd']:.4f}  95% CI of the mean=[{s['lo']:.4f},{s['hi']:.4f}]")
        print(f"  a defensible threshold sits at least 2 sd above 0.5, "
              f"i.e. >= {0.5 + 2 * s['sd']:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
