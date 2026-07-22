#!/usr/bin/env python3
"""Head-to-head: one of OUR checkpoints vs a ported foreign bot, in-engine.

Answers the question that decides whether training against a foreign opponent is
worth the compute: is it a real sparring partner or a punching bag? A bot far
weaker than the champion teaches the policy to beat something irrelevant.

Uses the same match accounting as scripts/matchwin_gate.py (split_matches /
match_record on match_mode rewards), so the win_share is directly comparable to
gate numbers. Our checkpoint always drives BLUE; the foreign bot drives ORANGE
(foreign opponents are 1v1-only, so every arena is 1v1).

Usage:
    .venv/bin/python scripts/bench_foreign.py checkpoints_entity/ck_000320471040.pt \
        --kind immortal --weights ~/.cache/construct/immortal_weights.npz
"""
import argparse
import os
import sys

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", help="our v1 checkpoint (drives blue)")
    ap.add_argument("--kind", default="immortal")
    ap.add_argument("--weights", default="~/.cache/construct/immortal_weights.npz")
    ap.add_argument("--arenas", type=int, default=32)
    ap.add_argument("--steps", type=int, default=45000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--period", type=int, default=1,
                    help="handicap: bot reacts every N decisions (1 = full strength)")
    args = ap.parse_args(argv)

    from construct._engine import Engine
    from construct.league.matches import load_sd, match_record, split_matches

    sd = load_sd(args.checkpoint)
    fw = {k: v.astype(np.float32)
          for k, v in np.load(os.path.expanduser(args.weights)).items()}

    eng = Engine(num_arenas=args.arenas, blue=1, orange=1,
                 schema_path="schema/v1.toml",
                 reward_config_path="configs/reward_v0.toml",
                 curriculum_config_path="configs/curriculum_v3_match.toml",
                 seed=args.seed, net_heads=4)
    eng.set_weights(sd)
    eng.set_foreign_opponents([fw], [args.kind], [args.period])
    # every arena driven by foreign slot 0 (encoded -2)
    out = eng.collect(args.steps, arena_opponents=[-2] * args.arenas)

    matches = split_matches(out["rewards"], out["terminated"])
    rec = match_record(matches)
    n = rec["wins"] + rec["draws"] + rec["losses"]
    if n == 0:
        print("no completed matches -- increase --steps")
        return 1
    share = (rec["wins"] + 0.5 * rec["draws"]) / n
    se = (share * (1 - share) / n) ** 0.5
    print(f"{args.checkpoint} (blue) vs {args.kind} (orange), period={args.period}")
    print(f"  {rec['wins']}W/{rec['draws']}D/{rec['losses']}L  n={n}  "
          f"win_share={share:.4f} +/- {se:.4f}")
    if share > 0.90:
        print("  -> PUNCHING BAG: the bot is far weaker; training against it "
              "teaches little.")
    elif share < 0.10:
        print("  -> WE ARE OUTCLASSED: the bot is far stronger.")
    else:
        print("  -> competitive sparring partner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
