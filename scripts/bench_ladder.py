"""Map the full opponent difficulty ladder: every (bot, handicap-period) "version"
measured by a reference policy's win_share, then sorted easiest -> hardest.

Purpose: design a SMOOTH auto-curriculum progression. Instead of "ramp Element's
period, then jump to Immortal", we enumerate all (bot x period) versions and order
them by difficulty, so the curriculum can move continuously across the whole space
(e.g. Element p1 might sit between Immortal p8 and Necto p12 in difficulty).

Reference = the champion (our strongest known policy; it spans the range -- beats
weak versions ~1.0, loses to strong ones ~0.0). The ladder is therefore "how hard
is each version for a champion-level policy", which is what the from-scratch net
will climb as it approaches and exceeds champion level.

Runs on an IDLE box (training is remote). Reuses one engine per bot for speed.
"""
import argparse
import numpy as np

CHAMPION = "checkpoints_entity/ck_000320471040.pt"
WEIGHTS = {
    "element": "~/.cache/construct/element_weights.npz",
    "immortal": "~/.cache/construct/immortal_weights.npz",
    "necto": "~/.cache/construct/necto_weights.npz",
    "nexto": "~/.cache/construct/nexto_weights.npz",
}


def main():
    import os
    from construct._engine import Engine
    from construct.league.matches import load_sd, match_record, split_matches

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=CHAMPION)
    ap.add_argument("--bots", default="element,immortal,necto,nexto")
    ap.add_argument("--periods", default="1,2,3,4,6,8,12")
    ap.add_argument("--arenas", type=int, default=12)
    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--out", default="logs/opponent_ladder.tsv")
    args = ap.parse_args()

    ref = load_sd(args.reference)
    bots = args.bots.split(",")
    periods = [int(p) for p in args.periods.split(",")]
    rows = []

    for bot in bots:
        wpath = os.path.expanduser(WEIGHTS[bot])
        if not os.path.exists(wpath):
            print(f"SKIP {bot}: weights absent ({wpath})", flush=True)
            continue
        fw = {k: v.astype(np.float32) for k, v in np.load(wpath).items()}
        eng = Engine(num_arenas=args.arenas, blue=1, orange=1,
                     schema_path="schema/v1.toml",
                     reward_config_path="configs/reward_v0.toml",
                     curriculum_config_path="configs/curriculum_v3_match.toml",
                     seed=11, net_heads=4)
        eng.set_weights(ref)
        for pd in periods:
            eng.set_foreign_opponents([fw], [bot], [pd])
            out = eng.collect(args.steps, arena_opponents=[-2] * args.arenas)
            rec = match_record(split_matches(out["rewards"], out["terminated"]))
            n = rec["wins"] + rec["draws"] + rec["losses"]
            ws = (rec["wins"] + 0.5 * rec["draws"]) / n if n else float("nan")
            rows.append((f"{bot}:p{pd}", bot, pd, ws, n))
            print(f"  {bot:9s} p{pd:<2d}  win_share {ws:.3f}  (n={n})", flush=True)

    # sort EASIEST (ref wins most) -> HARDEST (ref loses)
    rows.sort(key=lambda r: -r[3])
    print("\n=== OPPONENT DIFFICULTY LADDER (reference win_share, easiest -> hardest) ===")
    with open(args.out, "w") as f:
        f.write("version\tbot\tperiod\tref_win_share\tn\n")
        for name, bot, pd, ws, n in rows:
            print(f"  {name:14s}  ref_ws {ws:.3f}")
            f.write(f"{name}\t{bot}\t{pd}\t{ws:.4f}\t{n}\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
