"""One league tick: register newest checkpoints, play rating matches, print ladder.
Run manually or on a loop. Cheap: each match is headless engine time only."""
import argparse
import glob
import random

from construct.league.matches import MatchRunner, load_sd
from construct.league.registry import Registry
from construct.league.sampling import choose_opponents


def newest(pattern):
    cks = sorted(glob.glob(pattern))
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--registry", default="league/registry.jsonl")
    args = ap.parse_args()

    reg = Registry(path=args.registry)
    import torch
    for run, pattern in (("main", "checkpoints/ck_*.pt"), ("b", "checkpoints_b/ck_*.pt")):
        ck = newest(pattern)
        if ck:
            meta = torch.load(ck, map_location="cpu", weights_only=False)
            reg.add(ck, steps=meta["total_steps"], run=run,
                    reward_config=meta.get("reward_config_path", "unknown"))

    entries = reg.entries()
    if len(entries) >= 2:
        rng = random.Random()
        fresh = sorted(entries, key=lambda e: e["added_ts"])[-2:]
        mr = MatchRunner(num_arenas=8, seed=rng.randrange(1 << 30))
        played = 0
        for member in fresh:
            for opp in choose_opponents(reg, k=3, rng=rng):
                if opp["ck"] == member["ck"] or played >= args.matches:
                    continue
                ga, gb = mr.play(load_sd(member["ck"]), load_sd(opp["ck"]))
                reg.record_match(member["ck"], opp["ck"], ga, gb)
                print(f"{member['ck']}  {ga}:{gb}  {opp['ck']}")
                played += 1

    print(f"\n{'skill':>7}  {'mu':>6}  {'games':>5}  run   checkpoint")
    for e in reg.ladder()[:15]:
        print(f"{e['skill']:7.2f}  {e['mu']:6.2f}  {e['games']:5d}  {e['run']:<4}  {e['ck']}")


if __name__ == "__main__":
    main()
