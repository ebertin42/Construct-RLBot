"""One league tick: register newest checkpoints, play rating matches, print ladder.
Run manually or on a loop. Cheap: each match is headless engine time only.

Schema-aware: --schema-version 0 (default) is the legacy v0 MLP league
(checkpoints/ + checkpoints_b/, league/registry.jsonl by default). --schema-
version 1 is the entity-transformer league (checkpoints_entity/, league/
registry_v1.jsonl by default). v0 and v1 policies never share a registry file
or a match -- see construct.league.matches.play_entries's hard guard.
"""
import argparse
import glob
import random

from construct.league.matches import MatchRunner, load_sd, play_entries
from construct.league.registry import Registry
from construct.league.sampling import choose_opponents

# (run label, checkpoint glob) pairs scanned per schema version.
CHECKPOINT_SOURCES = {
    0: (("main", "checkpoints/ck_*.pt"), ("b", "checkpoints_b/ck_*.pt")),
    1: (("entity", "checkpoints_entity/ck_*.pt"),),
}
DEFAULT_REGISTRY_PATH = {0: "league/registry.jsonl", 1: "league/registry_v1.jsonl"}


def newest(pattern):
    cks = sorted(glob.glob(pattern))
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6)
    ap.add_argument("--schema-version", type=int, default=0, choices=sorted(CHECKPOINT_SOURCES))
    ap.add_argument("--registry", default=None,
                     help="defaults to league/registry.jsonl (v0) or "
                          "league/registry_v1.jsonl (v1)")
    ap.add_argument("--net-heads", type=int, default=4,
                     help="v1 only: attention head count the checkpoints were trained with")
    args = ap.parse_args()
    sv = args.schema_version

    registry_path = args.registry or DEFAULT_REGISTRY_PATH[sv]
    reg = Registry(path=registry_path)
    import torch
    for run, pattern in CHECKPOINT_SOURCES[sv]:
        ck = newest(pattern)
        if ck:
            meta = torch.load(ck, map_location="cpu", weights_only=False)
            reg.add(ck, steps=meta["total_steps"], run=run,
                    reward_config=meta.get("reward_config_path", "unknown"),
                    schema_version=sv)

    entries = reg.entries(schema_version=sv)
    if len(entries) >= 2:
        rng = random.Random()
        fresh = sorted(entries, key=lambda e: e["added_ts"])[-2:]
        mr = MatchRunner(num_arenas=8, seed=rng.randrange(1 << 30),
                          schema_version=sv, net_heads=args.net_heads)
        played = 0
        for member in fresh:
            for opp in choose_opponents(reg, k=3, rng=rng, schema_version=sv):
                if opp["ck"] == member["ck"] or played >= args.matches:
                    continue
                ga, gb = play_entries(mr, member, opp)
                reg.record_match(member["ck"], opp["ck"], ga, gb)
                print(f"{member['ck']}  {ga}:{gb}  {opp['ck']}")
                played += 1

    print(f"\n{'skill':>7}  {'mu':>6}  {'games':>5}  run   checkpoint")
    # defense-in-depth schema filter, in case --registry points at a mixed file
    ladder = [e for e in reg.ladder() if e["schema_version"] == sv]
    for e in ladder[:15]:
        print(f"{e['skill']:7.2f}  {e['mu']:6.2f}  {e['games']:5d}  {e['run']:<4}  {e['ck']}")


if __name__ == "__main__":
    main()
