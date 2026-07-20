"""One league tick: register newest checkpoints, play rating matches, print ladder.
Run manually or on a loop. Cheap: each match is headless engine time only.

Schema-aware: --schema-version 0 (default) is the legacy v0 MLP league
(checkpoints/ + checkpoints_b/, league/registry.jsonl by default). --schema-
version 1 is the entity-transformer league (checkpoints_entity/, league/
registry_v1.jsonl by default). --schema-version all runs BOTH pools in one
tick with the --matches budget split fairly between them (a pool that cannot
use its share rolls the leftover to the next pool). v0 and v1 policies never
share a match -- pairing is schema-pure by construction and
construct.league.matches.play_entries keeps a hard guard underneath.

All the logic lives in construct.league.tick; this file is just the CLI.
"""
import argparse
import random

from construct.league.tick import CHECKPOINT_SOURCES, run_tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", type=int, default=6,
                     help="total match budget for the tick, split across pools "
                          "when --schema-version all")
    ap.add_argument("--schema-version", default="0",
                     choices=[str(sv) for sv in sorted(CHECKPOINT_SOURCES)] + ["all"])
    ap.add_argument("--registry", default=None,
                     help="defaults to league/registry.jsonl (v0) or "
                          "league/registry_v1.jsonl (v1); with 'all', an explicit "
                          "path is shared by every pool (mixed-schema file)")
    ap.add_argument("--net-heads", type=int, default=4,
                     help="v1 only: attention head count the checkpoints were trained with")
    ap.add_argument("--seed", type=int, default=None,
                     help="seed match pairing/engine rng (default: fresh entropy per tick)")
    args = ap.parse_args()

    if args.schema_version == "all":
        svs = sorted(CHECKPOINT_SOURCES)
    else:
        svs = [int(args.schema_version)]
    run_tick(svs, args.matches, registry_path=args.registry,
             net_heads=args.net_heads, rng=random.Random(args.seed))


if __name__ == "__main__":
    main()
