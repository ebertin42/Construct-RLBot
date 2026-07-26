#!/usr/bin/env python3
"""Head-to-head: one of OUR checkpoints vs a ported foreign bot, in-engine.

Answers the question that decides whether training against a foreign opponent is
worth the compute: is it a real sparring partner or a punching bag? A bot far
weaker than the champion teaches the policy to beat something irrelevant.

Uses the same match accounting as scripts/matchwin_gate.py (split_matches /
match_record on match_mode rewards), so the win_share is directly comparable to
gate numbers. Our checkpoint always drives BLUE; the foreign bot drives ORANGE.
Every arena here is 1v1: this script has only ever built 1v1 arenas. Since
2026-07-26 the engine's guard is per KIND (nexto/necto take 2v2/3v3 arenas,
element/immortal do not), so a team-mode bench is now possible -- it is just not
wired here yet, and every historical number in logs/ is 1v1.

Usage:
    .venv/bin/python scripts/bench_foreign.py checkpoints_entity/ck_000320471040.pt \
        --kind immortal --weights ~/.cache/construct/immortal_weights.npz

The schema file is DERIVED FROM THE CHECKPOINT (construct.tables.schema_path),
not hardcoded: v9 launches on the 104-row v1-air action table and
`schema_version` is 1 for that AND for the 92-row v1.1 table, so a constant
"schema/v1.toml" here made the ruler unable to measure the one run that most
needs measuring. --schema overrides.

NECTO PROVENANCE (2026-07-26): the engine now clears necto's per-arena boost and
demo clocks on every episode reset, matching NectoObsBuilder.reset(). That is a
fidelity fix, and it means every necto row measured BEFORE 2026-07-26 was scored
on a build that leaked those clocks across resets and is NOT comparable to a row
measured after the v9 wheel is installed. Re-measure necto (only necto -- nexto,
element and immortal are byte-identical) before ranking it against an old number.
See engine/src/foreign.rs::reset_car for the measurement.
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
    ap.add_argument("--schema", default=None,
                    help="override the schema file; default is derived from the "
                         "checkpoint's own action table (schema/v1.toml for "
                         "construct_92_v1, schema/v1_air.toml for construct_104_v1air)")
    args = ap.parse_args(argv)

    import torch

    from construct._engine import Engine
    from construct.league.matches import load_sd, match_record, split_matches
    from construct.tables import schema_path, table_name

    sd = load_sd(args.checkpoint)
    fw = {k: v.astype(np.float32)
          for k, v in np.load(os.path.expanduser(args.weights)).items()}

    # THE SCHEMA COMES FROM THE CHECKPOINT, not from a constant. It used to be a
    # hardcoded "schema/v1.toml", which made this script -- the project's
    # absolute ruler (`champion-unbeaten-external-lever`, the only external
    # measurement that has ever been decisive) -- silently 92-row-only: on a
    # 104-row v1-air checkpoint the very next line, set_weights, dies on the
    # engine's cross-table guard. schema_version cannot be used to pick, since
    # it is 1 for both tables. --schema is the manual override.
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    schema = args.schema or schema_path(ck)

    eng = Engine(num_arenas=args.arenas, blue=1, orange=1,
                 schema_path=schema,
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
    print(f"{args.checkpoint} (blue) vs {args.kind} (orange), period={args.period}, "
          f"schema={schema} ({table_name(ck) or 'v0'})")
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
