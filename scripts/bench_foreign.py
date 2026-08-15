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

# A healthy match is 300 s of a 15 Hz agent clock. Kept in sync with
# scripts/matchwin_gate.py, which has censused blowups against it since the containment
# landed; split_matches' own docstring quotes the same 4500.
FULL_MATCH_STEPS = 4500


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

    # BLOWUP CENSUS. `matchwin_gate.py` has done this since the containment landed and this
    # script never did -- which is how a silently corrupted cell reached a published number
    # on 2026-08-09. A contained physics blowup (engine/src/episode.rs) terminates every car
    # in the arena and rebuilds it, so split_matches closes a record tens of steps into a
    # match instead of at FULL_MATCH_STEPS. One arena stuck in that loop shreds its 300 s
    # matches into ~6 s fragments, and since goals are reported PER MATCH the inflated
    # denominator drives goals_for and goals_against toward zero together while win_share
    # collapses to ~0.5 (fragments are mostly goalless, hence draws).
    #
    # Observed: ck_004070758400 vs necto read goals 0.1794/0.0882 win_share 0.5083 in one run
    # and 8.1451/3.9321 win_share 0.8935 in the next, SAME checkpoint, same seed -- the
    # engine is not bit-reproducible under ASLR, so this lands on some cells and not others.
    # It is therefore invisible to any single re-run and can only be caught by the census.
    #
    # A short record is NOT a 0-0 draw and is NOT dropped here: the blowup branch preserves
    # the score accumulated so far, so a blowup at 2-1 emits (2,1), a decided result. Counted
    # and reported, never silently filtered -- filtering would also move every historical
    # number this project has published.
    matches, durations = split_matches(out["rewards"], out["terminated"],
                                       with_durations=True)
    rec = match_record(matches)
    n = rec["wins"] + rec["draws"] + rec["losses"]
    if n == 0:
        print("no completed matches -- increase --steps")
        return 1
    short = sum(1 for d in durations if d < FULL_MATCH_STEPS)
    short_frac = short / len(durations)
    share = (rec["wins"] + 0.5 * rec["draws"]) / n
    se = (share * (1 - share) / n) ** 0.5
    print(f"{args.checkpoint} (blue) vs {args.kind} (orange), period={args.period}, "
          f"schema={schema} ({table_name(ck) or 'v0'})")
    print(f"  {rec['wins']}W/{rec['draws']}D/{rec['losses']}L  n={n}  "
          f"win_share={share:.4f} +/- {se:.4f}")
    # GOALS, because win_share IS CENSORED AT THE FLOOR and has already hit the bound.
    # At period 1 this bot reads 0W/0D/320L against all four opponents, so win_share is
    # exactly 0.0000 and carries no information about HOW badly we lose -- 1-0 and 9-0 are
    # the same number. `split_matches` has always returned (goals_for, goals_against) per
    # match and this script threw them away. goals-against is also the tighter channel by a
    # wide margin (CV 0.033-0.060 vs win_share's 0.217-0.307), so near a floor it is the
    # only signal with usable power. See memory: goals-against-is-the-gate,
    # absolute-strength-zero-at-period-1.
    gf = [m[0] for m in matches]
    ga = [m[1] for m in matches]
    mf, mg = sum(gf) / n, sum(ga) / n
    sef = (sum((x - mf) ** 2 for x in gf) / (n - 1) / n) ** 0.5 if n > 1 else 0.0
    seg = (sum((x - mg) ** 2 for x in ga) / (n - 1) / n) ** 0.5 if n > 1 else 0.0
    print(f"  goals_for={mf:.4f} +/- {sef:.4f}  goals_against={mg:.4f} +/- {seg:.4f}  "
          f"diff={mf - mg:+.4f}")
    print(f"  shutouts_against={sum(1 for x in gf if x == 0)}/{n}  "
          f"clean_sheets={sum(1 for x in ga if x == 0)}/{n}")
    # Machine-readable and on its own line so every harness can grep one key. The threshold
    # is deliberately tight: at 5% short records the per-match denominator is already
    # inflated enough to move goals_against by more than the 0.09 MDE these benches run at.
    print(f"  short_frac={short_frac:.4f}  short={short}/{len(durations)}"
          f"{'  CONTAMINATED' if short_frac > 0.05 else ''}")
    # FULL-MATCH-ONLY GOALS. The figures above divide by EVERY record, so a handful of arenas
    # stuck in a blowup loop can supply hundreds of ~6 s fragments and drag goals-per-match
    # toward zero -- necto against a nexto-clone read short_frac 0.82 and goals 1.76/0.66 on
    # 2026-08-15 for exactly that reason. A full match is a complete 300 s observation, so
    # restricting to `duration >= FULL_MATCH_STEPS` is the estimator that answers "how does a
    # match between these two go".
    #
    # It is printed ALONGSIDE, never instead of, the all-records figures: those are what every
    # historical number was computed on, and silently switching the definition would make new
    # rows incomparable with the series they are appended to -- the same trap that made a
    # column-index change produce a confident t=+19.76 once. Cross-era comparisons use the
    # all-records column; new analysis should prefer _full.
    #
    # Dropping short records is not free of assumptions: a blowup at 2-1 emits (2,1), so the
    # discarded partials carry real goals. What survives is an unbiased sample of FULL-match
    # play, which is the quantity of interest; what it cannot tell you is whether blowups
    # correlate with score, so a high short_frac still means "re-run this cell", not
    # "just read the _full column".
    full = [(g, a) for (g, a), d in zip(matches, durations) if d >= FULL_MATCH_STEPS]
    if full:
        nfull = len(full)
        mff = sum(g for g, _ in full) / nfull
        mgf = sum(a for _, a in full) / nfull
        seff = (sum((g - mff) ** 2 for g, _ in full) / (nfull - 1) / nfull) ** 0.5 \
            if nfull > 1 else 0.0
        segf = (sum((a - mgf) ** 2 for _, a in full) / (nfull - 1) / nfull) ** 0.5 \
            if nfull > 1 else 0.0
        print(f"  goals_for_full={mff:.4f} +/- {seff:.4f}  "
              f"goals_against_full={mgf:.4f} +/- {segf:.4f}  n_full={nfull}")
    else:
        print("  goals_for_full=NA  goals_against_full=NA  n_full=0"
              "   [every record was short -- no full match completed]")
    if short_frac > 0.05:
        print(f"  -> DO NOT USE THIS CELL. {short}/{len(durations)} records closed before "
              f"{FULL_MATCH_STEPS} steps (contained physics blowups), so goals-per-match "
              f"are divided by an inflated match count. Re-run it; the failure is "
              f"nondeterministic.")
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
