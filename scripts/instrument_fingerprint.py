"""Byte-level fingerprint of the gate instrument (the installed engine .so).

WHY: every gate result (null mean 0.502 sd 0.024, threshold 0.55, the champion
comparisons) was scored on one specific build of the engine. Any rebuild risks
silently changing that instrument -- see docs/training-journal.md's G2 bit-identity
episode, where a new engine shifted goal-share and forced a re-baseline.

This script replicates scripts/matchwin_gate.py::_play_order's EXACT MatchRunner
construction (same schema/net_heads/reward/curriculum), plays champion-vs-champion
at a fixed seed, and hashes every output array. Run it BEFORE and AFTER a rebuild:
identical hashes prove the instrument is unchanged and existing gate baselines
remain comparable.

YOU MUST RUN THIS UNDER A FIXED MEMORY LAYOUT (`setarch $(uname -m) -R`).

The engine is NOT bit-reproducible across processes under normal ASLR: RocketSim's
physics depends on memory layout, so the same seed can diverge run to run (measured
2026-07-22: 3 distinct hashes in 3 runs of the same build; the contained
physics-blowup tick itself moved between 440 and 648). Disabling ASLR makes it
deterministic (7/7 identical). Without `setarch -R` a "changed" result means
nothing and an "unchanged" result is luck.

Usage:
    setarch $(uname -m) -R .venv/bin/python scripts/instrument_fingerprint.py --save X
    # ... rebuild ...
    setarch $(uname -m) -R .venv/bin/python scripts/instrument_fingerprint.py --check X

Note this only pins BIT-reproducibility. Gate verdicts stay statistically valid
without it (n~640 matches, and the null baseline was measured empirically so it
already absorbs this variance) -- they are simply not bit-repeatable.
"""
import argparse, hashlib, json, pathlib, sys

import numpy as np

CHAMPION = "checkpoints_entity/ck_000320471040.pt"


def fingerprint(arenas=4, steps=3000, seed=11):
    from construct.league.matches import MatchRunner, load_sd

    sd = load_sd(CHAMPION)
    mr = MatchRunner(num_arenas=arenas, seed=seed, mode=1, schema_version=1,
                     net_heads=4, reward_config="configs/reward_v0.toml",
                     curriculum_config="configs/curriculum_v3_match.toml")
    mr.eng.set_weights(sd)
    mr.eng.set_opponents([sd])
    out = mr.eng.collect(steps, arena_opponents=mr.assignment)

    digests = {}
    for key in sorted(out.keys()):
        a = np.ascontiguousarray(np.asarray(out[key]))
        digests[key] = {
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "shape": list(a.shape),
            "dtype": str(a.dtype),
        }
    combined = hashlib.sha256(
        "".join(f"{k}:{v['sha256']}" for k, v in digests.items()).encode()
    ).hexdigest()
    return {"arenas": arenas, "steps": steps, "seed": seed,
            "champion": CHAMPION, "arrays": digests, "combined": combined}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arenas", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--save")
    ap.add_argument("--check")
    args = ap.parse_args()

    # Loud warning rather than a hard failure: someone may deliberately want a
    # single informational fingerprint. But a --check verdict without a fixed
    # layout is meaningless, so say so.
    import os
    if not os.environ.get("CONSTRUCT_FP_ALLOW_ASLR"):
        try:
            with open("/proc/sys/kernel/randomize_va_space") as f:
                aslr_on = f.read().strip() != "0"
        except OSError:
            aslr_on = True
        # setarch -R sets the process personality; detect via /proc/self/personality
        try:
            with open("/proc/self/personality") as f:
                fixed = (int(f.read().strip(), 16) & 0x0040000) != 0
        except OSError:
            fixed = False
        if aslr_on and not fixed:
            print("WARNING: running WITHOUT a fixed memory layout. The engine is not "
                  "bit-reproducible under ASLR; re-run under "
                  "`setarch $(uname -m) -R` or this comparison is meaningless.\n")

    fp = fingerprint(args.arenas, args.steps, args.seed)
    print(f"combined: {fp['combined']}")
    for k, v in fp["arrays"].items():
        print(f"  {k:14s} {v['dtype']:>8s} {str(v['shape']):>16s}  {v['sha256'][:16]}")

    if args.save:
        pathlib.Path(args.save).write_text(json.dumps(fp, indent=2))
        print(f"saved -> {args.save}")

    if args.check:
        old = json.loads(pathlib.Path(args.check).read_text())
        if old["combined"] == fp["combined"]:
            print(f"\nINSTRUMENT UNCHANGED (combined matches {args.check})")
            return 0
        print(f"\n*** INSTRUMENT CHANGED *** vs {args.check}")
        for k in sorted(set(old["arrays"]) | set(fp["arrays"])):
            o = old["arrays"].get(k, {}).get("sha256")
            n = fp["arrays"].get(k, {}).get("sha256")
            if o != n:
                print(f"  DIFFERS: {k}  old {str(o)[:16]}  new {str(n)[:16]}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
