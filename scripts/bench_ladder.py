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

NECTO PROVENANCE (2026-07-26): the engine now clears necto's per-arena boost and
demo clocks on every episode reset, matching NectoObsBuilder.reset(). That is a
fidelity fix, and it means every necto row measured BEFORE 2026-07-26 was scored
on a build that leaked those clocks across resets and is NOT comparable to a row
measured after the v9 wheel is installed. Re-measure necto (only necto -- nexto,
element and immortal are byte-identical) before ranking it against an old number.
See engine/src/foreign.rs::reset_car for the measurement.
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


def _engine_fingerprint():
    """md5 + mtime of the installed engine .so -- the cheap version of
    scripts/instrument_fingerprint.py. All it has to do is let a reader tell two
    ladder files apart when the build moved under them; necto is the reason (its
    rows changed on 2026-07-26 and nothing in the TSV said so)."""
    import hashlib
    import pathlib
    import construct
    so = next((q for q in pathlib.Path(construct.__file__).parent.glob("_engine*.so")), None)
    if so is None:
        return "unknown"
    return f"{so.name} md5:{hashlib.md5(so.read_bytes()).hexdigest()[:12]} mtime:{int(so.stat().st_mtime)}"

def main():
    import os

    import torch

    from construct._engine import Engine
    from construct.league.matches import load_sd, match_record, split_matches
    from construct.tables import schema_path

    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default=CHAMPION)
    ap.add_argument("--bots", default="element,immortal,necto,nexto")
    ap.add_argument("--periods", default="1,2,3,4,6,8,12")
    ap.add_argument("--arenas", type=int, default=12)
    ap.add_argument("--steps", type=int, default=18000)
    ap.add_argument("--out", default="logs/opponent_ladder.tsv")
    args = ap.parse_args()

    ref = load_sd(args.reference)
    # Schema from the REFERENCE checkpoint, same reason as bench_foreign.py: a
    # hardcoded schema/v1.toml makes this 92-row-only, and set_weights below
    # would die on the engine's cross-table guard for a v1-air reference.
    ref_schema = schema_path(
        torch.load(args.reference, map_location="cpu", weights_only=False)
    )
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
                     schema_path=ref_schema,
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
        # Stamp the instrument INTO the file: a ladder TSV outlives the shell it
        # was produced in, and necto's rows are only comparable within one engine
        # build (see the NECTO PROVENANCE note above).
        f.write(f"# engine {_engine_fingerprint()}\n")
        f.write("version\tbot\tperiod\tref_win_share\tn\n")
        for name, bot, pd, ws, n in rows:
            print(f"  {name:14s}  ref_ws {ws:.3f}")
            f.write(f"{name}\t{bot}\t{pd}\t{ws:.4f}\t{n}\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
