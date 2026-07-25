#!/usr/bin/env python3
"""Match-win gate: does a candidate WIN MORE FULL MATCHES than the champion?

The counterpart of the goal-share gate (champion_gate.py) for the match-win
campaign (task #56). A candidate is gated on the metric the objective is now
aligned with: fraction of full 300s matches won, not goal share.

Both side orders are mandatory (a single order swings the estimate), exactly as
the goal-share gate does. reward_v0 is forced as a neutral scoring tape and
curriculum_v3_match forces match_mode, so a `terminated` flag is a match
boundary and reward spikes past GOAL_THRESHOLD are goals -- independent of what
the candidate trained on.

Promotion threshold comes from the measured null (champion self-play): mean
0.5023, sd 0.0242 across 20 seeds x ~320 matches (journal 2026-07-21), so 0.55
sits ~2 sd above chance. A near-miss calls for more matches, never a lower bar.

THE CHAMPION MOVES. `--champion` defaults to `champion_ck` in
configs/champion.toml -- the project's single champion pointer -- not to a
hardcoded path, because a gate against a net we already beat 0.837 always passes
and measures nothing. `--promote-if-pass` moves that pointer on a PASS, through
champion_gate's atomic staged rewrite, and records `promoted` in the history row.
Promotion stays OPT-IN (see configs/champion.toml): a human sees every PASS
before the reference moves.

Deliberately NOT repointed at the moving champion: scripts/instrument_fingerprint.py
(proves engine bit-identity with a FIXED policy), scripts/match_gate_null.py and
perturb_null.py (null controls measured with that exact net), scripts/bench_ladder.py
(logs/opponent_ladder.tsv is champion-referenced). Those are pinned instruments;
repointing them would silently invalidate published numbers.

    matchwin_gate.py <candidate.pt> [--champion <ck>] [--arenas 32]
        [--steps 45000] [--seed 11] [--threshold 0.55] [--promote-if-pass]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Seeded champion, used only if configs/champion.toml is unreadable.
FALLBACK_CHAMPION = "checkpoints_entity/ck_000320471040.pt"
CHAMPION_CONFIG = "configs/champion.toml"


def _repo_root():
    return Path(__file__).resolve().parent.parent


def resolve_champion(config_path=None):
    """The current champion path from configs/champion.toml.

    Falls back to FALLBACK_CHAMPION (with a warning) rather than refusing to
    gate: a gate is a measurement, and losing the ability to measure because a
    pointer file moved is worse than measuring against the seeded reference.
    champion_gate.py keeps its own strict load_config -- it MUTATES the pointer,
    so guessing there would be wrong.
    """
    cfg = Path(config_path) if config_path else _repo_root() / CHAMPION_CONFIG
    try:
        import tomllib
        with cfg.open("rb") as f:
            ck = tomllib.load(f)["champion_ck"]
        return str(ck)
    except Exception as e:                                  # noqa: BLE001
        print(f"  (champion pointer unreadable [{cfg}]: {e}; "
              f"falling back to {FALLBACK_CHAMPION})")
        return FALLBACK_CHAMPION


def promote(candidate, config_path=None, add_to_league=True):
    """Move the champion pointer to `candidate` atomically, and enrol it in the
    league opponent pool so the CHAMPION ROTATES INTO TRAINING.

    Two separate effects, deliberately in this order:
      1. the pointer (what the next gate measures against) -- via champion_gate's
         staged write (temp file + os.replace), so a crash mid-promotion cannot
         leave a half-written pointer;
      2. the league registry (what the trainer plays against). Without this the
         pointer moves but the opponent pool never changes, so "rotating
         champion" would be rotating in name only.

    A registry failure does NOT undo the promotion -- the pointer is the
    authoritative record and the pool is a training convenience -- but it is
    reported. Returns the previous champion.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import champion_gate                                    # noqa: PLC0415
    cfg_path = Path(config_path) if config_path else _repo_root() / CHAMPION_CONFIG
    previous = resolve_champion(cfg_path)
    champion_gate.write_champion_ck(cfg_path, str(candidate))
    if add_to_league:
        try:
            cfg = champion_gate.load_config(cfg_path)
            steps, schema_version = _ck_provenance(candidate)
            champion_gate.add_to_league(cfg, str(candidate), steps, schema_version)
        except Exception as e:                              # noqa: BLE001
            print(f"  (champion pointer moved, but league enrolment failed: {e})")
    return previous


def _ck_provenance(ck):
    """(total_steps, schema_version) from a checkpoint. schema_version MUST be
    right: the trainer only ever considers pool entries tagged with its own
    schema version, since v0 and v1 policies cannot play each other (different
    obs). A mistagged entry is silently never selected."""
    import torch                                            # noqa: PLC0415
    d = torch.load(ck, map_location="cpu", weights_only=False)
    return int(d.get("total_steps", 0)), int(d.get("schema_version", 0))


def flip_to_candidate(matches):
    """Order 2 plays the CHAMPION in the learner row, so its (goals_a, goals_b)
    are (champion, candidate). Swap each pair so every match record is from the
    CANDIDATE's perspective before the two orders are summed. Getting this
    backwards silently inverts the verdict."""
    return [(b, a) for (a, b) in matches]


def aggregate(order1_wdl, order2_wdl, threshold):
    """Pure gate arithmetic over the two side orders, both already from the
    candidate's perspective. Draws count 0.5. `share` is None (never 0.0) when
    no match completed -- 0.0 would read as a total loss and could reject on
    zero evidence."""
    w = order1_wdl[0] + order2_wdl[0]
    d = order1_wdl[1] + order2_wdl[1]
    losses = order1_wdl[2] + order2_wdl[2]
    n = w + d + losses
    if n == 0:
        return {"wins": 0, "draws": 0, "losses": 0, "n": 0, "share": None,
                "se": None, "threshold": threshold, "verdict": "FAIL",
                "reason": "no completed matches"}
    share = (w + 0.5 * d) / n
    return {"wins": w, "draws": d, "losses": losses, "n": n, "share": share,
            "se": math.sqrt(0.25 / n), "threshold": threshold,
            "verdict": "PASS" if share >= threshold else "FAIL"}


def _play_order(champion_sd, candidate_sd, arenas, seed, steps, as_candidate_weights):
    """One side order. Returns (cand_wins, draws, cand_losses) over the matches
    played this order. `as_candidate_weights` True => candidate drives the
    learner row (its goals are the +spikes); False => champion does, and we
    flip the sign so the record is always from the CANDIDATE's perspective."""
    from construct.league.matches import MatchRunner, match_record, split_matches

    mr = MatchRunner(num_arenas=arenas, seed=seed, mode=1, schema_version=1,
                     net_heads=4, reward_config="configs/reward_v0.toml",
                     curriculum_config="configs/curriculum_v3_match.toml")
    if as_candidate_weights:
        mr.eng.set_weights(candidate_sd)
        mr.eng.set_opponents([champion_sd])
    else:
        mr.eng.set_weights(champion_sd)
        mr.eng.set_opponents([candidate_sd])
    out = mr.eng.collect(steps, arena_opponents=mr.assignment)
    matches = split_matches(out["rewards"], out["terminated"])
    if not as_candidate_weights:
        matches = flip_to_candidate(matches)
    rec = match_record(matches)
    return rec["wins"], rec["draws"], rec["losses"]


def gate(candidate, champion, arenas, steps, seed, threshold):
    from construct.league.matches import load_sd

    champ_sd = load_sd(champion)
    cand_sd = load_sd(candidate)
    o1 = _play_order(champ_sd, cand_sd, arenas, seed, steps, True)
    o2 = _play_order(champ_sd, cand_sd, arenas, seed + 1000, steps, False)
    r = aggregate(o1, o2, threshold)
    r["order1"], r["order2"] = o1, o2
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate")
    ap.add_argument("--champion", default=None,
                    help="default: champion_ck from configs/champion.toml")
    ap.add_argument("--arenas", type=int, default=32)
    ap.add_argument("--steps", type=int, default=45000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--promote-if-pass", action="store_true",
                    help="on PASS, move configs/champion.toml's champion_ck to the "
                         "candidate (atomic). Opt-in on purpose.")
    ap.add_argument("--champion-config", default=None,
                    help="override configs/champion.toml (tests)")
    args = ap.parse_args(argv)
    if args.champion is None:
        args.champion = resolve_champion(args.champion_config)

    r = gate(args.candidate, args.champion, args.arenas, args.steps, args.seed,
             args.threshold)
    print(f"candidate {args.candidate} vs champion {args.champion}")
    if r["share"] is None:
        print(f"  {r['reason']} -> {r['verdict']}")
        return 1
    print(f"  order1 (cand weights) W/D/L: {r['order1']}")
    print(f"  order2 (swapped)      W/D/L: {r['order2']}")
    print(f"  TOTAL {r['wins']}W/{r['draws']}D/{r['losses']}L  n={r['n']}  "
          f"win_share={r['share']:.4f} +/- {r['se']:.4f}")
    print(f"  verdict: {r['verdict']}  (threshold {r['threshold']:.3f}; "
          f"null mean 0.502 sd 0.024)")
    r["promoted"] = False
    if args.promote_if_pass and r["verdict"] == "PASS":
        try:
            was = promote(args.candidate, args.champion_config)
            r["promoted"] = True
            print(f"  PROMOTED: champion_ck {was} -> {args.candidate}")
        except Exception as e:                              # noqa: BLE001
            print(f"  promotion FAILED ({e}); champion pointer unchanged")
    elif args.promote_if_pass:
        print("  not promoted (verdict is not PASS)")
    _append_history(args, r)
    return 0


def _append_history(args, r):
    """One JSON line per gate to logs/matchwin_history.jsonl, so the dashboard's
    gate panel can show the absolute ruler over time instead of the number living
    only in whatever terminal ran it. Never fail the gate over bookkeeping."""
    import json
    import time
    from pathlib import Path
    row = {
        "ts": int(time.time()),
        "candidate": args.candidate, "champion": args.champion,
        "wins": r["wins"], "draws": r["draws"], "losses": r["losses"], "n": r["n"],
        "win_share": r["share"], "se": r["se"], "threshold": r["threshold"],
        "passed": r["verdict"] == "PASS",
        "promoted": bool(r.get("promoted")),
        "arenas": args.arenas, "steps": args.steps, "seed": args.seed,
    }
    try:
        p = Path(__file__).resolve().parent.parent / "logs" / "matchwin_history.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps(row) + "\n")
    except OSError as e:
        print(f"  (could not append gate history: {e})")


if __name__ == "__main__":
    sys.exit(main())
