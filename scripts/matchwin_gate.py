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

    matchwin_gate.py <candidate.pt> [--champion <ck>] [--arenas 32]
        [--steps 45000] [--seed 11] [--threshold 0.55]
"""
from __future__ import annotations

import argparse
import math
import sys


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
        matches = [(b, a) for (a, b) in matches]  # flip to candidate perspective
    rec = match_record(matches)
    return rec["wins"], rec["draws"], rec["losses"]


def gate(candidate, champion, arenas, steps, seed, threshold):
    from construct.league.matches import load_sd

    champ_sd = load_sd(champion)
    cand_sd = load_sd(candidate)
    w1, d1, l1 = _play_order(champ_sd, cand_sd, arenas, seed, steps, True)
    w2, d2, l2 = _play_order(champ_sd, cand_sd, arenas, seed + 1000, steps, False)
    wins, draws, losses = w1 + w2, d1 + d2, l1 + l2
    n = wins + draws + losses
    if n == 0:
        return {"n": 0, "share": None, "verdict": "FAIL", "reason": "no completed matches"}
    share = (wins + 0.5 * draws) / n
    se = math.sqrt(0.25 / n)
    verdict = "PASS" if share >= threshold else "FAIL"
    return {"wins": wins, "draws": draws, "losses": losses, "n": n,
            "share": share, "se": se, "threshold": threshold, "verdict": verdict,
            "order1": (w1, d1, l1), "order2": (w2, d2, l2)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidate")
    ap.add_argument("--champion", default="checkpoints_entity/ck_000320471040.pt")
    ap.add_argument("--arenas", type=int, default=32)
    ap.add_argument("--steps", type=int, default=45000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--threshold", type=float, default=0.55)
    args = ap.parse_args(argv)

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
