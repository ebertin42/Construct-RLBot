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

TEAM SIZES (--mode 1|2|3). This gate is NET vs NET (Engine.set_opponents), and
that is precisely why team gating is possible here and nowhere else: our entity
net is team-size agnostic, while bench_foreign / bench_ladder / version_ladder /
bot_tournament are structurally 1v1 because the ported bots' obs width is fixed
at one other car and the engine refuses them above 1v1
(engine/src/engine.rs:319-325). Four things to know before reading a team number:

  * n DOES NOT GROW WITH m. Matches per side order are arenas * floor(steps/4500)
    at every team size, so 2v2 buys no extra samples -- only ~1.5x (m=2) to ~2x
    (m=3) more wall clock for the same SE.
  * SHORT RECORDS ARE CENSUSED AND REPORTED, at every mode -- 1v1 is not immune.
    A contained physics blowup (engine/src/episode.rs:779-807) terminates the
    arena and rebuilds it, so a record closes tens of steps into a match instead
    of at 4500. It is NOT a 0-0 draw, and treating it as one is how this census
    was wrong before: the blowup branch zeroes only THAT STEP's reward and
    returns before the `if terminated { ... start_match() }` block, so nothing
    resets the score -- split_matches' accumulators still hold every goal scored
    earlier in the match, and a blowup after 2-1 emits a WIN (seen live: a real
    m=2 tape returned a short record of (0,1), a loss). The contamination
    therefore has no known sign, and a short record enters n as if it were a
    completed match, so se = sqrt(0.25/n) is quoted on a count of RECORDS rather
    than of matches -- which is why n_full is printed beside n. Rates measured:
    0-1.6% at m=1 and 3-22% at m=2/3 on 32 arenas x 9000 steps -- but 29% (m=1)
    and 62% (m=2) on a 4-arena run on a LOADED box, so the rate is a property of
    the run, not of the team size, and is not reproducible for a fixed seed.
    They are counted, never filtered: filtering would also move the 1v1 numbers
    and break comparability with every published gate result. Read a heavy
    census as "re-run on an idle box", never as a close call to correct for.
  * PROMOTION IS REFUSED ABOVE 1v1, at argparse time (see main).
  * DETERMINISM DOES NOT HOLD ABOVE 1v1. Arenas with 2+ cars per team are not
    bit-reproducible across independently constructed engines even at a fixed
    seed: RocketSim's ResetToRandomKickoff groups same-team cars by iterating a
    std::unordered_set<Car*>, so kickoff slot assignment follows heap addresses
    (tests/python/test_team_curriculum.py:28-50). 1v1 has no grouping ambiguity
    and stays reproducible. At --mode 2/3 a fixed --seed is a SAMPLE, not a
    fingerprint -- do not quote two team runs at one seed as a repeat measurement.

    matchwin_gate.py <candidate.pt> [--champion <ck>] [--arenas 32]
        [--steps 45000] [--seed 11] [--threshold 0.55] [--mode 1]
        [--promote-if-pass]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

# Seeded champion, used only if configs/champion.toml is unreadable.
FALLBACK_CHAMPION = "checkpoints_entity/ck_000320471040.pt"
CHAMPION_CONFIG = "configs/champion.toml"

# A full match is 300s at 120Hz with tick_skip 8 -> MAX_TICKS/tick_skip
# (engine/src/episode.rs:21). Any record shorter than this did not end on the
# clock; the only other producer of `terminated` is the contained-blowup path,
# which rebuilds the arena but does NOT reset the match score or restart the
# match clock (it returns before start_match()), so the record it closes carries
# the partial score rather than 0-0. Used for the census, never as a filter.
FULL_MATCH_STEPS = 4500

# The null is a property of a POLICY and a REGIME, so it cannot be borrowed
# across team sizes. 3v3 is deliberately absent rather than guessed -- printing
# a 2v2 null next to a 3v3 share would launder an uncalibrated verdict. Run
# scripts/match_gate_null.py --mode M --both-orders to fill it in.
#
#   1v1: champion self-play, 20 seeds x ~320 matches, single order
#        (journal 2026-07-21).
#   2v2: v9 ck_000572549120 self-play, 20 seeds x 640 matches, BOTH orders
#        (2026-07-27, logs/null_2v2.log). sd 0.0149 against an expected 0.0175,
#        inside the 95% chi-square band [0.0120, 0.0230]; mean 0.5020 with CI
#        [0.4954, 0.5085]; mirror symmetry 0.4986, CI [0.4927, 0.5046]; 0/12800
#        short records. NOT the duplicate-per-car signature, which would sit at
#        expected/sqrt(2) = 0.0124 AND would have reported 1280 matches/seed
#        rather than the correct 640 -- the match count is the check that
#        distinguishes them, not the sd alone, since 0.0124 also falls inside
#        the band.
#        Measured on a box running the viewer, dashboard and sync rather than an
#        idle one. For a NULL that is defensible where it would not be for a
#        gate: both-orders centres the mean on 0.5 by construction, so the
#        `gates-need-an-idle-box` win-share bias cancels, and load can only
#        inflate the sd -- which came in BELOW expectation, not above.
#
# The 2v2 sd is smaller than 1v1's because both-orders halves the variance
# (sqrt(2)) and 640 matches is 2x the 1v1 n. A 2v2 gate therefore needs a
# SMALLER margin to be significant, not a larger one: >= 0.530 at 2 sd.
NULL_BY_MODE = {1: (0.502, 0.024), 2: (0.502, 0.0149)}


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
            steps, schema_version, action_table = _ck_provenance(candidate)
            champion_gate.add_to_league(cfg, str(candidate), steps, schema_version,
                                        action_table=action_table)
        except Exception as e:                              # noqa: BLE001
            print(f"  (champion pointer moved, but league enrolment failed: {e})")
    return previous


def _ck_provenance(ck):
    """(total_steps, schema_version, action_table) from a checkpoint.

    schema_version MUST be right: the trainer only ever considers pool entries
    tagged with its own schema version, since v0 and v1 policies cannot play
    each other (different obs). A mistagged entry is silently never selected.

    action_table is the SAME hazard one level down and is not implied by the
    version -- that is 1 for both the 92-row `construct_92_v1` and the 104-row
    `construct_104_v1air`. `None` at v0 (no table name was ever recorded
    there), which add_to_league reads as "use the default"."""
    import torch                                            # noqa: PLC0415
    from construct.tables import table_name                 # noqa: PLC0415
    d = torch.load(ck, map_location="cpu", weights_only=False)
    return int(d.get("total_steps", 0)), int(d.get("schema_version", 0)), table_name(d)


def flip_to_candidate(matches):
    """Order 2 plays the CHAMPION in the learner rows (the whole blue team, at
    any team size), so its (goals_a, goals_b) are (champion, candidate). Swap
    each pair so every match record is from the CANDIDATE's perspective before
    the two orders are summed. Getting this backwards silently inverts the
    verdict. Team-size free: split_matches has already reduced each arena's m
    cars to one record per match."""
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


def _short_record_line(r):
    """The blowup census, printed at EVERY mode -- 1v1 is NOT immune. A raw
    data-quality flag on the run, deliberately with NO correction attached.

    A contained physics blowup (engine/src/episode.rs:779-807) terminates every
    car in the arena and rebuilds it, so split_matches closes a record tens of
    steps into a match instead of at 4500.

    WHAT A SHORT RECORD IS NOT: a 0-0 draw. This census used to say it was, and
    concluded from that "contamination is all draws, so it pulls the share toward
    0.5 and can never cause a false promotion". episode.rs says otherwise. The
    blowup branch sets `rewards[a] = 0.0` for the CURRENT STEP only and then
    `return`s -- before the `if terminated || truncated { ... start_match() }`
    block -- so score_blue/score_orange are never reset, and, what actually
    matters here, split_matches' own Python-side accumulators still hold every
    goal scored earlier in that match (in match mode a goal does NOT terminate,
    so nothing has flushed them). The emitted record is the partial match's real
    score: a blowup at 2-1 emits (2,1), a WIN. Nor is this a corner case -- the
    blowup this containment is named for is the mirrored-KICKOFF pinch, and in
    match mode a kickoff follows every goal, so blowups concentrate precisely
    where goals have already accumulated.

    Measured, not argued: real champion-self-play tapes (4 arenas, m=2) returned
    short records (0,0)@10 steps, (0,0)@22 and (0,1)@90 -- two draws and a LOSS.
    One decided result is all it takes; "all draws" was never true.

    Two consequences, and neither has a sign we can claim:
      * the contaminating records are ordinary wins/draws/losses drawn from a
        DIFFERENT (truncated-match) distribution, so the direction of the bias is
        unknown, not "toward 0.5";
      * a short record is a match that never finished, yet it enters n as if it
        had, so se = sqrt(0.25/n) is quoted on a denominator that is a count of
        RECORDS rather than of completed matches. (Whether the total yield also
        grows depends on where the shifted boundary lands: on that same tape,
        with --steps a whole number of matches, each short record DISPLACED a
        full one and the yield stayed at nominal.)
    Hence n_full is printed next to n, and a heavy census means RE-RUN ON AN IDLE
    BOX -- never "close call, correct for it".

    Durations are a census key, not a measurement of how much match was lost:
    rebuild_arena restarts the fresh arena's tick_count at 0 while
    match_start_tick keeps its old value, so after the FIRST match the
    `cur.tick_count - self.match_start_tick` at episode.rs:852 underflows (u64)
    and the next step terminates immediately. Pre-existing engine behaviour,
    outside this gate, but it is why a duration is only ever read as
    "< FULL_MATCH_STEPS" here and never as a quantity.

    THE RATE IS NOT A CONSTANT AND NOT REPRODUCIBLE. 0-1.6% at m=1 and 3-22% at
    m=2/3 on 32 arenas x 9000 steps; 29% (m=1) and 62% (m=2) on a 4-arena x
    9000-step run on a LOADED box; 0%/37.5%/12.5% at m=1/2/3 on 4 arenas x 9000
    on an idle one; and 0% over 16 arenas x 45000 at m=2. So: never budget
    against a remembered figure -- census every run.
    """
    n, short = r.get("records") or 0, r.get("short_records") or 0
    if not n:
        return "  short records: n/a (no records)"
    frac = short / n
    line = (f"  short records: {short}/{n} ({frac * 100:.1f}%)  n_full={n - short}"
            f"  [duration < {FULL_MATCH_STEPS} steps = contained physics blowups; "
            f"each carries the match's PARTIAL score, not 0-0]")
    if frac > 0.02:
        # No magnitude, no direction: both would be inventions. What IS known is
        # that these records came from truncated matches and that n counts them
        # as though they were completed ones.
        line += ("\n    heavy census: the verdict is contaminated by an unknown "
                 "amount in an unknown direction, and se is understated because "
                 "n counts short records -- re-run on an idle box")
    return line


def _gate_action_table(candidate, champion):
    """The action table BOTH sides of the gate must share, or a refusal.

    The gate plays candidate and champion through ONE engine, and one engine
    binds ONE decode table -- so a 104-row v1-air candidate against the 92-row
    champion is not a hard match, it is an impossible one. schema_version does
    not distinguish them (it is 1 for both), which is why this cannot be left
    to the existing version check. Same-table pairs return the shared name and
    it selects the schema file."""
    import torch                                             # noqa: PLC0415

    from construct.tables import table_name                  # noqa: PLC0415

    def _name(p):
        return table_name(torch.load(p, map_location="cpu", weights_only=False))

    tc, tch = _name(candidate), _name(champion)
    if tc != tch:
        raise SystemExit(
            f"gate refused: candidate {candidate} decodes with {tc!r} but champion "
            f"{champion} decodes with {tch!r}. The action table sets the policy's "
            f"OUTPUT DIMENSION and this gate drives both sides through one engine, "
            f"so they cannot be compared here. Cross-table evaluation needs one "
            f"engine per side and is NOT wired -- gate a v1-air lineage against a "
            f"v1-air champion, or use scripts/bench_foreign.py (a fixed external "
            f"opponent) as the common ruler."
        )
    return tc


def _play_order(champion_sd, candidate_sd, arenas, seed, steps, as_candidate_weights,
                mode=1, action_table=None):
    """One side order. Returns ((cand_wins, draws, cand_losses), census) over the
    matches played this order. `as_candidate_weights` True => candidate drives the
    learner rows (its goals are the +spikes); False => champion does, and we
    flip the sign so the record is always from the CANDIDATE's perspective.

    The W/D/L stays a plain 3-tuple so aggregate()'s pinned contract is untouched;
    the census rides alongside rather than inside it."""
    import numpy as np                                       # noqa: PLC0415

    from construct.league.matches import MatchRunner, match_record, split_matches

    # action_table selects the SCHEMA FILE; None keeps schema/v1.toml, i.e. every
    # historical invocation of this gate is byte-identical.
    mr = MatchRunner(num_arenas=arenas, seed=seed, mode=mode, schema_version=1,
                     net_heads=4, reward_config="configs/reward_v0.toml",
                     curriculum_config="configs/curriculum_v3_match.toml",
                     action_table=action_table)
    if as_candidate_weights:
        mr.eng.set_weights(candidate_sd)
        mr.eng.set_opponents([champion_sd])
    else:
        mr.eng.set_weights(champion_sd)
        mr.eng.set_opponents([candidate_sd])
    out = mr.eng.collect(steps, arena_opponents=mr.assignment)
    rew = np.asarray(out["rewards"])
    # mr.assignment is all-zeros (a native league opponent), so EVERY arena
    # contributes exactly `mode` learner columns and arena k owns
    # [k*mode, (k+1)*mode). One self-play (-1) arena would be 2*mode wide and
    # split_matches' reshape would silently mis-group every arena after it --
    # and divisibility alone does NOT catch that (31*2 + 4 = 66 is still even).
    # This equality does.
    assert rew.shape[1] == arenas * mode, (
        f"expected {arenas * mode} learner columns, got {rew.shape[1]}")
    matches, durations = split_matches(out["rewards"], out["terminated"],
                                       team_size=mode, with_durations=True)
    if not as_candidate_weights:
        matches = flip_to_candidate(matches)
    rec = match_record(matches)
    # Short records are matches cut off by a contained physics blowup, and they
    # keep whatever score had accumulated (see _short_record_line). Counted,
    # never dropped: a filter would also change the 1v1 numbers every published
    # gate result sits on.
    census = {"records": len(durations),
              "short": sum(1 for d in durations if d < FULL_MATCH_STEPS)}
    return (rec["wins"], rec["draws"], rec["losses"]), census


def gate(candidate, champion, arenas, steps, seed, threshold, mode=1):
    from construct.league.matches import load_sd

    table = _gate_action_table(candidate, champion)
    champ_sd = load_sd(champion)
    cand_sd = load_sd(candidate)
    o1, c1 = _play_order(champ_sd, cand_sd, arenas, seed, steps, True, mode=mode,
                         action_table=table)
    o2, c2 = _play_order(champ_sd, cand_sd, arenas, seed + 1000, steps, False,
                         mode=mode, action_table=table)
    r = aggregate(o1, o2, threshold)
    r["order1"], r["order2"] = o1, o2
    r["records"] = c1["records"] + c2["records"]
    r["short_records"] = c1["short"] + c2["short"]
    r["short_frac"] = (r["short_records"] / r["records"]) if r["records"] else None
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
    # The threshold does NOT move with team size. Measured z for 0.55 on the
    # n=640 two-order statistic is 2.80 / 2.75 / 2.84 at m=1/2/3 -- n is
    # team-size independent and only the draw rate shifts (0.155 -> 0.205), so
    # forcing the operator to retype a threshold at m>1 would fragment the
    # history file's scale for no statistical gain. What IS mode-aware is the
    # printed null (NULL_BY_MODE), which refuses to quote the 1v1 number.
    ap.add_argument("--threshold", type=float, default=0.55)
    ap.add_argument("--mode", type=int, choices=(1, 2, 3), default=1,
                    help="team size per side (1=1v1, 2=2v2, 3=3v3). NET vs NET "
                         "only. (The engine's foreign-bot guard became PER KIND "
                         "on 2026-07-26 -- nexto/necto do drive team arenas -- "
                         "but this gate has never used foreign opponents either "
                         "way.) Default 1 keeps every existing invocation "
                         "unchanged.")
    ap.add_argument("--promote-if-pass", action="store_true",
                    help="on PASS, move configs/champion.toml's champion_ck to the "
                         "candidate (atomic). Opt-in on purpose. 1v1 only.")
    ap.add_argument("--champion-config", default=None,
                    help="override configs/champion.toml (tests)")
    args = ap.parse_args(argv)
    # Refused HERE, not after the run: at --mode 2 the gate is 20-30 minutes of
    # engine time, and finding out afterwards that the result cannot promote is
    # the expensive way to learn it.
    if args.promote_if_pass and args.mode != 1:
        ap.error("--promote-if-pass is 1v1-only. configs/champion.toml's "
                 "champion_ck feeds the KL anchor, the league pool seed and the "
                 "deploy candidate, all measured in the 1v1 regime, and "
                 "a 2v2-gated net would silently become a 1v1 training "
                 "opponent. (Registry entries carry a `gated_mode` field since "
                 "2026-07-26, but nothing SELECTS on it yet, so the guard stays.) "
                 "The team gate also measures a DIFFERENT quantity: the champion "
                 "scores 7.1 goals/match at 1v1 and 4.5 at 3v3, and has never had "
                 "a teammate, so 3v3 is out-of-distribution for it. "
                 "Re-run at --mode 1 to promote.")
    if args.champion is None:
        args.champion = resolve_champion(args.champion_config)

    r = gate(args.candidate, args.champion, args.arenas, args.steps, args.seed,
             args.threshold, mode=args.mode)
    # Mode in the header so a pasted terminal block is self-identifying -- a 2v2
    # share read as the 1v1 ruler is the hazard this whole flag introduces.
    print(f"candidate {args.candidate} vs champion {args.champion}  "
          f"[{args.mode}v{args.mode}]")
    if r["share"] is None:
        print(f"  {r['reason']} -> {r['verdict']}")
        return 1
    print(f"  order1 (cand weights) W/D/L: {r['order1']}")
    print(f"  order2 (swapped)      W/D/L: {r['order2']}")
    print(f"  TOTAL {r['wins']}W/{r['draws']}D/{r['losses']}L  n={r['n']}  "
          f"win_share={r['share']:.4f} +/- {r['se']:.4f}")
    print(_short_record_line(r))
    null = NULL_BY_MODE.get(args.mode)
    # sd at 4dp: the 2v2 null is 0.0149 and 3dp rounds it to 0.015, which is
    # the one number a reader needs exactly in order to judge a margin.
    ann = (f"null mean {null[0]:.3f} sd {null[1]:.4f}" if null else
           f"null UNMEASURED at {args.mode}v{args.mode} -- run "
           f"scripts/match_gate_null.py --mode {args.mode} --both-orders")
    print(f"  verdict: {r['verdict']}  (threshold {r['threshold']:.3f}; {ann})")
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
    only in whatever terminal ran it. Never fail the gate over bookkeeping.

    ONE FILE, no backfill. Every mode goes in the same log because the threshold
    is the same at every team size, so the rows are on one scale; and rewriting
    already-published rows to add `mode` is worse than normalising on read.
    Readers treat a missing `mode` as 1 (scripts/dashboard.py::parse_gate_history).
    `records`/`short_records` are what let a reader tell a blowup-contaminated
    run from a genuinely close one after the fact.
    """
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
        "mode": args.mode,
        "records": r.get("records"), "short_records": r.get("short_records"),
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
