#!/usr/bin/env python3
"""Measure the match-win gate's null distribution: a policy against ITSELF.

Why this must exist before any promotion is trusted: on 2026-07-20 a pure
random perturbation of the champion PASSED the 52% goal-share gate at 56.3%.
That gate had SE ~3.7%. The match-win gate is far noisier per unit compute
(~19 matches in today's budget, SE ~11.5%), so a threshold picked by intuition
would promote noise routinely.

A policy played against itself must centre on 0.5. The SPREAD across seeds is
the number that matters -- it sets the smallest win-share difference this gate
can resolve, and therefore where a defensible threshold sits.

TEAM SIZES. --mode 2/3 measures the null for the 2v2/3v3 gate, and until it has
been run matchwin_gate.py prints "null UNMEASURED" rather than quoting the 1v1
number. The team null MUST use --both-orders, and the reason is the SPREAD, not
symmetry: the published 1v1 protocol measures one order at n~320 while the gate
statistic lives at n=640, so an sd quoted on the wrong n is off by sqrt(2) (this
is also why 0.55 is really ~2.8 sd above chance rather than 2 sd).

WHAT --both-orders DOES NOT TEST. It is not a mirror-symmetry test, and an
earlier version of this docstring claimed it was. This is SELF-PLAY -- the same
net on both sides -- so "order" here is only a reseed, and flipping order 1's
records to one fixed perspective makes any systematic blue edge b cancel
exactly: order 0 contributes 0.5+b, flipped order 1 contributes 0.5-b, so the
combined mean is centred on 0.5 however large b is. "Combined mean == 0.5" can
therefore only catch a flip/accounting bug; it cannot see asymmetry, and a
+/-0.011 criterion on it is satisfied trivially. The asymmetry test lives on the
UNFLIPPED blue-perspective shares, which this script pools and reports separately
(MIRROR SYMMETRY below) -- that pooled number is the quantity the published 1v1
single-order null 0.502 actually measured.

Protocol: 10 seeds x both orders x 32 arenas x 45000 steps (same total compute as
today's 20 x single-order design, 6400 matches, SE of the mean ~0.0057). Pass
criteria declared BEFORE the run, and every one of them quoted on the n THIS RUN
measures -- the script derives them from the observed draw rate and records per
seed, because a hardcoded single-order 0.0253 would reject a correct both-orders
run as a bug:

  * across-seed sd inside the printed 95% chi-square band around the expected sd
    (~0.0176 for the mandated both-orders m=2 protocol, NOT the single-order
    ~0.025; the same formula reproduces the published 1v1 band [0.018, 0.035]);
  * sd NOT near expected_sd/sqrt(m) -- that, and not "a tighter gate", is what a
    resurrected duplicate-record bug looks like;
  * pooled unflipped blue share consistent with 0.5, i.e. mirror symmetry
    MEASURED with its own CI rather than assumed. Above 1v1 it stops being a
    fact and becomes an assumption, which is why it is reported at all.
"""
from __future__ import annotations

import argparse
import math
import statistics
import sys

import numpy as np


def null_summary(shares):
    """Mean and spread of self-play win shares. sd/lo/hi are None below n=2,
    where a spread is undefined rather than zero."""
    shares = [float(s) for s in shares]
    n = len(shares)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "lo": None, "hi": None}
    mean = statistics.fmean(shares)
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "lo": None, "hi": None}
    sd = statistics.stdev(shares)
    half = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "lo": mean - half, "hi": mean + half}


def expected_sd(n_per_seed, draw_rate):
    """The across-seed sd a CORRECT null should land on, for the n this run
    actually measured.

    Per-match outcome X is 1/0.5/0 for win/draw/loss. Under the null the policy
    is symmetric, so E[X]=0.5 and Var(X) = (1 - draw_rate)/4; a seed's share is
    the mean of n_per_seed such draws, hence sd = sqrt((1 - p_d) / (4n)).

    n_per_seed COUNTS BOTH ORDERS when --both-orders is on. That is the whole
    reason this is computed rather than remembered: the published 1v1 figure
    ~0.025 is a SINGLE-ORDER number (n~320, p_d 0.155), and the mandated
    both-orders m=2 protocol sits at ~0.0176 (n~640, p_d 0.205). Quoting the
    single-order constant at a both-orders run would flag a clean run as broken
    -- 0.0253/sqrt(2) = 0.0179 is indistinguishable from the correct answer.
    """
    n_per_seed = float(n_per_seed)
    if n_per_seed <= 0:
        return None
    return math.sqrt(max(0.0, 1.0 - float(draw_rate)) / (4.0 * n_per_seed))


def sd_band(sd_expected, n_seeds, z=1.96):
    """Central 95% sampling band for an across-seed sd ESTIMATE built from
    n_seeds samples -- an sd from 20 seeds is itself noisy, so a bare
    "sd is 0.021, expected 0.0176" says nothing without this.

    (n-1)s^2/sigma^2 is chi-square with df = n_seeds-1; the quantiles come from
    Wilson-Hilferty rather than scipy (this repo's scripts must run on a bare
    venv), which is within ~0.1% of exact at df=19. Sanity check that this is
    the right generalisation and not a new invention: at the published 1v1
    numbers (sigma ~0.0257, 20 seeds) it returns [0.0176, 0.0338] -- the
    historical hardcoded band [0.018, 0.035] at the precision it was quoted to.
    """
    df = int(n_seeds) - 1
    if sd_expected is None or df < 2:
        return None

    def chi2_over_df(zq):
        a = 2.0 / (9.0 * df)
        return max(0.0, 1.0 - a + zq * math.sqrt(a)) ** 3

    return (sd_expected * math.sqrt(chi2_over_df(-z)),
            sd_expected * math.sqrt(chi2_over_df(z)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # DO NOT repoint this default at the moving champion: matchwin_gate.py:26-30
    # pins this script as an instrument measured with one exact net, and a null
    # sd belongs to the policy that produced it.
    ap.add_argument("--champion", default="checkpoints_entity/ck_000320471040.pt")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(11, 31)))
    ap.add_argument("--arenas", type=int, default=32)
    ap.add_argument("--steps", type=int, default=45000,
                    help="engine steps per seed; a match is ~4500 steps")
    ap.add_argument("--mode", type=int, choices=(1, 2, 3), default=1,
                    help="team size per side. Default 1 keeps the published "
                         "protocol byte-for-byte.")
    ap.add_argument("--both-orders", action="store_true",
                    help="play each seed in BOTH side orders, i.e. measure the "
                         "actual gate statistic (n=640) instead of one order "
                         "(n~320); the sd is smaller by sqrt(2) and that is the "
                         "whole point -- it is NOT a symmetry test, the flip "
                         "cancels blue advantage. Off by default so the published "
                         "1v1 null is reproducible; REQUIRED for a team null.")
    args = ap.parse_args(argv)
    if args.mode != 1:
        # The null sd is a property of the policy's draw rate, so a team null
        # measured against the seeded 1v1 reference would not describe the net
        # actually being gated. Force the operator to name it, and record it.
        if not any(a == "--champion" or a.startswith("--champion=")
                   for a in (argv if argv is not None else sys.argv[1:])):
            ap.error(f"--mode {args.mode} requires an explicit --champion: the null "
                     f"sd belongs to the policy that produced it, so a team null "
                     f"must be measured with the net being gated (and that "
                     f"checkpoint recorded alongside the result).")
        if not args.both_orders:
            print(f"WARNING: --mode {args.mode} without --both-orders measures a "
                  f"single-order sd at n~{args.arenas * args.steps // 4500}, but the "
                  f"gate statistic is the two-order share at twice that n, whose sd is "
                  f"smaller by sqrt(2). A threshold set from this run would sit ~1.4x "
                  f"too far above 0.5. (Mirror symmetry is reported either way -- it "
                  f"reads off the unflipped shares, which both protocols keep.) "
                  f"Re-run with --both-orders before trusting a verdict.",
                  flush=True)

    from construct.league.matches import MatchRunner, load_sd, match_record, split_matches
    from construct.tables import table_for_state_dict

    sd = load_sd(args.champion)
    # The null has to be re-measured per wheel and per net (`gates-need-an-idle-box`),
    # so this instrument has to work on whatever net is being gated -- including a
    # 104-row v1-air one. schema_version cannot pick the schema file; the weights'
    # own action-table buffer can, and reading it off `sd` needs no second load.
    table = table_for_state_dict(sd)
    shares = []
    # Unflipped, blue-perspective share of each (seed, order). The combined
    # `shares` above cannot carry this: the flip cancels blue advantage by
    # construction (see the docstring), so this list is the ONLY thing here that
    # can see an arena asymmetry.
    blue_shares = []
    per_seed_n, total_draws, total_matches = [], 0, 0
    total_records = total_short = 0
    for seed in args.seeds:
        # curriculum_v3_match => match_mode (G1): full 300s matches whose
        # terminated flag is the clock, so split_matches groups real matches.
        # reward_v0 is the neutral scoring tape (win_prob_weight=0, so the
        # shaping term is inert on the tape; goals still spike +/-10) -- and at
        # mode>1 it is also what keeps goal spikes at a raw +/-10 per teammate,
        # since reward_v0's team_spirit is 0.0 and nothing blends.
        matches, durations, blue_by_order = [], [], []
        # Self-play against IDENTICAL weights, so "both orders" is NOT a side
        # swap -- both iterations put the same net on both sides and differ only
        # by seed. It buys n (the gate statistic is the two-order share), and the
        # flip below is what makes this loop reproduce the gate's arithmetic. It
        # does NOT test blue/orange symmetry: the flip cancels exactly that.
        for order in range(2 if args.both_orders else 1):
            mr = MatchRunner(num_arenas=args.arenas, seed=seed + 1000 * order,
                             mode=args.mode, schema_version=1, net_heads=4,
                             reward_config="configs/reward_v0.toml",
                             curriculum_config="configs/curriculum_v3_match.toml",
                             action_table=table)
            mr.eng.set_weights(sd)
            mr.eng.set_opponents([sd])
            out = mr.eng.collect(args.steps, arena_opponents=mr.assignment)
            rew = np.asarray(out["rewards"])
            # Divisibility is not enough (31*2 + 4 = 66 is still even); this
            # equality is what proves every arena is `mode` columns wide.
            assert rew.shape[1] == args.arenas * args.mode, (
                f"expected {args.arenas * args.mode} learner columns, "
                f"got {rew.shape[1]}")
            m, d = split_matches(out["rewards"], out["terminated"],
                                 team_size=args.mode, with_durations=True)
            # BEFORE the flip: learner rows are the blue team in a league-opponent
            # arena, so this is blue's own record. Captured here because after the
            # flip the information is gone.
            raw = match_record(m)
            if raw["win_share"] is not None:
                blue_by_order.append(raw["win_share"])
            if order == 1:
                m = [(b, a) for (a, b) in m]     # flip to one fixed perspective
            matches += m
            durations += d
        rec = match_record(matches)
        # Short records are contained physics blowups (episode.rs:779-807). They
        # are NOT all 0-0 draws: the blowup branch zeroes only that step's reward
        # and returns before start_match(), so a record closed by a blowup carries
        # whatever the match had already scored. Census them as a data-quality
        # flag on the run -- a blowup-heavy null describes a blowup-heavy box, not
        # this policy -- and re-run on an idle box rather than correcting.
        short = sum(1 for x in durations if x < 4500)
        total_records += len(durations)
        total_short += short
        if rec["win_share"] is not None:
            shares.append(rec["win_share"])
            per_seed_n.append(rec["matches"])
            total_draws += rec["draws"]
            total_matches += rec["matches"]
            blue_shares += blue_by_order
            raw_note = ("  blue-raw=" + "/".join(f"{x:.3f}" for x in blue_by_order)
                        if args.both_orders else "")
            print(f"  seed {seed:5d}: {rec['wins']}W/{rec['draws']}D/{rec['losses']}L "
                  f"share={rec['win_share']:.3f}  short={short}/{len(durations)}"
                  f"{raw_note}", flush=True)

    if total_records:
        print(f"\nshort records overall: {total_short}/{total_records} "
              f"({100 * total_short / total_records:.1f}%)  [contained physics "
              f"blowups; the record carries the partial score, not 0-0 -- a heavy "
              f"census means re-run on an idle box]")
    s = null_summary(shares)
    if s["mean"] is None:
        print("\nNULL: no completed matches across any seed "
              "(match_mode not producing match outcomes -- see G1/G2 deployment gaps).")
        return 0
    orders = "both orders" if args.both_orders else "single order"
    print(f"\nNULL over {s['n']} seeds [{args.mode}v{args.mode}, {orders}, "
          f"{args.champion}]: mean={s['mean']:.4f}")
    if args.both_orders:
        # Say it out loud next to the number, because the number LOOKS like a
        # symmetry check and is not one: the flip cancels blue advantage, so this
        # mean is pinned to 0.5 whatever the arena does. It still earns its place
        # -- a flip or accounting bug shows up here immediately.
        print("    (both orders: this mean is centred on 0.5 BY CONSTRUCTION -- "
              "it checks the flip arithmetic, not mirror symmetry; see MIRROR "
              "SYMMETRY below)")
    if s["sd"] is not None:
        print(f"  sd={s['sd']:.4f}  95% CI of the mean=[{s['lo']:.4f},{s['hi']:.4f}]")
        print(f"  a defensible threshold sits at least 2 sd above 0.5, "
              f"i.e. >= {0.5 + 2 * s['sd']:.3f}")
        # Everything below is computed from THIS run's draw rate and records per
        # seed. Reference constants remembered from another protocol are exactly
        # how a correct both-orders team null gets misread as a duplicate-record
        # bug: the single-order 0.0253/sqrt(2) = 0.0179 is what a healthy
        # both-orders m=2 run measures.
        n_bar = statistics.fmean(per_seed_n) if per_seed_n else 0
        p_d = (total_draws / total_matches) if total_matches else 0.0
        exp = expected_sd(n_bar, p_d)
        if exp:
            band = sd_band(exp, s["n"])
            band_txt = f"  95% band [{band[0]:.4f},{band[1]:.4f}]" if band else ""
            print(f"  expected sd for this run ~{exp:.4f} "
                  f"(draw rate {p_d:.3f}, {n_bar:.0f} matches/seed, {orders})"
                  f"{band_txt}")
            if args.mode != 1:
                # sd near expected/sqrt(m) is not a tighter gate, it is the
                # duplicate-record bug back: m fake samples per arena shrink the
                # apparent sd by exactly sqrt(m) while measuring nothing new.
                # Scaled off `exp`, NOT off the single-order constant -- n is
                # team-size independent, but it is NOT order-count independent.
                print(f"  duplicate-record signature: sd near "
                      f"{exp / math.sqrt(args.mode):.4f} = expected/sqrt({args.mode}) "
                      f"means records are being duplicated per car -- investigate, "
                      f"do not celebrate.")
    raw = null_summary(blue_shares)
    if raw["mean"] is not None:
        # THE mirror-symmetry test. b is the arena's systematic blue edge; every
        # sample here is unflipped, so b survives instead of cancelling. At 1v1
        # symmetry is a fact of the geometry; at m>1 kickoff slot assignment
        # follows heap addresses (test_team_curriculum.py:28-50) and it becomes an
        # assumption, so it gets measured.
        print(f"\nMIRROR SYMMETRY [{args.mode}v{args.mode}]: pooled unflipped blue "
              f"share = {raw['mean']:.4f} over {raw['n']} seed-orders")
        if raw["lo"] is not None:
            ok = raw["lo"] <= 0.5 <= raw["hi"]
            print(f"  95% CI of the mean=[{raw['lo']:.4f},{raw['hi']:.4f}] -> "
                  + ("consistent with 0.5 (symmetric)" if ok else
                     f"ASYMMETRIC: blue edge {raw['mean'] - 0.5:+.4f}. The gate "
                     f"plays both orders so its share is still unbiased, but a "
                     f"single-order number at this mode is not."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
