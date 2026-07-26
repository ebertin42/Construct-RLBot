"""Head-to-head skill eval: play two frozen checkpoints against each other
through construct.league.matches.MatchRunner and report goal share.

WHY THIS EXISTS (see docs/training-journal.md, 2026-07-19 ~14:50 -- the
KL-anchor kill-switch): scripts/eval_metrics.py measures SELF-PLAY goals/min
(both sides run the SAME policy). Better defense suppresses that number with
zero skill loss, and both-sides-worse leaves it flat -- self-play goals/min
hid an 800M-step regression AND made a 3.5x-weaker policy look like it was
improving. Head-to-head against a frozen reference checkpoint doesn't have
that confound: a stronger policy wins more of a FIXED opponent regardless of
how either side's own defense trends over training. The league ladder
already computes exactly this at scale (many checkpoints, TrueSkill); this
script is the ad hoc / scriptable sibling -- one matchup at a time, no
rating bookkeeping, history appended for the dashboard.

SIDE BIAS IS REAL. A single mr.play(sd_a, sd_b) call is NOT a fair measure
-- the same pair of checkpoints can score wildly differently depending on
which side (which engine_kwargs slot) each plays. Observed on 2026-07-19:
19-60 one way, 16-59 the other; 36-59 vs 30-79 on another pair. Every
matchup here ALWAYS plays both side orders and sums the goals; the two
per-order shares are also reported so a large split (>15%) can be flagged as
an unstable measurement instead of silently averaged away.

DETERMINISM: MatchRunner's engine (Engine.collect -- RocketSim simulation +
the candle net forward pass) runs entirely on CPU, see `let dev =
Device::Cpu;` in engine/src/policy_v1.rs (and policy.rs for the v0 path) --
so a fixed seed gives an identical result run to run, and this NEVER
contends with GPU training. ctl.py wraps it with `nice -n 15`, same as
eval_metrics.py, so it doesn't starve anything else CPU-bound on the box
either.

CLI:
    h2h_eval.py CK_A CK_B [--steps 5400] [--arenas 8] [--seed 11]
    h2h_eval.py --vs-references CK [--steps 5400] [--arenas 8] [--seed 11]
        [--refs-config configs/h2h_references.toml] [--history logs/h2h_history.jsonl]

--vs-references plays CK against every reference listed in --refs-config
(see configs/h2h_references.toml) and appends one jsonl line per matchup to
--history (schema: ts, ck, ref, ref_label, goals_ck, goals_ref, share,
steps, seed) -- the dashboard's "Head-to-head (skill)" panel reads that
file.
"""
import argparse
import json
import sys
import time
import tomllib
from pathlib import Path

import torch

from construct.league.matches import MatchRunner, load_sd
from construct.tables import table_name

REPO = Path(__file__).resolve().parent.parent
DEFAULT_REFS_CONFIG = REPO / "configs" / "h2h_references.toml"
DEFAULT_HISTORY = REPO / "logs" / "h2h_history.jsonl"

# How far apart the two side orders' goal-shares can be before the combined
# result gets flagged unstable. Chosen against the observed swings in the
# module docstring (e.g. a ~24%/21% split is fine noise; the pairs that
# motivated this script sometimes swing much further and deserve a flag).
DISAGREEMENT_THRESHOLD = 0.15


class SchemaMismatchError(ValueError):
    """Two checkpoints that can't play each other: v0 (flat-obs MLP) and v1
    (entity-transformer) use structurally incompatible obs contracts (see
    the construct.league.matches module docstring), and two v1 checkpoints
    with different attention head counts can't share a candle-rebuilt net
    either. MatchRunner/play_entries already guard schema_version deep
    inside the engine boundary; this raises the same refusal early -- before
    an Engine is ever built -- with a clearer message."""


# ---------------------------------------------------------------------------
# pure functions (tested in tests/python/test_h2h.py with a mocked runner)
# ---------------------------------------------------------------------------

def checkpoint_meta(path):
    """Read {schema_version, heads, steps, action_table} from a checkpoint's own
    config -- same key path as scripts/eval_metrics.py's v1 branch
    (config.net.heads / schema_version). `heads` is None for schema_version 0
    (PolicyValueNet has no attention-head concept).

    `action_table` is a THIRD compatibility axis beside version and head count,
    and unlike them it is not visible in `schema_version`: that is 1 for both
    the 92-row `construct_92_v1` and the 104-row `construct_104_v1air`. It is
    read via construct.tables.table_name (recorded field, else the stored
    buffer's width, else the 92-row default), and it is what
    champion_gate/add_to_league stamps on the registry entry -- without it a
    promoted v1-air checkpoint is labelled 92-row and every downstream guard
    reads the wrong thing."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sv = int(ck.get("schema_version", 0))
    heads = int(ck["config"]["net"]["heads"]) if sv == 1 else None
    return {
        "path": str(path), "schema_version": sv, "heads": heads,
        "steps": int(ck.get("total_steps", 0)),
        "action_table": table_name(ck),
    }


def require_compatible(meta_a, meta_b):
    """Refuse a cross-schema (or, for v1, cross-head-count or cross-action-
    table) pair with a clear message, before touching the engine at all."""
    if meta_a["schema_version"] != meta_b["schema_version"]:
        raise SchemaMismatchError(
            f"cross-schema h2h refused: {meta_a['path']} is schema_version="
            f"{meta_a['schema_version']}, {meta_b['path']} is schema_version="
            f"{meta_b['schema_version']} -- v0 and v1 checkpoints use "
            f"incompatible obs contracts and can never play each other."
        )
    if meta_a["schema_version"] == 1 and meta_a["heads"] != meta_b["heads"]:
        raise SchemaMismatchError(
            f"h2h refused: {meta_a['path']} has net.heads={meta_a['heads']}, "
            f"{meta_b['path']} has net.heads={meta_b['heads']} -- both v1 "
            f"checkpoints must share an attention head count (candle "
            f"rebuilds attention from the head count, not tensor shape)."
        )
    # Third axis, invisible in schema_version (1 for BOTH v1 tables). play_h2h
    # drives both sides through ONE MatchRunner, i.e. one engine, i.e. one
    # decode table -- so a 92-row champion and a 104-row v9 candidate are not
    # merely unfair to each other, they are unplayable together. Uses .get so a
    # hand-built meta dict from an older caller still works.
    ta = meta_a.get("action_table")
    tb = meta_b.get("action_table")
    if meta_a["schema_version"] == 1 and ta != tb:
        raise SchemaMismatchError(
            f"h2h refused: {meta_a['path']} decodes with {ta!r}, "
            f"{meta_b['path']} with {tb!r} -- the action table sets the policy's "
            f"OUTPUT DIMENSION and one engine binds one table, so these two "
            f"cannot share a match. Gating a v1-air candidate against a 92-row "
            f"champion needs one engine per side; it is not wired here."
        )


def goal_share(a, b):
    """a's fraction of combined goals, or None when neither side scored."""
    total = a + b
    return a / total if total else None


def is_unstable(disagreement):
    """Strictly-greater-than: a disagreement exactly AT the threshold is not
    flagged, only one that exceeds it. None (one side never scored either
    order) is never unstable -- there's no share to disagree about."""
    return disagreement is not None and disagreement > DISAGREEMENT_THRESHOLD


def aggregate_sides(side1, side2):
    """Combine both side-order results into totals + a disagreement flag.

    side1 = (a1, b1) from mr.play(sd_a, sd_b, ...)  -- A in the "a" slot.
    side2 = (a2, b2) from mr.play(sd_b, sd_a, ...), already flipped back to
    (a-goals, b-goals) order by the caller -- A in the "b" slot this time.
    """
    a1, b1 = side1
    a2, b2 = side2
    a_total, b_total = a1 + a2, b1 + b2
    share1, share2 = goal_share(a1, b1), goal_share(a2, b2)
    disagreement = (
        abs(share1 - share2) if share1 is not None and share2 is not None else None
    )
    return {
        "a": a_total, "b": b_total, "share": goal_share(a_total, b_total),
        "side1": (a1, b1), "side2": (a2, b2),
        "share_side1": share1, "share_side2": share2,
        "disagreement": disagreement,
        "unstable": is_unstable(disagreement),
    }


def play_h2h(mr, sd_a, sd_b, steps):
    """Play both side orders on `mr` and return the aggregated result (see
    aggregate_sides). The only function in this module that touches the
    engine."""
    a1, b1 = mr.play(sd_a, sd_b, steps=steps)
    b2, a2 = mr.play(sd_b, sd_a, steps=steps)
    return aggregate_sides((a1, b1), (a2, b2))


def format_result(label_a, label_b, result):
    a1, b1 = result["side1"]
    a2, b2 = result["side2"]
    lines = [f"score: {a1}-{b1} / {a2}-{b2} (swapped) -- both as {label_a}-{label_b}"]
    if result["share"] is not None:
        lines.append(
            f"totals: {label_a}={result['a']} {label_b}={result['b']}  "
            f"share({label_a})={result['share'] * 100:.1f}%"
        )
    else:
        lines.append(f"totals: {label_a}={result['a']} {label_b}={result['b']}  share: n/a (0-0)")
    if result["unstable"]:
        lines.append(
            f"  unstable measurement: side orders disagree by "
            f"{result['disagreement'] * 100:.1f}% (side1 {result['share_side1'] * 100:.1f}% "
            f"vs side2 {result['share_side2'] * 100:.1f}%) -- treat this share with caution"
        )
    return "\n".join(lines)


# --- reference config --------------------------------------------------------

def load_references(path=DEFAULT_REFS_CONFIG):
    """Parse configs/h2h_references.toml -> list of {ck, label} dicts, in
    file order. A missing file returns an empty list (not an error --
    --vs-references with zero references configured is a no-op, not a
    crash)."""
    path = Path(path)
    if not path.is_file():
        return []
    with path.open("rb") as f:
        raw = tomllib.load(f)
    return [
        {"ck": str(r["ck"]), "label": str(r.get("label", r["ck"]))}
        for r in raw.get("reference", [])
    ]


# --- jsonl history ------------------------------------------------------------

def append_h2h_history(path, ck, ref, ref_label, goals_ck, goals_ref, steps, seed, ts=None):
    """Append one jsonl row per matchup. Schema: {ts, ck, ref, ref_label,
    goals_ck, goals_ref, share, steps, seed}. `steps` is per side order --
    both orders are always played, so total engine steps for the matchup is
    2x this value."""
    row = {
        "ts": int(ts if ts is not None else time.time()),
        "ck": Path(ck).name, "ref": Path(ref).name, "ref_label": ref_label,
        "goals_ck": int(goals_ck), "goals_ref": int(goals_ref),
        "share": goal_share(goals_ck, goals_ref),
        "steps": int(steps), "seed": int(seed),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
    return row


# ---------------------------------------------------------------------------
# engine-touching glue (not unit tested directly; exercised via the mocked
# pure functions above + one real-engine smoke in test_h2h.py)
# ---------------------------------------------------------------------------

def _build_runner(meta, arenas, seed, mode=1):
    # `mode` is trailing AND defaulted on purpose: champion_gate.py:358 calls this
    # positionally with three args, and the goal-share gate must stay untouched.
    #
    # There is deliberately NO --mode CLI flag on this script. h2h measures goal
    # SHARE, not match wins, so it is not where a team gate belongs
    # (scripts/matchwin_gate.py --mode is), and logs/h2h_history.jsonl carries
    # ABSOLUTE goal counts with no mode field -- adding the flag without that
    # schema change would silently mix 1v1 and 2v2 goal scales in one file. The
    # DETERMINISM paragraph in this module's docstring is likewise false at m>=2
    # (2+ cars per team are not bit-reproducible) but true for every invocation
    # that can actually occur, since mode never leaves 1 without a CLI surface.
    # Revisit all three together.
    # action_table picks the SCHEMA FILE (schema/v1.toml vs schema/v1_air.toml);
    # meta carries it because schema_version cannot. None at v0, where
    # _engine_kwargs ignores it -- and None at v1 too for a hand-built meta from
    # an older caller, which then gets the historical 92-row default.
    return MatchRunner(
        num_arenas=arenas, seed=seed, mode=mode,
        schema_version=meta["schema_version"],
        net_heads=meta["heads"] if meta["heads"] is not None else 4,
        action_table=meta.get("action_table"),
    )


def run_pairwise(ck_a, ck_b, steps, arenas, seed):
    meta_a, meta_b = checkpoint_meta(ck_a), checkpoint_meta(ck_b)
    require_compatible(meta_a, meta_b)
    mr = _build_runner(meta_a, arenas, seed)
    result = play_h2h(mr, load_sd(ck_a), load_sd(ck_b), steps)
    label_a, label_b = Path(ck_a).name, Path(ck_b).name
    print(f"=== {label_a} vs {label_b} "
          f"(schema v{meta_a['schema_version']}, {steps} steps/side, seed {seed}) ===")
    print(format_result(label_a, label_b, result))
    return result


def run_vs_references(ck, steps, arenas, seed, refs_config, history_path):
    meta_ck = checkpoint_meta(ck)
    refs = load_references(refs_config)
    if not refs:
        print(f"no references configured in {refs_config} -- nothing to play", file=sys.stderr)
        return []
    sd_ck = load_sd(ck)
    mr = None
    results = []
    for ref in refs:
        meta_ref = checkpoint_meta(ref["ck"])
        try:
            require_compatible(meta_ck, meta_ref)
        except SchemaMismatchError as e:
            print(f"skipping {ref['label']} ({ref['ck']}): {e}", file=sys.stderr)
            continue
        if mr is None:
            mr = _build_runner(meta_ck, arenas, seed)
        result = play_h2h(mr, sd_ck, load_sd(ref["ck"]), steps)
        label_ck, label_ref = Path(ck).name, ref["label"]
        print(f"=== {label_ck} vs {label_ref} ({Path(ref['ck']).name}) "
              f"({steps} steps/side, seed {seed}) ===")
        print(format_result(label_ck, label_ref, result))
        row = append_h2h_history(
            history_path, ck, ref["ck"], ref["label"],
            result["a"], result["b"], steps, seed,
        )
        results.append({"ref": ref, "result": result, "row": row})
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser():
    ap = argparse.ArgumentParser(
        description="Head-to-head skill eval: frozen checkpoints, both side orders, always summed."
    )
    ap.add_argument("checkpoints", nargs="+",
                     help="CK_A CK_B for a pairwise match, or just CK with --vs-references")
    ap.add_argument("--vs-references", action="store_true",
                     help="play the single given checkpoint against every reference in --refs-config")
    ap.add_argument("--steps", type=int, default=5400,
                     help="engine steps per side order (both orders are always played)")
    ap.add_argument("--arenas", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--refs-config", default=str(DEFAULT_REFS_CONFIG))
    ap.add_argument("--history", default=str(DEFAULT_HISTORY))
    return ap


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.vs_references:
        if len(args.checkpoints) != 1:
            parser.error("--vs-references takes exactly one checkpoint")
        run_vs_references(args.checkpoints[0], args.steps, args.arenas, args.seed,
                           args.refs_config, args.history)
        return
    if len(args.checkpoints) != 2:
        parser.error("pairwise mode takes exactly two checkpoints (CK_A CK_B)")
    try:
        run_pairwise(args.checkpoints[0], args.checkpoints[1], args.steps, args.arenas, args.seed)
    except SchemaMismatchError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
