"""Head-to-head matches between two frozen checkpoints, via opponent arenas.

The engine's reward stream is used purely as a scoring tape: with reward_v0
(goal=10, bias 0, shaping << 9.4) a learner-row reward >= 9.4 means A scored,
<= -9.4 means B scored. Matches always use reward_v0 regardless of what the
policies were trained on -- reward is computed by reward::compute() from raw
game state (engine/src/episode.rs step_impl), never from the obs encoding, so
the tape trick is schema-independent and holds unchanged for a v1 engine.

v0 and v1 policies can NEVER play each other (different obs contracts): each
MatchRunner is built for exactly one schema_version, and play_entries() below
is the hard guard against feeding it a cross-schema pair.

The SAME is true one level down, and `schema_version` cannot express it: a
MatchRunner is also built for exactly one ACTION TABLE, because one engine binds
one decode table and both sides of a match go through this single engine.
Version 1 admits two -- the 92-row `construct_92_v1` and the 104-row
`construct_104_v1air` -- so `action_table=` selects the schema file (default:
the 92-row one, i.e. every pre-v9 caller is unchanged), `_require_table` refuses
a state dict of the other width, and play_entries refuses the pairing on
metadata before either side is loaded.
"""
import numpy as np
import torch

from construct._engine import Engine
from construct.tables import (
    DEFAULT_V1_TABLE,
    V0_SCHEMA_PATH,
    V1_TABLE_ROWS as V1_TABLE_ROWS_BY_NAME,
    V1_TABLE_SCHEMA,
    schema_path_for_table,
)

# Goal detection threshold: goal pays ±10; same-step shaping can offset a concede
# by up to +0.55 (touch 0.5 + vel_to_ball 0.05), so a concede row can be as small
# as -9.45 in magnitude. Non-goal rows never exceed |0.55|. 9.4 sits safely
# inside the [0.55, 9.45] gap on both sides.
#
# 9.4 SURVIVES AT 2v2/3v3 UNCHANGED, and that is a property of the reward config,
# not of the team size: matches force reward_v0, whose team_spirit/opp_spirit are
# 0.0 (engine/src/reward.rs:267,393, asserted by the Rust test at :572), so nothing
# blends across teammates on the scoring tape and EVERY car of the scoring team
# gets a raw +/-10 row. THE TRAP, by name: configs/reward_v8_team.toml:33 sets
# team_spirit = 0.3, which turns a lone scorer's spike into 0.7*10 + 0.3*(10/m)
# = 8.0 at m=2 -- BELOW this bar, so goals would silently vanish from the tape.
# The gate must never be repointed at a blended reward; _all_or_nothing() below is
# the detector if someone tries.
GOAL_THRESHOLD = 9.4

_SCHEMA_PATHS = {0: V0_SCHEMA_PATH, 1: V1_TABLE_SCHEMA[DEFAULT_V1_TABLE]}


def _engine_kwargs(num_arenas, seed, reward_config, mode, schema_version, net_heads,
                    curriculum_config=None, action_table=None):
    """Assemble the kwargs dict for the Engine constructor.

    Pulled out of MatchRunner.__init__ so it can be unit-tested without
    constructing a real Engine (expensive, and match_mode support in the
    local engine build is a separate deployment step -- see matches.py
    module docstring / deployment-gap G1).

    curriculum_config is only threaded through when explicitly given: the
    default (None) omits the `curriculum_config_path` key entirely rather
    than passing None, so legacy single-goal-boundary matches (the existing
    goal-share gate) stay byte-for-byte unchanged.

    `mode` is the team size per side and goes straight into blue=/orange=, so
    mode=2 builds 2v2 arenas and mode=3 builds 3v3. No code change was needed
    for team matches here -- the fix lives in the per-arena reduction in
    MatchRunner.play / split_matches, not in engine construction.

    `action_table` picks the SCHEMA FILE at schema_version 1, where the version
    number alone cannot: it is 1 for both the 92-row `construct_92_v1` and the
    104-row `construct_104_v1air`, and one engine binds one decode table. None
    (the default) keeps the historical 92-row schema/v1.toml, so every existing
    caller is byte-identical; h2h_eval/_build_runner threads the checkpoint's
    own table through so a v9 net can be gated at all. Ignored at v0, which has
    exactly one table.
    """
    schema_path = _SCHEMA_PATHS[schema_version]
    if schema_version == 1 and action_table is not None:
        schema_path = schema_path_for_table(action_table)
    engine_kwargs = dict(
        num_arenas=num_arenas, blue=mode, orange=mode,
        schema_path=schema_path, reward_config_path=reward_config,
        seed=seed,
    )
    if schema_version == 1:
        # candle rebuilds attention from the raw state dict; head count is
        # not recoverable from tensor shapes alone (same reason Trainer
        # passes it -- see train.py's engine_kwargs["net_heads"]).
        engine_kwargs["net_heads"] = net_heads
    if curriculum_config is not None:
        engine_kwargs["curriculum_config_path"] = curriculum_config
    return engine_kwargs


def load_sd(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    # Generic tensor->numpy conversion: works unchanged for both v0
    # (PolicyValueNet) and v1 (EntityPolicyNet, whose state_dict also carries
    # the non-trainable `action_table` buffer -- the engine's v1 EntityPolicy
    # requires and consumes that key, see engine/src/policy_v1.rs).
    return {k: v.numpy().astype(np.float32) for k, v in ck["model"].items()}


class MatchRunner:
    def __init__(self, num_arenas=8, seed=0, reward_config="configs/reward_v0.toml", mode=1,
                 schema_version=0, net_heads=4, curriculum_config=None,
                 action_table=None):
        # Goal events pay EVERY learner agent on the scoring team in that arena at
        # the same step, so a raw `(rew >= GOAL_THRESHOLD).sum()` over the whole
        # tape counts one 2v2 goal twice and one 3v3 goal three times. That is why
        # mode>1 used to be refused. It is now handled instead of forbidden: every
        # league-opponent arena contributes exactly `mode` learner columns, laid
        # out contiguously (arena k owns [k*mode, (k+1)*mode) -- measured live at
        # m=2: goal-step column groups [6,7],[4,5],[0,1]; at m=3: [6,7,8],[3,4,5]),
        # so reducing per (step, arena) BEFORE summing counts each goal exactly
        # once. See play() and split_matches().
        #
        # The uniform column width is a PRECONDITION, not a fact of the engine:
        # self.assignment below is all-zeros (a native league opponent), and only
        # that makes every arena `mode` wide. A single self-play (-1) arena
        # contributes blue+orange = 2*mode learner rows instead
        # (engine/src/engine.rs:1408-1410), which would slide every later arena's
        # columns and silently mis-group the reshape. Callers that build their own
        # assignment must assert ncol == num_arenas * mode (see
        # scripts/matchwin_gate.py::_play_order).
        assert mode in (1, 2, 3), f"MatchRunner mode must be 1, 2 or 3 (got {mode})"
        assert schema_version in _SCHEMA_PATHS, (
            f"MatchRunner schema_version must be one of {sorted(_SCHEMA_PATHS)}, "
            f"got {schema_version}"
        )
        self.schema_version = schema_version
        # Which decode table this runner's engine binds. `play()` checks every
        # state dict against it, because ONE ENGINE BINDS ONE TABLE and both
        # sides of a match go through this single engine (set_weights +
        # set_opponents). None at v0 (one table exists); at v1 it defaults to
        # the 92-row table, exactly what every pre-v9 caller got.
        self.action_table = (
            (action_table or DEFAULT_V1_TABLE) if schema_version == 1 else None
        )
        # Both consumed downstream: play() needs `mode` for the per-arena reshape,
        # and `num_arenas` lets a caller assert the exact learner-column count
        # (divisibility alone is NOT enough -- 31 arenas at m=2 plus one self-play
        # arena is 31*2 + 4 = 66 columns, still even, still mis-grouped).
        self.mode = mode
        self.num_arenas = num_arenas
        # curriculum_config is None by default: omitting curriculum_config_path
        # entirely (not passing it as None) keeps legacy construction -- and
        # therefore the existing goal-share gate's per-goal boundaries --
        # byte-for-byte unchanged. Passing it enables match_mode (full 300s
        # matches, deployment-gap G1); the engine build that supports it is a
        # separate step (G2).
        engine_kwargs = _engine_kwargs(
            num_arenas, seed, reward_config, mode, schema_version, net_heads,
            curriculum_config=curriculum_config, action_table=action_table,
        )
        self.eng = Engine(**engine_kwargs)
        self.assignment = [0] * num_arenas

    def _require_table(self, sd, which):
        """Refuse a state dict whose action table is not the one this engine
        binds. v0 state dicts carry no such buffer and are skipped."""
        if self.action_table is None or "action_table" not in sd:
            return
        want = V1_TABLE_ROWS_BY_NAME[self.action_table]
        got = int(sd["action_table"].shape[0])
        if got != want:
            raise ValueError(
                f"action-table mismatch on the {which} side: this MatchRunner "
                f"binds {self.action_table} ({want} rows) but the state dict "
                f"carries {got} rows. One engine binds ONE decode table, so a "
                f"92-row and a 104-row policy cannot share a match -- build one "
                f"runner per table (MatchRunner(action_table=...))."
            )

    def play(self, sd_a, sd_b, steps=2700):
        # Arenas are not reset between calls: match N+1's collect() continues
        # from wherever match N left the ball/cars, and set_weights/set_opponents
        # swap the policies driving those arenas mid-episode. Intentional --
        # avoids a reset() round-trip per match -- and deterministic given a
        # fixed seed + call sequence (see test_match_deterministic); the only
        # cost is a bit of extra noise in per-match goal counts, which washes
        # out over the TrueSkill ladder's many matches.
        # Width check before the engine boundary. The engine guards this too,
        # but it can only report "the state dict carries N rows"; here we can
        # also name WHICH SIDE is wrong and which table this runner binds.
        self._require_table(sd_a, "learner (sd_a)")
        self._require_table(sd_b, "opponent (sd_b)")
        self.eng.set_weights(sd_a)
        self.eng.set_opponents([sd_b])
        out = self.eng.collect(steps, arena_opponents=self.assignment)
        rew = np.asarray(out["rewards"])
        # One goal pays all `mode` cars of the scoring team on the same step, so
        # reduce per (step, arena) BEFORE summing or both counts come out m-times
        # inflated. The RATIO survives that inflation, which is exactly why it
        # would have gone unnoticed: an h2h goal share would look fine while the
        # absolute totals were wrong -- and those totals feed champion_gate's
        # min_total_goals sample-size guard (m x easier to satisfy), gate_stats'
        # Wilson n, and the goal counts written to logs/h2h_history.jsonl.
        # At mode=1 reshape(T, -1, 1).any(axis=2) is the identity, so this is
        # arithmetically the old `(rew >= TH).sum()` -- including for a 1-D tape,
        # which reshape(T, -1, 1) absorbs the same way the old .sum() did.
        by_arena = rew.reshape(rew.shape[0], -1, self.mode)
        goals_a = int((by_arena >= GOAL_THRESHOLD).any(axis=2).sum())
        goals_b = int((by_arena <= -GOAL_THRESHOLD).any(axis=2).sum())
        return goals_a, goals_b


def play_entries(mr: "MatchRunner", entry_a: dict, entry_b: dict, steps: int = 2700):
    """Run a match between two registry entries via `mr`, refusing to pit
    different schema_version OR different action_table checkpoints against
    each other.

    v0 and v1 obs are structurally different (flat 94-float vector vs entity
    tensors) -- a cross-schema match would either crash deep in the engine
    boundary (missing/extra state_dict keys) or, worse, silently misbehave.
    This checks entry metadata *before* touching disk or the engine, so the
    refusal is immediate and doesn't depend on `mr` being usable at all
    (checked first, ahead of the mr.schema_version comparison below).

    ACTION TABLE IS A SECOND AXIS. schema_version is 1 for BOTH v1 tables --
    the 92-row `construct_92_v1` and the 104-row `construct_104_v1air` -- so
    the version check alone does NOT catch a v1.1-vs-v1-air pairing. That
    pairing is unplayable rather than merely unfair: `MatchRunner.play` drives
    both sides through ONE engine (set_weights + set_opponents), and one
    engine binds one decode table. The engine also guards this, but refusing
    on metadata names the two checkpoints instead of a tensor width.
    """
    va = entry_a.get("schema_version", 0)
    vb = entry_b.get("schema_version", 0)
    if va != vb:
        raise ValueError(
            f"cross-schema match refused: {entry_a['ck']!r} is schema_version={va}, "
            f"{entry_b['ck']!r} is schema_version={vb}"
        )
    # Entry-vs-entry, so it belongs with the version check above and BEFORE any
    # `mr` dereference -- same "refuse without needing a usable mr" contract.
    # Default: pre-v9 entries predate the field and are all the 92-row table.
    ta = entry_a.get("action_table", "construct_92_v1")
    tb = entry_b.get("action_table", "construct_92_v1")
    if ta != tb:
        raise ValueError(
            f"cross-action-table match refused: {entry_a['ck']!r} uses {ta!r}, "
            f"{entry_b['ck']!r} uses {tb!r}. The action table sets the policy's "
            f"output dimension and one engine binds one table, so these two "
            f"cannot be played against each other -- run each side on its own "
            f"schema (schema/v1.toml vs schema/v1_air.toml)."
        )
    if va != mr.schema_version:
        raise ValueError(
            f"entry schema_version={va} does not match MatchRunner schema_version="
            f"{mr.schema_version}"
        )
    return mr.play(load_sd(entry_a["ck"]), load_sd(entry_b["ck"]), steps=steps)


def _all_or_nothing(mask, team_size):
    """True iff, in every arena, all `team_size` cars agree.

    A goal pays EVERY car on the scoring team at the same step, and reward_v0
    does not blend (team_spirit 0.0), so a group is all-fired or none-fired --
    verified live at m=1,2,3 (per-column blue-goal totals [87,87] and
    [53,53,53] within an arena). A PARTIAL group means the tape is not a clean
    scoring tape; the known cause is a team-spirit-blended reward config
    (reward_v8_team, tau=0.3 -> a lone scorer's spike drops to 8.0, under
    GOAL_THRESHOLD). Refusing there is the whole point: a blended tape would
    still produce plausible-looking win shares, just silently missing goals.

    This is the reason the per-arena reduction is `any` rather than `sum // m`.
    Both are exact while the assumption holds; only `any` leaves a place to
    check that it still does.
    """
    k = mask.sum(axis=1)
    return bool(((k == 0) | (k == team_size)).all())


def split_matches(rewards, terminated, threshold=GOAL_THRESHOLD, team_size=1,
                  with_durations=False):
    """Group a reward tape into per-match (goals_a, goals_b) using terminated
    flags as match boundaries.

    In match mode `terminated` means "the clock expired", so it is exactly the
    match boundary. A goal is still a reward spike past `threshold` -- matches
    always run reward_v0 as a neutral scoring tape (see module doc), so this
    holds whatever the policies trained on.

    Records are PER ARENA, *NOT* per column -- the distinction is invisible at
    1v1, where an arena is exactly one learner column, and is the whole content
    of the team-size fix. Each arena plays its own sequence of matches, and a
    300s clock makes all arenas terminate on the same step, so a naive
    `terminated.any()` + arena-summed count would collapse all N arenas' goals
    into ONE record -- N-fold fewer samples, and an aggregate-goal comparison
    rather than the per-match W/D/L we want. Accumulate and emit per arena.

    `team_size` (m) is how many learner columns one arena owns. At m>1 the old
    per-column code was wrong TWICE: it counted every goal m times (all m
    teammates get paid) AND emitted m duplicate records per arena. The duplicate
    records each carried the CORRECT score, so `win_share` was unbiased and the
    bug was invisible in the headline number -- but n was m x too large, so every
    SE and CI derived from it was understated by sqrt(m). That is what
    test_se_matches_binomial_at_half exists to protect.

    Match YIELD is team-size independent: n_matches = arenas * floor(steps/4500)
    per side order at every m (measured: 64 records at m=1, 2 and 3 for 32 arenas
    x 9000 steps). Team size therefore buys no extra samples -- aggregate()'s se
    does not move with m, only wall clock does (x1.49 at m=2, x1.96 at m=3).

    Column layout is arena-major / car-minor: arena k owns columns
    [k*m, (k+1)*m), so a plain reshape IS the grouping. GUARD LIMIT: the
    `ncol % team_size` check below is NECESSARY BUT NOT SUFFICIENT -- 31 arenas
    at m=2 plus one self-play (-1) arena gives 31*2 + 4 = 66 columns, still
    divisible by 2 and still mis-grouped. The sufficient check is
    `ncol == num_arenas * team_size` and it belongs at the call site, where
    num_arenas is known (scripts/matchwin_gate.py::_play_order does it).

    A trailing partial match is DISCARDED: it has no outcome, and scoring it as
    a draw would bias every gate toward 0.5.

    `with_durations=True` additionally returns per-record durations in steps,
    aligned one-to-one with the records BY CONSTRUCTION (same loop, same append)
    rather than by two loops kept in lockstep. Durations are how the caller
    censuses contained physics blowups: engine/src/episode.rs:779-807 sets
    terminated on every car of a blown-up arena and rebuilds it, so a record
    closes tens of steps into a match instead of at 4500.

    A short record is NOT a 0-0 draw, and the accumulators below are the reason.
    The blowup branch zeroes only the CURRENT step's reward and returns before
    the engine's `start_match()`, and in match mode a goal never terminates, so
    `a`/`b` still hold every goal scored earlier in that match when the
    terminated flag arrives. What gets emitted is the partial match's real score
    -- a blowup at 2-1 emits (2,1). Callers must treat short records as
    unsigned contamination (scripts/matchwin_gate.py::_short_record_line), not as
    draws that conveniently pull toward 0.5.

    The default return type is unchanged (a plain list of 2-tuples) because
    flip_to_candidate and match_record both destructure 2-tuples.
    """
    rewards = np.asarray(rewards)
    terminated = np.asarray(terminated)
    if rewards.ndim == 1:            # a single-arena tape may arrive as (T,)
        # A (T,) tape is unambiguous only at 1v1. At m>1 it could be one m-car
        # arena or m single-car arenas and nothing in the array distinguishes
        # them -- refuse rather than guess.
        if team_size != 1:
            raise ValueError("a 1-D reward tape is ambiguous at team_size > 1")
        rewards = rewards[:, None]
        terminated = terminated[:, None]
    T, ncol = rewards.shape
    if ncol % team_size:
        raise ValueError(
            f"{ncol} learner columns is not a whole number of {team_size}-car "
            f"arenas -- the engine's team size disagrees with team_size={team_size}"
        )
    n = ncol // team_size
    out, durations = [], []
    a = np.zeros(n, dtype=int)       # per-ARENA accumulators -- NOT per column
    b = np.zeros(n, dtype=int)
    start = np.zeros(n, dtype=int)
    for t in range(T):
        # arena-major, car-minor, so reshape alone is the right gather: engine.rs
        # builds learner_idx by walking arenas in order, and the multi-thread
        # merge preserves it because workers own contiguous arena ranges.
        sa = (rewards[t] >= threshold).reshape(n, team_size)
        sb = (rewards[t] <= -threshold).reshape(n, team_size)
        ends = terminated[t].reshape(n, team_size)
        if team_size != 1:
            # Vacuous at m=1 (a group of one is always all-or-nothing), so it is
            # skipped there. The reshape+any that replaced the flat compare does
            # cost something -- measured 0.16s -> 0.43s over a real 45000x32 gate
            # tape -- but that is 0.3s against ~10 min of engine time per side
            # order, and it buys one implementation instead of two.
            for m_, what in ((sa, "blue goal"), (sb, "orange goal"), (ends, "terminated")):
                if not _all_or_nothing(m_, team_size):
                    raise ValueError(
                        f"{what} fired for only SOME cars of an arena at step {t}; "
                        f"the tape is not an unblended reward_v0 scoring tape"
                    )
        # .any(axis=1) over a width-1 group is the identity map on booleans, so
        # m=1 goes through literally the old expression with no float arithmetic
        # reordered; .astype(int) is kept verbatim so the accumulator dtype and
        # the add are the old ones too.
        a += sa.any(axis=1).astype(int)
        b += sb.any(axis=1).astype(int)
        # `any`, not `all`: the two differ only once the all-or-nothing
        # assumption is already broken, and LOSING a boundary desynchronises that
        # arena from the 4500-step grid for the rest of the tape. Failing toward
        # "keep the boundary" is the recoverable direction; the check above is
        # what reports the anomaly rather than the split absorbing it silently.
        for arena in np.nonzero(ends.any(axis=1))[0]:
            out.append((int(a[arena]), int(b[arena])))
            durations.append(t + 1 - int(start[arena]))
            a[arena] = 0
            b[arena] = 0
            start[arena] = t + 1
    return (out, durations) if with_durations else out


def match_record(matches):
    """Win/draw/loss counts and win share (draws count 0.5).

    `win_share` is None when no match completed -- 0.0 would read as a total
    loss and could drive a promotion decision off zero evidence.
    """
    wins = sum(1 for a, b in matches if a > b)
    losses = sum(1 for a, b in matches if a < b)
    draws = len(matches) - wins - losses
    share = None if not matches else (wins + 0.5 * draws) / len(matches)
    return {"wins": wins, "draws": draws, "losses": losses,
            "matches": len(matches), "win_share": share}
