import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from construct.league.matches import match_record, split_matches  # noqa: E402

TH = 9.4


def tape(*rows):
    """(T,1) reward tape for a single arena."""
    return np.array(rows, dtype=np.float32).reshape(-1, 1)


def flags(*rows):
    return np.array(rows, dtype=bool).reshape(-1, 1)


def test_splits_on_terminated_boundaries():
    r = tape(10.0, 0.1, -10.0, 0.0, 10.0, 0.0)
    t = flags(False, False, False, True, False, True)
    assert split_matches(r, t, TH) == [(1, 1), (1, 0)]


def test_sub_threshold_rows_are_not_goals():
    """Shaping noise must never be counted as a goal."""
    r = tape(0.55, -0.55, 9.39, -9.39, 0.0)
    t = flags(False, False, False, False, True)
    assert split_matches(r, t, TH) == [(0, 0)]


def test_trailing_incomplete_match_is_discarded():
    """A match still in progress at the end of the tape has no outcome and must
    not be scored as a draw -- that would bias every gate toward 0.5."""
    r = tape(10.0, 0.0, 10.0)
    t = flags(False, True, False)
    assert split_matches(r, t, TH) == [(1, 0)]


def test_multi_arena_produces_one_record_per_arena_not_a_sum():
    """The whole point of the per-arena split. Two arenas terminate on the same
    step (the lockstep 300s clock); each must yield its OWN record, not a single
    summed one. A naive terminated.any()+sum collapses them to [(3, 1)]."""
    r = np.array([[10.0, -10.0], [10.0, 0.0], [10.0, 0.0]], dtype=np.float32)
    t = np.array([[False, False], [False, False], [True, True]], dtype=bool)
    assert sorted(split_matches(r, t, TH)) == [(0, 1), (3, 0)]


def test_match_record_counts_wins_draws_losses():
    rec = match_record([(2, 1), (0, 0), (1, 3), (1, 0)])
    assert rec["wins"] == 2 and rec["draws"] == 1 and rec["losses"] == 1
    assert rec["win_share"] == pytest.approx((2 + 0.5) / 4)


def test_a_draw_counts_as_half():
    """Standard convention, and it keeps a self-play null control at exactly
    0.5 rather than pushing it around by the draw rate."""
    assert match_record([(0, 0), (0, 0)])["win_share"] == pytest.approx(0.5)


def test_no_completed_matches_is_none_not_zero():
    """0 wins of 0 matches is not 0% -- returning 0.0 would read as a total
    loss and could promote or reject on nothing."""
    assert match_record([])["win_share"] is None


# --- team sizes: 2v2 / 3v3 ---------------------------------------------------
#
# An arena owns `team_size` CONTIGUOUS learner columns (arena k -> [k*m, (k+1)*m);
# measured goal-step column groups at m=2: [6,7],[4,5],[0,1]; at m=3:
# [6,7,8],[3,4,5]). A goal pays every car of the scoring team on the same step, so
# the per-column code these replace was wrong twice at m>1: m-times goal counts AND
# m duplicate records per arena. The duplicates carried the right score, so
# win_share looked fine while n -- and every SE derived from it -- was m x too big.


def rows(*r):
    return np.array(r, dtype=np.float32)


def terms(*r):
    return np.array(r, dtype=bool)


def test_team_size_1_default_matches_legacy_output():
    """The comparability constraint: logs/matchwin_history.jsonl and the 0.8367
    champion result must stay on ONE scale, so an explicit team_size=1 and the
    default have to be the same computation, not merely similar."""
    fixtures = [
        (tape(10.0, 0.1, -10.0, 0.0, 10.0, 0.0),
         flags(False, False, False, True, False, True)),
        (tape(0.55, -0.55, 9.39, -9.39, 0.0), flags(False, False, False, False, True)),
        (tape(10.0, 0.0, 10.0), flags(False, True, False)),
        (rows([10.0, -10.0], [10.0, 0.0], [10.0, 0.0]),
         terms([False, False], [False, False], [True, True])),
    ]
    for r, t in fixtures:
        assert split_matches(r, t, TH, team_size=1) == split_matches(r, t, TH)


def test_2v2_one_goal_across_both_columns_is_one_goal():
    """THE m>1 bug in one tape. Both teammates are paid the same goal, so the
    per-column code returns [(1, 1), (1, 1)]: the goal counted twice and the
    single arena emitted twice."""
    r = rows([10.0, 10.0], [0.0, 0.0], [-10.0, -10.0])
    t = terms([False, False], [False, False], [True, True])
    assert split_matches(r, t, TH, team_size=2) == [(1, 1)]


def test_2v2_two_arenas_four_columns():
    """Grouping is arena-major / car-minor, so a plain reshape is the right
    gather: arena k reads columns [2k, 2k+2), never a stride or an interleave."""
    r = rows([10.0, 10.0, 0.0, 0.0],       # arena 0 scores
             [0.0, 0.0, -10.0, -10.0],     # arena 1 concedes
             [10.0, 10.0, 0.0, 0.0])       # arena 0 scores again
    t = terms([False] * 4, [False] * 4, [True] * 4)
    assert split_matches(r, t, TH, team_size=2) == [(2, 0), (0, 1)]


def test_3v3_contiguous_layout():
    """9 columns -> 3 arenas, using the measured [3,4,5] / [6,7,8] group pattern."""
    r = rows([0, 0, 0, 0, 0, 0, 10.0, 10.0, 10.0],      # arena 2 scores
             [0, 0, 0, -10.0, -10.0, -10.0, 0, 0, 0],   # arena 1 concedes
             [0] * 9)
    t = terms([False] * 9, [False] * 9, [True] * 9)
    assert split_matches(r, t, TH, team_size=3) == [(0, 0), (0, 1), (1, 0)]


def test_record_count_is_independent_of_team_size():
    """Team size buys NO extra samples -- 6 arenas x 2 terminations is 12 records
    at m=1, 2 and 3 alike (measured live: 64 records at each m for 32 arenas x
    9000 steps). This is why aggregate()'s se does not move with m, and why the
    old duplicate-record path inflated n by exactly m."""
    counts = []
    for m in (1, 2, 3):
        ncol = 6 * m
        r = np.zeros((4, ncol), dtype=np.float32)
        t = np.zeros((4, ncol), dtype=bool)
        t[1, :] = True
        t[3, :] = True
        counts.append(len(split_matches(r, t, TH, team_size=m)))
    assert counts == [12, 12, 12]


def test_partial_group_is_refused():
    """The blended-reward detector. This exact tape is a LEGAL m=1 tape (two
    one-car arenas, one scoring, one conceding) and an ILLEGAL m=2 one: a goal
    that paid only one car of a pair means the tape is not unblended reward_v0
    -- reward_v8_team's team_spirit=0.3 shrinks a lone scorer's spike to 8.0,
    under GOAL_THRESHOLD, so goals would silently disappear."""
    r = rows([10.0, -10.0], [10.0, 0.0], [10.0, 0.0])
    t = terms([False, False], [False, False], [True, True])
    assert sorted(split_matches(r, t, TH)) == [(0, 1), (3, 0)]      # legal at m=1
    with pytest.raises(ValueError, match="only SOME cars"):
        split_matches(r, t, TH, team_size=2)


def test_ncol_not_divisible_raises():
    """The engine/CLI mode-disagreement guard: 3 learner columns cannot be a
    whole number of 2-car arenas."""
    r = np.zeros((2, 3), dtype=np.float32)
    t = np.zeros((2, 3), dtype=bool)
    with pytest.raises(ValueError, match="whole number"):
        split_matches(r, t, TH, team_size=2)


def test_1d_tape_at_team_size_2_raises():
    """A (T,) tape could be one 2-car arena or two 1-car arenas and nothing in
    the array says which. Refuse rather than guess."""
    with pytest.raises(ValueError, match="ambiguous"):
        split_matches(np.zeros(4, dtype=np.float32), np.zeros(4, dtype=bool),
                      TH, team_size=2)


def test_1d_tape_still_promoted_at_team_size_1():
    """Real single-arena callers pass (T,); that behaviour has to survive."""
    r = np.array([10.0, 0.0, -10.0, 0.0], dtype=np.float32)
    t = np.array([False, True, False, True], dtype=bool)
    assert split_matches(r, t, TH) == [(1, 0), (0, 1)]


def test_durations_align_one_to_one_with_records():
    """Alignment is by construction (one loop, one append site) -- the reason
    with_durations lives inside split_matches instead of a second function that
    would have to be kept in lockstep. Arenas terminate on a STAGGERED schedule
    here precisely so a per-column or per-arena mixup shows up."""
    r = np.zeros((5, 4), dtype=np.float32)
    t = np.zeros((5, 4), dtype=bool)
    t[1, 0:2] = True        # arena 0 ends at t=1  -> duration 2
    t[2, 2:4] = True        # arena 1 ends at t=2  -> duration 3
    t[4, 0:2] = True        # arena 0 ends at t=4  -> duration 5-2 = 3
    recs, durs = split_matches(r, t, TH, team_size=2, with_durations=True)
    assert len(recs) == len(durs) == 3
    assert durs == [2, 3, 3]


def test_a_blowup_closed_record_carries_the_partial_score_not_a_draw():
    """A contained physics blowup emits the match's REAL score so far, not 0-0.

    The engine's blowup branch (episode.rs:779-807) zeroes only the current
    step's reward and returns before start_match(), and in match mode a goal
    never terminates -- so the accumulators here still hold every goal the match
    had scored when the terminated flag lands. This tape is that situation: two
    blue goals and one orange, then a blowup step (all-zero rewards, terminated).
    The record must be (2, 1), a WIN, and it must be SHORT.

    Pinned because the census in scripts/matchwin_gate.py used to claim short
    records were "all draws" and therefore could "never cause a false promotion".
    """
    r = np.array([[10.0], [-10.0], [10.0], [0.0]], dtype=np.float32)
    t = flags(False, False, False, True)
    recs, durs = split_matches(r, t, TH, with_durations=True)
    assert recs == [(2, 1)], "a blowup closes the record at the partial score"
    assert durs == [4] and durs[0] < 4500, "and it is a SHORT record"


def test_durations_default_off_returns_plain_list():
    """flip_to_candidate and match_record both destructure 2-tuples, so the
    DEFAULT return type must stay a plain list of pairs."""
    got = split_matches(tape(10.0, 0.0), flags(False, True), TH)
    assert isinstance(got, list) and got == [(1, 0)]


# --- MatchRunner.play's goal counts ------------------------------------------

class _StubEngine:
    """Just enough engine for play()'s arithmetic -- no physics, no weights."""

    def __init__(self, rewards):
        self._rew = rewards

    def set_weights(self, sd):
        pass

    def set_opponents(self, sds):
        pass

    def collect(self, steps, arena_opponents=None):
        return {"rewards": self._rew}


def _stub_runner(mode, rewards):
    from construct.league.matches import MatchRunner
    mr = MatchRunner.__new__(MatchRunner)      # bypass real Engine construction
    mr.mode = mode
    mr.num_arenas = rewards.shape[1] // mode
    mr.assignment = [0] * mr.num_arenas
    mr.eng = _StubEngine(rewards)
    return mr


def test_play_counts_a_2v2_goal_once_not_twice():
    """The same m x inflation on the goal-share path. Old code: (2, 2).
    The ratio survives the inflation, which is why it would have looked fine
    while champion_gate's min_total_goals guard got 2x easier to satisfy."""
    r = rows([10.0, 10.0], [0.0, 0.0], [-10.0, -10.0])
    assert _stub_runner(2, r).play(None, None) == (1, 1)


def test_play_at_mode_1_is_the_legacy_sum():
    r = rows([10.0, -10.0], [10.0, 0.0], [0.0, 0.0])
    assert _stub_runner(1, r).play(None, None) == (2, 1)


def test_play_at_3v3_counts_per_arena():
    r = rows([10.0, 10.0, 10.0, 0.0, 0.0, 0.0],     # arena 0 scores
             [0.0, 0.0, 0.0, 10.0, 10.0, 10.0])     # arena 1 scores
    assert _stub_runner(3, r).play(None, None) == (2, 0)
