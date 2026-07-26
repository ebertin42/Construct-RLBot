"""Per-BLOCK opponent placement (v9 D8, train.plan_block_assignment).

WHAT WENT WRONG BEFORE. v8 stamped foreign arenas onto the FRONT of the whole
arena list and league arenas onto the BACK. Arenas are laid out in 1v1/2v2/3v3
blocks, so with team_size_weights [0.85, 0.1, 0.05] that put every league arena
in the 3v3/2v2 tail: v8 ended up with ZERO 3v3 self-play arenas and 100% of its
3v3 experience against one frozen, 1v1-trained, near-random opponent. Nobody
decided that -- it fell out of tail assignment.

WHAT PER-BLOCK BUYS. With foreign and league placed as the same fraction inside
each block,

    rows_m = a_m * m * (2 - phi - lambda)

so equal learner rows <=> a_m proportional to 1/m <=> weights [6, 3, 2],
INDEPENDENT of both opponent fractions. The invariant test below is the one that
matters: retune either fraction and the 1s/2s/3s split must stay balanced.
"""
import pytest

from construct.learn.train import plan_block_assignment, team_size_blocks

# engine::allocate_team_sizes(144, [6,3,2]) -> 79 / 39 / 26
V9_SIZES = [1] * 79 + [2] * 39 + [3] * 26
PHI = 0.3333
LAMBDA = 0.115
# 8 slots = (bot x team size): element/immortal/necto/nexto at 1v1,
# necto/nexto at 2v2 and again at 3v3.
V9_MODES = [1, 1, 1, 1, 2, 2, 3, 3]


def _counts(a, sizes):
    """(foreign, league, self-play) arena counts per team size."""
    out = {}
    for m in sorted(set(sizes)):
        idx = [i for i, s in enumerate(sizes) if s == m]
        out[m] = (
            sum(1 for i in idx if a[i] <= -2),
            sum(1 for i in idx if a[i] >= 0),
            sum(1 for i in idx if a[i] == -1),
        )
    return out


def _rows(a, sizes):
    """Learner rows per team size: a self-play arena yields 2m, an opponent
    arena (foreign or league) yields m -- its orange side does not learn."""
    out = {}
    for m in sorted(set(sizes)):
        out[m] = sum((2 * m if a[i] == -1 else m)
                     for i, s in enumerate(sizes) if s == m)
    return out


def test_blocks_are_detected():
    assert team_size_blocks(V9_SIZES) == [(1, 0, 79), (2, 79, 39), (3, 118, 26)]
    assert team_size_blocks([1, 1, 1]) == [(1, 0, 3)]


def test_unsorted_team_sizes_are_refused():
    # Everything downstream indexes arenas by block; an unsorted list would
    # place opponents in the wrong regime silently.
    with pytest.raises(AssertionError, match="1s/2s/3s blocks"):
        team_size_blocks([1, 3, 2])


def test_v9_layout_matches_the_config_header():
    """The exact numbers configs/train_v9_fromscratch.toml documents."""
    a = plan_block_assignment(V9_SIZES, V9_MODES, PHI, 4, LAMBDA)
    assert _counts(a, V9_SIZES) == {1: (26, 9, 44), 2: (13, 4, 22), 3: (9, 3, 14)}
    assert _rows(a, V9_SIZES) == {1: 123, 2: 122, 3: 120}
    total = sum(_rows(a, V9_SIZES).values())
    assert total == 365
    shares = [r / total for r in _rows(a, V9_SIZES).values()]
    for s in shares:
        assert abs(s - 1 / 3) < 0.01, f"row shares must be ~equal thirds, got {shares}"


def test_equal_rows_is_independent_of_both_opponent_fractions():
    """THE invariant. `rows_m = a_m * m * (2 - phi - lambda)` has no phi or
    lambda dependence in the RATIO, so retuning either must leave the split
    balanced. Under v8's front/tail stamping this is false."""
    for phi in (0.0, 1 / 6, 1 / 3, 0.5):
        for lam in (0.0, 0.115, 0.25):
            if phi + lam > 0.95:
                continue
            a = plan_block_assignment(V9_SIZES, V9_MODES, phi, 4, lam)
            rows = _rows(a, V9_SIZES)
            spread = (max(rows.values()) - min(rows.values())) / max(rows.values())
            assert spread < 0.05, f"phi={phi} lambda={lam}: rows {rows}"


def test_a_block_only_receives_slots_of_its_own_mode():
    """This is what confines element/immortal (fixed 107-float obs, panics at
    2v2) to the 1v1 block. A block must never borrow another mode's bot -- that
    would hard-error deep inside a worker instead of here."""
    a = plan_block_assignment(V9_SIZES, V9_MODES, PHI, 4, LAMBDA)
    for i, m in enumerate(V9_SIZES):
        if a[i] <= -2:
            slot = -a[i] - 2
            assert V9_MODES[slot] == m, (
                f"arena {i} is {m}v{m} but carries slot {slot} (mode {V9_MODES[slot]})"
            )


def test_block_without_a_matching_slot_gets_no_foreign_arenas():
    # Only 1v1 bots configured: the 2v2/3v3 blocks stay foreign-free rather than
    # being handed an incompatible bot. This is the D9-slips fallback (§3.4).
    a = plan_block_assignment(V9_SIZES, [1, 1], PHI, 4, LAMBDA)
    c = _counts(a, V9_SIZES)
    assert c[1][0] == 26
    assert c[2][0] == 0 and c[3][0] == 0


def test_foreign_and_league_never_overlap():
    a = plan_block_assignment(V9_SIZES, V9_MODES, PHI, 4, LAMBDA)
    assert len(a) == len(V9_SIZES)
    for k in a:
        assert k == -1 or k >= 0 or k <= -2  # exactly one encoding per arena
    # foreign takes the front of each block, league the back
    for m, start, count in team_size_blocks(V9_SIZES):
        block = a[start:start + count]
        n_for = sum(1 for k in block if k <= -2)
        n_lg = sum(1 for k in block if k >= 0)
        assert all(k <= -2 for k in block[:n_for])
        assert all(k >= 0 for k in block[count - n_lg:]) if n_lg else True


def test_block_overflow_is_asserted_per_block_not_globally():
    """A GLOBAL check passes while a single block overflows -- which is exactly
    the shape of the v8 trap (foreign_frac 0.55 of 192 arenas = 106, more than
    the 64 1v1 arenas that equal thirds would leave)."""
    with pytest.raises(AssertionError, match="block"):
        plan_block_assignment(V9_SIZES, V9_MODES, 0.9, 4, 0.2)


def test_single_block_reduces_to_the_historical_front_tail_layout():
    """Every pre-v9 config is one 1v1 block, and must be placed exactly as
    before: foreign at the front, league at the tail, both round-robin."""
    sizes = [1] * 4
    a = plan_block_assignment(sizes, [], 0.0, 2, 0.5)
    assert a == [-1, -1, 0, 1], "league tail round-robin, unchanged"
    a = plan_block_assignment(sizes, [1, 1], 0.5, 0, 0.0)
    assert a == [-2, -3, -1, -1], "foreign front round-robin, unchanged"


def test_zero_league_slots_places_no_league_arenas():
    a = plan_block_assignment(V9_SIZES, V9_MODES, PHI, 0, LAMBDA)
    assert all(k < 0 for k in a)
