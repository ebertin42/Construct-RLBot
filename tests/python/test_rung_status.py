"""The reporting bug this script exists to prevent.

Every rung report on 2026-07-27 ran `... | head -4` against a 12-slot roster.
Slots 0-3 are exactly the four 1v1 slots, so the eight 2v2/3v3 slots were
invisible and the run was described as "converged" off a quarter of the data.
The tests below pin the two properties that made that possible: reporting a
SUBSET without saying so, and reading the wrong side of a rung MOVE.
"""
import importlib.util
import pathlib

spec = importlib.util.spec_from_file_location(
    "rung_status", pathlib.Path(__file__).resolve().parents[2] / "scripts" / "rung_status.py")
rs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rs)


def line(segs):
    return "auto-curriculum: " + " | ".join(segs)


def roster(over=None):
    """A full 12-segment line, every slot at the floor unless overridden.

    `over` is an int-keyed dict, so it cannot be passed as **kwargs -- Python
    requires string keyword names.
    """
    over = over or {}
    segs = []
    for i, (kind, m) in enumerate(rs.SLOTS):
        segs.append(over.get(i, f"{kind} wr0.50/ema0.50 1c/p24"))
    return line(segs)


def test_every_slot_is_parsed_not_just_the_first_four():
    rows = rs.parse([roster()])
    assert len(rows) == 1
    assert len(rows[0]) == 12, "a 12-slot roster must yield 12 records"
    # and the mode mapping must be the config's order, not four-of-each
    assert [m for _, m in rs.SLOTS] == [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]


def test_a_rung_move_reports_where_the_slot_landed():
    """`1c/p20->1c/p16` means the slot MOVED to p16. Reading the first rung
    reports where it just left, which reads as "no movement" forever."""
    rows = rs.parse([roster({0: "element wr0.83/ema0.68 1c/p20->1c/p16"})])
    assert rows[0][0]["p"] == 16
    assert rows[0][0]["cars"] == 1


def test_an_unmeasured_slot_carries_its_rung_with_no_win_rate():
    """The staggered eval measures ONE team size per cycle, so 8 of 12 segments
    read `n/a` on any line. They must keep their rung and NOT report a wr --
    a fabricated 0.0 there would look like a collapse."""
    rows = rs.parse([roster({4: "necto n/a 1c/p24"})])
    d = rows[0][4]
    assert d["wr"] is None and d["ema"] is None
    assert (d["cars"], d["p"]) == (1, 24)


def test_a_line_with_the_wrong_slot_count_is_skipped_not_misaligned():
    """An older roster (or a banner) must not be silently mapped onto the
    current 12-slot order -- that would attribute one bot's rung to another."""
    assert rs.parse([line(["element wr0.5/ema0.5 1c/p24"] * 4)]) == []


def test_harder_is_lexicographic_cars_then_period():
    # more cars is harder regardless of period; then lower period is harder
    assert rs.harder((2, 24), (1, 1)), "car count dominates"
    assert rs.harder((1, 12), (1, 24)), "lower period is harder at equal cars"
    assert not rs.harder((1, 24), (1, 24)), "equal is not harder"


def test_the_floor_is_not_mistaken_for_progress():
    """p24 is the EASIEST rung on the ladder. A slot sitting there has not
    advanced, however healthy its win rate looks."""
    assert rs.LADDER[-1] == 24 and rs.LADDER[0] == 1
    assert rs.harder((1, rs.LADDER[0]), (1, rs.LADDER[-1]))
