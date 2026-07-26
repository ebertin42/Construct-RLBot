"""The auto-curriculum controller must survive a restart (D6), and must not act
on the first evals after one (D5).

THE FAILURE THESE PIN. Nothing in `save_checkpoint` recorded the controller, so
every resume rebuilt all twelve rungs from `decision_periods` in the TOML and
cleared every EMA and dwell counter. The v9 restart at 222M reset all twelve
slots to the seed p24 -- so the restart was not the single-variable change it
was performed as, and the run then spent tens of millions of steps re-walking a
ladder it had already climbed (one rung per `adjust_dwell + 1` measurements, and
with the staggered eval a slot is measured once every `adjust_every * n_modes`
iterations).

Written against a REAL Trainer on the real engine rather than a stub, because
the thing under test is the checkpoint seam itself: which keys are written, and
what `Trainer.__init__` does with them on the way back in.
"""
import pathlib

import pytest
import torch

from construct.learn.train import Trainer

CACHE = pathlib.Path.home() / ".cache/construct"
ELEMENT, NECTO = CACHE / "element_weights.npz", CACHE / "necto_weights.npz"

needs_weights = pytest.mark.skipif(
    not (ELEMENT.exists() and NECTO.exists()),
    reason="foreign bot weights absent (unlicensed, never committed)",
)

# The ladder the fixture ships with. Deliberately NOT v9's, so these tests pin
# the mechanism rather than today's config; the trim case below supplies its own.
LADDER = "[1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32]"


def write_cfg(tmp_path, name="t.toml", ladder=LADDER, slots=("element", "necto"),
              periods="[24, 24]"):
    """A minimal two-slot foreign config on disk, so `load_checkpoint` (which
    takes a config PATH) exercises the real resume path."""
    rows = ",\n  ".join(
        f'{{ kind = "{k}", mode = 1, weights = "{CACHE / (k + "_weights.npz")}" }}'
        for k in slots
    )
    p = tmp_path / name
    p.write_text(f"""
schema_path = "schema/v1.toml"
reward_config_path = "configs/reward_v0.toml"

[env]
num_arenas = 4
blue = 1
orange = 1
seed = 0

[net]
d_model = 32
layers = 1
heads = 2
ff = 64

[ppo]
rollout_steps = 8
gamma = 0.99
lam = 0.95
lr = 3e-4
clip = 0.2
entropy_coef = 0.003
value_coef = 1.0
epochs = 1
minibatch_size = 256

[run]
device = "cpu"
checkpoint_dir = "{tmp_path}"
save_every_iters = 1000

[foreign]
enabled = true
opponent_frac = 0.5
decision_periods = {periods}
foreign_cars = {[1] * len(slots)}
slots = [
  {rows}
]

[foreign.auto_curriculum]
enabled = false
period_ladder = {ladder}
win_lo = 0.35
win_hi = 0.65
adjust_every = 1
adjust_dwell = 1
""")
    return str(p)


def moved_trainer(cfg_path):
    """A Trainer whose controller has been walked away from its config seeds in
    every persisted field -- the rungs, and the measurement state that is written
    for provenance but deliberately not restored."""
    from construct.learn.config import TrainConfig
    t = Trainer(TrainConfig.load(cfg_path))
    t._rung_harder(0)          # element: p24 -> p20
    t._rung_harder(0)          # -> p16
    t._rung_easier(1)          # necto: p24 -> p32
    t._ac_wr_ema = [0.62, 0.41]
    t._ac_since = [3, 0]
    t._ac_cycle = 7
    t.total_steps = 275_000_000
    return t


def curriculum_step(t, wrs):
    """One `_auto_curriculum_step` on a real (resumed) Trainer with the eval
    replaced by a fixed reading. The config ships `enabled = false` so the
    controller is inert during construction; these tests are about what it does
    with the state a resume handed it."""
    t._ac_on = True
    t._ac_every = 1
    t._ac_eval = object()               # non-None -> the eval branch is taken
    t._ac_wr_alpha = 0.3                # the shipped smoothing
    t._measure_foreign_winrates = lambda modes=None: list(wrs)
    t._auto_curriculum_step(1, ep_reward_mean=0.0)


@needs_weights
def test_rungs_survive_a_checkpoint_round_trip(tmp_path):
    cfg_path = write_cfg(tmp_path)
    t = moved_trainer(cfg_path)
    want = {f: list(getattr(t, f)) if isinstance(getattr(t, f), list) else getattr(t, f)
            for f in Trainer._CURRICULUM_FIELDS}
    assert want["_foreign_periods"] == [16, 32], "precondition: the rungs actually moved"

    p = str(tmp_path / "ck.pt")
    t.save_checkpoint(p)
    t2 = Trainer.load_checkpoint(p, cfg_path=cfg_path)

    for f in Trainer._CURRICULUM_FIELDS:
        got = getattr(t2, f)
        assert (list(got) if isinstance(got, list) else got) == want[f], (
            f"{f} did not survive the restart: {got!r} != {want[f]!r}"
        )
    # And the headline consequence, stated as itself: the seed is p24, so a
    # controller that had NOT been restored would read [24, 24] here.
    assert t2._foreign_periods == [16, 32]


@needs_weights
def test_the_ema_and_the_dwell_do_not_survive_a_restart(tmp_path):
    """The rung is knowledge about the OPPONENT; the EMA and the dwell counter
    are properties of a POLICY, and a restart exists in order to change that
    policy (this one changes entropy_coef).

    They are still WRITTEN, so a checkpoint can be read post-hoc, and they are
    ignored on the way back in.
    """
    cfg_path = write_cfg(tmp_path)
    t = moved_trainer(cfg_path)
    p = str(tmp_path / "ck.pt")
    t.save_checkpoint(p)

    state = torch.load(p, map_location="cpu", weights_only=False)
    assert state["curriculum"]["_ac_wr_ema"] == [0.62, 0.41], "written for provenance"
    assert state["curriculum"]["_ac_since"] == [3, 0]

    t2 = Trainer.load_checkpoint(p, cfg_path=cfg_path)
    assert t2._foreign_periods == [16, 32], "the rungs still come back"
    assert t2._ac_wr_ema == [None, None], (
        "a smoothed win rate measured by a different policy is not evidence"
    )
    assert t2._ac_since == [0, 0], "and every slot re-serves its dwell"


@needs_weights
def test_a_restored_slot_cannot_move_a_rung_on_its_first_post_warmup_eval(tmp_path):
    """THE CONSEQUENCE, end to end, and the thing a six-field round-trip test
    cannot see.

    `_ac_since` only ever resets on a move, so in steady state it comes back well
    past the dwell -- measured, all twelve slots in the last four
    `auto-curriculum:` lines of the v9 log print with no `dwellN/M` suffix. Had
    the counter and the EMA been restored, the first eval after the 20M warmup
    would decide on `0.7 * (a pre-restart reading) + 0.3 * (today's)`: a stored
    0.90 plus a fresh 0.50 -- dead centre of [0.35, 0.65] -- gives 0.78 and
    HARDENS a rung that today's measurement says is fair.
    """
    cfg_path = write_cfg(tmp_path)
    t = moved_trainer(cfg_path)
    t._ac_wr_ema, t._ac_since = [0.90, 0.90], [37, 37]   # the logged steady state
    p = str(tmp_path / "ck.pt")
    t.save_checkpoint(p)

    t2 = Trainer.load_checkpoint(p, cfg_path=cfg_path)
    t2.total_steps = t2._ac_start_steps + t2._ac_warmup_steps    # warmup served
    curriculum_step(t2, [0.50, 0.50])

    assert t2._foreign_periods == [16, 32], (
        "an in-band measurement must HOLD; 0.7*0.90 + 0.3*0.50 = 0.78 would harden"
    )
    assert t2._ac_wr_ema == [0.50, 0.50], "the fresh reading seeds the EMA on its own"
    assert t2._ac_since == [1, 1], "and the first measurement is spent on the dwell"


@needs_weights
def test_a_checkpoint_written_before_this_existed_falls_back_to_the_config_seeds(tmp_path):
    """Degrade gracefully: every checkpoint in the repo predates the key, and
    resuming one must behave exactly as it did before -- config seeds, cleared
    EMA, dwell counters at their fresh-run value."""
    cfg_path = write_cfg(tmp_path)
    t = moved_trainer(cfg_path)
    p = str(tmp_path / "ck.pt")
    t.save_checkpoint(p)

    state = torch.load(p, map_location="cpu", weights_only=False)
    del state["curriculum"]          # exactly what a pre-2026-07-26 file looks like
    torch.save(state, p)

    t2 = Trainer.load_checkpoint(p, cfg_path=cfg_path)
    assert t2._foreign_periods == [24, 24], "must fall back to decision_periods"
    assert t2._foreign_cars == [1, 1]
    assert t2._ac_wr_ema == [None, None], "no stale EMA out of nowhere"
    assert t2._ac_since == [0, 0]
    assert t2._ac_cycle == 0
    assert t2.total_steps == 275_000_000, "the rest of the resume is unaffected"


@needs_weights
def test_a_changed_slot_list_refuses_the_restore_rather_than_misassigning_rungs(tmp_path):
    """Rung state is POSITIONAL. Restoring slot 1's ladder index onto a
    different bot would hand it another bot's difficulty with nothing in the log
    to say so, so a slot list that has changed shape or order is refused whole."""
    cfg_path = write_cfg(tmp_path)
    t = moved_trainer(cfg_path)
    p = str(tmp_path / "ck.pt")
    t.save_checkpoint(p)

    other = write_cfg(tmp_path, name="swapped.toml", slots=("necto", "element"))
    t2 = Trainer.load_checkpoint(p, cfg_path=other)
    assert t2._foreign_periods == [24, 24], "a reordered roster must not inherit rungs"
    assert t2._ac_wr_ema == [None, None]


@needs_weights
def test_a_trimmed_ladder_remaps_to_the_nearest_surviving_rung(tmp_path):
    """THE D7 RISK, checked rather than assumed.

    Trimming the ladder's easy tail (v9 drops p32 and p48 so p24 is the floor)
    strands any slot parked out there, and its stored ladder INDEX now points
    past the end of the array. The restore re-derives the index from the stored
    PERIOD, so the slot lands on the nearest surviving rung -- p32 -> p24, one
    rung in the old ladder -- instead of raising, or silently keeping an index
    that names a different difficulty.
    """
    wide = write_cfg(tmp_path, name="wide.toml", ladder="[1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32]")
    t = moved_trainer(wide)
    assert t._foreign_periods[1] == 32 and t._ac_pidx[1] == 10, "precondition: parked at p32"
    p = str(tmp_path / "ck.pt")
    t.save_checkpoint(p)

    trimmed = write_cfg(tmp_path, name="trimmed.toml",
                        ladder="[1, 2, 3, 4, 6, 8, 12, 16, 20, 24]")
    t2 = Trainer.load_checkpoint(p, cfg_path=trimmed)
    assert t2._foreign_periods[1] == 24, "p32 must land on the nearest surviving rung"
    assert t2._ac_pidx[1] == 9, "and its index must name that rung, in range"
    assert 0 <= t2._ac_pidx[1] < len(t2._ac_ladder)
    # the slot that was inside the trim is untouched
    assert t2._foreign_periods[0] == 16 and t2._ac_ladder[t2._ac_pidx[0]] == 16
    # THE TRIM IS ITSELF A RUNG MOVE -- p32 -> p24 is a real difficulty increase,
    # made by editing a TOML rather than by a measurement -- so the EMA that
    # described p32 must not survive it. This is the same rule
    # `_auto_curriculum_step` applies to its own moves, and without it the first
    # post-warmup eval would harden p24 on evidence gathered at p32, having never
    # measured p24.
    assert t2._ac_wr_ema[1] is None
    assert t2._ac_since[1] == 0
    # and the controller still works from there: one rung easier is the floor,
    # and a second call saturates instead of indexing off the end.
    t2._rung_easier(1)
    assert t2._foreign_periods[1] == 24 and t2._ac_pidx[1] == 9


@needs_weights
def test_the_ladder_index_always_names_the_stored_period(tmp_path):
    """The invariant that makes the remap safe: `ladder[pidx] == period` after a
    restore, for every slot, whatever the ladder did in between. A pair that
    disagrees is the silent mis-decode -- the engine runs one difficulty and the
    controller reasons about another."""
    wide = write_cfg(tmp_path, name="wide.toml")
    t = moved_trainer(wide)
    p = str(tmp_path / "ck.pt")
    t.save_checkpoint(p)
    for ladder in ("[1, 2, 3, 4, 6, 8, 12, 16, 20, 24]",     # trimmed
                   "[2, 4, 8, 16]",                          # coarse and short
                   "[1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48]"):   # extended
        cfg_path = write_cfg(tmp_path, name=f"l{len(ladder)}.toml", ladder=ladder)
        t2 = Trainer.load_checkpoint(p, cfg_path=cfg_path)
        for s in range(len(t2._foreign_periods)):
            assert 0 <= t2._ac_pidx[s] < len(t2._ac_ladder), (ladder, s)
            assert t2._ac_ladder[t2._ac_pidx[s]] == t2._foreign_periods[s], (ladder, s)


@needs_weights
def test_a_fresh_run_starts_with_a_full_dwell_and_a_warmup(tmp_path):
    """D5's two seeds, on a real Trainer. `_ac_since` used to start at 10**6 --
    "free to move at start" -- which let the FIRST eval a process ever ran move
    a rung, and that eval is the least trustworthy one there is (measured across
    restarts: 0.11-0.28 followed by 0.67-0.92 one rung apart)."""
    from construct.learn.config import TrainConfig
    t = Trainer(TrainConfig.load(write_cfg(tmp_path)))
    assert t._ac_since == [0, 0], "no slot may act on its first measurement"
    assert t._ac_warmup_steps == 20_000_000
    assert t._ac_start_steps == 0, "a fresh run is gated for its first 20M too"


@needs_weights
def test_the_warmup_anchor_is_this_process_not_the_lineage(tmp_path):
    """A resume at 275M must be gated to 295M, not treated as long past 20M.
    That is the whole point: the unreliable evals are the ones just after a
    RESTART, and `total_steps` alone cannot see a restart."""
    cfg_path = write_cfg(tmp_path)
    t = moved_trainer(cfg_path)
    p = str(tmp_path / "ck.pt")
    t.save_checkpoint(p)
    t2 = Trainer.load_checkpoint(p, cfg_path=cfg_path)
    assert t2._ac_start_steps == 275_000_000
    assert t2.total_steps - t2._ac_start_steps < t2._ac_warmup_steps


# --------------------------------------------------------------- D4 / D7 ----
# The shipped v9 config, asserted where a reader can see the reasoning next to
# the number. `cfg.foreign` IS re-read from the TOML on a resume (load_checkpoint
# only substitutes net/ppo/env), so these values take effect on the next restart.


def v9_ac():
    import tomllib
    with open("configs/train_v9_fromscratch.toml", "rb") as f:
        return tomllib.load(f)["foreign"]


def test_v9_ladder_floors_at_p24():
    """D7. v9's element/1v1 slot walked down to p32 and sat at a comfortable
    0.567 for 210M steps -- in band, therefore never touched again. A rung the
    controller is happy with is a rung the policy is never forced past; v8's
    floor was p12 and its 1v1 slots had to learn their way out."""
    lad = v9_ac()["auto_curriculum"]["period_ladder"]
    assert lad == sorted(lad), "index 0 must stay the HARDEST rung"
    assert lad[-1] == 24, f"p24 must be the floor, got {lad[-1]}"
    assert 32 not in lad and 48 not in lad


def test_v9_seed_period_is_on_the_trimmed_ladder():
    """The trim's one hard requirement: `decision_periods` must still name a
    ladder entry, or every slot cold-starts on a nearest-match the config never
    asked for. p24 is the last entry, so a fresh run seeds at index len-1 --
    the easiest expressible rung -- and no index can run off the end."""
    fg = v9_ac()
    lad = fg["auto_curriculum"]["period_ladder"]
    for p in fg["decision_periods"]:
        assert p in lad, f"seed p{p} is not on the ladder {lad}"
    assert lad.index(24) == len(lad) - 1


def test_v9_cadence_and_dwell_move_together():
    """D4. The dwell is denominated in MEASUREMENTS but protects against acting
    on a policy that has not changed, which is a quantity of ITERATIONS. 20/2
    held a slot 3 measurements x 60 iters = 180; 40/1 holds it 2 x 120 = 240 --
    shorter dwell, longer wait, half as many full eval passes."""
    ac = v9_ac()["auto_curriculum"]
    assert ac["adjust_every"] == 40
    assert ac["adjust_dwell"] == 1
    n_modes = len({s["mode"] for s in v9_ac()["slots"]})
    assert ac["eval_stagger"] is True
    iters_between_moves = ac["adjust_every"] * n_modes * (ac["adjust_dwell"] + 1)
    assert iters_between_moves >= 180, (
        f"a slot would be free to move every {iters_between_moves} iters, "
        "faster than the 180 the L4 dwell simulation was run against"
    )
