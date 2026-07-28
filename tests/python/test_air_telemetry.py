"""D3: the two instruments the exploration diagnosis needed and did not have.

1. A GATED aerial-touch fraction. `air_tch_frac`
   (`airborne_touch_events / touch_events`) is ANTI-CORRELATED with aerial
   skill -- a RANDOM policy reads 0.87-0.93 on it (the curriculum starts half
   the cars airborne, and a tumbling car's contacts are all "airborne") while
   the champion reads 0.0117 and makes 63x more real aerial touches than v9's
   0.0345. The missing half of the gate is the height, and `air_setup` was
   staged on off a 5%-at-150M threshold read from that broken number.

2. `p_jump`, the softmax mass on jump rows. Under a FROZEN config it was probed
   at 6.5e-6 (222M), 4.36e-2 (248M) and 1.08e-5 (256M) -- a 4,000x oscillation
   with 67 of 104 action rows dead, and not one field in the log line moved
   with it.

Plus the learner/opponent split those two are read through: the counters sum
over every car in every arena, and on the champion the arena-wide
r_aerial_touch of +2.23 was ENTIRELY the ported bot's.
"""
import types

import pytest
import torch

from construct.learn.config import TrainConfig
from construct.learn.train import Trainer


def line(terms: dict) -> str:
    """`_reward_terms_line` unbound against a stub engine -- it reads nothing
    but `self.engine.reward_terms()`."""
    stub = types.SimpleNamespace(
        engine=types.SimpleNamespace(reward_terms=lambda: terms))
    return Trainer._reward_terms_line(stub)


# --- (a) the gated fraction ------------------------------------------------

def test_gated_fraction_separates_air_play_from_being_off_the_ground():
    """The champion's signature: every airborne touch it makes is under the
    z_lo = 400 ramp, which is why its own share of r_aerial_touch is exactly
    0.0000. The old fraction cannot see the difference; the gated one reads 0."""
    s = line({
        "agent_steps": 9000.0, "touch_events": 300.0,
        "airborne_touch_events": 261.0,          # 0.87 -- the random-policy read
        "gated_aerial_touch_events": 0.0,
    })
    assert "air_gate_frac 0.0000" in s
    assert "air_tch_frac 0.870" in s, "the legacy number stays, for log continuity"


def test_gated_fraction_moves_when_real_aerials_happen():
    s = line({
        "agent_steps": 9000.0, "touch_events": 300.0,
        "airborne_touch_events": 120.0, "gated_aerial_touch_events": 45.0,
    })
    assert "air_gate_frac 0.1500" in s


def test_an_engine_without_the_gated_counter_reads_exactly_as_before():
    """Degrade gracefully. The remote runs an INSTALLED .so, not the repo, so
    train.py must be shippable ahead of a rebuilt wheel -- and when it is, the
    line must be the old line rather than a crash or a fabricated zero."""
    s = line({"agent_steps": 9000.0, "touch_events": 300.0,
              "airborne_touch_events": 120.0})
    assert "air_gate_frac" not in s
    assert "tch/min/car 30.0" in s and "air_tch_frac 0.400" in s
    assert " lrn " not in s, "no split to report either"


# --- (a2) the airborne fraction, on the SAME population as the gated one -----

def test_air_any_frac_is_learner_split_so_it_shares_a_population_with_the_gate():
    """THE division nobody had made. Ordering the aerial gate's two conditions
    (airborne, then height) needs `air_gate_frac / air_any_frac`, and every
    attempt divided the learner-only gated fraction by the ARENA-WIDE
    `air_tch_frac` -- mixing a 21%-foreign arena into a statement about our
    policy. Same category error that made an arena-wide r_aerial_touch of +2.23
    read as ours when 0.0000 of it was.

    Here the bot is airborne on 80% of its touches and we are on 10% of ours.
    The arena-wide blend says 0.240; only the split says which is which.
    """
    s = line({
        "agent_steps": 5000.0, "learner_agent_steps": 4000.0, "opp_agent_steps": 1000.0,
        "touch_events": 250.0, "learner_touch_events": 200.0, "opp_touch_events": 50.0,
        "airborne_touch_events": 60.0,
        "learner_airborne_touch_events": 20.0, "opp_airborne_touch_events": 40.0,
        "gated_aerial_touch_events": 12.0,
        "learner_gated_aerial_touch_events": 2.0, "opp_gated_aerial_touch_events": 10.0,
    })
    assert "air_any_frac lrn 0.100 opp 0.800" in s
    assert "air_tch_frac 0.240" in s, "the legacy arena-wide number is untouched"
    # and now the quotient means something: 2/20 of OUR airborne touches clear
    # the ramp, against 10/40 of the bot's.
    assert "air_gate_frac lrn 0.0100 opp 0.2000" in s


def test_air_any_frac_falls_back_to_the_arena_number_on_an_unsplit_engine():
    """An older `.so` has no `learner_` counters, so the split degrades to the
    arena-wide value under a name that no longer claims to be learner-only. It
    must not fabricate a zero."""
    s = line({"agent_steps": 9000.0, "touch_events": 300.0,
              "airborne_touch_events": 120.0})
    assert "air_any_frac 0.400" in s and " lrn " not in s


def test_air_any_frac_does_not_divide_by_zero_on_a_touchless_iteration():
    s = line({"agent_steps": 4000.0, "learner_agent_steps": 2000.0,
              "opp_agent_steps": 2000.0,
              "touch_events": 0.0, "learner_touch_events": 0.0,
              "opp_touch_events": 0.0, "airborne_touch_events": 0.0,
              "learner_airborne_touch_events": 0.0, "opp_airborne_touch_events": 0.0})
    assert "air_any_frac lrn 0.000 opp 0.000" in s


# --- (a3) the touch-height histogram ----------------------------------------

def _z(**kw):
    """Six learner z-buckets, defaulting to zero."""
    Z = ("lt150", "150_300", "300_500", "500_800", "800_1200", "ge1200")
    return {f"learner_air_touch_z_{b}": float(kw.get(b, 0)) for b in Z}


def _z_line(terms_seq):
    """Feed several iterations through ONE stub trainer so the accumulator
    persists, and return the last line."""
    stub = types.SimpleNamespace(_AIR_Z_MIN=Trainer._AIR_Z_MIN)
    out = ""
    for terms in terms_seq:
        stub.engine = types.SimpleNamespace(reward_terms=lambda t=terms: t)
        out = Trainer._reward_terms_line(stub)
    return out


def _iter_terms(airborne, hist):
    return {
        "agent_steps": 90000.0, "learner_agent_steps": 90000.0,
        "touch_events": 400.0, "learner_touch_events": 400.0,
        "airborne_touch_events": float(airborne),
        "learner_airborne_touch_events": float(airborne),
        **_z(**hist),
    }


def test_air_z_accumulates_across_iterations_instead_of_never_firing():
    """THE BUG THIS REPLACES. The first version only printed when a SINGLE
    iteration cleared the floor, but the live rate is ~8 airborne touches per
    iteration and reward_terms() resets on read -- so it was installed on the
    training box and stayed silent forever.

    25 iterations x 8 events = 200 = _AIR_Z_MIN, so the last one reports."""
    per = {"lt150": 6, "150_300": 2}
    seq = [_iter_terms(8, per) for _ in range(25)]
    assert "air_z" not in _z_line(seq[:24]), "must stay silent below the floor"
    s = _z_line(seq)
    assert "air_z 75.0/25.0/0.0/0.0/0.0/0.0 n200" in s


def test_air_z_resets_after_reporting_so_windows_do_not_overlap():
    per = {"lt150": 8}
    stub = types.SimpleNamespace(_AIR_Z_MIN=Trainer._AIR_Z_MIN)
    fired = 0
    for _ in range(75):                      # three full windows
        terms = _iter_terms(8, per)
        stub.engine = types.SimpleNamespace(reward_terms=lambda t=terms: t)
        if "air_z" in Trainer._reward_terms_line(stub):
            fired += 1
    assert fired == 3, f"expected one report per 200 events, got {fired}"


def test_air_z_banks_nothing_from_an_iteration_whose_buckets_do_not_sum():
    """The wiring self-check. A polluted accumulator would outlive the single
    bad iteration that polluted it, so a mismatch must bank nothing at all."""
    good = _iter_terms(8, {"lt150": 8})
    bad = _iter_terms(8, {"lt150": 4})       # sums to 4, not 8
    seq = [bad] * 25 + [good] * 24
    assert "air_z" not in _z_line(seq), "bad iterations must not count toward n"


def test_air_z_absent_on_an_engine_without_the_buckets():
    """The remote runs an INSTALLED .so; train.py must be shippable ahead of a
    rebuilt wheel and degrade to the old line rather than crash."""
    s = line({"agent_steps": 9000.0, "learner_agent_steps": 9000.0,
              "touch_events": 300.0, "learner_touch_events": 300.0,
              "airborne_touch_events": 120.0,
              "learner_airborne_touch_events": 120.0})
    assert "air_z" not in s and "air_any_frac" in s


# --- (c) learner vs opponent ------------------------------------------------

def test_touch_stats_split_learner_from_opponent():
    """Half the agent-steps are the ported bot's, and it is doing all of the
    aerial work. Unsplit, the line says air play is happening."""
    s = line({
        "agent_steps": 8000.0, "learner_agent_steps": 4000.0,
        "touch_events": 200.0, "learner_touch_events": 80.0,
        "airborne_touch_events": 100.0,
        "gated_aerial_touch_events": 60.0, "learner_gated_aerial_touch_events": 0.0,
        "aerial_touch": 2.23, "learner_aerial_touch": 0.0,
    })
    # 80 touches over 4000 steps at 15 decisions/s = 18.0/min/car for us,
    # 120 over 4000 = 27.0 for the bot.
    assert "tch/min/car lrn 18.0 opp 27.0" in s
    assert "air_gate_frac lrn 0.0000 opp 0.5000" in s
    assert "r_aerial_touch lrn 0.00 opp 2.23" in s, (
        "the champion's exact reading: the whole term was the opponent's"
    )


def test_a_self_play_only_run_reports_no_opponent_column():
    """With no opponent-driven cars the two counter sets are equal and an `opp`
    column would be a column of zeros pretending to mean something."""
    s = line({
        "agent_steps": 4000.0, "learner_agent_steps": 4000.0,
        "touch_events": 100.0, "learner_touch_events": 100.0,
        "airborne_touch_events": 20.0,
        "gated_aerial_touch_events": 5.0, "learner_gated_aerial_touch_events": 5.0,
    })
    assert "opp" not in s
    assert "tch/min/car lrn 22.5" in s


def test_a_zero_touch_iteration_does_not_divide_by_zero():
    s = line({"agent_steps": 4000.0, "learner_agent_steps": 2000.0,
              "touch_events": 0.0, "learner_touch_events": 0.0,
              "gated_aerial_touch_events": 0.0,
              "learner_gated_aerial_touch_events": 0.0})
    assert "air_gate_frac lrn 0.0000 opp 0.0000" in s


def test_the_opponent_column_is_the_bots_own_counter_not_a_subtraction():
    """A PARTIAL foreign team has a third kind of car: orange cars the bot cannot
    drive are mirrored through OUR net and then dropped from the training set. In
    v9's shipped config 31 of the 79 non-learner rows -- 39% -- are those
    mirrors, so `total - learner` prints a blend of the bot and ourselves under a
    label that says it is the bot.

    Sized on the live run: 391 learner rows, 48 bot cars, 31 mirrors. At a true
    bot aerial-gate rate of 0.24 and ours of 0.02, the subtraction reads 0.153.
    """
    # touches proportional to steps; the bot gates 0.24 of its, we gate 0.02.
    s = line({
        "agent_steps": 4700.0, "learner_agent_steps": 3910.0, "opp_agent_steps": 480.0,
        "touch_events": 470.0, "learner_touch_events": 391.0, "opp_touch_events": 48.0,
        "airborne_touch_events": 300.0,
        "gated_aerial_touch_events": 19.34,          # 0.02*391 + 0.24*48 + 0.02*31
        "learner_gated_aerial_touch_events": 7.82,
        "opp_gated_aerial_touch_events": 11.52,
    })
    assert "air_gate_frac lrn 0.0200 opp 0.2400" in s, (
        "the bot's own rate; (470-391) touches and (19.34-7.82) gated would say 0.1458"
    )


def test_an_engine_without_the_opp_counters_still_subtracts():
    """The remote runs an INSTALLED .so. Until it is rebuilt there are no `opp_`
    keys, and the line must fall back to the subtraction rather than print zeros
    -- diluted, but the same number the log has been carrying."""
    s = line({
        "agent_steps": 4700.0, "learner_agent_steps": 3910.0,
        "touch_events": 470.0, "learner_touch_events": 391.0,
        "airborne_touch_events": 300.0,
        "gated_aerial_touch_events": 19.34,
        "learner_gated_aerial_touch_events": 7.82,
    })
    assert "air_gate_frac lrn 0.0200 opp 0.1458" in s


# --- (b) p_jump -------------------------------------------------------------

def air_cfg(tmp_path):
    cfg = TrainConfig.load("configs/train_v1.toml")
    cfg.schema_path = "schema/v1_air.toml"      # the 104-row table v9 launches on
    cfg.env.update(num_arenas=2, blue=1, orange=1)
    cfg.env.pop("team_size_weights", None)
    cfg.curriculum_config_path = ""
    cfg.net = {"d_model": 32, "layers": 1, "heads": 2, "ff": 64}
    cfg.ppo.update(rollout_steps=8, minibatch_size=256)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=1000)
    cfg.league, cfg.kickstart = {}, {}
    return cfg


def test_p_jump_is_the_softmax_mass_on_jump_rows(tmp_path):
    t = Trainer(air_cfg(tmp_path))
    batch = t.collect(8)
    got = t._p_jump(batch)

    obs = batch["obs"]
    any_obs = next(iter(obs.values()))
    idx = t._p_jump_idx(any_obs.shape[0], any_obs.device)
    with torch.no_grad():
        logits, _ = t.net(**{k: v[idx] for k, v in obs.items()})
    want = float(torch.softmax(logits, -1)[:, t.net.action_table[:, 5] > 0].sum(-1).mean())
    assert got == pytest.approx(want, rel=1e-6)
    assert 0.0 < got < 1.0


def test_p_jump_samples_every_learner_column_not_every_32nd():
    """THE DEGENERATE CASE THE FIXED STRIDE HAD. Rows are arena-major/car-minor
    per timestep, so row r's learner column is `r % N`, and a stride of
    `step = (T*N) // 8192` reaches only `N / gcd(step, N)` columns. At T=256 the
    step is N//32, so whenever N is a multiple of 32 it DIVIDES N and the sample
    collapses to 32 columns -- and always car 0 of every g-th arena, i.e.
    systematically the blue car that takes the kickoff.

    `--num-arenas` is a routine `resume_train.py` override and N also shifts with
    the foreign fraction and the team-size mix, so the instrument's sampling
    character could change between two runs being compared with nothing in the
    log to say so. It did not bite at the live N of 391; it was a lottery on a
    number nobody was watching.
    """
    n, T = 384, 256                     # 384 = 12 * 32
    rows = n * T
    t = types.SimpleNamespace(total_steps=296_184_064,
                              _P_JUMP_ROWS=Trainer._P_JUMP_ROWS)

    stride = torch.arange(0, rows, max(1, rows // Trainer._P_JUMP_ROWS))
    assert len({int(i) % n for i in stride}) == 32, "the old behaviour, pinned"

    idx = Trainer._p_jump_idx(t, rows, torch.device("cpu"))
    assert len(idx) == Trainer._P_JUMP_ROWS
    assert len(set(idx.tolist())) == Trainer._P_JUMP_ROWS, "no row sampled twice"
    assert len({int(i) % n for i in idx}) == n, "every learner column is reachable"
    assert len({int(i) // n for i in idx}) == T, "and every timestep"


def test_p_jump_sampling_is_reproducible_from_the_step_count():
    """Seeded on `total_steps`, so the same checkpoint re-measures the same
    states -- and two different iterations do not measure the same ones."""
    mk = lambda steps: types.SimpleNamespace(
        total_steps=steps, _P_JUMP_ROWS=Trainer._P_JUMP_ROWS)
    rows, dev = 384 * 256, torch.device("cpu")
    a = Trainer._p_jump_idx(mk(1000), rows, dev)
    assert torch.equal(a, Trainer._p_jump_idx(mk(1000), rows, dev))
    assert not torch.equal(a, Trainer._p_jump_idx(mk(1001), rows, dev))


def test_p_jump_takes_the_whole_batch_when_it_is_smaller_than_the_sample():
    t = types.SimpleNamespace(total_steps=7, _P_JUMP_ROWS=Trainer._P_JUMP_ROWS)
    idx = Trainer._p_jump_idx(t, 100, torch.device("cpu"))
    assert sorted(idx.tolist()) == list(range(100))


def test_p_jump_does_not_draw_from_the_global_rng(tmp_path):
    """It must be measurable without changing the run. ppo_update's minibatch
    permutation draws from the same global generator, so one random number
    consumed here would shift every subsequent update -- a telemetry line that
    changes the training it reports on."""
    t = Trainer(air_cfg(tmp_path))
    batch = t.collect(8)
    before = torch.random.get_rng_state()
    t._p_jump(batch)
    assert torch.equal(before, torch.random.get_rng_state())


def test_p_jump_leaves_no_gradients_behind(tmp_path):
    t = Trainer(air_cfg(tmp_path))
    batch = t.collect(8)
    t.opt.zero_grad(set_to_none=True)
    t._p_jump(batch)
    assert all(p.grad is None for p in t.net.parameters()), (
        "an extra forward for telemetry must not put anything in .grad"
    )


def test_p_jump_is_none_for_a_table_with_no_jump_column(tmp_path):
    """v0 nets, and any future table without the column: no field, no crash."""
    t = Trainer(air_cfg(tmp_path))
    batch = t.collect(8)
    t.net.action_table = t.net.action_table[:, :4]
    assert t._p_jump(batch) is None
