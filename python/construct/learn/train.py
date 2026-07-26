import os
import random
import time
import tomllib

import numpy as np
import torch

from construct._engine import Engine, action_table_v1, action_table_v1_air, schema_dict
from construct.learn.config import TrainConfig
from construct.learn.gae import compute_gae
from construct.learn.kickstart import (
    TEACHER_OBS_SIZE,
    KickstartSchedule,
    KickstartTeacher,
    kickstart_losses,
)
from construct.learn.kl_prior import KLPrior, kl_student_prior
from construct.learn.model import PolicyValueNet
from construct.learn.model_v1 import EntityPolicyNet
from construct.learn.ppo import ppo_update
from construct.tables import DEFAULT_V1_TABLE


def compose_extra_loss(net, batch, *, kickstart, lambda_k, lambda_v,
                       prior_logits, lambda_p):
    """Build the per-minibatch extra-loss hook for ppo_update, composing the
    kickstart distillation term (KL(teacher‖student)+value MSE, annealed) and
    the BC-prior anchor (KL(student‖prior), constant lambda_p). Returns None
    when neither is active — ppo_update's no-hook path stays byte-identical
    (v0 runs and prior-less v1 runs are untouched, per T7's original
    byte-identity guarantee, now extended to cover the prior term too).

    kickstart: None or (t_logits, t_values) precomputed full-batch teacher
    outputs; prior_logits: None or precomputed full-batch prior logits."""
    kick_on = kickstart is not None and (lambda_k > 0.0 or lambda_v > 0.0)
    prior_on = prior_logits is not None and lambda_p > 0.0
    if not kick_on and not prior_on:
        return None

    def fn(idx):
        # One student forward serves both terms (full [B,92] distribution).
        # Re-runs the student net's forward directly (rather than threading
        # logits out through evaluate()/Categorical) to get the full [B,92]
        # distribution both KL terms need -- evaluate() only returns the
        # sampled action's logprob. This costs a second forward pass per
        # minibatch versus changing model.py/model_v1.py's evaluate()
        # signature to also return logits; traded deliberately for keeping
        # those files (and ppo.py's default no-hook path) untouched.
        # batch["obs"] is the entity dict (keys match EntityPolicyNet.
        # forward's params) -- both kickstart and the prior imply v1.
        s_logits, s_value = net(**{k: v[idx] for k, v in batch["obs"].items()})
        # Roots `loss` in the autograd graph unconditionally (via a *0.0 no-op
        # so it never contributes value) so `loss.requires_grad` holds even in
        # a hypothetical future branch that adds neither kl nor kl_p; not
        # load-bearing today since kick_on/prior_on guarantee at least one of
        # kl or kl_p is added below, and either already carries the grad path.
        loss = s_logits.sum() * 0.0
        info = {}
        if kick_on:
            t_logits, t_values = kickstart
            kl, v_mse = kickstart_losses(s_logits, s_value.squeeze(-1),
                                         t_logits[idx], t_values[idx])
            loss = loss + lambda_k * kl + lambda_v * v_mse
            info["kick_kl"] = kl.item()
        if prior_on:
            kl_p = kl_student_prior(s_logits, prior_logits[idx])
            loss = loss + lambda_p * kl_p
            info["kl_pri"] = kl_p.item()
        return loss, info

    return fn


def check_win_prob_gamma(reward_cfg, ppo_cfg):
    """Refuse to train if the shaping gamma differs from the training gamma.

    Potential-based shaping preserves the optimal policy only when the gamma
    inside `gamma * PHI(s') - PHI(s)` is the SAME gamma the returns are
    discounted with. A mismatch silently optimises a different objective and
    produces a run that looks entirely healthy, so this fails loud at startup.

    `win_prob_gamma` is the SHARED potential gamma: v9's `air_setup` shaping
    (engine/src/reward.rs::air_shaping) reads the same field, so this guard now
    fires for either term. There is one potential gamma per run by design -- two
    would be two chances to get this exact silent failure.
    """
    terms = [k for k in ("win_prob_weight", "air_setup")
             if float(reward_cfg.get(k, 0.0)) != 0.0]
    if not terms:
        return
    shaping = float(reward_cfg.get("win_prob_gamma", 0.0))
    training = float(ppo_cfg["gamma"])
    if abs(shaping - training) > 1e-9:
        raise ValueError(
            f"win_prob_gamma ({shaping}) != ppo gamma ({training}), with "
            f"potential-based shaping active via {'+'.join(terms)}. "
            f"Potential-based shaping only preserves the optimal policy when "
            f"these are identical; a mismatch trains a different objective and "
            f"the run will look normal. Fix the reward config."
        )


def action_table_for(schema: dict):
    """The [N,8] action table this schema selects, as a numpy array.

    Schema version 1 admits TWO action tables and they are told apart by
    `action_count`, not by `version` -- the obs contract is byte-identical
    between them, only the policy's output dimension differs:

      * 92  -> `construct_92_v1`,     the frozen v1.1 table. Every existing
               checkpoint (v8, the champion, every league arm) is this.
      * 104 -> `construct_104_v1air`, v1.1's 92 rows APPENDED with a doubled
               12-row clean-air block. See engine/src/actions.rs
               ::make_lookup_table_v1_air for the derivation.

    Dispatching here rather than hardcoding `action_table_v1()` is what makes
    the two tables coexist. The action table sets the policy's OUTPUT
    DIMENSION, so it is fixed for the life of a lineage and can only be chosen
    at a from-scratch launch; the engine-side guard (lib.rs::validate_schema)
    accepts both widths precisely so a v1.1 champion stays loadable as the
    ruler a v1-air run gets gated against.
    """
    n = int(schema["action_count"])
    if n == 104:
        return action_table_v1_air()
    if n == 92:
        return action_table_v1()
    raise ValueError(
        f"schema action_count={n} is not a known v1 action table "
        "(92 = construct_92_v1, 104 = construct_104_v1air)"
    )


def lr_at(total_steps, ppo: dict) -> float:
    """Learning rate for this iteration: `ppo["lr"]`, optionally annealed.

    WHY THIS EXISTS (v9 D8). Until 2026-07-26 the learning rate was set ONCE,
    at `torch.optim.Adam(..., lr=cfg.ppo["lr"])`, and was then unchangeable for
    the life of a lineage -- TWICE over:
      1. `resume_train.py` does `cfg.ppo = state["config"]["ppo"]`, so the toml's
         lr loses to the checkpoint's;
      2. even if you edited that dict, `self.opt.load_state_dict(...)` restores
         Adam's `param_groups`, which carry `lr` -- so the restored value wins
         again. `--reset-optimizer` was the only lever, and it throws away the
         moment estimates as well.
    Every lr this project has ever run was therefore frozen at its launch value.

    WHY IT MATTERS HERE. ZealanL's RLGym-PPO-Guide stages the learning rate by
    what the policy can do: 2e-4 while it cannot score, 1e-4 once it is trying
    to, <=0.8e-4 for complex mechanics. v9 is a single run that starts at the
    first stage and is aimed squarely at the third, so no single constant is
    right for both ends -- but the two ends are ~200M steps apart, which is
    exactly what a step-keyed anneal expresses and a constant cannot.

    SHAPE. Flat at `lr` through `lr_hold_steps`, then LINEAR to `lr_final` over
    `lr_anneal_steps`, then flat at `lr_final`. Linear, not cosine: the hold is
    doing the real work (it keeps the launch window byte-identical to a flat
    config) and a curve inside the ramp is unmeasurable next to it.

    Absent `lr_final` this returns `ppo["lr"]` unchanged, so every pre-v9 config
    is byte-identical and the mechanism costs a dict lookup.
    """
    final = ppo.get("lr_final")
    base = float(ppo["lr"])
    if final is None:
        return base
    final = float(final)
    hold = float(ppo.get("lr_hold_steps", 0))
    span = float(ppo.get("lr_anneal_steps", 0))
    if total_steps <= hold:
        return base
    if span <= 0:
        return final
    frac = min(1.0, (float(total_steps) - hold) / span)
    return base + (final - base) * frac


def check_match_mode_required(reward_cfg, curriculum_cfg):
    """Refuse win-probability shaping without a match layer.

    PHI is a function of score and remaining clock. With match_mode off there
    is no score and no clock, PHI is pinned at 0 forever, and the shaping
    term is identically zero -- so the run trains on nothing at all while every
    log line looks normal. Fail at startup instead.
    """
    if float(reward_cfg.get("win_prob_weight", 0.0)) == 0.0:
        return
    if not bool(curriculum_cfg.get("match_mode", False)):
        raise ValueError(
            "win_prob_weight is set but the curriculum has match_mode = false. "
            "Win-probability shaping needs a score and a match clock; without "
            "them PHI is constant, the shaping term is exactly zero, and the "
            "run trains on nothing while appearing healthy. "
            "Use configs/curriculum_v3_match.toml."
        )


def team_size_blocks(team_sizes):
    """`[(team_size, start, count), ...]` for the contiguous 1s/2s/3s blocks.

    `engine.team_sizes` is already sorted ascending by construction
    (`engine::allocate_team_sizes` emits a 1s block, then 2s, then 3s), so a
    single pass is enough. Asserted, because everything below indexes arenas by
    block and a silently unsorted list would place opponents in the wrong regime.
    """
    sizes = list(team_sizes)
    assert sizes == sorted(sizes), (
        f"arena team sizes must be ordered in 1s/2s/3s blocks, got {sizes}"
    )
    out, start = [], 0
    while start < len(sizes):
        s = sizes[start]
        n = 0
        while start + n < len(sizes) and sizes[start + n] == s:
            n += 1
        out.append((s, start, n))
        start += n
    return out


def league_cold_start_warning(entries, schema_version, action_table, registry_path):
    """The reason a configured league can never pick an opponent, or None if it
    can. Printed ONCE at Trainer init.

    WHY THIS IS NOT AN ASSERT. `_refresh_opponents` already handles an empty
    pick correctly -- the arenas fall back to self-play and it prints "league:
    no opponents available" -- but it says that once every `refresh_iters`
    (200 for v9) in a run whose first ~20M steps are explicitly to be ignored,
    so the message lands where nobody is looking. And for a from-scratch v9 the
    empty pool is EXPECTED at launch and TEMPORARY: the registry holds only the
    92-row champion, `choose_opponents` correctly filters it out of a 104-row
    run (one engine binds one decode table), and the pool fills the first time
    v9 promotes a checkpoint of its own table. Failing to launch over that
    would be wrong; launching without saying so is how a run's real arena
    composition silently differs from its documented one for 600M steps.

    The distinction that matters and that this message draws: entries exist but
    NONE MATCH (a cold start, or a misconfigured registry path) versus the
    registry being empty outright.
    """
    same_schema = [e for e in entries if e.get("schema_version", 0) == schema_version]
    legal = [
        e for e in same_schema
        if action_table is None
        or e.get("action_table", DEFAULT_V1_TABLE) == action_table
    ]
    if legal:
        return None
    tables = sorted({e.get("action_table", DEFAULT_V1_TABLE) for e in same_schema})
    return (
        "\n"
        "!! LEAGUE IS INERT AT LAUNCH -- every league arena will run as SELF-PLAY.\n"
        f"   registry            {registry_path}\n"
        f"   entries             {len(entries)} total, {len(same_schema)} at "
        f"schema_version={schema_version}\n"
        f"   this run decodes    {action_table}\n"
        f"   pool offers         {tables or ['(nothing)']}\n"
        "   The action table sets the policy's OUTPUT DIMENSION and one engine\n"
        "   binds one table, so a pool entry on another table is not a weaker\n"
        "   opponent, it is an unplayable one. This is the expected cold start\n"
        "   for a from-scratch run on a new table: it clears the first time a\n"
        "   checkpoint of THIS table is promoted into that registry (which\n"
        "   requires champion_gate/tick to stamp action_table on the entry --\n"
        "   they do since 2026-07-26). Until then the run's arena composition\n"
        "   is the one documented for a dead league, not the one with it.\n"
    )


def plan_block_assignment(team_sizes, slot_modes, foreign_frac,
                          n_league_slots, league_frac):
    """Place foreign and league opponents PER BLOCK, and return the assignment.

    Encoding matches engine/src/engine.rs: `-1` self-play, `k >= 0` native
    (league) slot `k`, `-(f) - 2` foreign slot `f`.

    WHY PER BLOCK (v9 D8). v8 stamped foreign arenas onto the FRONT of the whole
    arena list and league arenas onto the BACK. Arenas are laid out in
    1v1/2v2/3v3 blocks, so with weights [0.85, 0.1, 0.05] that put every league
    arena in the 3v3/2v2 tail and every foreign arena in the 1v1 head: v8 ended
    up with ZERO 3v3 self-play arenas, and 100% of its 3v3 experience came from
    one frozen 1v1-trained near-random opponent. That was an accident of tail
    assignment, not a decision.

    It also makes the equal-learner-rows invariant hold. With foreign and league
    placed as the same fraction INSIDE each block,

        rows_m = a_m * m * (2 - phi - lambda)

    so equal rows <=> a_m proportional to 1/m <=> team_size_weights [6, 3, 2],
    INDEPENDENT of `foreign_frac` and `league_frac`. Retune either fraction and
    the 1s/2s/3s split stays balanced. Under front/tail stamping that is false.

    `slot_modes[f]` is the team size foreign slot `f` plays at; a block only
    receives slots whose mode matches it, which is also what keeps
    element/immortal (fixed 107-float obs) confined to the 1v1 block. A block
    with no matching slots simply gets no foreign arenas -- it does NOT silently
    borrow another mode's bot, which would hard-error deep inside a worker.

    Foreign takes the front of each block and league the back, so the two can
    never overlap; asserted per block rather than globally, because a global
    check passes while a single block overflows.
    """
    a = [-1] * len(team_sizes)
    for m, start, count in team_size_blocks(team_sizes):
        mine = [f for f, mode in enumerate(slot_modes) if mode == m]
        n_for = round(foreign_frac * count) if mine else 0
        n_lg = round(league_frac * count) if n_league_slots else 0
        assert n_for + n_lg <= count, (
            f"{m}v{m} block: foreign ({n_for}) + league ({n_lg}) arenas exceed the "
            f"block size ({count}). Lower opponent_frac, or re-weight team_size_weights."
        )
        for i in range(n_for):
            a[start + i] = -(mine[i % len(mine)]) - 2
        for i in range(n_lg):
            a[start + count - n_lg + i] = i % n_league_slots
    return a


class Trainer:
    def __init__(self, cfg: TrainConfig, _state: dict | None = None):
        self.cfg = cfg
        self.schema = schema_dict(cfg.schema_path)
        self.is_v1 = self.schema["version"] == 1
        # Checkpoint/config schema guard: a checkpoint only makes sense with
        # the net family it was trained as. In particular a v0 MLP state dict
        # can NOT seed a v1 EntityPolicyNet (different obs, different net) --
        # the sanctioned v0->v1 bridge is kickstart distillation, where the
        # v0 checkpoint is the frozen TEACHER ([kickstart].teacher in the
        # config / resume_train.py --kickstart-teacher), not the resume state.
        if _state is not None:
            ck_ver = int(_state.get("schema_version", 0))
            if ck_ver != self.schema["version"]:
                raise ValueError(
                    f"checkpoint schema_version={ck_ver} does not match config "
                    f"schema version={self.schema['version']} ({cfg.schema_path}). "
                    "A v0 MLP checkpoint cannot seed a v1 entity net directly; "
                    "start a fresh v1 run and pass the v0 checkpoint as the "
                    "kickstart teacher ([kickstart].teacher / "
                    "resume_train.py --kickstart-teacher) instead."
                )
        # Objective guards (T5): load the reward and curriculum configs as
        # dicts (cfg only carries their paths -- the engine consumes the
        # paths directly, nothing on the Python side parsed them before this)
        # and fail loud on combinations that would silently train the wrong
        # objective. Both checks are inert (no-op) whenever win-prob shaping
        # is off, i.e. every historical reward config -- see each function's
        # docstring for the failure mode it closes. Placed before any
        # engine/net construction so a bad config never wastes GPU time.
        with open(cfg.reward_config_path, "rb") as f:
            reward_cfg = tomllib.load(f)
        curriculum_cfg = {}
        if cfg.curriculum_config_path:
            with open(cfg.curriculum_config_path, "rb") as f:
                curriculum_cfg = tomllib.load(f)
        check_win_prob_gamma(reward_cfg, cfg.ppo)
        check_match_mode_required(reward_cfg, curriculum_cfg)

        # Kickstart distillation (T7): only meaningful for a v1-schema run
        # with a [kickstart] block that names a teacher (the config template
        # ships with `teacher` commented out, so a bare `steps=` block stays
        # inactive). `emit_v0_obs` asks the engine to also record the legacy
        # 94-float obs per collected step (T6), which the frozen v0 MLP
        # teacher needs. For v0-schema runs this is always False, so the
        # Engine constructor call, collect() output, and run() loop below
        # are byte-identical to pre-T7 behavior.
        kickstart_active = bool(cfg.kickstart.get("teacher")) and self.is_v1
        engine_kwargs = dict(
            num_arenas=cfg.env["num_arenas"], blue=cfg.env["blue"], orange=cfg.env["orange"],
            schema_path=cfg.schema_path, reward_config_path=cfg.reward_config_path,
            seed=cfg.env["seed"],
            team_size_weights=cfg.env.get("team_size_weights"),
            curriculum_config_path=cfg.curriculum_config_path or None,
        )
        if self.is_v1:
            # candle rebuilds attention from the state dict; the head count is
            # not recoverable from tensor shapes, so it rides along explicitly.
            engine_kwargs["net_heads"] = int(cfg.net["heads"])
        if kickstart_active:
            engine_kwargs["emit_v0_obs"] = True
        self.engine = Engine(**engine_kwargs)
        dev = cfg.run.get("device", "cuda")
        self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
        if self.is_v1:
            # Net dims come from [net] (T1 gate: 128/2/4/512 launch config);
            # the action table is compiled into the engine and is carried by
            # the net as a non-trainable buffer (set_weights hands it back to
            # candle with the rest of the state dict).
            #
            # Selected by the SCHEMA, not hardcoded: v1 has two legal tables,
            # the frozen 92-row v1.1 and the 104-row v1-air (v1.1 + a doubled
            # clean-air block; see engine/src/actions.rs). The table sets the
            # policy's OUTPUT DIMENSION, so it can only ever be chosen at the
            # start of a lineage -- getting this from the schema is what lets
            # v9 launch on v1-air while every existing checkpoint keeps
            # loading against v1.1.
            self.net = EntityPolicyNet(
                d_model=int(cfg.net["d_model"]), layers=int(cfg.net["layers"]),
                heads=int(cfg.net["heads"]), ff=int(cfg.net["ff"]),
                action_table=action_table_for(self.schema),
            ).to(self.device)
        else:
            self.net = PolicyValueNet(
                self.engine.obs_size, self.engine.action_count, tuple(cfg.net["hidden"])
            ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo["lr"])
        self.total_steps = 0

        self.kickstart: dict | None = None
        if kickstart_active:
            self.kickstart = {
                "teacher": KickstartTeacher(cfg.kickstart["teacher"], device=str(self.device)),
                "schedule": KickstartSchedule(
                    lambda_k0=float(cfg.kickstart.get("lambda_k", 1.0)),
                    kickstart_steps=int(cfg.kickstart.get("steps", 500_000_000)),
                    lambda_v=float(cfg.kickstart.get("lambda_v", 0.5)),
                ),
            }

        # BC-prior anchor (K3): only meaningful for a v1-schema run with a
        # [kl_prior] block that names a checkpoint (the config template ships
        # with `ck` absent, so cfg.kl_prior stays {} and this block is
        # inactive). Gate mirrors kickstart's: cfg.kl_prior.get("ck") truthy
        # AND self.is_v1. For v0 runs and prior-less v1 runs self.kl_prior
        # stays None, so run()'s prior_logits stays None and
        # compose_extra_loss's prior_on branch is dead -- byte-identical to
        # pre-K3 behavior.
        self.kl_prior: dict | None = None
        if cfg.kl_prior.get("ck") and self.is_v1:
            self.kl_prior = {
                # expect_net closes the Global Constraint "prior dims must equal
                # student dims": cfg.net IS the student's dims (restored from the
                # resume checkpoint by resume_train.py before Trainer init).
                "net": KLPrior(cfg.kl_prior["ck"], device=str(self.device),
                               expect_net=cfg.net),
                "lambda": float(cfg.kl_prior.get("lambda", 0.05)),
                # anneal_iters > 0 -> lambda_p decays lambda -> lambda_floor over
                # this many iters, then stays at the floor (stabilise early, free late)
                "anneal_iters": int(cfg.kl_prior.get("anneal_iters", 0)),
                "lambda_floor": float(cfg.kl_prior.get("lambda_floor", 0.0)),
            }
            _an = self.kl_prior["anneal_iters"]
            print(f"kl_prior: anchored to {cfg.kl_prior['ck']} "
                  f"lambda_p={self.kl_prior['lambda']}"
                  + (f" annealing -> {self.kl_prior['lambda_floor']} over {_an} iters"
                     if _an > 0 else ""), flush=True)

        # Opponent-pool ("league") integration. Disabled by default (cfg.league == {}
        # or {"enabled": False}) -> self._assignment stays None -> engine.collect's
        # arena_opponents=None path, byte-identical to pre-league behavior.
        self.num_arenas = cfg.env["num_arenas"]
        self._assignment: list[int] | None = None
        self._league: dict | None = None
        lg = cfg.league
        if lg.get("enabled"):
            from construct.league.registry import Registry
            from construct.league.sampling import choose_opponents
            from construct.league.matches import load_sd
            frac = float(lg.get("opponent_frac", 0.2))
            assert 0 <= frac < 1, f"league.opponent_frac must be in [0, 1), got {frac}"
            slots = int(lg.get("slots", 4))
            # engine.set_opponents rejects more than 8 slots (lib.rs) -- catch it at
            # init time rather than mid-run.
            assert 1 <= slots <= 8, f"league.slots must be in [1, 8], got {slots}"
            self._league = {
                "registry": Registry(path=lg.get("registry", "league/registry.jsonl")),
                "choose": choose_opponents, "load_sd": load_sd,
                "frac": frac,
                # floor at 1: refresh_iters=0 would make `it % refresh` divide by zero.
                "refresh": max(1, int(lg.get("refresh_iters", 200))),
                "slots": slots,
            }
            # Say it AT LAUNCH, not 200 iterations in. See
            # league_cold_start_warning for why this is a warning and not an
            # assert.
            warning = league_cold_start_warning(
                self._league["registry"].entries(),
                self.schema["version"],
                self.schema["action_table"] if self.is_v1 else None,
                self._league["registry"].path,
            )
            if warning:
                print(warning, flush=True)

        # Foreign opponents: ported community bots (docs/foreign-opponents.md).
        # They occupy a SEPARATE engine slot space, addressed as -(slot)-2, and
        # take arenas from the FRONT OF EACH TEAM-SIZE BLOCK (league takes the
        # BACK of each block), so the two can run together in every regime.
        # Disabled by default -> assignment untouched.
        # Per-arena team size (1/2/3), straight from the engine's own allocator.
        # Everything below places opponents PER BLOCK against this; Python could
        # not previously see it (`allocate_team_sizes` was Rust-only), which is
        # why v8 had to stamp fronts and tails.
        self._team_sizes = list(getattr(self.engine, "team_sizes", [1] * self.num_arenas))
        self._foreign_frac = 0.0
        self._foreign_slots = 0
        self._foreign_modes: list[int] = []
        self._foreign_cars: list[int] = []
        fg = cfg.foreign
        if fg.get("enabled"):
            # TWO config shapes. v9's `slots = [{kind, mode, weights}, ...]`
            # carries the TEAM SIZE each bot plays at, which is what per-block
            # placement and the (bot x mode) rung ladder need. The legacy
            # parallel-lists form (kinds/weights/decision_periods) is still
            # accepted unchanged and is treated as all-1v1, so every pre-v9
            # config keeps working byte-for-byte.
            slots_cfg = list(fg.get("slots", []))
            if slots_cfg:
                kinds = [str(s["kind"]) for s in slots_cfg]
                paths = [str(s["weights"]) for s in slots_cfg]
                modes = [int(s.get("mode", 1)) for s in slots_cfg]
            else:
                kinds = list(fg.get("kinds", []))
                paths = list(fg.get("weights", []))
                modes = [1] * len(kinds)
            assert kinds and len(kinds) == len(paths), (
                f"foreign slots must be non-empty with one weights path each, "
                f"got {len(kinds)} kinds and {len(paths)} weights"
            )
            # 12 is the cap in engine/src/lib.rs
            # (`set_foreign_opponents`: `opponents.len() > 12`), raised from 8
            # on 2026-07-26 when element/immortal joined the 2v2 and 3v3
            # blocks. v9 sits EXACTLY at it with 12 -- a fifth bot, or a bot at
            # a fourth team size, needs that one-line change first. Catch it
            # here rather than mid-run.
            assert len(kinds) <= 12, (
                f"at most 12 foreign slots (engine/src/lib.rs), got {len(kinds)}"
            )
            # NOTE: element/immortal at mode > 1 used to be refused here. They
            # are now allowed -- above 1v1 they run on a TRUNCATED
            # AdvancedObs-107 showing only the opponent nearest the ball
            # (engine obs_advanced::build_advanced_obs_one_other). Above 1v1
            # they are a degraded local variant rather than the ported bot; see
            # docs/foreign-opponents.md for the exploitability tripwire.
            for k, m in zip(kinds, modes):
                assert m in (1, 2, 3), f"foreign slot mode must be 1/2/3, got {m}"
            frac = float(fg.get("opponent_frac", 0.25))
            assert 0 <= frac < 1, f"foreign.opponent_frac must be in [0, 1), got {frac}"
            # Difficulty handicap: react every N decisions. Every ported bot beats
            # the champion 96-0, so a period > 1 is normally required to get a
            # contestable teacher (Element at period 4 -> we win ~0.375).
            periods = list(fg.get("decision_periods", [1] * len(kinds)))
            assert len(periods) == len(kinds), (
                f"foreign.decision_periods must match slots ({len(kinds)}), "
                f"got {len(periods)}"
            )
            assert all(int(p) >= 1 for p in periods), "decision_periods must be >= 1"
            # How many ORANGE cars each bot drives. The knob that actually works
            # above 1v1: at 3v3 a FULL nexto team out-scores us even at p96
            # (goal share 0.371), while decision_period sweeps 0.02 -> 0.95 at
            # 1v1. Default 1 car everywhere; clamped to the slot's mode.
            cars = list(fg.get("foreign_cars", [1] * len(kinds)))
            assert len(cars) == len(kinds), (
                f"foreign.foreign_cars must match slots ({len(kinds)}), got {len(cars)}"
            )
            cars = [max(1, min(int(c), m)) for c, m in zip(cars, modes)]
            sds = []
            for p in paths:
                path = os.path.expanduser(str(p))
                sds.append({k: v.astype(np.float32) for k, v in np.load(path).items()})
            # Fail loudly here: a foreign pool that silently doesn't load would make
            # the run quietly identical to plain self-play and waste the experiment.
            self.engine.set_foreign_opponents(
                sds, kinds, [int(p) for p in periods], list(cars))
            self._foreign_frac = frac
            self._foreign_slots = len(sds)
            self._foreign_modes = list(modes)
            self._foreign_cars = list(cars)
            self._assignment = self._apply_foreign([-1] * self.num_arenas)
            n_for = sum(1 for k in self._assignment if k <= -2)
            print(f"foreign: {list(zip(kinds, modes))} periods={periods} cars={cars} "
                  f"on {n_for}/{self.num_arenas} arenas (frac={frac} PER BLOCK)",
                  flush=True)
            # --- auto-curriculum: hold the opponent near a ~50% win rate by
            # tuning its decision_period, so the gradient is always maximally
            # informative (never saturated at a 0% or 100% win rate). Because
            # the reward is zero-sum, self-play arenas cancel in the mean episode
            # reward, leaving ep_reward_mean as a clean "are we beating the
            # foreign opponent" margin -- the control signal. See
            # docs/foreign-opponents.md.
            self._foreign_sds = sds
            self._foreign_kinds = kinds
            self._foreign_periods = [int(p) for p in periods]
            ac = fg.get("auto_curriculum", {})
            self._ac_on = bool(ac.get("enabled", False))
            self._ac_period_min = int(ac.get("period_min", 1))
            self._ac_period_max = int(ac.get("period_max", 12))
            self._ac_every = max(1, int(ac.get("adjust_every", 20)))
            # Target a GOAL-based win rate, measured on a separate eval engine, so
            # the signal is clean regardless of the (possibly shaped, non-zero-sum)
            # TRAINING reward. Winning > win_hi -> opponent harder; < win_lo ->
            # easier. Falls back to the ep_reward_mean signal (band) only if the
            # eval engine can't be built.
            self._ac_win_hi = float(ac.get("win_hi", 0.55))
            self._ac_win_lo = float(ac.get("win_lo", 0.45))
            self._ac_band = float(ac.get("band", 5.0))     # ep_rew fallback only
            self._ac_ema = None
            self._ac_alpha = float(ac.get("ema_alpha", 0.2))
            # PER-BOT control (2026-07-25). Each foreign slot gets its OWN period,
            # tuned from its OWN measured win rate, because difficulty is wildly
            # uneven at a shared period: measured at p4 on ck_001171502080 we beat
            # immortal 0.896 and element 0.677 but lose to nexto 0.146. One shared
            # knob cannot put four different bots in the same band -- it parks two
            # of them as punching bags and one as an unwinnable wall, and only the
            # middle pair produce useful gradient. Per-slot integer knobs also
            # restore RESOLUTION: a single shared knob saturates once the true
            # equilibrium falls between two integers (observed: period thrashing
            # 3<->4 for 258M steps while the real win rate kept creeping up).
            # Each slot's win rate is noisy (few matches per bot per eval), so
            # adjust on a per-slot EMA rather than a single raw measurement.
            self._ac_wr_ema = [None] * self._foreign_slots
            self._ac_wr_alpha = float(ac.get("wr_ema_alpha", 0.3))
            # DWELL: evals a slot must sit still after a change before it may move
            # again. One eval scores only ~18 matches per bot, so a single
            # measurement's sd (~0.118 at wr 0.5) dwarfs a +/-0.05 band -- without
            # a dwell the controller moves on ~85-90% of evals and wanders over 4
            # rungs, which is exactly why the logged period turned into unreadable
            # noise. Dwelling lets the (post-change, freshly reset) EMA accumulate
            # 2-3 samples before acting: simulated move rate drops to 0.11-0.22 and
            # the period settles into a 2-rung band at the SAME mean win rate
            # (~0.5). See the 2026-07-25 controller simulation.
            self._ac_dwell = int(ac.get("adjust_dwell", 2))
            self._ac_since = [10 ** 6] * self._foreign_slots   # free to move at start
            # LEXICOGRAPHIC rung control (v9 D11). Two knobs per slot:
            # `foreign_cars` (coarse) and an index into `period_ladder` (fine).
            #
            #   too easy: period index -= 1; if already at the hardest period and
            #             cars < mode, cars += 1 and the period resets to the
            #             EASIEST rung of the new car count
            #   too hard: period index += 1; if already at the easiest period and
            #             cars > 1, cars -= 1 and the period resets to the HARDEST
            #
            # Monotone BY CONSTRUCTION OF THE RULE, so we never have to know a
            # priori whether (2 cars, p12) is harder than (1 car, p1) -- the
            # controller finds out. Absent `period_ladder` the legacy +/-1
            # integer-period controller runs unchanged, which is what keeps every
            # pre-v9 config (and test_auto_curriculum's stub) behaving exactly
            # as before.
            ladder = ac.get("period_ladder")
            self._ac_ladder = [int(x) for x in ladder] if ladder else None
            if self._ac_ladder:
                assert self._ac_ladder == sorted(self._ac_ladder), (
                    "period_ladder must be ascending (index 0 = hardest)"
                )
                # Seed each slot at the ladder entry nearest its configured
                # period, so config and controller agree from iteration 0.
                self._ac_pidx = [
                    min(range(len(self._ac_ladder)),
                        key=lambda i, p=p: abs(self._ac_ladder[i] - p))
                    for p in self._foreign_periods
                ]
                self._foreign_periods = [self._ac_ladder[i] for i in self._ac_pidx]
            else:
                self._ac_pidx = [0] * self._foreign_slots
            # A 2-rung move when |ema - 0.5| > this. v9 sets 0.35: single-eval sd
            # is 0.118, so that is ~3 sd outside the band -- not a noise-driven
            # move -- and it buys back the ratchet speed lost to the staggered
            # (one mode per cycle) eval below.
            #
            # DEFAULT 1.0 == never fires, because |ema - 0.5| <= 0.5 always. The
            # single-rung law is the one L4's dwell arithmetic and simulation
            # were done against, so a config that does not ask for the 2-rung
            # extension must not get it: adding a knob must never silently
            # change a run that predates it.
            self._ac_big = float(ac.get("big_move_margin", 1.0))
            self._ac_eval = None
            self._ac_cycle = 0
            if self._ac_on:
                # L2: THE EVAL MUST RUN THE REGIME IT GATES. One engine per team
                # size, each built with the TRAINING curriculum (match_mode) and
                # reward_v0 as the neutral tape, each running its slots at their
                # CURRENT (foreign_cars, decision_period).
                #
                # The pre-v9 eval was hardcoded blue=1, orange=1. That was
                # harmless only while foreign arenas above 1v1 were refused
                # outright; the moment (bot x mode) slots exist it would gate a
                # 3v3 rung on 1v1 evidence. A goal-TERMINATED eval already cost
                # us this once (2026-07-23: it said 50% at p6 while real
                # match_mode was ~12%, and over-ratcheted the opponent ~2 rungs).
                by_mode = list(ac.get("eval_arenas_by_mode",
                                      [int(ac.get("eval_arenas", 16)), 0, 0]))
                self._ac_stagger = bool(ac.get("eval_stagger", False))
                engines, arenas_by_mode = {}, {}
                for m in sorted(set(self._foreign_modes)):
                    n_ar = int(by_mode[m - 1]) if m - 1 < len(by_mode) else 0
                    mine = [f for f, mm in enumerate(self._foreign_modes) if mm == m]
                    if n_ar <= 0 or not mine:
                        continue
                    try:
                        ev = Engine(num_arenas=n_ar, blue=m, orange=m,
                                    schema_path=self.cfg.schema_path,
                                    reward_config_path="configs/reward_v0.toml",
                                    curriculum_config_path=(self.cfg.curriculum_config_path or None),
                                    seed=12345 + m, num_threads=0,
                                    net_heads=int(self.cfg.net.get("heads", 4)))
                        ev.set_foreign_opponents(
                            [sds[f] for f in mine], [kinds[f] for f in mine],
                            [self._foreign_periods[f] for f in mine],
                            [self._foreign_cars[f] for f in mine])
                        engines[m] = ev
                        arenas_by_mode[m] = n_ar
                    except Exception as e:
                        print(f"auto-curriculum: mode-{m} eval engine failed ({e}); "
                              f"slots at that mode will fall back to the ep_rew signal",
                              flush=True)
                if engines:
                    self._ac_eval = {"engines": engines, "arenas": arenas_by_mode,
                                     "steps": int(ac.get("eval_steps", 14000))}
                    print("auto-curriculum: match-mode eval engines built "
                          f"{ {m: arenas_by_mode[m] for m in engines} } arenas, "
                          f"{self._ac_eval['steps']} steps"
                          + (", ROUND-ROBIN one mode per cycle" if self._ac_stagger else ""),
                          flush=True)
                else:
                    print("auto-curriculum: no eval engine; falling back to ep_rew signal",
                          flush=True)
                rung = ("ladder " + str(self._ac_ladder) if self._ac_ladder
                        else f"period [{self._ac_period_min},{self._ac_period_max}]")
                print(f"auto-curriculum ON: hold win rate in "
                      f"[{self._ac_win_lo},{self._ac_win_hi}], {rung}, "
                      f"adjust every {self._ac_every} iters", flush=True)

        if _state:
            self.net.load_state_dict(_state["model"])
            if _state["optimizer"] is not None:  # None = deliberate reset (regime swap)
                self.opt.load_state_dict(_state["optimizer"])
            self.total_steps = _state["total_steps"]

        # cfg.ppo["lr"] is the AUTHORITY, not Adam's restored param_groups.
        # `load_state_dict` above carries the checkpoint's `lr` back in, which is
        # why the rate was previously unchangeable on a resume without also
        # dropping the moment estimates (--reset-optimizer). This line is a
        # no-op for every historical resume -- resume_train.py takes cfg.ppo FROM
        # the checkpoint, so the two values are the same number -- and it is what
        # makes `resume_train.py --lr` and the lr_final anneal actually bite.
        self._apply_lr(lr_at(self.total_steps, self.cfg.ppo))
        if self.cfg.ppo.get("lr_final") is not None:
            print(f"lr anneal ON: {float(self.cfg.ppo['lr']):.2e} held to "
                  f"{int(self.cfg.ppo.get('lr_hold_steps', 0)):,} steps, then linear to "
                  f"{float(self.cfg.ppo['lr_final']):.2e} over "
                  f"{int(self.cfg.ppo.get('lr_anneal_steps', 0)):,} steps "
                  f"(now {lr_at(self.total_steps, self.cfg.ppo):.2e})", flush=True)

    def _apply_lr(self, lr: float) -> float:
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

    def _measure_foreign_winrates(self, modes=None) -> list[float | None] | None:
        """PER-SLOT blue win share, each bot at its own current rung, measured on
        the per-mode goal-based eval engines. ~0.5 = even.

        Eval arenas are round-robined across the foreign slots OF THAT MODE
        exactly like the training assignment, so every bot is measured -- the
        pre-2026-07-25 version collected with `[-2] * arenas`, i.e. slot 0 ONLY,
        so the controller was blind to every bot but the first and adding three
        more opponents moved the period by exactly nothing.

        `modes` restricts the measurement to those team sizes (the staggered
        eval measures ONE mode per cycle: a full 8-slot eval at hard rungs costs
        ~116s wall against ~100s of training per 20 iters, and staggering cuts
        that ~3x while keeping 6 arenas -> ~18 matches per slot, which is what
        the [0.35, 0.65] band and the 2-eval dwell are calibrated against).
        Slots not measured this cycle come back as None and are skipped.

        T2 -- THE MIS-ATTRIBUTION FIX, which had to land in the same change as
        the per-kind guard. The old code used ARENA indices as COLUMN indices.
        That was correct ONLY because the eval was blue=1 (one learner column
        per arena) and foreign arenas above 1v1 were refused outright; its own
        comment said so. The moment a mode-2/3 eval engine exists that premise
        is false and each slot would be handed some OTHER slot's columns -- a
        mis-attribution, not a mis-scaling, that mis-tunes every bot's rung
        while looking completely normal. At mode m an arena owns m contiguous
        columns (arena-major, car-minor), so the fix is to select WHOLE ARENAS
        and hand `split_matches` the real `team_size`.

        Returns a list aligned with `self._foreign_kinds`, or None if there is
        no eval engine at all.
        """
        ev = self._ac_eval
        if ev is None:
            return None
        import numpy as _np
        from construct.league.matches import match_record, split_matches
        sd = {k: v.detach().cpu().numpy().astype(_np.float32)
              for k, v in self.net.state_dict().items()}
        wrs: list[float | None] = [None] * self._foreign_slots
        for m, eng in ev["engines"].items():
            if modes is not None and m not in modes:
                continue
            mine = [f for f, mm in enumerate(self._foreign_modes) if mm == m]
            if not mine:
                continue
            eng.set_weights(sd)
            # L2 again: the eval runs the slot's CURRENT rung -- both knobs.
            eng.set_foreign_opponents(
                [self._foreign_sds[f] for f in mine],
                [self._foreign_kinds[f] for f in mine],
                [self._foreign_periods[f] for f in mine],
                [self._foreign_cars[f] for f in mine])
            n_ar = ev["arenas"][m]
            assign = [-(i % len(mine)) - 2 for i in range(n_ar)]
            out = eng.collect(ev["steps"], arena_opponents=assign)
            rew = _np.asarray(out["rewards"])
            term = _np.asarray(out["terminated"])
            # Every eval arena is a foreign arena, so it contributes exactly its
            # m BLUE cars: ncol == n_ar * m, arena-major / car-minor.
            assert rew.shape[1] == n_ar * m, (
                f"mode {m} eval: {rew.shape[1]} learner columns for {n_ar} arenas "
                f"at team size {m}; the column layout is not what the split assumes"
            )
            for j, f in enumerate(mine):
                arenas = [i for i, k in enumerate(assign) if k == -j - 2]
                cols = [a * m + c for a in arenas for c in range(m)]
                rec = match_record(
                    split_matches(rew[:, cols], term[:, cols], team_size=m))
                n = rec["wins"] + rec["draws"] + rec["losses"]
                wrs[f] = (rec["wins"] + 0.5 * rec["draws"]) / n if n else None
        return wrs

    # --- lexicographic rung control (D11) ---------------------------------
    # `foreign_cars` is the COARSE knob and the period ladder the fine one.
    # Both helpers are no-ops past their end of the ladder, so the controller
    # can call them blindly and the rung simply saturates.
    #
    # Without a `period_ladder` these degrade to the legacy +/-1 integer period
    # clamped to [period_min, period_max] -- byte-identical control law, which
    # is what keeps every pre-v9 config and test behaving as before.

    def _rung_harder(self, s: int):
        if self._ac_ladder:
            if self._ac_pidx[s] > 0:
                self._ac_pidx[s] -= 1
            elif self._foreign_cars[s] < self._foreign_modes[s]:
                # Out of period headroom: add a car and drop back to the
                # EASIEST period of the new car count. The controller does not
                # need to know whether that is a net step up or down -- the next
                # measurement tells it, and the rule stays monotone either way.
                self._foreign_cars[s] += 1
                self._ac_pidx[s] = len(self._ac_ladder) - 1
            else:
                return
            self._foreign_periods[s] = self._ac_ladder[self._ac_pidx[s]]
        else:
            self._foreign_periods[s] = max(self._ac_period_min,
                                           self._foreign_periods[s] - 1)

    def _rung_easier(self, s: int):
        if self._ac_ladder:
            if self._ac_pidx[s] < len(self._ac_ladder) - 1:
                self._ac_pidx[s] += 1
            elif self._foreign_cars[s] > 1:
                self._foreign_cars[s] -= 1
                self._ac_pidx[s] = 0
            else:
                return
            self._foreign_periods[s] = self._ac_ladder[self._ac_pidx[s]]
        else:
            self._foreign_periods[s] = min(self._ac_period_max,
                                           self._foreign_periods[s] + 1)

    def _rung_str(self, s: int) -> str:
        p = self._foreign_periods[s]
        if getattr(self, "_foreign_cars", None) and getattr(self, "_ac_ladder", None):
            return f"{self._foreign_cars[s]}c/p{p}"
        return f"p{p}"

    def _auto_curriculum_step(self, it: int, ep_reward_mean: float):
        """Hold EACH foreign opponent near a ~50% win rate by tuning ITS OWN
        decision_period. Preferred signal is a per-bot GOAL-based win rate on a
        separate eval engine (clean regardless of the training reward), smoothed
        per slot by an EMA because each bot gets only a few matches per eval.
        Falls back to the zero-sum ep_reward_mean margin (one shared knob) if the
        eval engine is absent or no match completed."""
        if not getattr(self, "_ac_on", False) or self._foreign_slots == 0:
            return
        # ep_rew fallback signal accumulates every iter (EMA)
        n_for = max(1, round(self._foreign_frac * self.num_arenas))
        signal = ep_reward_mean / n_for
        self._ac_ema = (signal if self._ac_ema is None
                        else (1 - self._ac_alpha) * self._ac_ema
                        + self._ac_alpha * signal)
        if it % self._ac_every != 0:
            return
        # Staggered eval: one MODE per cycle, round-robin. Each slot is then
        # measured every `adjust_every * n_modes` iters and (with the 2-eval
        # dwell) can move at most every 3 measurements. The 2-rung jump rule
        # below and honest cold-start seeds are what pay for that latency.
        modes = None
        if getattr(self, "_ac_stagger", False) and self._ac_eval:
            avail = sorted(self._ac_eval["engines"])
            if avail:
                modes = {avail[self._ac_cycle % len(avail)]}
                self._ac_cycle += 1
        wrs = self._measure_foreign_winrates(modes)
        if wrs is not None and not any(w is not None for w in wrs):
            # An eval engine EXISTS but no match completed. Do NOT fall through
            # to the ep_rew fallback below: that is one shared knob applied to
            # every slot, driven by a reward the whole auto-curriculum design
            # exists to avoid trusting (L8 -- win-prob shaping once drove ep_rew
            # +510 while real skill FELL). With the staggered eval this case is
            # also routine-looking: a cycle that measures only mode 3 leaves
            # every mode-1 slot at None, and moving all eight on a shared ep_rew
            # margin would be silent nonsense. Hold, and say so.
            print("auto-curriculum: no completed match this cycle -- hold "
                  + " ".join(f"{k} {self._rung_str(s)}"
                             for s, k in enumerate(self._foreign_kinds)), flush=True)
            return
        if wrs is not None:
            # PER-BOT: each slot's period follows its OWN smoothed win rate, so a
            # punching bag gets harder while an unwinnable wall gets easier in the
            # same pass (they used to share one knob and fight each other).
            old = list(self._foreign_periods)
            old_cars = list(getattr(self, "_foreign_cars", []))
            parts = []
            for s, w in enumerate(wrs):
                if w is None:
                    parts.append(f"{self._foreign_kinds[s]} n/a {self._rung_str(s)}")
                    continue
                e = self._ac_wr_ema[s]
                e = w if e is None else (1 - self._ac_wr_alpha) * e + self._ac_wr_alpha * w
                self._ac_wr_ema[s] = e
                cur = self._foreign_periods[s]
                before = self._rung_str(s)
                if self._ac_since[s] < self._ac_dwell:
                    # still settling after its last change: keep folding
                    # measurements into the EMA, but don't act on them yet
                    self._ac_since[s] += 1
                    parts.append(f"{self._foreign_kinds[s]} wr{w:.2f}/ema{e:.2f} "
                                 f"{before} dwell{self._ac_since[s]}/{self._ac_dwell}")
                    continue
                # ONE EXTENSION to the L4-tested law: a 2-rung move when
                # |ema - 0.5| > big_move_margin (default 0.35, i.e. ~3 sd outside
                # the band at a single-eval sd of 0.118). That is not a
                # noise-driven move; it is the only thing that buys back the
                # ratchet speed the staggered eval costs.
                n_rungs = 2 if abs(e - 0.5) > self._ac_big else 1
                if e > self._ac_win_hi:
                    for _ in range(n_rungs):
                        self._rung_harder(s)
                elif e < self._ac_win_lo:
                    for _ in range(n_rungs):
                        self._rung_easier(s)
                parts.append(f"{self._foreign_kinds[s]} wr{w:.2f}/ema{e:.2f} "
                             f"{before}->{self._rung_str(s)}")
                moved = (self._foreign_periods[s] != cur
                         or (old_cars and self._foreign_cars[s] != old_cars[s]))
                if not moved:
                    self._ac_since[s] += 1
                else:
                    self._ac_since[s] = 0
                    # The difficulty just changed, so every sample in this slot's
                    # EMA describes the OLD rung. Keeping it would ratchet again
                    # on stale evidence -- e.g. seed 0.90 at p4, harden to p3, and
                    # the still-0.66 EMA immediately hardens to p2 having never
                    # measured p3 (caught by test_auto_curriculum). Smoothing is
                    # for noise at a FIXED difficulty; a change invalidates it.
                    # This applies to a `foreign_cars` change for exactly the same
                    # reason, which is why `moved` tests BOTH knobs.
                    self._ac_wr_ema[s] = None
            cars_now = list(getattr(self, "_foreign_cars", []))
            if self._foreign_periods != old or cars_now != old_cars:
                self.engine.set_foreign_opponents(
                    self._foreign_sds, self._foreign_kinds, self._foreign_periods,
                    cars_now or None)
            print("auto-curriculum: " + " | ".join(parts), flush=True)
            return
        # fallback: ep_rew margin -- reached ONLY when there is no eval engine at
        # all (`wrs is None`). One shared knob, as before, since there is no
        # per-bot signal to split on. Note this signal is not zero-sum under a
        # shaped reward and is exactly what L8 warns about; it is a degraded
        # mode, not a design.
        cur = self._foreign_periods[0]
        new = cur
        if self._ac_ema > self._ac_band:
            new = max(self._ac_period_min, cur - 1)
        elif self._ac_ema < -self._ac_band:
            new = min(self._ac_period_max, cur + 1)
        reason = f"ep_rew_ema {self._ac_ema:+.2f}"
        if new != cur:
            self._foreign_periods = [new] * len(self._foreign_kinds)
            self.engine.set_foreign_opponents(
                self._foreign_sds, self._foreign_kinds, self._foreign_periods)
            print(f"auto-curriculum: {reason} -> period {cur} -> {new}", flush=True)
        else:
            print(f"auto-curriculum: {reason} -> hold period {cur}", flush=True)

    def _plan(self, n_league_slots: int) -> list[int]:
        """Full arena assignment: foreign at the front of each BLOCK, league at
        the back of each block. See `plan_block_assignment` for why per-block.

        With a single 1v1 block (every pre-v9 config) this reduces exactly to
        the old front/tail stamping, so those runs are unchanged.
        """
        return plan_block_assignment(
            self._team_sizes,
            self._foreign_modes if self._foreign_frac > 0 else [],
            self._foreign_frac,
            n_league_slots,
            float(self._league["frac"]) if self._league else 0.0,
        )

    def _apply_foreign(self, a: list[int]) -> list[int]:
        """Foreign-only assignment (no league opponents loaded yet).

        Kept as a named method because __init__ and `_refresh_opponents`'
        no-picks path both need "the assignment with foreign arenas but nothing
        else"; the placement itself lives in `plan_block_assignment`.
        """
        if self._foreign_slots == 0 or self._foreign_frac <= 0:
            return a
        return self._plan(0)

    def _refresh_opponents(self, it: int = 0):
        """Pull a fresh opponent pool from the registry and rebuild the arena
        assignment. Called every `league.refresh_iters` iterations from run().

        Reloads the registry from disk on every call: scripts/league_tick.py
        appends to the same jsonl out-of-process (via its own Registry
        instance), and the in-memory snapshot taken at Trainer.__init__ would
        otherwise never see those entries for the life of the run.
        Registry._save writes via a temp-file + os.replace, so a concurrent
        read here always sees either the old or the new file, never a torn
        one. If the reload itself fails (e.g. transient I/O error), keep the
        previous in-memory snapshot and warn, rather than taking down the run.

        Opponent picks are seeded from (run seed, iteration) so assignments
        are reproducible across identical runs but still vary refresh to
        refresh -- part of the repo's fixed-config determinism contract.

        Assignment (v9): the last `round(frac * block)` arenas OF EACH TEAM-SIZE
        BLOCK get opponent slots round-robin; the rest stay self-play (-1). See
        `plan_block_assignment`.

        This used to take the tail of the WHOLE arena list, which with mixed
        team sizes biased league arenas toward the largest teams -- and at
        [0.85, 0.1, 0.05] that meant every 3v3 arena was a league arena, so v8
        had zero 3v3 self-play and 100% of its 3v3 experience against one frozen
        1v1-trained near-random opponent. Per-block placement is what makes the
        1s/2s/3s row split independent of both opponent fractions.

        A pruned/corrupted checkpoint on disk must never take down a long
        run at a refresh boundary: each pick is loaded independently, bad
        ones are skipped with a warning, and if every pick fails the
        previous assignment (and opponent weights already in the engine)
        is left in place rather than falling back to an empty pool. The same
        holds if every pick loads fine but `engine.set_opponents` itself
        raises (e.g. an opponent state dict that's incompatible with the
        engine's current obs mode) -- that's also caught and logged, leaving
        the previous assignment and in-engine opponent weights untouched.
        """
        L = self._league
        from construct.league.registry import Registry
        try:
            L["registry"] = Registry(path=L["registry"].path)
        except Exception as e:
            print(f"league: failed to reload registry from disk ({e}), "
                  "keeping previous snapshot", flush=True)
        rng = random.Random(hash((self.cfg.env["seed"], it)))
        # Belt: only ever consider opponents tagged with this trainer's own
        # schema_version (self.schema["version"], derived from cfg.schema_path)
        # -- v0 and v1 policies can never play each other (different obs). The
        # try/except below (skipping bad loads, degrading set_opponents
        # failures to "keep previous assignment") is the suspenders: it still
        # catches anything that slips past this filter (e.g. a hand-edited
        # registry, or a future schema_version this filter doesn't know about).
        # ...and, for v1 ONLY, by ACTION TABLE -- which schema_version does not
        # distinguish: it is 1 for both the 92-row v1.1 and the 104-row v1-air
        # table. A 92-row arm picked into a 104-row run is rejected by
        # set_opponents below, which degrades to "keep previous assignment", so
        # without this filter a v1-air run would look like it simply had no
        # league opponents rather than an incompatible pool. Not applied on v0:
        # its table name ("rlgym_lookup_90") was never recorded on registry
        # entries, so filtering on it would reject every v0 arm.
        picks = L["choose"](L["registry"], k=L["slots"], rng=rng,
                             schema_version=self.schema["version"],
                             action_table=(self.schema["action_table"]
                                           if self.is_v1 else None))
        if not picks:
            # Keep any foreign arenas: "no league opponents" must not silently
            # drop the ported bots too.
            self._assignment = (
                self._apply_foreign([-1] * self.num_arenas)
                if self._foreign_slots else None
            )
            print("league: no opponents available (self-play + foreign only)", flush=True)
            return
        sds, names = [], []
        for p in picks:
            try:
                sds.append(L["load_sd"](p["ck"]))
                names.append(p["ck"])
            except Exception as e:
                print(f"league: skipping opponent {p['ck']!r} ({e})", flush=True)
        if not sds:
            print("league: all picks failed to load, keeping previous assignment", flush=True)
            return
        try:
            self.engine.set_opponents(sds)
        except Exception as e:
            # A misconfigured/incompatible opponent pool (e.g. a future
            # league-on-v1 obs/schema mismatch) must degrade to "keep
            # whatever assignment and engine-side opponent weights were
            # already in place" rather than take down the whole run.
            print(f"league: set_opponents failed ({e}), keeping previous assignment", flush=True)
            return
        self._assignment = self._plan(len(sds))
        n_lg = sum(1 for k in self._assignment if k >= 0)
        print(f"league: opponents {names} on {n_lg}/{self.num_arenas} arenas "
              f"(frac={L['frac']} PER BLOCK)", flush=True)

    def collect(self, T: int) -> dict:
        # state_dict() includes non-trainable buffers (v1's `action_table`)
        # by design -- the engine's candle net consumes them too.
        self.engine.set_weights(
            {k: v.detach().cpu().numpy().astype(np.float32)
             for k, v in self.net.state_dict().items()}
        )
        out = self.engine.collect(T, arena_opponents=self._assignment)
        # Buffer width shrinks from the full agent count when opponent arenas are
        # active (opponent-driven rows aren't learner transitions) -- see
        # Engine.collect's `learner_agents` docstring. All reshapes below, and
        # total_steps accounting in run(), must use N, not engine.num_agents.
        N = out["learner_agents"]

        values_ext = np.concatenate(
            [out["values"], out["last_values"][None, :]], axis=0
        )
        adv, ret = compute_gae(
            out["rewards"], values_ext, out["final_values"],
            out["terminated"], out["truncated"],
            self.cfg.ppo["gamma"], self.cfg.ppo["lam"],
        )
        dev = self.device
        # Obs plumbing dispatch. v0: the flat (T,N,94) tensor, exactly as
        # always. v1: the engine returns entity tensors instead of a flat obs
        # (see lib.rs collect) -- flatten each over (T,N) and pack them as a
        # dict whose KEYS MATCH EntityPolicyNet.evaluate/forward's keyword
        # parameters (ents/mask/query/prev); ppo_update indexes each tensor
        # by the same minibatch permutation and splats the dict into
        # net.evaluate. Everything GAE needs (rewards/values/final_values/
        # flags) is obs-layout-independent, and the engine computes value
        # bootstraps (final_values for truncations, last_values) in-engine
        # for both modes, so nothing else changes shape.
        if self.is_v1:
            obs = {
                "ents": torch.as_tensor(out["ents"].reshape(T * N, *out["ents"].shape[2:]), device=dev),
                "mask": torch.as_tensor(out["mask"].reshape(T * N, out["mask"].shape[2]), device=dev),
                "query": torch.as_tensor(out["query"].reshape(T * N, out["query"].shape[2]), device=dev),
                "prev": torch.as_tensor(out["prev"].reshape(T * N, out["prev"].shape[2]), device=dev),
            }
        else:
            obs = torch.as_tensor(out["obs"].reshape(T * N, self.engine.obs_size), device=dev)
        done_frac = (out["terminated"] | out["truncated"]).sum()
        result = {
            "obs": obs,
            "actions": torch.as_tensor(out["actions"].reshape(-1), device=dev),
            "logprobs": torch.as_tensor(out["logprobs"].reshape(-1), device=dev),
            "advantages": torch.as_tensor(adv, device=dev).reshape(-1),
            "returns": torch.as_tensor(ret, device=dev).reshape(-1),
            "values": torch.as_tensor(out["values"].reshape(-1), device=dev),
            "ep_reward_mean": float(out["rewards"].sum() / max(1, done_frac)),
            "n_agents": N,
            "forward_rows": int(out.get("forward_rows", N)),
        }
        # Per-ROW team-size label for the per-group advantage standardisation.
        #
        # THE FLATTENING IS THE PLACE A SILENT MIS-LABEL WOULD HIDE, so state it:
        # every other tensor above is `(T, N, ...)` flattened C-order, i.e. row
        # `t*N + n`. The engine gives ONE label per learner COLUMN (shape (N,)),
        # so it must be TILED over T -- `np.tile(lab, T)` produces exactly
        # `lab[r % N]` at flat index `r = t*N + n`, which matches. Do NOT
        # `repeat`; that would give `lab[r // T]` and mislabel almost every row.
        #
        # And it must come FROM THE ENGINE. Reconstructing it in Python would
        # mean re-deriving the worker split `(num_arenas - assigned)/(threads - t)`
        # with `threads` auto-detected from the machine -- it would agree on this
        # box and disagree on another, mis-scaling every gradient with no error.
        if "learner_team_size" in out:
            lab = np.asarray(out["learner_team_size"]).astype(np.int64)
            assert lab.shape == (N,), f"learner_team_size {lab.shape} != ({N},)"
            result["group"] = torch.as_tensor(np.tile(lab, T), device=dev)
        if "obs_v0" in out:
            result["obs_v0"] = torch.as_tensor(
                out["obs_v0"].reshape(T * N, TEACHER_OBS_SIZE), device=dev
            )
        return result

    def _group_line(self, batch) -> str:
        """Per-team-size rows and mean|A_norm|.

        The ONLY view of "is learning actually equal across 1s/2s/3s". G2's
        go/no-go is rows within +/-2% of 33.3% and mean|A_norm| within +/-10% of
        each other; before the per-group standardisation the latter differed by
        a measured 9.48x and there was no way to see it from the log.
        """
        g, adv = batch["group"], batch["advantages"]
        # Standardise the same way the update did, so the logged |A_norm| is the
        # number the entropy coefficient is actually competing with.
        from construct.learn.ppo import standardise_advantages
        p = self.cfg.ppo
        a, _ = standardise_advantages(
            adv, adv_norm=p.get("adv_norm", "global"), group=g,
            min_group_rows=int(p.get("min_group_rows", 2048)),
            group_std_floor=float(p.get("group_std_floor", 0.05)))
        total = g.numel()
        parts = []
        for m in torch.unique(g).tolist():
            sel = g == m
            k = int(sel.sum())
            parts.append(f"{m}v{m} {100.0 * k / max(1, total):.1f}%|A|{a[sel].abs().mean():.3f}")
        return "grp " + " ".join(parts)

    def _reward_terms_line(self) -> str:
        """Per-term reward sums + the §8.3 farming tripwires, per iteration.

        `touches_per_min_per_car` is tripwire 1 and the cleanest farm detector:
        continuous ball contact fires 12-14 touch events/s = 720-860/min, while
        healthy play is 20-40/min. `airborne_touch_frac` is G4's "is air play
        real" number (target 10-25%; >50% with goals flat is a hover farm).
        Empty string when the engine predates `reward_terms()`.

        READ `r_aerial_touch` AS A NET, NOT A VOLUME. Since 2026-07-26 the term
        is SIGNED (driving the ball at your own net is charged what driving it at
        theirs pays -- that symmetry is what makes a closed cycle sum to zero), so
        a small sum means "balanced", not "inert". `air_tch_frac` is the liveness
        signal; if it is healthy and r_aerial_touch is ~0, the policy is going up
        and achieving nothing, which is a different problem from not going up.
        Tripwire 1 alone would NOT have caught the lateral farm this replaced: a
        6000uu bat at 2400uu/s is 24 touches/min/car, inside the healthy band.
        """
        get = getattr(self.engine, "reward_terms", None)
        if get is None:
            return ""
        try:
            t = get()
        except Exception:
            return ""
        steps = t.get("agent_steps", 0.0)
        if steps <= 0:
            return ""
        # tick_skip 8 at 120Hz -> 15 decisions/s per car
        minutes = steps / 15.0 / 60.0
        out = [f"tch/min/car {t.get('touch_events', 0.0) / max(1e-9, minutes):.1f}"]
        tev = t.get("touch_events", 0.0)
        if tev > 0:
            out.append(f"air_tch_frac {t.get('airborne_touch_events', 0.0) / tev:.3f}")
        for k in ("aerial_touch", "air_setup", "touch_accel", "touch"):
            if t.get(k, 0.0):
                out.append(f"r_{k} {t[k]:.2f}")
        ge = t.get("goal_events", 0.0)
        if ge > 0:
            # L8's master detector: farming shows as shaping-per-goal RISING
            # while the gate win-share is flat or falling. Win-prob shaping once
            # drove ep_rew +510 while real skill fell.
            #
            # The `goal` term is DELIBERATELY EXCLUDED. These counters are summed
            # over every car in every arena, and goal/concede are exact negations,
            # so the goal term sums to ~0 by symmetry (measured: 218 goal events,
            # summed goal reward 0.00) and would only add noise to the ratio.
            shaping = sum(t.get(k, 0.0) for k in
                          ("touch", "vel_to_ball", "touch_accel", "vel_ball_to_goal",
                           "offensive_potential", "aerial_touch", "air_setup", "win_prob"))
            out.append(f"shaping/goal {shaping / ge:.2f}")
        return " " + " ".join(out)

    def run(self, max_iterations: int | None = None):
        it = 0
        p = self.cfg.ppo
        while max_iterations is None or it < max_iterations:
            t0 = time.perf_counter()
            # Keyed on total_steps, not on `it`: `it` restarts at 0 on every
            # resume, so an iteration-keyed schedule would silently rewind to the
            # launch rate every time the box is restarted. total_steps is restored
            # from the checkpoint, so the anneal is a pure function of how much
            # experience the LINEAGE has seen. No-op unless lr_final is set.
            lr = self._apply_lr(lr_at(self.total_steps, p))
            if self._league and it % self._league["refresh"] == 0:
                self._refresh_opponents(it)
            batch = self.collect(p["rollout_steps"])

            # --- extra-loss hook: kickstart distillation (T7) + BC-prior anchor (K3) ---
            # Both terms funnel through compose_extra_loss (module-level, see its
            # docstring). Kickstart is guarded on self.kickstart (None for every v0
            # run, and for any v1 run without a [kickstart] config) AND on "obs_v0"
            # actually being in the batch -- collect() only puts it there when the
            # engine was built with emit_v0_obs=True, i.e. exactly the
            # kickstart_active case from __init__. The prior is guarded on
            # self.kl_prior (None for every v0 run, and for any v1 run without a
            # [kl_prior] config). compose_extra_loss returns None when neither is
            # active, which makes ppo_update's hook parameter a complete no-op: v0
            # runs and prior-less v1 runs are byte-identical to pre-K3 behavior.
            kickstart_outs = None
            lambda_k = lambda_v = 0.0
            if self.kickstart is not None and "obs_v0" in batch:
                lambda_k, lambda_v = self.kickstart["schedule"].coef(self.total_steps)
                if lambda_k > 0.0 or lambda_v > 0.0:
                    with torch.no_grad():
                        kickstart_outs = self.kickstart["teacher"].logits_values(batch["obs_v0"])
            prior_logits, lambda_p = None, 0.0
            if self.kl_prior is not None:
                lambda_p = self.kl_prior["lambda"]
                # Optional anchor ANNEAL: keep the anchor strong early -- it holds
                # the policy still while the value head recalibrates to a new reward
                # (a cold regime change otherwise collapses: the stale critic feeds
                # garbage advantages) -- then decay it to `lambda_floor` so the
                # policy is free to move past the anchor, driven by the curriculum.
                anneal = self.kl_prior.get("anneal_iters", 0)
                if anneal > 0:
                    floor = self.kl_prior.get("lambda_floor", 0.0)
                    frac = max(0.0, 1.0 - it / anneal)
                    lambda_p = floor + (self.kl_prior["lambda"] - floor) * frac
                with torch.no_grad():
                    prior_logits = self.kl_prior["net"].logits(batch["obs"])
            extra_loss_fn = compose_extra_loss(
                self.net, batch,
                kickstart=kickstart_outs, lambda_k=lambda_k, lambda_v=lambda_v,
                prior_logits=prior_logits, lambda_p=lambda_p,
            )

            stats = ppo_update(
                self.net, self.opt, batch, clip=p["clip"], entropy_coef=p["entropy_coef"],
                value_coef=p["value_coef"], epochs=p["epochs"], minibatch_size=p["minibatch_size"],
                extra_loss_fn=extra_loss_fn,
                # Default "global" == the historical single-scalar
                # standardisation, byte-identical. v9 declares "group".
                adv_norm=p.get("adv_norm", "global"),
                group=batch.get("group"),
                min_group_rows=int(p.get("min_group_rows", 2048)),
                group_std_floor=float(p.get("group_std_floor", 0.05)),
            )
            # Steps count LEARNER transitions (batch["n_agents"] == out["learner_agents"]
            # from collect()), not the raw engine agent count -- opponent arenas
            # contribute fewer learner rows than self-play arenas.
            n = p["rollout_steps"] * batch["n_agents"]
            self.total_steps += n
            it += 1
            # auto-curriculum: retune the opponent difficulty from this iter's margin
            self._auto_curriculum_step(it, batch["ep_reward_mean"])
            if it % self.cfg.run.get("log_every_iters", 1) == 0:
                sps = n / (time.perf_counter() - t0)
                msg = (
                    f"iter {it} steps {self.total_steps:,} sps {sps:,.0f} "
                    f"ep_rew {batch['ep_reward_mean']:.3f} "
                    f"pi_loss {stats['policy_loss']:.4f} v_loss {stats['value_loss']:.4f} "
                    f"ent {stats['entropy']:.3f} clip {stats['clip_frac']:.3f}"
                )
                if kickstart_outs is not None:
                    msg += f" kick_kl {stats.get('kick_kl', 0.0):.4f} lambda_k {lambda_k:.3f}"
                if self.kl_prior is not None:
                    msg += f" kl_pri {stats.get('kl_pri', 0.0):.4f} lambda_p {lambda_p:.3f}"
                if p.get("lr_final") is not None:
                    # Only when a schedule is configured, so every existing run's
                    # log line -- and ctl.py/dashboard.py's regex, which anchors on
                    # `clip` and tolerates suffixes -- is unchanged.
                    msg += f" lr {lr:.2e}"
                if batch.get("group") is not None:
                    msg += " " + self._group_line(batch)
                if stats.get("adv_group_floor_hits", 0.0):
                    # G1's go/no-go asserts this is 0. A floor hit means a
                    # team-size group had (nearly) no reward events and its pure
                    # critic noise was about to be amplified to unit scale.
                    msg += f" adv_group_floor_hits {stats['adv_group_floor_hits']:.0f}"
                if batch.get("forward_rows", 0) != batch["n_agents"]:
                    msg += f" fwd_rows {batch['forward_rows']}"
                msg += self._reward_terms_line()
                print(msg, flush=True)
            if it % self.cfg.run.get("save_every_iters", 20) == 0:
                os.makedirs(self.cfg.run["checkpoint_dir"], exist_ok=True)
                self.save_checkpoint(
                    os.path.join(self.cfg.run["checkpoint_dir"], f"ck_{self.total_steps:012d}.pt")
                )

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.opt.state_dict(),
                "total_steps": self.total_steps,
                "schema_version": self.schema["version"],
                # Which action table this lineage was trained against. Needed
                # because `schema_version` is 1 for BOTH v1 tables -- the
                # 92-row v1.1 and the 104-row v1-air share an obs contract and
                # differ only in the policy's output dimension. Downstream
                # (deploy, league, benches) dispatches on this NAME rather than
                # inferring from a row count, so a checkpoint always says which
                # table decodes its logits. Absent on every pre-v9 checkpoint;
                # readers must default to "construct_92_v1".
                "action_table": self.schema["action_table"],
                "config": {"net": self.cfg.net, "ppo": self.cfg.ppo, "env": self.cfg.env},
                # provenance only (resume takes the path from CLI/config, not from here):
                # records which reward regime produced this checkpoint
                "reward_config_path": self.cfg.reward_config_path,
                "curriculum_config_path": self.cfg.curriculum_config_path,
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, cfg_path: str = "configs/train_v0.toml") -> "Trainer":
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TrainConfig.load(cfg_path)
        # Restore the exact env/net/ppo config the checkpoint was trained under
        # (not just net) — otherwise resuming with the on-disk default config
        # silently changes num_arenas/rollout_steps and desyncs total_steps.
        cfg.net = state["config"]["net"]
        cfg.ppo = state["config"]["ppo"]
        cfg.env = state["config"]["env"]
        return cls(cfg, _state=state)


if __name__ == "__main__":
    import sys

    cfg = TrainConfig.load(sys.argv[1] if len(sys.argv) > 1 else "configs/train_v0.toml")
    Trainer(cfg).run()
