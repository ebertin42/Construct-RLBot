import tomllib
from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    schema_path: str
    reward_config_path: str
    env: dict = field(default_factory=dict)
    net: dict = field(default_factory=dict)
    ppo: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)
    curriculum_config_path: str = ""
    # keys: enabled (bool), opponent_frac (float, default 0.2), registry (path),
    # refresh_iters (int, default 200), slots (int <= 8, default 4). See
    # Trainer._refresh_opponents / Trainer.collect for how these are consumed.
    league: dict = field(default_factory=dict)
    # kickstart distillation (v1-schema runs only; see kickstart.py + Trainer.__init__).
    # keys: teacher (v0 checkpoint path, required to activate), steps (anneal
    # horizon, default 500_000_000), lambda_k (initial KL weight, default 1.0),
    # lambda_v (value-regression weight, default 0.5).
    kickstart: dict = field(default_factory=dict)
    # KL-prior anchor (v1-schema runs only; frozen BC net, see kl_prior.py +
    # Trainer.__init__). Keys: ck (checkpoint path), lambda (float, default 0.05).
    kl_prior: dict = field(default_factory=dict)
    # Foreign opponents: externally-trained community bots ported into the engine
    # (see docs/foreign-opponents.md). Keys: enabled (bool), opponent_frac (float,
    # fraction of EACH TEAM-SIZE BLOCK's arenas they drive, default 0.25),
    # decision_periods (list[int], one per slot, default all 1 -- a bot reacts
    # every Nth decision, the difficulty handicap, since every bot beats us 96-0),
    # foreign_cars (list[int], one per slot, how many ORANGE cars each bot drives;
    # the rest of that team mirrors our own policy).
    #
    # The roster comes in one of two shapes:
    #   v9:     slots = [{ kind = "nexto", mode = 3, weights = "...npz" }, ...]
    #           -- `mode` is the TEAM SIZE that slot plays at, which is what
    #           per-block placement and the (bot x mode) rung ladder need.
    #   legacy: kinds = [...] + weights = [...] (parallel lists), treated as
    #           all mode 1. Still accepted, unchanged.
    # At most 8 slots (the cap in engine/src/lib.rs).
    #
    # Independent of `league`: within EACH block, foreign arenas are taken from
    # the front and league opponent arenas from the back, so both can run at once
    # in every regime.
    foreign: dict = field(default_factory=dict)
    # On-policy distillation FROM a foreign bot (learn/foreign_distill.py). Distinct from
    # `foreign` above, which puts those bots in the ORANGE cars as opponents: this queries
    # one on the LEARNER's own car every step and asks "what would you do here", then trains
    # the student's action head on the answer. Keys: kind ("nexto"/"immortal"/..., required
    # to activate), weights (path to the .npz, default ~/.cache/construct/<kind>_weights.npz).
    # The strength lives in ppo.foreign_distill_coef so it can be annealed like every other
    # loss weight. Weights are UNLICENSED for redistribution and live outside the repo.
    #
    # MULTI-TEACHER: `mixture = [{kind, weight, weights?}, ...]` distils from several bots
    # at once. The engine holds exactly ONE teacher (Engine::set_teacher takes a single
    # state dict), so the mixture is realised by SAMPLING a teacher per iteration in
    # proportion to `weight` -- an unbiased estimator of the weighted-average CE, at
    # iteration granularity. `kind` alone still means one teacher, set once, on the
    # pre-mixture code path.
    #
    # WHICH BOTS CAN ACTUALLY TEACH: the label is an index into OUR action table, so a
    # teacher whose controls fall off that table emits -1 and the frame is dropped.
    # Measured on ck_004068864000, 4 arenas, 2 iterations (fd_lab):
    #     nexto 1.000   immortal 0.87   necto 0.24   element 0.20
    # nexto shares our table by construction; immortal's 126-row table overlaps it far
    # more than its separate provenance suggests. necto (multi-discrete decode) and
    # element (continuous controls) are NOT usable: a 20% label rate is not a random
    # subsample, it is exactly the frames our table cannot express, so training on it
    # teaches a biased subset while looking healthy.
    foreign_distill: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(**raw)
