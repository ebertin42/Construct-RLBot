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

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(**raw)
