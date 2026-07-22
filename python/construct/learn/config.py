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
    # (see docs/foreign-opponents.md). Keys: enabled (bool), kinds (list[str], e.g.
    # ["immortal"]), weights (list[path] to the extracted npz, same length/order as
    # kinds), opponent_frac (float, fraction of arenas they drive, default 0.25),
    # decision_periods (list[int] same length as kinds, default all 1; a bot reacts
    # every Nth decision -- the difficulty handicap, since every bot beats us 96-0).
    # Independent of `league`: foreign arenas are taken from the FRONT of the arena
    # list and league opponent arenas from the BACK, so both can run at once.
    foreign: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(**raw)
