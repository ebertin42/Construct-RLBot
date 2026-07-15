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

    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, "rb") as f:
            raw = tomllib.load(f)
        return cls(**raw)
