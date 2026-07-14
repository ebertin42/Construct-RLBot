"""Head-to-head matches between two frozen checkpoints, via opponent arenas.

The engine's reward stream is used purely as a scoring tape: with reward_v0
(goal=10, bias 0, shaping << 9.5) a learner-row reward >= 9.5 means A scored,
<= -9.5 means B scored. Matches always use reward_v0 regardless of what the
policies were trained on.
"""
import numpy as np
import torch

from construct._engine import Engine


def load_sd(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    return {k: v.numpy().astype(np.float32) for k, v in ck["model"].items()}


class MatchRunner:
    def __init__(self, num_arenas=8, seed=0, reward_config="configs/reward_v0.toml", mode=1):
        # Goal events pay every learner agent on the scoring team in that arena.
        # At mode=1 (1v1) each opponent arena has exactly one learner row (blue),
        # so the >=9.5 / <=-9.5 count below is exact. 2v2+ would multi-count (both
        # teammates get paid the same goal reward) -- divide by team size when that
        # arrives. YAGNI today: assert mode==1 until then.
        assert mode == 1, "MatchRunner only supports 1v1 (mode=1); 2v2+ would multi-count goals"
        self.eng = Engine(num_arenas=num_arenas, blue=mode, orange=mode,
                          schema_path="schema/v0.toml", reward_config_path=reward_config,
                          seed=seed)
        self.assignment = [0] * num_arenas

    def play(self, sd_a, sd_b, steps=2700):
        self.eng.set_weights(sd_a)
        self.eng.set_opponents([sd_b])
        out = self.eng.collect(steps, arena_opponents=self.assignment)
        rew = np.asarray(out["rewards"])
        goals_a = int((rew >= 9.5).sum())
        goals_b = int((rew <= -9.5).sum())
        return goals_a, goals_b
