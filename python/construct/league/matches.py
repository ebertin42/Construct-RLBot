"""Head-to-head matches between two frozen checkpoints, via opponent arenas.

The engine's reward stream is used purely as a scoring tape: with reward_v0
(goal=10, bias 0, shaping << 9.4) a learner-row reward >= 9.4 means A scored,
<= -9.4 means B scored. Matches always use reward_v0 regardless of what the
policies were trained on.
"""
import numpy as np
import torch

from construct._engine import Engine

# Goal detection threshold: goal pays ±10; same-step shaping can offset a concede
# by up to +0.55 (touch 0.5 + vel_to_ball 0.05), so a concede row can be as small
# as -9.45 in magnitude. Non-goal rows never exceed |0.55|. 9.4 sits safely
# inside the [0.55, 9.45] gap on both sides.
GOAL_THRESHOLD = 9.4


def load_sd(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    return {k: v.numpy().astype(np.float32) for k, v in ck["model"].items()}


class MatchRunner:
    def __init__(self, num_arenas=8, seed=0, reward_config="configs/reward_v0.toml", mode=1):
        # Goal events pay every learner agent on the scoring team in that arena.
        # At mode=1 (1v1) each opponent arena has exactly one learner row (blue),
        # so the GOAL_THRESHOLD count below is exact. 2v2+ would multi-count (both
        # teammates get paid the same goal reward) -- divide by team size when that
        # arrives. YAGNI today: assert mode==1 until then.
        assert mode == 1, "MatchRunner only supports 1v1 (mode=1); 2v2+ would multi-count goals"
        self.eng = Engine(num_arenas=num_arenas, blue=mode, orange=mode,
                          schema_path="schema/v0.toml", reward_config_path=reward_config,
                          seed=seed)
        self.assignment = [0] * num_arenas

    def play(self, sd_a, sd_b, steps=2700):
        # Arenas are not reset between calls: match N+1's collect() continues
        # from wherever match N left the ball/cars, and set_weights/set_opponents
        # swap the policies driving those arenas mid-episode. Intentional --
        # avoids a reset() round-trip per match -- and deterministic given a
        # fixed seed + call sequence (see test_match_deterministic); the only
        # cost is a bit of extra noise in per-match goal counts, which washes
        # out over the TrueSkill ladder's many matches.
        self.eng.set_weights(sd_a)
        self.eng.set_opponents([sd_b])
        out = self.eng.collect(steps, arena_opponents=self.assignment)
        rew = np.asarray(out["rewards"])
        goals_a = int((rew >= GOAL_THRESHOLD).sum())
        goals_b = int((rew <= -GOAL_THRESHOLD).sum())
        return goals_a, goals_b
