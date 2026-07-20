"""Head-to-head matches between two frozen checkpoints, via opponent arenas.

The engine's reward stream is used purely as a scoring tape: with reward_v0
(goal=10, bias 0, shaping << 9.4) a learner-row reward >= 9.4 means A scored,
<= -9.4 means B scored. Matches always use reward_v0 regardless of what the
policies were trained on -- reward is computed by reward::compute() from raw
game state (engine/src/episode.rs step_impl), never from the obs encoding, so
the tape trick is schema-independent and holds unchanged for a v1 engine.

v0 and v1 policies can NEVER play each other (different obs contracts): each
MatchRunner is built for exactly one schema_version, and play_entries() below
is the hard guard against feeding it a cross-schema pair.
"""
import numpy as np
import torch

from construct._engine import Engine

# Goal detection threshold: goal pays ±10; same-step shaping can offset a concede
# by up to +0.55 (touch 0.5 + vel_to_ball 0.05), so a concede row can be as small
# as -9.45 in magnitude. Non-goal rows never exceed |0.55|. 9.4 sits safely
# inside the [0.55, 9.45] gap on both sides.
GOAL_THRESHOLD = 9.4

_SCHEMA_PATHS = {0: "schema/v0.toml", 1: "schema/v1.toml"}


def load_sd(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    # Generic tensor->numpy conversion: works unchanged for both v0
    # (PolicyValueNet) and v1 (EntityPolicyNet, whose state_dict also carries
    # the non-trainable `action_table` buffer -- the engine's v1 EntityPolicy
    # requires and consumes that key, see engine/src/policy_v1.rs).
    return {k: v.numpy().astype(np.float32) for k, v in ck["model"].items()}


class MatchRunner:
    def __init__(self, num_arenas=8, seed=0, reward_config="configs/reward_v0.toml", mode=1,
                 schema_version=0, net_heads=4):
        # Goal events pay every learner agent on the scoring team in that arena.
        # At mode=1 (1v1) each opponent arena has exactly one learner row (blue),
        # so the GOAL_THRESHOLD count below is exact. 2v2+ would multi-count (both
        # teammates get paid the same goal reward) -- divide by team size when that
        # arrives. YAGNI today: assert mode==1 until then.
        assert mode == 1, "MatchRunner only supports 1v1 (mode=1); 2v2+ would multi-count goals"
        assert schema_version in _SCHEMA_PATHS, (
            f"MatchRunner schema_version must be one of {sorted(_SCHEMA_PATHS)}, "
            f"got {schema_version}"
        )
        self.schema_version = schema_version
        engine_kwargs = dict(
            num_arenas=num_arenas, blue=mode, orange=mode,
            schema_path=_SCHEMA_PATHS[schema_version], reward_config_path=reward_config,
            seed=seed,
        )
        if schema_version == 1:
            # candle rebuilds attention from the raw state dict; head count is
            # not recoverable from tensor shapes alone (same reason Trainer
            # passes it -- see train.py's engine_kwargs["net_heads"]).
            engine_kwargs["net_heads"] = net_heads
        self.eng = Engine(**engine_kwargs)
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


def play_entries(mr: "MatchRunner", entry_a: dict, entry_b: dict, steps: int = 2700):
    """Run a match between two registry entries via `mr`, refusing to pit
    different schema_version checkpoints against each other.

    v0 and v1 obs are structurally different (flat 94-float vector vs entity
    tensors) -- a cross-schema match would either crash deep in the engine
    boundary (missing/extra state_dict keys) or, worse, silently misbehave.
    This checks entry metadata *before* touching disk or the engine, so the
    refusal is immediate and doesn't depend on `mr` being usable at all
    (checked first, ahead of the mr.schema_version comparison below).
    """
    va = entry_a.get("schema_version", 0)
    vb = entry_b.get("schema_version", 0)
    if va != vb:
        raise ValueError(
            f"cross-schema match refused: {entry_a['ck']!r} is schema_version={va}, "
            f"{entry_b['ck']!r} is schema_version={vb}"
        )
    if va != mr.schema_version:
        raise ValueError(
            f"entry schema_version={va} does not match MatchRunner schema_version="
            f"{mr.schema_version}"
        )
    return mr.play(load_sd(entry_a["ck"]), load_sd(entry_b["ck"]), steps=steps)


def split_matches(rewards, terminated, threshold=GOAL_THRESHOLD):
    """Group a reward tape into per-match (goals_a, goals_b) using terminated
    flags as match boundaries.

    In match mode `terminated` means "the clock expired", so it is exactly the
    match boundary. A goal is still a reward spike past `threshold` -- matches
    always run reward_v0 as a neutral scoring tape (see module doc), so this
    holds whatever the policies trained on.

    A trailing partial match is DISCARDED: it has no outcome, and scoring it as
    a draw would bias every gate toward 0.5.
    """
    rewards = np.asarray(rewards)
    terminated = np.asarray(terminated)
    out, a, b = [], 0, 0
    for t in range(rewards.shape[0]):
        a += int((rewards[t] >= threshold).sum())
        b += int((rewards[t] <= -threshold).sum())
        if bool(terminated[t].any()):
            out.append((a, b))
            a, b = 0, 0
    return out


def match_record(matches):
    """Win/draw/loss counts and win share (draws count 0.5).

    `win_share` is None when no match completed -- 0.0 would read as a total
    loss and could drive a promotion decision off zero evidence.
    """
    wins = sum(1 for a, b in matches if a > b)
    losses = sum(1 for a, b in matches if a < b)
    draws = len(matches) - wins - losses
    share = None if not matches else (wins + 0.5 * draws) / len(matches)
    return {"wins": wins, "draws": draws, "losses": losses,
            "matches": len(matches), "win_share": share}
