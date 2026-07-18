"""Headless behavior eval: touches/min and mean dist-to-ball over N eval steps.

Dispatches on the checkpoint's `schema_version` (v0, the default when the key
is absent, or v1 as written by Trainer.save_checkpoint since the
entity-transformer plan). v0 builds a PolicyValueNet + flat-obs Engine and
runs a manual step loop, unchanged from before. v1 builds an EntityPolicyNet
+ entity-obs Engine (schema/v1.toml); the v1 Engine's reset()/step() are
v0-only (obs are entity tensors, driven engine-side -- see policy_v1.rs / the
Engine.v0_only guard in lib.rs), so the v1 path drives the same eval tape
through engine.collect() instead: one set_weights, one collect() call for the
whole eval window, sampled actions (matches training and the v0 tape's
sampled-action convention -- see the v0 branch's determinism comment below).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from construct._engine import Engine
from construct.learn.model import PolicyValueNet

ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
is_v1 = int(ck.get("schema_version", 0)) == 1
STEPS = 4500  # 5 min of game time

if is_v1:
    from construct.learn.model_v1 import EntityPolicyNet

    net_cfg = ck["config"]["net"]
    heads = int(net_cfg["heads"])
    try:
        from construct._engine import action_table_v1
        table = action_table_v1()
    except ImportError:
        # T8 says action_table_v1() exists; fall back to the checkpoint's own
        # buffer (register_buffer'd, so it's in the state dict) if it doesn't.
        table = ck["model"]["action_table"].numpy()
    net = EntityPolicyNet(
        d_model=int(net_cfg["d_model"]), layers=int(net_cfg["layers"]),
        heads=heads, ff=int(net_cfg["ff"]), action_table=table,
    )
    net.load_state_dict(ck["model"])
    net.eval()

    eng = Engine(num_arenas=16, blue=1, orange=1, schema_path="schema/v1.toml",
                 reward_config_path="configs/reward_v0.toml", seed=1234, net_heads=heads)
    eng.set_weights(
        {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    )
    out = eng.collect(STEPS)
    touches = int((out["rewards"] >= 0.5).sum())  # touch weight fires at >= 0.5
    goals = int((out["rewards"] >= 9.5).sum())    # goal pays +10 to the scorer, unambiguous

    # dist comes from obs[:, 28:31] in v0 ((ball - car) * pos_norm, see the v0
    # branch below); v1 has no flat obs, so read the same quantity off the
    # entity tensor instead. Per obs_v1.rs: entity row 0 is always self
    # (SELF_IDX) and row 6 is always the ball (BALL_IDX); both rows carry
    # their position at cols [5:8], already scaled by the same pos_norm
    # (independently scaling self and ball by the same scalar and then
    # subtracting is identical to scaling the difference).
    SELF_IDX, BALL_IDX, POS_LO, POS_HI = 0, 6, 5, 8
    ents = out["ents"]  # (T, N, MAX_ENT, ENT_FEAT)
    self_pos = ents[:, :, SELF_IDX, POS_LO:POS_HI]
    ball_pos = ents[:, :, BALL_IDX, POS_LO:POS_HI]
    pos_norm = 1 / 2300  # schema/v1.toml normalization.pos_norm, same as v0.toml's
    dist_mean = float(np.linalg.norm(ball_pos - self_pos, axis=-1).mean() / pos_norm)

    minutes = STEPS / 15 / 60 * eng.num_agents
    match_minutes = STEPS / 15 / 60 * 16  # 16 arenas = 16 concurrent matches
    touches_min, goals_min, dist = touches / minutes, goals / match_minutes, dist_mean
else:
    eng = Engine(num_arenas=16, blue=1, orange=1, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=1234)
    net = PolicyValueNet(eng.obs_size, eng.action_count, tuple(ck["config"]["net"]["hidden"]))
    net.load_state_dict(ck["model"])
    net.eval()

    obs = torch.as_tensor(eng.reset())
    touches = 0
    goals = 0
    dist_sum = 0.0
    torch.manual_seed(0)  # sampled actions, reproducible: argmax play deadlocks in
    # mirror-symmetric standoffs (both cars park at the ball) and zeroes the metric
    for _ in range(STEPS):
        with torch.no_grad():
            logits = net(obs)[0]
            acts = torch.distributions.Categorical(logits=logits).sample().numpy().astype(np.int64)
        nobs, rew, term, trunc, _ = eng.step(acts)
        touches += int((rew >= 0.5).sum())  # touch weight fires at >= 0.5
        goals += int((rew >= 9.5).sum())    # goal pays +10 to the scorer, unambiguous
        # obs[28:31] is (ball - car) * pos_norm
        dist_sum += float(np.linalg.norm(nobs[:, 28:31], axis=1).mean() / (1 / 2300))
        obs = torch.as_tensor(nobs)

    minutes = STEPS / 15 / 60 * eng.num_agents
    match_minutes = STEPS / 15 / 60 * 16  # 16 arenas = 16 concurrent matches
    touches_min, goals_min, dist = touches / minutes, goals / match_minutes, dist_sum / STEPS

# stdout is parsed by other tooling (dashboard EvalRunner used to regex it) —
# keep the three lines byte-identical to before.
print(f"touches/min/agent: {touches_min:.2f}")
print(f"mean dist to ball: {dist:.0f} uu")
print(f"goals/min/match: {goals_min:.2f}")

# Structured seam for the dashboard's eval-history panel: one jsonl row per run.
_hist = Path(__file__).resolve().parent.parent / "logs" / "eval_history.jsonl"
_hist.parent.mkdir(parents=True, exist_ok=True)
with _hist.open("a") as f:
    f.write(json.dumps({
        "ts": int(time.time()), "ck": Path(sys.argv[1]).name,
        "goals_min": round(goals_min, 4), "touches_min": round(touches_min, 4),
        "dist": round(dist, 1),
    }) + "\n")
