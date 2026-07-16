"""Resume training from a checkpoint, optionally overriding env settings.

Usage: python scripts/resume_train.py <checkpoint.pt> [--num-arenas N] [--device D]

The checkpoint stores the env/ppo config it was trained with; load_checkpoint
restores it for safety. num_arenas is safe to override (model shapes don't
depend on it — bigger = larger inference batches = better throughput).
"""
import argparse

import torch

from construct.learn.config import TrainConfig
from construct.learn.train import Trainer

p = argparse.ArgumentParser()
p.add_argument("checkpoint")
p.add_argument("--num-arenas", type=int, default=None)
p.add_argument("--seed", type=int, default=None)
p.add_argument("--checkpoint-dir", default=None)
p.add_argument("--device", default=None)
p.add_argument("--config", default="configs/train_v0.toml")
p.add_argument("--reward-config", default=None)
p.add_argument("--team-sizes", default=None, help="W1,W2,W3 weights for 1v1/2v2/3v3 arena mix")
p.add_argument("--curriculum-config", default=None)
p.add_argument("--reset-optimizer", action="store_true",
               help="drop Adam state (use when swapping reward regimes — stale "
                    "moments belong to the old loss landscape)")
p.add_argument("--league", action="store_true",
               help="enable opponent-pool sampling with config-file/default "
                    "league settings (registry/opponent_frac/refresh_iters/slots)")
p.add_argument("--kickstart-teacher", default=None,
               help="path to a frozen v0 MLP checkpoint; enables kickstart "
                    "distillation (annealed KL + value regression to this "
                    "teacher). Only takes effect on a v1-schema run.")
p.add_argument("--kickstart-steps", type=int, default=None,
               help="steps over which the kickstart KL weight anneals to 0 "
                    "(default 500_000_000; only used with --kickstart-teacher)")
args = p.parse_args()

state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
cfg = TrainConfig.load(args.config)
cfg.net = state["config"]["net"]
cfg.env = state["config"]["env"]
cfg.ppo = state["config"]["ppo"]
if args.num_arenas:
    cfg.env["num_arenas"] = args.num_arenas
if args.seed is not None:
    cfg.env["seed"] = args.seed
if args.checkpoint_dir:
    cfg.run["checkpoint_dir"] = args.checkpoint_dir
if args.device:
    cfg.run["device"] = args.device
if args.reward_config:
    cfg.reward_config_path = args.reward_config
if args.team_sizes:
    cfg.env["team_size_weights"] = [float(x) for x in args.team_sizes.split(",")]
if args.curriculum_config:
    cfg.curriculum_config_path = args.curriculum_config
if args.reset_optimizer:
    state["optimizer"] = None
if args.league:
    cfg.league = {**cfg.league, "enabled": True}
if args.kickstart_teacher:
    cfg.kickstart = {**cfg.kickstart, "teacher": args.kickstart_teacher}
    if args.kickstart_steps is not None:
        cfg.kickstart["steps"] = args.kickstart_steps

t = Trainer(cfg, _state=state)
print(f"resumed at {t.total_steps:,} steps | arenas={cfg.env['num_arenas']} "
      f"agents={t.engine.num_agents} device={t.device}", flush=True)
t.run()
