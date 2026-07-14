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

t = Trainer(cfg, _state=state)
print(f"resumed at {t.total_steps:,} steps | arenas={cfg.env['num_arenas']} "
      f"agents={t.engine.num_agents} device={t.device}", flush=True)
t.run()
