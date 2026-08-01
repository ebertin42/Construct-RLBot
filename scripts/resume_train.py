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
p.add_argument("--kl-prior", default=None,
               help="v1 BC checkpoint to use as a frozen KL prior anchor")
p.add_argument("--kl-prior-lambda", type=float, default=None,
               help="KL(student‖prior) coefficient (default 0.05)")
p.add_argument("--entropy-coef", type=float, default=None,
               help="override PPO entropy_coef. The resumed CHECKPOINT's ppo block "
                    "normally wins over the toml, so this is the only way to change it "
                    "on a resume (see the 2026-07-19 state-blindness diagnosis).")
p.add_argument("--lr", type=float, default=None,
               help="override the Adam learning rate. Needed because lr was "
                    "unchangeable on a resume TWICE over: cfg.ppo comes from the "
                    "checkpoint, and Adam's restored param_groups carry lr as well "
                    "-- so before 2026-07-26 the only way to change it was "
                    "--reset-optimizer, which also discards the moment estimates. "
                    "Trainer now applies cfg.ppo['lr'] after the optimizer restore.")
p.add_argument("--lr-final", type=float, default=None,
               help="turn on (or retarget) the linear lr anneal; pair with "
                    "--lr-hold-steps / --lr-anneal-steps. See train.lr_at.")
p.add_argument("--lr-hold-steps", type=int, default=None,
               help="hold --lr flat for this many TOTAL lineage steps before the "
                    "anneal starts (keyed on total_steps, which survives a resume)")
p.add_argument("--lr-anneal-steps", type=int, default=None,
               help="length of the linear ramp from --lr to --lr-final")
p.add_argument("--adv-norm", choices=["global", "group"], default=None,
               help="override PPO adv_norm ('global' = one scalar over the whole "
                    "mixed batch, historical; 'group' = standardise within each "
                    "team size, v9's D6). Same reason as --entropy-coef: the "
                    "resumed CHECKPOINT's ppo block wins over the toml, so on a "
                    "resume this flag is the ONLY way to change it -- and it is "
                    "also the kill switch if per-group standardisation misbehaves.")
p.add_argument("--aux", action="store_true",
               help="enable the aux heads (aux_reward + aux_recon, see learn/aux.py). "
                    "REQUIRED ON A RESUME: line ~75 does `cfg.net = state[\'config\'][\'net\']`, "
                    "so the CHECKPOINT's net block overwrites the toml and an `aux = true` "
                    "there is silently discarded -- the run would train with dead heads and "
                    "produce a guaranteed null. Same trap class as --entropy-coef/--adv-norm.")
p.add_argument("--aux-recon-coef", type=float, default=None,
               help="weight on the masked entity-reconstruction aux loss")
p.add_argument("--aux-reward-coef", type=float, default=None,
               help="weight on the multi-horizon return-prediction aux loss")
p.add_argument("--max-iterations", type=int, default=None,
               help="stop after N iterations (for bounded A/B experiments); default: run forever")
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
# Applied AFTER `cfg.net = state["config"]["net"]` above, which is the whole point.
if args.aux:
    cfg.net = {**cfg.net, "aux": True}
if args.aux_recon_coef is not None:
    cfg.ppo = {**cfg.ppo, "aux_recon_coef": args.aux_recon_coef}
if args.aux_reward_coef is not None:
    cfg.ppo = {**cfg.ppo, "aux_reward_coef": args.aux_reward_coef}
if args.reset_optimizer:
    state["optimizer"] = None
if args.league:
    cfg.league = {**cfg.league, "enabled": True}
if args.entropy_coef is not None:
    cfg.ppo = {**cfg.ppo, "entropy_coef": args.entropy_coef}
if args.adv_norm is not None:
    cfg.ppo = {**cfg.ppo, "adv_norm": args.adv_norm}
for _flag, _key in (("lr", "lr"), ("lr_final", "lr_final"),
                    ("lr_hold_steps", "lr_hold_steps"),
                    ("lr_anneal_steps", "lr_anneal_steps")):
    _v = getattr(args, _flag)
    if _v is not None:
        cfg.ppo = {**cfg.ppo, _key: _v}
if args.kickstart_teacher:
    cfg.kickstart = {**cfg.kickstart, "teacher": args.kickstart_teacher}
    if args.kickstart_steps is not None:
        cfg.kickstart["steps"] = args.kickstart_steps
if args.kl_prior:
    cfg.kl_prior = {**cfg.kl_prior, "ck": args.kl_prior}
if args.kl_prior_lambda is not None:
    cfg.kl_prior = {**cfg.kl_prior, "lambda": args.kl_prior_lambda}

t = Trainer(cfg, _state=state)
print(f"resumed at {t.total_steps:,} steps | arenas={cfg.env['num_arenas']} "
      f"agents={t.engine.num_agents} device={t.device}", flush=True)
t.run(max_iterations=args.max_iterations)
