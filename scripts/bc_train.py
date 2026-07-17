"""Behavior-cloning pretrain of EntityPolicyNet on bc-export shards.

Usage: python scripts/bc_train.py [--config configs/bc_v1.toml]
           [--data-dir DIR] [--device D] [--epochs N] [--checkpoint-dir DIR]

Streams `bc_*.npz` (see replay/src/bin/bc_export.rs) and trains the policy
head with inverse-action-frequency-weighted cross-entropy; the value head
stays at init (deferred -- see construct/learn/bc.py). Writes one
v1-schema checkpoint per epoch into checkpoint_dir. Run niced next to the
live trainer: `nice -n 10 .venv/bin/python scripts/bc_train.py --device cpu`.
"""
import argparse

from construct._engine import action_table_v1
from construct.learn.bc import BCConfig, BCTrainer


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/bc_v1.toml")
    p.add_argument("--data-dir", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--checkpoint-dir", default=None)
    args = p.parse_args()

    cfg = BCConfig.load(args.config)
    if args.data_dir:
        cfg.data_dir = args.data_dir
    if args.device:
        cfg.run["device"] = args.device
    if args.epochs is not None:
        cfg.train["epochs"] = args.epochs
    if args.checkpoint_dir:
        cfg.run["checkpoint_dir"] = args.checkpoint_dir

    BCTrainer(cfg, action_table_v1()).run()


if __name__ == "__main__":
    main()
