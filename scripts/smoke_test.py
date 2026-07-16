"""End-to-end smoke: 1k env-steps of real training must run without error
and produce a loadable checkpoint. Run after any engine/learn change.

Default = the v0 MLP path (configs/train_v0.toml, byte-identical to the
historical smoke). `--schema v1` runs the same tiny loop on the entity-
transformer path (configs/train_v1.toml shrunk: pure 1v1, no curriculum,
kickstart stays inactive because the config template ships `teacher`
commented out).
"""
import argparse
import tempfile

from construct.learn.config import TrainConfig
from construct.learn.train import Trainer

p = argparse.ArgumentParser()
p.add_argument("--schema", choices=["v0", "v1"], default="v0")
args = p.parse_args()

with tempfile.TemporaryDirectory() as d:
    if args.schema == "v1":
        cfg = TrainConfig.load("configs/train_v1.toml")
        cfg.env.update(blue=1, orange=1)
        cfg.env.pop("team_size_weights", None)
        cfg.curriculum_config_path = ""
    else:
        cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=8)
    cfg.ppo.update(rollout_steps=64, minibatch_size=512)
    cfg.run.update(checkpoint_dir=d, save_every_iters=2, device="cpu")
    t = Trainer(cfg)
    t.run(max_iterations=2)
    assert t.total_steps == 2 * 64 * 16
    ck = f"{d}/ck_{t.total_steps:012d}.pt"
    t2 = Trainer.load_checkpoint(
        ck, cfg_path="configs/train_v1.toml" if args.schema == "v1" else "configs/train_v0.toml"
    )
    assert t2.total_steps == t.total_steps
    print(f"SMOKE OK ({args.schema}):", t.total_steps, "steps")
