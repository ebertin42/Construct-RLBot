"""End-to-end smoke: 1k env-steps of real training must run without error
and produce a loadable checkpoint. Run after any engine/learn change."""
import tempfile

from construct.learn.config import TrainConfig
from construct.learn.train import Trainer

with tempfile.TemporaryDirectory() as d:
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=8)
    cfg.ppo.update(rollout_steps=64, minibatch_size=512)
    cfg.run.update(checkpoint_dir=d, save_every_iters=2, device="cpu")
    t = Trainer(cfg)
    t.run(max_iterations=2)
    assert t.total_steps == 2 * 64 * 16
    print("SMOKE OK:", t.total_steps, "steps")
