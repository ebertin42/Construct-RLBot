import numpy as np
import torch
from construct.learn.config import TrainConfig
from construct.learn.train import Trainer

def small_cfg(tmp_path):
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=4)
    cfg.ppo.update(rollout_steps=16, minibatch_size=128)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=1)
    return cfg

def test_one_iteration_runs_and_steps_counted(tmp_path):
    t = Trainer(small_cfg(tmp_path))
    t.run(max_iterations=1)
    assert t.total_steps == 16 * 8  # T * num_agents

def test_checkpoint_roundtrip_resumes(tmp_path):
    t = Trainer(small_cfg(tmp_path))
    t.run(max_iterations=1)
    p = f"{tmp_path}/ck.pt"
    t.save_checkpoint(p)
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["schema_version"] == 0 and ck["total_steps"] == 128
    t2 = Trainer.load_checkpoint(p)
    assert t2.total_steps == 128
    before = [x.clone() for x in t2.net.parameters()]
    t2.run(max_iterations=1)
    assert t2.total_steps == 256
    assert any(not torch.equal(a, b) for a, b in zip(before, t2.net.parameters()))
