import os
import time

import numpy as np
import torch

from construct._engine import Engine, schema_dict
from construct.learn.config import TrainConfig
from construct.learn.gae import compute_gae
from construct.learn.model import PolicyValueNet
from construct.learn.ppo import ppo_update


class Trainer:
    def __init__(self, cfg: TrainConfig, _state: dict | None = None):
        self.cfg = cfg
        self.schema = schema_dict(cfg.schema_path)
        self.engine = Engine(
            num_arenas=cfg.env["num_arenas"], blue=cfg.env["blue"], orange=cfg.env["orange"],
            schema_path=cfg.schema_path, reward_config_path=cfg.reward_config_path,
            seed=cfg.env["seed"],
        )
        dev = cfg.run.get("device", "cuda")
        self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
        self.net = PolicyValueNet(
            self.engine.obs_size, self.engine.action_count, tuple(cfg.net["hidden"])
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo["lr"])
        self.total_steps = 0
        if _state:
            self.net.load_state_dict(_state["model"])
            self.opt.load_state_dict(_state["optimizer"])
            self.total_steps = _state["total_steps"]

    def collect(self, T: int) -> dict:
        N, D = self.engine.num_agents, self.engine.obs_size
        self.engine.set_weights(
            {k: v.detach().cpu().numpy().astype(np.float32)
             for k, v in self.net.state_dict().items()}
        )
        out = self.engine.collect(T)

        values_ext = np.concatenate(
            [out["values"], out["last_values"][None, :]], axis=0
        )
        adv, ret = compute_gae(
            out["rewards"], values_ext, out["final_values"],
            out["terminated"], out["truncated"],
            self.cfg.ppo["gamma"], self.cfg.ppo["lam"],
        )
        dev = self.device
        flat_obs = torch.as_tensor(out["obs"].reshape(T * N, D), device=dev)
        done_frac = (out["terminated"] | out["truncated"]).sum()
        return {
            "obs": flat_obs,
            "actions": torch.as_tensor(out["actions"].reshape(-1), device=dev),
            "logprobs": torch.as_tensor(out["logprobs"].reshape(-1), device=dev),
            "advantages": torch.as_tensor(adv, device=dev).reshape(-1),
            "returns": torch.as_tensor(ret, device=dev).reshape(-1),
            "values": torch.as_tensor(out["values"].reshape(-1), device=dev),
            "ep_reward_mean": float(out["rewards"].sum() / max(1, done_frac)),
        }

    def run(self, max_iterations: int | None = None):
        it = 0
        p = self.cfg.ppo
        while max_iterations is None or it < max_iterations:
            t0 = time.perf_counter()
            batch = self.collect(p["rollout_steps"])
            stats = ppo_update(
                self.net, self.opt, batch, clip=p["clip"], entropy_coef=p["entropy_coef"],
                value_coef=p["value_coef"], epochs=p["epochs"], minibatch_size=p["minibatch_size"],
            )
            n = p["rollout_steps"] * self.engine.num_agents
            self.total_steps += n
            it += 1
            if it % self.cfg.run.get("log_every_iters", 1) == 0:
                sps = n / (time.perf_counter() - t0)
                print(
                    f"iter {it} steps {self.total_steps:,} sps {sps:,.0f} "
                    f"ep_rew {batch['ep_reward_mean']:.3f} "
                    f"pi_loss {stats['policy_loss']:.4f} v_loss {stats['value_loss']:.4f} "
                    f"ent {stats['entropy']:.3f} clip {stats['clip_frac']:.3f}",
                    flush=True,
                )
            if it % self.cfg.run.get("save_every_iters", 20) == 0:
                os.makedirs(self.cfg.run["checkpoint_dir"], exist_ok=True)
                self.save_checkpoint(
                    os.path.join(self.cfg.run["checkpoint_dir"], f"ck_{self.total_steps:012d}.pt")
                )

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "model": self.net.state_dict(),
                "optimizer": self.opt.state_dict(),
                "total_steps": self.total_steps,
                "schema_version": self.schema["version"],
                "config": {"net": self.cfg.net, "ppo": self.cfg.ppo, "env": self.cfg.env},
            },
            path,
        )

    @classmethod
    def load_checkpoint(cls, path: str, cfg_path: str = "configs/train_v0.toml") -> "Trainer":
        state = torch.load(path, map_location="cpu", weights_only=False)
        cfg = TrainConfig.load(cfg_path)
        # Restore the exact env/net/ppo config the checkpoint was trained under
        # (not just net) — otherwise resuming with the on-disk default config
        # silently changes num_arenas/rollout_steps and desyncs total_steps.
        cfg.net = state["config"]["net"]
        cfg.ppo = state["config"]["ppo"]
        cfg.env = state["config"]["env"]
        return cls(cfg, _state=state)


if __name__ == "__main__":
    import sys

    cfg = TrainConfig.load(sys.argv[1] if len(sys.argv) > 1 else "configs/train_v0.toml")
    Trainer(cfg).run()
