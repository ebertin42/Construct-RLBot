import os
import random
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
            team_size_weights=cfg.env.get("team_size_weights"),
            curriculum_config_path=cfg.curriculum_config_path or None,
        )
        dev = cfg.run.get("device", "cuda")
        self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
        self.net = PolicyValueNet(
            self.engine.obs_size, self.engine.action_count, tuple(cfg.net["hidden"])
        ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo["lr"])
        self.total_steps = 0

        # Opponent-pool ("league") integration. Disabled by default (cfg.league == {}
        # or {"enabled": False}) -> self._assignment stays None -> engine.collect's
        # arena_opponents=None path, byte-identical to pre-league behavior.
        self.num_arenas = cfg.env["num_arenas"]
        self._assignment: list[int] | None = None
        self._league: dict | None = None
        lg = cfg.league
        if lg.get("enabled"):
            from construct.league.registry import Registry
            from construct.league.sampling import choose_opponents
            from construct.league.matches import load_sd
            frac = float(lg.get("opponent_frac", 0.2))
            assert 0 <= frac < 1, f"league.opponent_frac must be in [0, 1), got {frac}"
            slots = int(lg.get("slots", 4))
            # engine.set_opponents rejects more than 8 slots (lib.rs) -- catch it at
            # init time rather than mid-run.
            assert 1 <= slots <= 8, f"league.slots must be in [1, 8], got {slots}"
            self._league = {
                "registry": Registry(path=lg.get("registry", "league/registry.jsonl")),
                "choose": choose_opponents, "load_sd": load_sd,
                "frac": frac,
                # floor at 1: refresh_iters=0 would make `it % refresh` divide by zero.
                "refresh": max(1, int(lg.get("refresh_iters", 200))),
                "slots": slots,
            }

        if _state:
            self.net.load_state_dict(_state["model"])
            if _state["optimizer"] is not None:  # None = deliberate reset (regime swap)
                self.opt.load_state_dict(_state["optimizer"])
            self.total_steps = _state["total_steps"]

    def _refresh_opponents(self, it: int = 0):
        """Pull a fresh opponent pool from the registry and rebuild the arena
        assignment. Called every `league.refresh_iters` iterations from run().

        Reloads the registry from disk on every call: scripts/league_tick.py
        appends to the same jsonl out-of-process (via its own Registry
        instance), and the in-memory snapshot taken at Trainer.__init__ would
        otherwise never see those entries for the life of the run.
        Registry._save writes via a temp-file + os.replace, so a concurrent
        read here always sees either the old or the new file, never a torn
        one. If the reload itself fails (e.g. transient I/O error), keep the
        previous in-memory snapshot and warn, rather than taking down the run.

        Opponent picks are seeded from (run seed, iteration) so assignments
        are reproducible across identical runs but still vary refresh to
        refresh -- part of the repo's fixed-config determinism contract.

        Assignment: the last `round(frac * num_arenas)` arenas (arenas are
        ordered in 1v1/2v2/3v3 blocks -- see Engine's team_size_weights
        allocation) get opponent slots round-robin; the rest stay self-play
        (-1). Taking the tail biases opponent arenas toward the largest-team
        arenas first (3v3 before 2v2 before 1v1) when team sizes are mixed --
        acceptable v1 behavior per the design doc, revisit if team-size-aware
        placement is ever needed.

        A pruned/corrupted checkpoint on disk must never take down a long
        run at a refresh boundary: each pick is loaded independently, bad
        ones are skipped with a warning, and if every pick fails the
        previous assignment (and opponent weights already in the engine)
        is left in place rather than falling back to an empty pool.
        """
        L = self._league
        from construct.league.registry import Registry
        try:
            L["registry"] = Registry(path=L["registry"].path)
        except Exception as e:
            print(f"league: failed to reload registry from disk ({e}), "
                  "keeping previous snapshot", flush=True)
        rng = random.Random(hash((self.cfg.env["seed"], it)))
        picks = L["choose"](L["registry"], k=L["slots"], rng=rng)
        if not picks:
            self._assignment = None
            print("league: no opponents available (pure self-play)", flush=True)
            return
        sds, names = [], []
        for p in picks:
            try:
                sds.append(L["load_sd"](p["ck"]))
                names.append(p["ck"])
            except Exception as e:
                print(f"league: skipping opponent {p['ck']!r} ({e})", flush=True)
        if not sds:
            print("league: all picks failed to load, keeping previous assignment", flush=True)
            return
        self.engine.set_opponents(sds)
        n = self.num_arenas
        n_opp = round(L["frac"] * n)
        a = [-1] * n
        start = n - n_opp
        for i in range(n_opp):
            a[start + i] = i % len(sds)
        self._assignment = a
        print(f"league: opponents {names}", flush=True)

    def collect(self, T: int) -> dict:
        D = self.engine.obs_size
        self.engine.set_weights(
            {k: v.detach().cpu().numpy().astype(np.float32)
             for k, v in self.net.state_dict().items()}
        )
        out = self.engine.collect(T, arena_opponents=self._assignment)
        # Buffer width shrinks from the full agent count when opponent arenas are
        # active (opponent-driven rows aren't learner transitions) -- see
        # Engine.collect's `learner_agents` docstring. All reshapes below, and
        # total_steps accounting in run(), must use N, not engine.num_agents.
        N = out["learner_agents"]

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
            "n_agents": N,
        }

    def run(self, max_iterations: int | None = None):
        it = 0
        p = self.cfg.ppo
        while max_iterations is None or it < max_iterations:
            t0 = time.perf_counter()
            if self._league and it % self._league["refresh"] == 0:
                self._refresh_opponents(it)
            batch = self.collect(p["rollout_steps"])
            stats = ppo_update(
                self.net, self.opt, batch, clip=p["clip"], entropy_coef=p["entropy_coef"],
                value_coef=p["value_coef"], epochs=p["epochs"], minibatch_size=p["minibatch_size"],
            )
            # Steps count LEARNER transitions (batch["n_agents"] == out["learner_agents"]
            # from collect()), not the raw engine agent count -- opponent arenas
            # contribute fewer learner rows than self-play arenas.
            n = p["rollout_steps"] * batch["n_agents"]
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
                # provenance only (resume takes the path from CLI/config, not from here):
                # records which reward regime produced this checkpoint
                "reward_config_path": self.cfg.reward_config_path,
                "curriculum_config_path": self.cfg.curriculum_config_path,
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
