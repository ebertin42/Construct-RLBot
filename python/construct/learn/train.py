import os
import random
import time
import tomllib

import numpy as np
import torch

from construct._engine import Engine, action_table_v1, schema_dict
from construct.learn.config import TrainConfig
from construct.learn.gae import compute_gae
from construct.learn.kickstart import (
    TEACHER_OBS_SIZE,
    KickstartSchedule,
    KickstartTeacher,
    kickstart_losses,
)
from construct.learn.kl_prior import KLPrior, kl_student_prior
from construct.learn.model import PolicyValueNet
from construct.learn.model_v1 import EntityPolicyNet
from construct.learn.ppo import ppo_update


def compose_extra_loss(net, batch, *, kickstart, lambda_k, lambda_v,
                       prior_logits, lambda_p):
    """Build the per-minibatch extra-loss hook for ppo_update, composing the
    kickstart distillation term (KL(teacher‖student)+value MSE, annealed) and
    the BC-prior anchor (KL(student‖prior), constant lambda_p). Returns None
    when neither is active — ppo_update's no-hook path stays byte-identical
    (v0 runs and prior-less v1 runs are untouched, per T7's original
    byte-identity guarantee, now extended to cover the prior term too).

    kickstart: None or (t_logits, t_values) precomputed full-batch teacher
    outputs; prior_logits: None or precomputed full-batch prior logits."""
    kick_on = kickstart is not None and (lambda_k > 0.0 or lambda_v > 0.0)
    prior_on = prior_logits is not None and lambda_p > 0.0
    if not kick_on and not prior_on:
        return None

    def fn(idx):
        # One student forward serves both terms (full [B,92] distribution).
        # Re-runs the student net's forward directly (rather than threading
        # logits out through evaluate()/Categorical) to get the full [B,92]
        # distribution both KL terms need -- evaluate() only returns the
        # sampled action's logprob. This costs a second forward pass per
        # minibatch versus changing model.py/model_v1.py's evaluate()
        # signature to also return logits; traded deliberately for keeping
        # those files (and ppo.py's default no-hook path) untouched.
        # batch["obs"] is the entity dict (keys match EntityPolicyNet.
        # forward's params) -- both kickstart and the prior imply v1.
        s_logits, s_value = net(**{k: v[idx] for k, v in batch["obs"].items()})
        # Roots `loss` in the autograd graph unconditionally (via a *0.0 no-op
        # so it never contributes value) so `loss.requires_grad` holds even in
        # a hypothetical future branch that adds neither kl nor kl_p; not
        # load-bearing today since kick_on/prior_on guarantee at least one of
        # kl or kl_p is added below, and either already carries the grad path.
        loss = s_logits.sum() * 0.0
        info = {}
        if kick_on:
            t_logits, t_values = kickstart
            kl, v_mse = kickstart_losses(s_logits, s_value.squeeze(-1),
                                         t_logits[idx], t_values[idx])
            loss = loss + lambda_k * kl + lambda_v * v_mse
            info["kick_kl"] = kl.item()
        if prior_on:
            kl_p = kl_student_prior(s_logits, prior_logits[idx])
            loss = loss + lambda_p * kl_p
            info["kl_pri"] = kl_p.item()
        return loss, info

    return fn


def check_win_prob_gamma(reward_cfg, ppo_cfg):
    """Refuse to train if the shaping gamma differs from the training gamma.

    Potential-based shaping preserves the optimal policy only when the gamma
    inside `gamma * PHI(s') - PHI(s)` is the SAME gamma the returns are
    discounted with. A mismatch silently optimises a different objective and
    produces a run that looks entirely healthy, so this fails loud at startup.
    """
    if float(reward_cfg.get("win_prob_weight", 0.0)) == 0.0:
        return
    shaping = float(reward_cfg.get("win_prob_gamma", 0.0))
    training = float(ppo_cfg["gamma"])
    if abs(shaping - training) > 1e-9:
        raise ValueError(
            f"win_prob_gamma ({shaping}) != ppo gamma ({training}). "
            f"Potential-based shaping only preserves the optimal policy when "
            f"these are identical; a mismatch trains a different objective and "
            f"the run will look normal. Fix the reward config."
        )


def check_match_mode_required(reward_cfg, curriculum_cfg):
    """Refuse win-probability shaping without a match layer.

    PHI is a function of score and remaining clock. With match_mode off there
    is no score and no clock, PHI is pinned at 0 forever, and the shaping
    term is identically zero -- so the run trains on nothing at all while every
    log line looks normal. Fail at startup instead.
    """
    if float(reward_cfg.get("win_prob_weight", 0.0)) == 0.0:
        return
    if not bool(curriculum_cfg.get("match_mode", False)):
        raise ValueError(
            "win_prob_weight is set but the curriculum has match_mode = false. "
            "Win-probability shaping needs a score and a match clock; without "
            "them PHI is constant, the shaping term is exactly zero, and the "
            "run trains on nothing while appearing healthy. "
            "Use configs/curriculum_v3_match.toml."
        )


class Trainer:
    def __init__(self, cfg: TrainConfig, _state: dict | None = None):
        self.cfg = cfg
        self.schema = schema_dict(cfg.schema_path)
        self.is_v1 = self.schema["version"] == 1
        # Checkpoint/config schema guard: a checkpoint only makes sense with
        # the net family it was trained as. In particular a v0 MLP state dict
        # can NOT seed a v1 EntityPolicyNet (different obs, different net) --
        # the sanctioned v0->v1 bridge is kickstart distillation, where the
        # v0 checkpoint is the frozen TEACHER ([kickstart].teacher in the
        # config / resume_train.py --kickstart-teacher), not the resume state.
        if _state is not None:
            ck_ver = int(_state.get("schema_version", 0))
            if ck_ver != self.schema["version"]:
                raise ValueError(
                    f"checkpoint schema_version={ck_ver} does not match config "
                    f"schema version={self.schema['version']} ({cfg.schema_path}). "
                    "A v0 MLP checkpoint cannot seed a v1 entity net directly; "
                    "start a fresh v1 run and pass the v0 checkpoint as the "
                    "kickstart teacher ([kickstart].teacher / "
                    "resume_train.py --kickstart-teacher) instead."
                )
        # Objective guards (T5): load the reward and curriculum configs as
        # dicts (cfg only carries their paths -- the engine consumes the
        # paths directly, nothing on the Python side parsed them before this)
        # and fail loud on combinations that would silently train the wrong
        # objective. Both checks are inert (no-op) whenever win-prob shaping
        # is off, i.e. every historical reward config -- see each function's
        # docstring for the failure mode it closes. Placed before any
        # engine/net construction so a bad config never wastes GPU time.
        with open(cfg.reward_config_path, "rb") as f:
            reward_cfg = tomllib.load(f)
        curriculum_cfg = {}
        if cfg.curriculum_config_path:
            with open(cfg.curriculum_config_path, "rb") as f:
                curriculum_cfg = tomllib.load(f)
        check_win_prob_gamma(reward_cfg, cfg.ppo)
        check_match_mode_required(reward_cfg, curriculum_cfg)

        # Kickstart distillation (T7): only meaningful for a v1-schema run
        # with a [kickstart] block that names a teacher (the config template
        # ships with `teacher` commented out, so a bare `steps=` block stays
        # inactive). `emit_v0_obs` asks the engine to also record the legacy
        # 94-float obs per collected step (T6), which the frozen v0 MLP
        # teacher needs. For v0-schema runs this is always False, so the
        # Engine constructor call, collect() output, and run() loop below
        # are byte-identical to pre-T7 behavior.
        kickstart_active = bool(cfg.kickstart.get("teacher")) and self.is_v1
        engine_kwargs = dict(
            num_arenas=cfg.env["num_arenas"], blue=cfg.env["blue"], orange=cfg.env["orange"],
            schema_path=cfg.schema_path, reward_config_path=cfg.reward_config_path,
            seed=cfg.env["seed"],
            team_size_weights=cfg.env.get("team_size_weights"),
            curriculum_config_path=cfg.curriculum_config_path or None,
        )
        if self.is_v1:
            # candle rebuilds attention from the state dict; the head count is
            # not recoverable from tensor shapes, so it rides along explicitly.
            engine_kwargs["net_heads"] = int(cfg.net["heads"])
        if kickstart_active:
            engine_kwargs["emit_v0_obs"] = True
        self.engine = Engine(**engine_kwargs)
        dev = cfg.run.get("device", "cuda")
        self.device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")
        if self.is_v1:
            # Net dims come from [net] (T1 gate: 128/2/4/512 launch config);
            # the 92-row v1.1 action table is compiled into the engine and is
            # carried by the net as a non-trainable buffer (set_weights hands
            # it back to candle with the rest of the state dict).
            self.net = EntityPolicyNet(
                d_model=int(cfg.net["d_model"]), layers=int(cfg.net["layers"]),
                heads=int(cfg.net["heads"]), ff=int(cfg.net["ff"]),
                action_table=action_table_v1(),
            ).to(self.device)
        else:
            self.net = PolicyValueNet(
                self.engine.obs_size, self.engine.action_count, tuple(cfg.net["hidden"])
            ).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.ppo["lr"])
        self.total_steps = 0

        self.kickstart: dict | None = None
        if kickstart_active:
            self.kickstart = {
                "teacher": KickstartTeacher(cfg.kickstart["teacher"], device=str(self.device)),
                "schedule": KickstartSchedule(
                    lambda_k0=float(cfg.kickstart.get("lambda_k", 1.0)),
                    kickstart_steps=int(cfg.kickstart.get("steps", 500_000_000)),
                    lambda_v=float(cfg.kickstart.get("lambda_v", 0.5)),
                ),
            }

        # BC-prior anchor (K3): only meaningful for a v1-schema run with a
        # [kl_prior] block that names a checkpoint (the config template ships
        # with `ck` absent, so cfg.kl_prior stays {} and this block is
        # inactive). Gate mirrors kickstart's: cfg.kl_prior.get("ck") truthy
        # AND self.is_v1. For v0 runs and prior-less v1 runs self.kl_prior
        # stays None, so run()'s prior_logits stays None and
        # compose_extra_loss's prior_on branch is dead -- byte-identical to
        # pre-K3 behavior.
        self.kl_prior: dict | None = None
        if cfg.kl_prior.get("ck") and self.is_v1:
            self.kl_prior = {
                # expect_net closes the Global Constraint "prior dims must equal
                # student dims": cfg.net IS the student's dims (restored from the
                # resume checkpoint by resume_train.py before Trainer init).
                "net": KLPrior(cfg.kl_prior["ck"], device=str(self.device),
                               expect_net=cfg.net),
                "lambda": float(cfg.kl_prior.get("lambda", 0.05)),
            }
            print(f"kl_prior: anchored to {cfg.kl_prior['ck']} "
                  f"lambda_p={self.kl_prior['lambda']}", flush=True)

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
        is left in place rather than falling back to an empty pool. The same
        holds if every pick loads fine but `engine.set_opponents` itself
        raises (e.g. an opponent state dict that's incompatible with the
        engine's current obs mode) -- that's also caught and logged, leaving
        the previous assignment and in-engine opponent weights untouched.
        """
        L = self._league
        from construct.league.registry import Registry
        try:
            L["registry"] = Registry(path=L["registry"].path)
        except Exception as e:
            print(f"league: failed to reload registry from disk ({e}), "
                  "keeping previous snapshot", flush=True)
        rng = random.Random(hash((self.cfg.env["seed"], it)))
        # Belt: only ever consider opponents tagged with this trainer's own
        # schema_version (self.schema["version"], derived from cfg.schema_path)
        # -- v0 and v1 policies can never play each other (different obs). The
        # try/except below (skipping bad loads, degrading set_opponents
        # failures to "keep previous assignment") is the suspenders: it still
        # catches anything that slips past this filter (e.g. a hand-edited
        # registry, or a future schema_version this filter doesn't know about).
        picks = L["choose"](L["registry"], k=L["slots"], rng=rng,
                             schema_version=self.schema["version"])
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
        try:
            self.engine.set_opponents(sds)
        except Exception as e:
            # A misconfigured/incompatible opponent pool (e.g. a future
            # league-on-v1 obs/schema mismatch) must degrade to "keep
            # whatever assignment and engine-side opponent weights were
            # already in place" rather than take down the whole run.
            print(f"league: set_opponents failed ({e}), keeping previous assignment", flush=True)
            return
        n = self.num_arenas
        n_opp = round(L["frac"] * n)
        a = [-1] * n
        start = n - n_opp
        for i in range(n_opp):
            a[start + i] = i % len(sds)
        self._assignment = a
        print(f"league: opponents {names}", flush=True)

    def collect(self, T: int) -> dict:
        # state_dict() includes non-trainable buffers (v1's `action_table`)
        # by design -- the engine's candle net consumes them too.
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
        # Obs plumbing dispatch. v0: the flat (T,N,94) tensor, exactly as
        # always. v1: the engine returns entity tensors instead of a flat obs
        # (see lib.rs collect) -- flatten each over (T,N) and pack them as a
        # dict whose KEYS MATCH EntityPolicyNet.evaluate/forward's keyword
        # parameters (ents/mask/query/prev); ppo_update indexes each tensor
        # by the same minibatch permutation and splats the dict into
        # net.evaluate. Everything GAE needs (rewards/values/final_values/
        # flags) is obs-layout-independent, and the engine computes value
        # bootstraps (final_values for truncations, last_values) in-engine
        # for both modes, so nothing else changes shape.
        if self.is_v1:
            obs = {
                "ents": torch.as_tensor(out["ents"].reshape(T * N, *out["ents"].shape[2:]), device=dev),
                "mask": torch.as_tensor(out["mask"].reshape(T * N, out["mask"].shape[2]), device=dev),
                "query": torch.as_tensor(out["query"].reshape(T * N, out["query"].shape[2]), device=dev),
                "prev": torch.as_tensor(out["prev"].reshape(T * N, out["prev"].shape[2]), device=dev),
            }
        else:
            obs = torch.as_tensor(out["obs"].reshape(T * N, self.engine.obs_size), device=dev)
        done_frac = (out["terminated"] | out["truncated"]).sum()
        result = {
            "obs": obs,
            "actions": torch.as_tensor(out["actions"].reshape(-1), device=dev),
            "logprobs": torch.as_tensor(out["logprobs"].reshape(-1), device=dev),
            "advantages": torch.as_tensor(adv, device=dev).reshape(-1),
            "returns": torch.as_tensor(ret, device=dev).reshape(-1),
            "values": torch.as_tensor(out["values"].reshape(-1), device=dev),
            "ep_reward_mean": float(out["rewards"].sum() / max(1, done_frac)),
            "n_agents": N,
        }
        if "obs_v0" in out:
            result["obs_v0"] = torch.as_tensor(
                out["obs_v0"].reshape(T * N, TEACHER_OBS_SIZE), device=dev
            )
        return result

    def run(self, max_iterations: int | None = None):
        it = 0
        p = self.cfg.ppo
        while max_iterations is None or it < max_iterations:
            t0 = time.perf_counter()
            if self._league and it % self._league["refresh"] == 0:
                self._refresh_opponents(it)
            batch = self.collect(p["rollout_steps"])

            # --- extra-loss hook: kickstart distillation (T7) + BC-prior anchor (K3) ---
            # Both terms funnel through compose_extra_loss (module-level, see its
            # docstring). Kickstart is guarded on self.kickstart (None for every v0
            # run, and for any v1 run without a [kickstart] config) AND on "obs_v0"
            # actually being in the batch -- collect() only puts it there when the
            # engine was built with emit_v0_obs=True, i.e. exactly the
            # kickstart_active case from __init__. The prior is guarded on
            # self.kl_prior (None for every v0 run, and for any v1 run without a
            # [kl_prior] config). compose_extra_loss returns None when neither is
            # active, which makes ppo_update's hook parameter a complete no-op: v0
            # runs and prior-less v1 runs are byte-identical to pre-K3 behavior.
            kickstart_outs = None
            lambda_k = lambda_v = 0.0
            if self.kickstart is not None and "obs_v0" in batch:
                lambda_k, lambda_v = self.kickstart["schedule"].coef(self.total_steps)
                if lambda_k > 0.0 or lambda_v > 0.0:
                    with torch.no_grad():
                        kickstart_outs = self.kickstart["teacher"].logits_values(batch["obs_v0"])
            prior_logits, lambda_p = None, 0.0
            if self.kl_prior is not None:
                lambda_p = self.kl_prior["lambda"]
                with torch.no_grad():
                    prior_logits = self.kl_prior["net"].logits(batch["obs"])
            extra_loss_fn = compose_extra_loss(
                self.net, batch,
                kickstart=kickstart_outs, lambda_k=lambda_k, lambda_v=lambda_v,
                prior_logits=prior_logits, lambda_p=lambda_p,
            )

            stats = ppo_update(
                self.net, self.opt, batch, clip=p["clip"], entropy_coef=p["entropy_coef"],
                value_coef=p["value_coef"], epochs=p["epochs"], minibatch_size=p["minibatch_size"],
                extra_loss_fn=extra_loss_fn,
            )
            # Steps count LEARNER transitions (batch["n_agents"] == out["learner_agents"]
            # from collect()), not the raw engine agent count -- opponent arenas
            # contribute fewer learner rows than self-play arenas.
            n = p["rollout_steps"] * batch["n_agents"]
            self.total_steps += n
            it += 1
            if it % self.cfg.run.get("log_every_iters", 1) == 0:
                sps = n / (time.perf_counter() - t0)
                msg = (
                    f"iter {it} steps {self.total_steps:,} sps {sps:,.0f} "
                    f"ep_rew {batch['ep_reward_mean']:.3f} "
                    f"pi_loss {stats['policy_loss']:.4f} v_loss {stats['value_loss']:.4f} "
                    f"ent {stats['entropy']:.3f} clip {stats['clip_frac']:.3f}"
                )
                if kickstart_outs is not None:
                    msg += f" kick_kl {stats.get('kick_kl', 0.0):.4f} lambda_k {lambda_k:.3f}"
                if self.kl_prior is not None:
                    msg += f" kl_pri {stats.get('kl_pri', 0.0):.4f} lambda_p {lambda_p:.3f}"
                print(msg, flush=True)
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
