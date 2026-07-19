#!/usr/bin/env python3
"""PPO learning-health diagnostic — READ-ONLY, offline, no weight updates.

WHY THIS EXISTS (docs/training-journal.md, 2026-07-19 ~17:40): head-to-head
tournaments established that PPO self-play has NEVER demonstrably improved this
policy. The kickstart-era checkpoints (100M/200M/320M/440M) sit flat within
43-54% of each other and beat every post-500M checkpoint by 80-90%, across four
different reward designs. The policy was carried to a plateau by KL
distillation from a v0 teacher and has decayed since the anchor annealed off at
500M. Before spending more GPU on reward/regime variants we need to know
whether the RL loop can improve ANYTHING.

Already ruled out by inspection, do not re-litigate here:
  * advantages ARE normalized (learn/ppo.py:42)
  * loss assembly is standard PPO (clip + value MSE + entropy bonus)
  * engine weights ARE synced before every collect (Trainer.collect calls
    set_weights(net.state_dict()) immediately before engine.collect)

So this looks elsewhere. It loads a checkpoint the way Trainer does, builds an
Engine exactly as Trainer.collect does (same schema/reward/curriculum/team-size
config, so the rollout is representative of real training data), runs ONE
collect, and reports five things the training loop never logs:

  1. VALUE-HEAD EXPLAINED VARIANCE — the single most diagnostic PPO number,
     logged nowhere. ev = 1 - Var(returns - values) / Var(returns), over GAE
     returns computed with the exact math train.py uses (construct.learn.gae).
     ev < 0.1 means the advantages the policy gradient rides on are essentially
     noise, and NO reward design can fix that.
  2. IMPORTANCE-RATIO SANITY — the engine samples actions in Rust/candle (f32,
     CPU) while the learner recomputes logprobs in PyTorch. At epoch 0 the
     ratio exp(pi_torch - pi_engine) MUST be ~1.0. A systematic offset means
     every PPO update has been applying wrong importance weights.
  3. ADVANTAGE + RETURN DISTRIBUTION — spread, zero-fraction, and reward
     sparsity. A near-all-zero advantage vector explains a random-walking
     policy.
  4. ACTION DISTRIBUTION — entropy vs ln(action_count), concentration, dead
     actions.
  5. VALUE PREDICTION SCALE — predicted values vs actual returns; a value head
     miscalibrated to a changed reward scale shows up here.

Nothing here trains, saves, or mutates anything on disk. The engine and the
torch forward both run at whatever nice level you launch with; a collect is
CPU-bound (RocketSim + candle), so `nice -n 15` it when the box is busy.

CLI:
    diagnose_ppo.py CK [--config configs/train_v1.toml] [--steps 256]
                       [--arenas 32] [--compare CK2] [--own-reward]
                       [--device cpu|cuda] [--seed N]

--compare runs the identical protocol on a second checkpoint and prints a diff
table. That contrast (a strong early checkpoint vs a degraded late one) is the
point of the tool: absolute PPO numbers are hard to judge, but the DIFFERENCE
between a policy we know is good and one we know is bad is not.

--own-reward uses each checkpoint's OWN recorded reward/curriculum config
(saved as provenance in the checkpoint) instead of the one in --config. Off by
default: the shared-config run is the apples-to-apples comparison. Turn it on
to ask "how well does each value head explain the regime it was actually
trained under" — for a checkpoint trained on a different reward, the default
run understates its value head.
"""
import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]

# --------------------------------------------------------------------------
# thresholds — every verdict line prints the threshold it used, so these are
# the only place a band is defined.
# --------------------------------------------------------------------------
EV_PASS = 0.80          # healthy value head
EV_WEAK = 0.30          # below this the advantages are mostly noise
EV_NOISE = 0.10         # below this they are essentially pure noise

RATIO_PASS_MEAN = 1e-4  # |mean signed logprob delta| — systematic offset
RATIO_PASS_FRAC = 1e-3  # fraction of ratios outside [0.9, 1.1]
RATIO_WEAK_MEAN = 1e-2
RATIO_WEAK_FRAC = 1e-2

ADV_ZERO_PASS = 0.05    # fraction of exactly-zero advantages
ADV_ZERO_WEAK = 0.50
ADV_STD_MIN = 1e-6      # a degenerate advantage vector

ENT_HIGH_FAIL = 0.95    # entropy / ln(A): above this the policy is ~uniform
ENT_HIGH_WEAK = 0.85
ENT_LOW_WEAK = 0.15
ENT_LOW_FAIL = 0.05     # collapsed onto a handful of actions

# state-dependence: I(S;A) / ln(A), i.e. how much of the available action
# entropy the policy actually spends on telling states apart. Judgment-call
# bands, not literature: 0 is provably a state-blind policy (it samples the
# same distribution everywhere and PPO cannot be improving it), and anything
# under a quarter of the action entropy means most of what the policy emits is
# unconditioned noise.
MI_PASS = 0.25
MI_FAIL = 0.05

VSCALE_PASS_STD = (0.5, 2.0)   # std(values) / std(returns)
VSCALE_WEAK_STD = (0.2, 5.0)
VSCALE_PASS_BIAS = 0.5         # |mean(values) - mean(returns)| / std(returns)
VSCALE_WEAK_BIAS = 1.5

RATIO_BAND = (0.9, 1.1)


# ==========================================================================
# pure functions — no engine, no torch, no I/O. Tested in
# tests/python/test_diagnose_ppo.py against synthetic arrays.
# ==========================================================================

def explained_variance(returns, values) -> float:
    """ev = 1 - Var(returns - values) / Var(returns).

    1.0 = the value head explains the returns perfectly; 0.0 = it does no
    better than predicting the mean return; negative = worse than the mean.
    Returns nan when the returns have no variance to explain (a constant
    return vector makes the ratio 0/0 — that is a degenerate rollout, not a
    healthy value head, and the verdict treats nan as FAIL).
    """
    returns = np.asarray(returns, dtype=np.float64).ravel()
    values = np.asarray(values, dtype=np.float64).ravel()
    assert returns.shape == values.shape, (
        f"shape mismatch: returns {returns.shape} vs values {values.shape}")
    var_r = returns.var()
    if var_r <= 0.0:
        return float("nan")
    return float(1.0 - (returns - values).var() / var_r)


def ratio_delta_stats(engine_logprobs, torch_logprobs, band=RATIO_BAND) -> dict:
    """Compare the logprobs the engine (candle, f32, CPU) returned for the
    sampled actions against the ones the learner's torch net recomputes for the
    same (obs, action) pairs.

    delta = torch - engine, so ratio = exp(delta) is exactly the importance
    weight PPO's first epoch applies. A known ~1e-7 candle/torch gemm
    difference is harmless; what matters is a SYSTEMATIC offset (mean_delta far
    from 0) or a heavy tail (frac_outside).
    """
    e = np.asarray(engine_logprobs, dtype=np.float64).ravel()
    t = np.asarray(torch_logprobs, dtype=np.float64).ravel()
    assert e.shape == t.shape, f"shape mismatch: {e.shape} vs {t.shape}"
    d = t - e
    ratio = np.exp(d)
    lo, hi = band
    return {
        "n": int(d.size),
        "mean_delta": float(d.mean()),
        "std_delta": float(d.std()),
        "max_abs_delta": float(np.abs(d).max()),
        "ratio_mean": float(ratio.mean()),
        "ratio_p1": float(np.percentile(ratio, 1)),
        "ratio_p99": float(np.percentile(ratio, 99)),
        "frac_outside": float(((ratio < lo) | (ratio > hi)).mean()),
    }


def advantage_stats(adv, ret) -> dict:
    """Spread and degeneracy of the advantage/return vectors GAE produced."""
    a = np.asarray(adv, dtype=np.float64).ravel()
    r = np.asarray(ret, dtype=np.float64).ravel()
    pcts = [1, 5, 25, 50, 75, 95, 99]
    return {
        "n": int(a.size),
        "adv_mean": float(a.mean()),
        "adv_std": float(a.std()),
        "adv_pct": {p: float(np.percentile(a, p)) for p in pcts},
        "adv_frac_zero": float((a == 0.0).mean()),
        "adv_max_abs": float(np.abs(a).max()),
        "ret_mean": float(r.mean()),
        "ret_std": float(r.std()),
        "ret_pct": {p: float(np.percentile(r, p)) for p in pcts},
    }


def reward_sparsity(rewards, event_scale=1.0) -> dict:
    """How often the reward signal is nonzero at all, and how often it carries
    a goal-scale event (|r| > event_scale). A dense shaping term keeps
    frac_nonzero near 1; a goals-only regime drops it toward 0.
    """
    r = np.asarray(rewards, dtype=np.float64).ravel()
    a = np.abs(r)
    return {
        "n": int(r.size),
        "frac_nonzero": float((a > 0.0).mean()),
        "frac_event": float((a > event_scale).mean()),
        "mean_abs": float(a.mean()),
        "max_abs": float(a.max()) if r.size else 0.0,
        "mean": float(r.mean()),
    }


def action_stats(actions, n_actions, top_k=5, cover=0.90,
                 per_state_entropy=None) -> dict:
    """Empirical action distribution over one collect.

    entropy is the MARGINAL entropy in nats (over all sampled actions, all
    states pooled); entropy_ratio compares it to ln(n_actions), the
    uniform-policy maximum. n_cover is how many distinct actions it takes to
    account for `cover` of the sampled mass; n_never counts actions the policy
    never sampled at all.

    per_state_entropy (optional): the mean of H(pi(.|s)) over the same rollout,
    i.e. what the training log's `ent` column reports. Given both, the
    difference is the mutual information between state and action,

        I(S;A) = H(A) - E_s[H(A|s)]  (nats),

    which is the sharpest available answer to "is this a POLICY or a dice
    roll?". A policy that ignores its input has I(S;A) = 0: it samples the same
    distribution everywhere, so pooling states costs no entropy. Per-state
    entropy alone cannot distinguish "high entropy because the policy hedges
    sensibly in each state" from "high entropy because it is barely
    state-dependent" — the marginal is the missing half of that read, and it is
    free once both numbers exist.
    """
    a = np.asarray(actions, dtype=np.int64).ravel()
    counts = np.bincount(a, minlength=n_actions).astype(np.float64)
    assert counts.size == n_actions, (
        f"action id >= n_actions: max {a.max()} with n_actions={n_actions}")
    p = counts / max(counts.sum(), 1.0)
    nz = p[p > 0.0]
    entropy = float(-(nz * np.log(nz)).sum())
    max_entropy = float(math.log(n_actions))
    order = np.sort(p)[::-1]
    n_cover = int(np.searchsorted(np.cumsum(order), cover) + 1)
    out = {
        "n": int(a.size),
        "n_actions": int(n_actions),
        "entropy": entropy,
        "max_entropy": max_entropy,
        "entropy_ratio": float(entropy / max_entropy) if max_entropy > 0 else float("nan"),
        "top_k": top_k,
        "top_k_share": float(order[:top_k].sum()),
        "top1_share": float(order[0]),
        "n_cover": n_cover,
        "cover": cover,
        "n_never": int((counts == 0).sum()),
        "per_state_entropy": None,
        "per_state_ratio": None,
        "state_dependence": None,
        "mi_ratio": None,
    }
    if per_state_entropy is not None:
        h_s = float(per_state_entropy)
        out["per_state_entropy"] = h_s
        out["per_state_ratio"] = float(h_s / max_entropy) if max_entropy > 0 else float("nan")
        out["state_dependence"] = entropy - h_s
        out["mi_ratio"] = (float((entropy - h_s) / max_entropy)
                           if max_entropy > 0 else float("nan"))
    return out


def value_scale_stats(values, returns) -> dict:
    """Is the value head even predicting on the same SCALE as the returns?
    A head resumed across a reward-regime change (e.g. v3's +10/-8 goal to
    v4's effective +20/-20) can be badly miscalibrated while still correlating,
    which shows up here rather than in ev alone.
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    r = np.asarray(returns, dtype=np.float64).ravel()
    assert v.shape == r.shape, f"shape mismatch: {v.shape} vs {r.shape}"
    ret_std = r.std()
    std_ratio = float(v.std() / ret_std) if ret_std > 0 else float("nan")
    bias = float((v.mean() - r.mean()) / ret_std) if ret_std > 0 else float("nan")
    corr = float("nan")
    if v.std() > 0 and ret_std > 0:
        corr = float(np.corrcoef(v, r)[0, 1])
    return {
        "val_mean": float(v.mean()),
        "val_std": float(v.std()),
        "ret_mean": float(r.mean()),
        "ret_std": float(ret_std),
        "std_ratio": std_ratio,
        "bias_in_ret_std": bias,
        "corr": corr,
        "rmse": float(np.sqrt(((v - r) ** 2).mean())),
    }


# --------------------------------------------------------------------------
# verdicts — each returns (label, threshold_text). Labels are exactly one of
# PASS / WEAK / FAIL so a caller can grep the output.
# --------------------------------------------------------------------------

def verdict_ev(ev) -> tuple:
    thr = f"PASS ev>={EV_PASS}, WEAK ev>={EV_WEAK}, FAIL below (ev<{EV_NOISE} = pure noise)"
    if ev is None or (isinstance(ev, float) and math.isnan(ev)):
        return "FAIL", thr
    if ev >= EV_PASS:
        return "PASS", thr
    if ev >= EV_WEAK:
        return "WEAK", thr
    return "FAIL", thr


def verdict_ratio(s: dict) -> tuple:
    thr = (f"PASS |mean delta|<{RATIO_PASS_MEAN:g} and frac outside "
           f"[{RATIO_BAND[0]},{RATIO_BAND[1]}]<{RATIO_PASS_FRAC:g}, "
           f"WEAK <{RATIO_WEAK_MEAN:g}/{RATIO_WEAK_FRAC:g}, FAIL beyond")
    m, f = abs(s["mean_delta"]), s["frac_outside"]
    if not math.isfinite(m) or not math.isfinite(f):
        return "FAIL", thr
    if m < RATIO_PASS_MEAN and f < RATIO_PASS_FRAC:
        return "PASS", thr
    if m < RATIO_WEAK_MEAN and f < RATIO_WEAK_FRAC:
        return "WEAK", thr
    return "FAIL", thr


def verdict_adv(s: dict) -> tuple:
    thr = (f"PASS frac_zero<{ADV_ZERO_PASS:g} and std>{ADV_STD_MIN:g}, "
           f"WEAK frac_zero<{ADV_ZERO_WEAK:g}, FAIL beyond")
    if s["adv_std"] <= ADV_STD_MIN:
        return "FAIL", thr
    if s["adv_frac_zero"] < ADV_ZERO_PASS:
        return "PASS", thr
    if s["adv_frac_zero"] < ADV_ZERO_WEAK:
        return "WEAK", thr
    return "FAIL", thr


def verdict_actions(s: dict) -> tuple:
    """Judges the per-state entropy when it is available (that is the number
    the training log prints and the one that means "how sharp is the policy"),
    falling back to the marginal otherwise — plus a floor on state-dependence
    I(S;A)/ln(A), because a high-entropy policy that is also nearly
    state-INDEPENDENT is not exploring, it is guessing."""
    thr = (f"PASS {ENT_LOW_WEAK}<=H/ln(A)<={ENT_HIGH_WEAK} and I(S;A)/ln(A)>={MI_PASS}, "
           f"WEAK outside that, FAIL H/ln(A)>{ENT_HIGH_FAIL} (~uniform), "
           f"<{ENT_LOW_FAIL} (collapsed), or I(S;A)/ln(A)<{MI_FAIL} (state-blind)")
    r = s.get("per_state_ratio")
    if r is None:
        r = s["entropy_ratio"]
    mi = s.get("mi_ratio")
    if not math.isfinite(r) or (mi is not None and not math.isfinite(mi)):
        return "FAIL", thr
    if r > ENT_HIGH_FAIL or r < ENT_LOW_FAIL:
        return "FAIL", thr
    if mi is not None and mi < MI_FAIL:
        return "FAIL", thr
    if r > ENT_HIGH_WEAK or r < ENT_LOW_WEAK:
        return "WEAK", thr
    if mi is not None and mi < MI_PASS:
        return "WEAK", thr
    return "PASS", thr


def verdict_value_scale(s: dict) -> tuple:
    thr = (f"PASS std ratio in {VSCALE_PASS_STD} and |bias|<{VSCALE_PASS_BIAS} ret-std, "
           f"WEAK in {VSCALE_WEAK_STD} and |bias|<{VSCALE_WEAK_BIAS}, FAIL beyond")
    sr, b = s["std_ratio"], abs(s["bias_in_ret_std"])
    if not math.isfinite(sr) or not math.isfinite(b):
        return "FAIL", thr
    if VSCALE_PASS_STD[0] <= sr <= VSCALE_PASS_STD[1] and b < VSCALE_PASS_BIAS:
        return "PASS", thr
    if VSCALE_WEAK_STD[0] <= sr <= VSCALE_WEAK_STD[1] and b < VSCALE_WEAK_BIAS:
        return "WEAK", thr
    return "FAIL", thr


def overall_verdict(labels) -> str:
    """Worst label wins — one FAIL is enough to condemn the loop."""
    labels = list(labels)
    if "FAIL" in labels:
        return "FAIL"
    if "WEAK" in labels:
        return "WEAK"
    return "PASS"


# ==========================================================================
# I/O side: checkpoint -> engine -> one collect -> stats
# ==========================================================================

def _resolve(p) -> str:
    """Config paths in the checkpoints/toml are repo-relative; make them work
    regardless of the cwd the diagnostic was launched from."""
    p = str(p)
    if not p:
        return p
    q = Path(p)
    return str(q if q.is_absolute() or q.exists() else REPO / p)


def load_checkpoint(path: str):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_net(state, device):
    """Load the checkpoint the way Trainer does for the v1 path: EntityPolicyNet
    dims come from the checkpoint's own config.net, the action table from the
    engine (construct._engine.action_table_v1)."""
    from construct._engine import action_table_v1
    from construct.learn.model_v1 import EntityPolicyNet

    net_cfg = state["config"]["net"]
    net = EntityPolicyNet(
        d_model=int(net_cfg["d_model"]), layers=int(net_cfg["layers"]),
        heads=int(net_cfg["heads"]), ff=int(net_cfg["ff"]),
        action_table=action_table_v1(),
    ).to(device)
    net.load_state_dict(state["model"])
    net.eval()
    return net


def build_engine(cfg, state, *, arenas, seed, own_reward):
    """Mirror Trainer.__init__'s Engine construction exactly (minus kickstart's
    emit_v0_obs, which the diagnostic has no use for): same schema, reward,
    curriculum, team-size mix and net_heads — only num_arenas is scaled down,
    which changes throughput but not the data distribution."""
    from construct._engine import Engine

    if own_reward:
        reward_path = state.get("reward_config_path") or cfg.reward_config_path
        curric = state.get("curriculum_config_path") or cfg.curriculum_config_path
    else:
        reward_path = cfg.reward_config_path
        curric = cfg.curriculum_config_path
    reward_path = _resolve(reward_path)
    curric = _resolve(curric) if curric else None

    env = state["config"]["env"]
    eng = Engine(
        num_arenas=arenas,
        blue=env["blue"], orange=env["orange"],
        schema_path=_resolve(cfg.schema_path),
        reward_config_path=reward_path,
        seed=seed if seed is not None else env["seed"],
        team_size_weights=env.get("team_size_weights"),
        curriculum_config_path=curric,
        net_heads=int(state["config"]["net"]["heads"]),
    )
    return eng, reward_path, curric


@torch.no_grad()
def torch_logprobs_values(net, obs: dict, actions, device, chunk=8192):
    """Recompute logprobs/values for the collected (obs, action) pairs with the
    LEARNER's torch net — the same call ppo_update makes on its first epoch,
    before any gradient step. Chunked so a 30k-row collect fits comfortably."""
    n = actions.shape[0]
    lps, vals, ents = [], [], []
    for s in range(0, n, chunk):
        sl = slice(s, min(s + chunk, n))
        kw = {k: torch.as_tensor(v[sl]).to(device) for k, v in obs.items()}
        lp, ent, val = net.evaluate(**kw, actions=torch.as_tensor(actions[sl]).to(device))
        lps.append(lp.float().cpu().numpy())
        vals.append(val.float().cpu().numpy())
        ents.append(ent.float().cpu().numpy())
    return np.concatenate(lps), np.concatenate(vals), np.concatenate(ents)


def diagnose(ck_path: str, cfg, *, steps: int, arenas: int, device, seed=None,
             own_reward=False) -> dict:
    """One checkpoint, one collect, all five checks. No training, no writes."""
    from construct.learn.gae import compute_gae

    t0 = time.perf_counter()
    state = load_checkpoint(ck_path)
    ck_ver = int(state.get("schema_version", 0))
    if ck_ver != 1:
        raise SystemExit(
            f"{ck_path}: schema_version={ck_ver}; this diagnostic implements the "
            "v1 entity path only (Trainer's v0 MLP path has a different net and obs).")

    net = build_net(state, device)
    engine, reward_path, curric = build_engine(
        cfg, state, arenas=arenas, seed=seed, own_reward=own_reward)

    # Exactly what Trainer.collect does: push the learner weights into the
    # engine, then collect. (The sync is not the suspect — it is verified here
    # anyway, via the epoch-0 importance ratio.)
    engine.set_weights({k: v.detach().cpu().numpy().astype(np.float32)
                        for k, v in net.state_dict().items()})
    t_collect = time.perf_counter()
    out = engine.collect(steps, arena_opponents=None)
    collect_s = time.perf_counter() - t_collect

    T, N = steps, out["learner_agents"]
    values_ext = np.concatenate([out["values"], out["last_values"][None, :]], axis=0)
    gamma, lam = cfg.ppo["gamma"], cfg.ppo["lam"]
    adv, ret = compute_gae(out["rewards"], values_ext, out["final_values"],
                           out["terminated"], out["truncated"], gamma, lam)
    # Companion measure. GAE returns are PARTLY MADE OF the values they are
    # scored against (ret = adv + V, and with lam=0.95/gamma=0.9954 the TD
    # correction only integrates ~1/(1-gamma*lam) ~ 18 steps of real reward),
    # so ev against them is optimistically biased — it measures ~18-step
    # self-consistency, not "does the value head know who is winning". The
    # lam=1.0 return is the Monte-Carlo return of the rollout (value enters
    # only through the tail bootstrap, weight gamma^T), so ev_mc is the
    # stricter read. A big ev/ev_mc gap = a value head that is locally smooth
    # but blind to the actual outcome.
    _, ret_mc = compute_gae(out["rewards"], values_ext, out["final_values"],
                            out["terminated"], out["truncated"], gamma, 1.0)

    obs = {
        "ents": out["ents"].reshape(T * N, *out["ents"].shape[2:]),
        "mask": out["mask"].reshape(T * N, out["mask"].shape[2]),
        "query": out["query"].reshape(T * N, out["query"].shape[2]),
        "prev": out["prev"].reshape(T * N, out["prev"].shape[2]),
    }
    actions = out["actions"].reshape(-1)
    eng_lp = out["logprobs"].reshape(-1)
    eng_val = out["values"].reshape(-1)
    flat_adv, flat_ret = adv.reshape(-1), ret.reshape(-1)

    t_fwd = time.perf_counter()
    tor_lp, tor_val, tor_ent = torch_logprobs_values(net, obs, actions, device)
    fwd_s = time.perf_counter() - t_fwd

    # goal-scale event threshold: |r| > 1 (all live regimes put shaping well
    # under 1/step and goals at 8-20, so this cleanly separates the two).
    ev = explained_variance(flat_ret, eng_val)
    ev_mc = explained_variance(ret_mc.reshape(-1), eng_val)
    ratio = ratio_delta_stats(eng_lp, tor_lp)
    advs = advantage_stats(flat_adv, flat_ret)
    sparse = reward_sparsity(out["rewards"], event_scale=1.0)
    acts = action_stats(actions, engine.action_count,
                        per_state_entropy=float(tor_ent.mean()))
    vscale = value_scale_stats(eng_val, flat_ret)
    vparity = ratio_delta_stats(eng_val, tor_val)  # candle-vs-torch value drift

    labels = {
        "ev": verdict_ev(ev)[0],
        "ratio": verdict_ratio(ratio)[0],
        "adv": verdict_adv(advs)[0],
        "act": verdict_actions(acts)[0],
        "vscale": verdict_value_scale(vscale)[0],
    }
    done = out["terminated"] | out["truncated"]
    return {
        "labels": labels,
        "overall": overall_verdict(labels.values()),
        "ck": ck_path,
        "total_steps": int(state.get("total_steps", 0)),
        "reward_config": reward_path,
        "curriculum_config": curric,
        "ck_reward_config": state.get("reward_config_path"),
        "T": T, "N": N, "rows": T * N,
        "arenas": arenas,
        "gamma": gamma, "lam": lam,
        "device": str(device),
        "collect_s": collect_s, "fwd_s": fwd_s,
        "wall_s": time.perf_counter() - t0,
        "episodes": int(done.sum()),
        "ep_reward_mean": float(out["rewards"].sum() / max(1, int(done.sum()))),
        "ev": ev,
        "ev_mc": ev_mc,
        "ratio": ratio,
        "adv": advs,
        "sparsity": sparse,
        "actions": acts,
        "vscale": vscale,
        "vparity": vparity,
        "torch_entropy": float(tor_ent.mean()),
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _line(label, thr):
    return f"  verdict: {label:4s}   [{thr}]"


def render(d: dict) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A(f"PPO LEARNING-HEALTH DIAGNOSTIC — {d['ck']}")
    A("=" * 78)
    A(f"checkpoint steps   {d['total_steps']:,}")
    A(f"reward config      {d['reward_config']}  (checkpoint provenance: {d['ck_reward_config']})")
    A(f"curriculum         {d['curriculum_config']}")
    A(f"rollout            T={d['T']} x N={d['N']} learner agents = {d['rows']:,} rows "
      f"({d['arenas']} arenas)")
    A(f"gae                gamma={d['gamma']} lam={d['lam']}")
    A(f"episodes ended     {d['episodes']}   ep_reward_mean {d['ep_reward_mean']:.3f}")
    A(f"device             {d['device']}   collect {d['collect_s']:.1f}s  "
      f"torch fwd {d['fwd_s']:.1f}s")
    A("")

    ev = d["ev"]
    lab, thr = verdict_ev(ev)
    A("[1] VALUE-HEAD EXPLAINED VARIANCE")
    A(f"  ev = 1 - Var(returns - values)/Var(returns) = {ev:+.4f}   "
      f"(GAE lam={d['lam']}, the returns PPO actually regresses on)")
    A(f"  ev_mc (same rollout, lam=1.0 Monte-Carlo returns) = {d.get('ev_mc', float('nan')):+.4f}")
    A(f"  Var(returns) {d['adv']['ret_std'] ** 2:.4f}   "
      f"Var(residual) {(d['vscale']['rmse'] ** 2):.4f} (rmse {d['vscale']['rmse']:.4f})")
    A(f"  corr(values, returns) = {d['vscale']['corr']:+.4f}")
    A("  ev is optimistically biased: GAE returns are built FROM the values "
      "(ret = adv + V) and")
    A(f"  the TD correction only spans ~{1 / max(1e-9, 1 - d['gamma'] * d['lam']):.0f} steps, "
      "so ev mostly measures short-horizon")
    A("  self-consistency. ev_mc is the stricter read — a large ev/ev_mc gap means the "
      "value")
    A("  head is locally smooth but blind to how the episode actually turns out.")
    if math.isfinite(ev) and ev < EV_NOISE:
        A(f"  >> ev < {EV_NOISE}: the value baseline explains almost nothing, so the "
          "advantages")
        A("     the policy gradient rides on are essentially noise. No reward design "
          "can fix this.")
    A(_line(lab, thr))
    A("")

    r = d["ratio"]
    lab_r, thr_r = verdict_ratio(r)
    A("[2] IMPORTANCE-RATIO SANITY (engine candle f32 vs learner torch, epoch 0)")
    A(f"  logprob delta (torch - engine): mean {r['mean_delta']:+.3e}  "
      f"std {r['std_delta']:.3e}  max|d| {r['max_abs_delta']:.3e}")
    A(f"  ratio exp(delta): mean {r['ratio_mean']:.6f}  p1 {r['ratio_p1']:.6f}  "
      f"p99 {r['ratio_p99']:.6f}")
    A(f"  frac outside [{RATIO_BAND[0]},{RATIO_BAND[1]}]: {r['frac_outside']:.6f}"
      f"  ({int(r['frac_outside'] * r['n']):,} of {r['n']:,})")
    vp = d["vparity"]
    A(f"  (value parity, same forward: max|torch-engine| {vp['max_abs_delta']:.3e}, "
      f"mean {vp['mean_delta']:+.3e})")
    A("  note: a ~1e-7 candle/torch gemm difference is expected and harmless; only a "
      "systematic")
    A("  offset or a heavy tail means PPO has been applying wrong importance weights.")
    A(_line(lab_r, thr_r))
    A("")

    a, sp = d["adv"], d["sparsity"]
    lab_a, thr_a = verdict_adv(a)
    A("[3] ADVANTAGE + RETURN DISTRIBUTION")
    A(f"  advantages: mean {a['adv_mean']:+.4f}  std {a['adv_std']:.4f}  "
      f"max|a| {a['adv_max_abs']:.4f}  frac exactly zero {a['adv_frac_zero']:.4f}")
    A("  adv pct  " + "  ".join(f"p{p}={a['adv_pct'][p]:+.4f}" for p in (1, 5, 25, 50, 75, 95, 99)))
    A(f"  returns:    mean {a['ret_mean']:+.4f}  std {a['ret_std']:.4f}")
    A("  ret pct  " + "  ".join(f"p{p}={a['ret_pct'][p]:+.4f}" for p in (1, 5, 25, 50, 75, 95, 99)))
    A(f"  reward sparsity: |r|>0 on {sp['frac_nonzero']:.4f} of steps, "
      f"|r|>1 (goal-scale) on {sp['frac_event']:.6f}")
    A(f"  reward mean {sp['mean']:+.5f}  mean|r| {sp['mean_abs']:.5f}  "
      f"max|r| {sp['max_abs']:.3f}")
    A(_line(lab_a, thr_a))
    A("")

    ac = d["actions"]
    lab_c, thr_c = verdict_actions(ac)
    A("[4] ACTION DISTRIBUTION")
    A(f"  marginal entropy  H(A)    {ac['entropy']:.4f} nats of ln({ac['n_actions']}) = "
      f"{ac['max_entropy']:.4f} max  ->  {ac['entropy_ratio'] * 100:.1f}% of uniform")
    if ac["per_state_entropy"] is not None:
        A(f"  per-state entropy E[H(A|s)] {ac['per_state_entropy']:.4f} nats "
          f"->  {ac['per_state_ratio'] * 100:.1f}% of uniform "
          "(this is the training log's `ent`)")
        A(f"  state-dependence  I(S;A) = H(A) - E[H(A|s)] = {ac['state_dependence']:.4f} "
          f"nats  ->  {ac['mi_ratio'] * 100:.1f}% of ln({ac['n_actions']})")
        A("  I(S;A) is how much the sampled action actually depends on the state. "
          "At 0 the")
        A("  policy emits the same distribution everywhere — a dice roll with a "
          "state-shaped")
        A("  label on it, which no amount of PPO on any reward can be improving.")
    A(f"  top-1 share {ac['top1_share']:.4f}   top-{ac['top_k']} share {ac['top_k_share']:.4f}")
    A(f"  {ac['n_cover']} of {ac['n_actions']} actions cover "
      f"{int(ac['cover'] * 100)}% of sampled mass")
    A(f"  never sampled: {ac['n_never']} of {ac['n_actions']} actions")
    A(_line(lab_c, thr_c))
    A("")

    v = d["vscale"]
    lab_v, thr_v = verdict_value_scale(v)
    A("[5] VALUE PREDICTION SCALE")
    A(f"  predicted values: mean {v['val_mean']:+.4f}  std {v['val_std']:.4f}")
    A(f"  actual returns:   mean {v['ret_mean']:+.4f}  std {v['ret_std']:.4f}")
    A(f"  std ratio {v['std_ratio']:.4f}   bias {v['bias_in_ret_std']:+.4f} return-stds"
      f"   rmse {v['rmse']:.4f}")
    A(_line(lab_v, thr_v))
    A("")

    labels = [lab, lab_r, lab_a, lab_c, lab_v]
    A(f"SUMMARY  {Path(d['ck']).name}  @ {d['total_steps']:,} steps")
    A("  " + "  ".join(f"[{i}]{n}={l}" for i, (n, l) in enumerate(
        zip(("ev", "ratio", "adv", "act", "vscale"), labels), start=1)))
    A(f"  OVERALL: {overall_verdict(labels)}")
    return "\n".join(L)


def render_compare(a: dict, b: dict) -> str:
    rows = [
        ("total steps", f"{a['total_steps']:,}", f"{b['total_steps']:,}"),
        ("explained variance", f"{a['ev']:+.4f}", f"{b['ev']:+.4f}"),
        ("ev_mc (lam=1 returns)", f"{a.get('ev_mc', float('nan')):+.4f}",
         f"{b.get('ev_mc', float('nan')):+.4f}"),
        ("corr(value, return)", f"{a['vscale']['corr']:+.4f}", f"{b['vscale']['corr']:+.4f}"),
        ("value rmse", f"{a['vscale']['rmse']:.4f}", f"{b['vscale']['rmse']:.4f}"),
        ("value std / return std", f"{a['vscale']['std_ratio']:.4f}", f"{b['vscale']['std_ratio']:.4f}"),
        ("value bias (ret std)", f"{a['vscale']['bias_in_ret_std']:+.4f}", f"{b['vscale']['bias_in_ret_std']:+.4f}"),
        ("ratio mean", f"{a['ratio']['ratio_mean']:.6f}", f"{b['ratio']['ratio_mean']:.6f}"),
        ("ratio max|logp delta|", f"{a['ratio']['max_abs_delta']:.3e}", f"{b['ratio']['max_abs_delta']:.3e}"),
        ("ratio frac outside", f"{a['ratio']['frac_outside']:.6f}", f"{b['ratio']['frac_outside']:.6f}"),
        ("advantage std", f"{a['adv']['adv_std']:.4f}", f"{b['adv']['adv_std']:.4f}"),
        ("advantage frac zero", f"{a['adv']['adv_frac_zero']:.4f}", f"{b['adv']['adv_frac_zero']:.4f}"),
        ("return std", f"{a['adv']['ret_std']:.4f}", f"{b['adv']['ret_std']:.4f}"),
        ("reward frac |r|>0", f"{a['sparsity']['frac_nonzero']:.4f}", f"{b['sparsity']['frac_nonzero']:.4f}"),
        ("reward frac |r|>1", f"{a['sparsity']['frac_event']:.6f}", f"{b['sparsity']['frac_event']:.6f}"),
        ("mean|reward|", f"{a['sparsity']['mean_abs']:.5f}", f"{b['sparsity']['mean_abs']:.5f}"),
        ("marginal H(A) (nats)", f"{a['actions']['entropy']:.4f}", f"{b['actions']['entropy']:.4f}"),
        ("marginal % of uniform", f"{a['actions']['entropy_ratio'] * 100:.1f}%", f"{b['actions']['entropy_ratio'] * 100:.1f}%"),
        ("per-state E[H(A|s)]", f"{a['actions']['per_state_entropy']:.4f}", f"{b['actions']['per_state_entropy']:.4f}"),
        ("state-dependence I(S;A)", f"{a['actions']['state_dependence']:.4f}", f"{b['actions']['state_dependence']:.4f}"),
        ("I(S;A) % of ln(A)", f"{a['actions']['mi_ratio'] * 100:.1f}%", f"{b['actions']['mi_ratio'] * 100:.1f}%"),
        ("actions covering 90%", f"{a['actions']['n_cover']}", f"{b['actions']['n_cover']}"),
        ("actions never sampled", f"{a['actions']['n_never']}", f"{b['actions']['n_never']}"),
        ("top-1 action share", f"{a['actions']['top1_share']:.4f}", f"{b['actions']['top1_share']:.4f}"),
        ("episodes ended", f"{a['episodes']}", f"{b['episodes']}"),
        ("OVERALL", a.get("overall", "?"), b.get("overall", "?")),
    ]
    na, nb = Path(a["ck"]).name, Path(b["ck"]).name
    w = max(len(r[0]) for r in rows)
    wa = max(len(na), max(len(r[1]) for r in rows))
    wb = max(len(nb), max(len(r[2]) for r in rows))
    L = ["", "=" * 78, "COMPARISON", "=" * 78,
         f"{'metric':<{w}}  {na:>{wa}}  {nb:>{wb}}",
         f"{'-' * w}  {'-' * wa}  {'-' * wb}"]
    for k, x, y in rows:
        L.append(f"{k:<{w}}  {x:>{wa}}  {y:>{wb}}")
    L.append("")
    L.append("Read this table as a contrast, not as absolutes: the metrics that DIFFER "
             "between a")
    L.append("checkpoint known to be strong and one known to be degraded are the ones "
             "that carry")
    L.append("the regression; the ones that are equally bad in BOTH are properties of "
             "the loop")
    L.append("itself and were never fixed by any reward variant.")
    return "\n".join(L)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Read-only PPO learning-health diagnostic (no training, no writes).")
    p.add_argument("checkpoint")
    p.add_argument("--config", default=str(REPO / "configs" / "train_v1.toml"))
    p.add_argument("--steps", type=int, default=256, help="rollout T for the single collect")
    p.add_argument("--arenas", type=int, default=32)
    p.add_argument("--compare", default=None, help="second checkpoint to diff against")
    p.add_argument("--own-reward", action="store_true",
                   help="use each checkpoint's own recorded reward/curriculum config "
                        "instead of --config's")
    p.add_argument("--device", default=None, help="torch device for the learner-side "
                                                  "recompute (default: config's)")
    p.add_argument("--seed", type=int, default=None, help="engine seed override")
    args = p.parse_args(argv)

    from construct.learn.config import TrainConfig
    cfg = TrainConfig.load(_resolve(args.config))
    dev = args.device or cfg.run.get("device", "cpu")
    device = torch.device(dev if (dev != "cuda" or torch.cuda.is_available()) else "cpu")

    kw = dict(steps=args.steps, arenas=args.arenas, device=device, seed=args.seed,
              own_reward=args.own_reward)
    a = diagnose(args.checkpoint, cfg, **kw)
    print(render(a), flush=True)
    if args.compare:
        b = diagnose(args.compare, cfg, **kw)
        print(render(b), flush=True)
        print(render_compare(a, b), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
