"""Auxiliary losses for the entity net (Lucy-SKG style): entity reconstruction and
multi-horizon return prediction.

WHY. `scripts/diagnose_ppo.py` on run A at 2.77B steps: the value head explains **0.84** of
the GAE returns PPO regresses on but only **~0.35** of true Monte-Carlo returns. It is
locally smooth and long-horizon weak, and conceding -- the measured way this bot loses -- is
a long-horizon outcome 3-8s after the positional error.

The two heads have existed in `model_v1.py` since the entity-transformer plan but were DEAD:
`aux_outputs()` was called nowhere, no loss used them, and `train.py` built the net without
passing `aux` at all. Enabling the flag alone would have added ~57k parameters that receive
no gradient.

WHAT THEY PREDICT. The original spec declared the shapes (`aux_reward` d->3, `aux_recon`
d->17*26) but never said what the outputs mean. Chosen here:

  * `aux_recon` -- reconstruct the entity matrix from `pooled`. Forces the pooled embedding
    to retain the full state rather than only what the policy needs. MASKED: absent entity
    slots are all-zero rows and an unmasked MSE would be dominated by them, teaching the
    trunk to predict zeros.
  * `aux_reward` -- the discounted return at THREE horizons (default 1, 15, 150 decisions =
    0.067s / 1s / 10s at tick_skip 8 and 120Hz; 150 is ~gamma's half-life at gamma=0.9954).
    This aims straight at the measured deficiency: a representation that can predict the
    10s-horizon return is one the value head can read a long-horizon value off.

BOTH ARE AUXILIARY. They are added to the PPO loss through the existing `extra_loss_fn`
seam and shape the SHARED trunk; neither replaces the value head, and their weights default
to 0.0 so an unconfigured run is byte-identical to before.
"""
from __future__ import annotations

import numpy as np
import torch

# Horizons in DECISIONS. tick_skip=8 at 120Hz => 15 decisions/s, so these are
# 0.067s / 1s / 10s. The last is ~the half-life of gamma=0.9954 (150.3 decisions).
DEFAULT_HORIZONS = (1, 15, 150)


def horizon_returns(
    rewards: np.ndarray,      # (T, N)
    values: np.ndarray,       # (T+1, N) -- includes the bootstrap row
    terminated: np.ndarray,   # (T, N) bool
    truncated: np.ndarray,    # (T, N) bool
    gamma: float,
    horizons=DEFAULT_HORIZONS,
) -> np.ndarray:              # (T, N, len(horizons))
    """n-step discounted returns, one column per horizon.

        G_t^(h) = sum_{i<h} gamma^i * r_{t+i}   + gamma^h * V(s_{t+h})

    truncated at the first episode end at or after t. A terminated step has NO successor
    value, so the sum stops there and the bootstrap is dropped -- bootstrapping past a
    termination leaks the NEXT episode's value backwards, which is the one error here that
    would be invisible in the loss curve.

    Rows near the end of the rollout have a shorter effective horizon (they run out of
    data); they bootstrap on `values` at whatever step they reach, which is the same
    convention `compute_gae` uses for the tail.
    """
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    T, N = rewards.shape
    done = np.asarray(terminated) | np.asarray(truncated)
    hs = tuple(int(h) for h in horizons)
    out = np.zeros((T, N, len(hs)), dtype=np.float32)

    for j, h in enumerate(hs):
        acc = np.zeros(N, dtype=np.float64)          # running discounted reward sum
        # Walk backwards is not usable here (each t has its OWN window), so walk forward
        # per t but reuse the fact that h is small (<= 150) relative to T.
        for t in range(T):
            g = 1.0
            acc[:] = 0.0
            alive = np.ones(N, dtype=bool)
            steps = 0
            for i in range(h):
                if t + i >= T:
                    break
                acc += np.where(alive, g * rewards[t + i], 0.0)
                steps = i + 1
                # after consuming step t+i, an episode end stops this window
                ended = np.asarray(done[t + i]) & alive
                alive = alive & ~ended
                g *= gamma
                if not alive.any():
                    break
            # bootstrap only where the window never hit an episode end
            boot_idx = min(t + steps, T)
            acc += np.where(alive, g * values[boot_idx], 0.0)
            out[t, :, j] = acc
    return out


def aux_losses(
    net,
    pooled: torch.Tensor,        # [B, d]
    ents: torch.Tensor,          # [B, E, F] reconstruction target
    mask: torch.Tensor,          # [B, E] bool, True = absent/ignore
    reward_target: torch.Tensor, # [B, H]
    *,
    w_recon: float,
    w_reward: float,
) -> tuple[torch.Tensor, dict]:
    """Weighted aux loss plus a stats dict. Returns a 0-d tensor that is safe to add to the
    PPO loss even when both weights are 0.0 (it stays rooted in the autograd graph via a
    *0.0 no-op, matching how `compose_extra_loss` roots its own loss)."""
    loss = pooled.sum() * 0.0
    info: dict[str, float] = {}
    if w_recon == 0.0 and w_reward == 0.0:
        return loss, info

    pred_r, pred_e = net.aux_outputs(pooled)

    if w_recon != 0.0:
        B, E, F = ents.shape
        pred_e = pred_e.view(B, E, F)
        keep = (~mask).unsqueeze(-1).to(pred_e.dtype)      # [B,E,1], 1.0 where present
        # Mean over PRESENT elements only. Denominator counts elements, not rows, so a
        # batch with more absent slots is not silently down-weighted.
        denom = keep.sum() * F
        recon = (((pred_e - ents) ** 2) * keep).sum() / denom.clamp(min=1.0)
        loss = loss + w_recon * recon
        info["aux_recon"] = float(recon.detach())

    if w_reward != 0.0:
        rl = torch.nn.functional.mse_loss(pred_r, reward_target)
        loss = loss + w_reward * rl
        info["aux_reward"] = float(rl.detach())
        # PER-HORIZON EXPLAINED VARIANCE, because the aggregate above is a TRAP.
        # It averages three horizons whose targets have wildly different variance: the
        # h=1 target is nearly trivial (r + gamma*V', which the value head already fits)
        # while h=150 is the hard one that ev_mc says is the actual deficiency. A drop
        # from 34 to 3 in the mean is consistent with the easy horizon carrying all of it
        # and the long one not moving -- unattributed aggregates have already misled this
        # investigation twice.
        #
        # ev = 1 - MSE/Var is scale-free, comparable ACROSS horizons, and on the same
        # scale as the diagnose_ppo ev_mc (~0.35) this whole arm exists to move.
        with torch.no_grad():
            se = ((pred_r - reward_target) ** 2).mean(dim=0)          # [H]
            var = reward_target.var(dim=0, unbiased=False)            # [H]
            ev = 1.0 - se / var.clamp(min=1e-8)
            # A degenerate (constant) target makes ev meaningless, not 1.0 -- report nan
            # so a flat column is visible as absent rather than as a perfect fit.
            ev = torch.where(var > 1e-8, ev, torch.full_like(ev, float("nan")))
            # SCALARS, one key per horizon -- ppo_update averages stats as floats, so a
            # list here would break the aggregation. nan on a degenerate column
            # propagates through that mean and prints as nan, which is the honest
            # rendering of "not measurable" rather than a silent 0.0 or a fake 1.0.
            for _i in range(ev.shape[0]):
                info[f"aux_ev{_i}"] = float(ev[_i])

    return loss, info
