"""Kickstart distillation (Schmitt et al. 2018): PPO + an annealed on-policy
KL to a frozen v0 MLP teacher, plus a value-head regression term, so the
obs-v1 entity-transformer student starts at (not below) the current MLP's
performance instead of re-learning from scratch.

See docs/superpowers/plans/2026-07-16-entity-transformer-obs-v1.md, Task T7.

This module only builds the teacher + the loss primitives. Wiring into the
PPO update loop lives in train.py (see the `extra_loss_fn` hook documented
there); this module has no dependency on the v1 net or engine.
"""

import torch
import torch.nn.functional as F

from construct.learn.model import PolicyValueNet

# The kickstart teacher is always the legacy v0 MLP -- fixed obs/action
# widths, independent of whatever schema the student trains under.
TEACHER_OBS_SIZE = 94
TEACHER_ACTION_COUNT = 90

# Padding value for the teacher's missing (stall) action slots. Deliberately
# NOT -inf: log_softmax's normalizer (logsumexp) stays finite either way,
# but exp(-1e9) underflows cleanly to a probability of exactly 0.0 while
# -inf's log-softmax output is itself -inf, and 0.0 * -inf = NaN the moment
# it's multiplied into the KL sum below. -1e9 keeps every intermediate
# value finite (same trick model_v1.py's MHA already uses for masked keys).
NEG_INF_PAD = -1e9


class KickstartTeacher:
    """Frozen v0 MLP policy/value net used only to produce soft targets for
    the student during kickstart distillation. Never trained further."""

    def __init__(self, ck_path: str, device: str = "cpu"):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        hidden = tuple(ck["config"]["net"]["hidden"])
        self.device = torch.device(device)
        self.net = PolicyValueNet(TEACHER_OBS_SIZE, TEACHER_ACTION_COUNT, hidden).to(self.device)
        self.net.load_state_dict(ck["model"])
        self.net.eval()
        self.net.requires_grad_(False)

    @torch.no_grad()
    def logits_values(self, obs_v0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs_v0: [B,94] -> (logits [B,90], values [B])."""
        obs_v0 = obs_v0.to(self.device)
        logits, values = self.net(obs_v0)
        return logits, values


def pad_teacher_logits(logits90: torch.Tensor, out_dim: int = 92) -> torch.Tensor:
    """[B,90] -> [B,out_dim], padding the appended (e.g. stall) action slots
    with NEG_INF_PAD rather than -inf -- see module docstring."""
    b, n = logits90.shape
    pad = out_dim - n
    assert pad >= 0, f"out_dim={out_dim} smaller than teacher width {n}"
    if pad == 0:
        return logits90
    filler = torch.full((b, pad), NEG_INF_PAD, dtype=logits90.dtype, device=logits90.device)
    return torch.cat([logits90, filler], dim=1)


class KickstartSchedule:
    """Linear anneal of the KL weight lambda_k from lambda_k0 -> 0 over
    kickstart_steps. lambda_v (value-regression weight) stays fixed while
    lambda_k > 0, then drops to 0 once the KL pull has fully annealed off."""

    def __init__(
        self,
        lambda_k0: float = 1.0,
        kickstart_steps: int = 500_000_000,
        lambda_v: float = 0.5,
    ):
        self.lambda_k0 = lambda_k0
        self.kickstart_steps = kickstart_steps
        self.lambda_v = lambda_v

    def coef(self, total_steps: int) -> tuple[float, float]:
        if self.kickstart_steps <= 0:
            frac = 1.0
        else:
            frac = min(1.0, max(0.0, total_steps / self.kickstart_steps))
        lambda_k = max(0.0, self.lambda_k0 * (1.0 - frac))
        lambda_v = self.lambda_v if lambda_k > 0.0 else 0.0
        return lambda_k, lambda_v


def kickstart_losses(
    student_logits: torch.Tensor,    # [B,92]
    student_values: torch.Tensor,    # [B]
    teacher_logits90: torch.Tensor,  # [B,90]
    teacher_values: torch.Tensor,    # [B]
) -> tuple[torch.Tensor, torch.Tensor]:
    """KL(teacher || student) = sum(p_t * (log p_t - log p_s)), mean over the
    batch, plus MSE(student_values, teacher_values)."""
    teacher_logits = pad_teacher_logits(teacher_logits90, out_dim=student_logits.shape[-1])
    p_t = torch.softmax(teacher_logits, dim=-1)
    log_p_t = torch.log_softmax(teacher_logits, dim=-1)
    log_p_s = torch.log_softmax(student_logits, dim=-1)
    kl = (p_t * (log_p_t - log_p_s)).sum(dim=-1).mean()
    v_mse = F.mse_loss(student_values, teacher_values)
    return kl, v_mse
