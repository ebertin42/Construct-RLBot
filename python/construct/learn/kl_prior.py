"""Frozen BC-prior anchor for PPO (spec §5.5).

KL DIRECTION: kl_student_prior computes KL(student ‖ prior) — mode-seeking.
The student is penalized for putting mass where the human prior has none,
but is free to concentrate on a subset of human-like actions. This is the
opposite direction from kickstart_losses (KL(teacher‖student), mode-covering)
because an anchor constrains support; a distillation target demands coverage.
"""
import torch
import torch.nn.functional as F

from construct.learn.model_v1 import EntityPolicyNet


class KLPrior:
    """Loads a frozen v1 (BC) checkpoint; serves full-distribution logits."""

    def __init__(self, ck_path: str, device: str = "cpu"):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        sv = ck.get("schema_version")
        assert sv == 1, f"kl_prior needs a v1 checkpoint, got schema_version={sv} ({ck_path})"
        dims = ck["config"]["net"]
        self.net = EntityPolicyNet(
            d_model=int(dims["d_model"]), layers=int(dims["layers"]),
            heads=int(dims["heads"]), ff=int(dims["ff"]),
            action_table=ck["model"]["action_table"].numpy(),
        )
        self.net.load_state_dict(ck["model"])  # strict: dims mismatch raises
        self.net.to(device).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.device = device

    @torch.no_grad()
    def logits(self, obs: dict) -> torch.Tensor:
        logits, _ = self.net(**obs)
        return logits


def kl_student_prior(student_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """Mean over batch of KL(student ‖ prior) on [B,92] logits."""
    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_p = F.log_softmax(prior_logits, dim=-1)
    p_s = log_p_s.exp()
    return (p_s * (log_p_s - log_p_p)).sum(-1).mean()
