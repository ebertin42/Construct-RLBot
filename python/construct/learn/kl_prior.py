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

# Fixed forward-chunk size for KLPrior.logits(). At remote deploy scale
# (138-167k row batches, RTX 3060 12GB) forwarding the whole batch at once
# blows the transient ff activations ([B,17,512] pairs in model_v1.Block,
# ~12GB peak) past the card's memory and OOMs mid-iteration. All of
# EntityPolicyNet's ops (LayerNorm, attention, ff) are row-independent, so
# chunking the batch dim is exactly equivalent to a single full forward --
# it's purely a memory/throughput knob, not a numerics change. Fixed (not
# adaptive) so behavior stays deterministic across runs/devices.
KL_PRIOR_CHUNK = 16384


class KLPrior:
    """Loads a frozen v1 (BC) checkpoint; serves full-distribution logits."""

    def __init__(self, ck_path: str, device: str = "cpu", expect_net: dict | None = None):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        sv = ck.get("schema_version")
        assert sv == 1, f"kl_prior needs a v1 checkpoint, got schema_version={sv} ({ck_path})"
        dims = ck["config"]["net"]
        if expect_net is not None:
            keys = ("d_model", "layers", "heads", "ff")
            assert all(int(dims[k]) == int(expect_net[k]) for k in keys), (
                f"kl_prior dims mismatch: checkpoint net={dims} != student "
                f"expect_net={expect_net} ({ck_path})"
            )
        self.net = EntityPolicyNet(
            d_model=int(dims["d_model"]), layers=int(dims["layers"]),
            heads=int(dims["heads"]), ff=int(dims["ff"]),
            action_table=ck["model"]["action_table"].numpy(),
        )
        self.net.load_state_dict(ck["model"])  # strict: dims mismatch raises
        self.device = torch.device(device)
        self.net.to(self.device).eval()
        self.net.requires_grad_(False)

    @torch.no_grad()
    def logits(self, obs: dict[str, torch.Tensor], _chunk: int = KL_PRIOR_CHUNK) -> torch.Tensor:
        obs = {k: v.to(self.device) for k, v in obs.items()}
        b = next(iter(obs.values())).shape[0]
        outs = []
        for s in range(0, b, _chunk):
            e = min(s + _chunk, b)
            sl = {k: v[s:e] for k, v in obs.items()}
            logits, _ = self.net(**sl)
            outs.append(logits)
        return torch.cat(outs, dim=0)


def kl_student_prior(student_logits: torch.Tensor, prior_logits: torch.Tensor) -> torch.Tensor:
    """Mean over batch of KL(student ‖ prior) on [B,92] logits."""
    log_p_s = F.log_softmax(student_logits, dim=-1)
    log_p_p = F.log_softmax(prior_logits, dim=-1)
    p_s = log_p_s.exp()
    return (p_s * (log_p_s - log_p_p)).sum(-1).mean()
