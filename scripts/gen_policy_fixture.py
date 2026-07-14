# scripts/gen_policy_fixture.py
"""Golden fixture for Rust<->PyTorch policy parity: random small net + inputs
+ expected outputs, all in one JSON."""
import json
from pathlib import Path

import torch

from construct.learn.model import PolicyValueNet

torch.manual_seed(7)
net = PolicyValueNet(obs_size=94, action_count=90, hidden=(32, 32)).eval()
obs = torch.randn(5, 94)
with torch.no_grad():
    logits, values = net(obs)

fx = {
    "obs_size": 94, "action_count": 90, "hidden": [32, 32], "batch": 5,
    "state_dict": {k: v.flatten().tolist() for k, v in net.state_dict().items()},
    "shapes": {k: list(v.shape) for k, v in net.state_dict().items()},
    "obs": obs.flatten().tolist(),
    "expected_logits": logits.flatten().tolist(),
    "expected_values": values.flatten().tolist(),
}
out = Path("engine/tests/fixtures/policy_fixture.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(fx))
print(f"wrote {out} ({out.stat().st_size} bytes)")
