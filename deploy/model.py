import torch
import torch.nn as nn


class PolicyValueNet(nn.Module):
    def __init__(self, obs_size: int, action_count: int, hidden: tuple[int, ...] = (512, 512)):
        super().__init__()
        layers: list[nn.Module] = []
        last = obs_size
        for h in hidden:
            layers += [nn.Linear(last, h), nn.ReLU()]
            last = h
        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(last, action_count)
        self.value_head = nn.Linear(last, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.trunk(obs)
        return self.policy_head(z), self.value_head(z).squeeze(-1)

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate(self, obs: torch.Tensor, actions: torch.Tensor):
        logits, value = self(obs)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions), dist.entropy(), value


def load_policy(checkpoint_path: str, obs_size: int, action_count: int) -> "PolicyValueNet":
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert ck["schema_version"] == 0, f"schema mismatch: {ck['schema_version']}"
    net = PolicyValueNet(obs_size, action_count, tuple(ck["config"]["net"]["hidden"]))
    net.load_state_dict(ck["model"])
    net.eval()
    torch.set_num_threads(1)
    return net
