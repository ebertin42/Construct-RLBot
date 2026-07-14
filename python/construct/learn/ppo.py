import torch


def ppo_update(
    net,
    optimizer,
    batch: dict,          # obs, actions, logprobs, advantages, returns, values (flat tensors)
    clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 1.0,
    epochs: int = 3,
    minibatch_size: int = 4096,
    max_grad_norm: float = 0.5,
) -> dict:
    n = batch["obs"].shape[0]
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0, "updates": 0}
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["obs"].device)
        for s in range(0, n, minibatch_size):
            idx = perm[s : s + minibatch_size]
            logprobs, entropy, values = net.evaluate(batch["obs"][idx], batch["actions"][idx])
            ratio = torch.exp(logprobs - batch["logprobs"][idx])
            a = adv[idx]
            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, batch["returns"][idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.mean().item()
            stats["clip_frac"] += ((ratio - 1).abs() > clip).float().mean().item()
            stats["updates"] += 1
    for k in ("policy_loss", "value_loss", "entropy", "clip_frac"):
        stats[k] /= max(stats["updates"], 1)
    return stats
