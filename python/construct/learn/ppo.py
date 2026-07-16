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
    # Optional hook: idx (LongTensor of minibatch indices into `batch`) ->
    # (extra_loss: 0-d Tensor, info: dict[str, float]). Added to the PPO loss
    # before the backward pass, and `info` is mean-accumulated into the
    # returned stats dict alongside policy_loss/value_loss/etc. This is the
    # kickstart-distillation seam (see train.py's `extra_loss_fn` closure and
    # kickstart.py) -- chosen over e.g. having evaluate() also return logits
    # so ppo_update/model.py/model_v1.py's public signatures stay untouched
    # when the hook is unused (`None` here is a complete no-op, byte-identical
    # to pre-hook behavior).
    extra_loss_fn=None,
) -> dict:
    n = batch["obs"].shape[0]
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0,
             "updates": 0, "skipped": 0}
    extra_keys: set[str] = set()
    for _ in range(epochs):
        perm = torch.randperm(n, device=batch["obs"].device)
        for s in range(0, n, minibatch_size):
            idx = perm[s : s + minibatch_size]
            logprobs, entropy, values = net.evaluate(batch["obs"][idx], batch["actions"][idx])
            # clamp the logratio: with very sharp policies (|logits| ~ 100 after
            # billions of steps) or stale old_logprobs (reward-regime swap),
            # exp() overflows to inf and one bad minibatch NaNs every weight
            # via clip_grad_norm. NOTE: PPO's gradient is NOT always zero
            # outside the clip range (it is nonzero when the unclipped branch
            # wins the min, e.g. A<0 with ratio >> 1+eps), so this clamp CAN
            # bias truly pathological minibatches — but at |logratio|=20 the
            # ratio is ~5e8, astronomically past the trust region; in normal
            # operation the clamp never fires. Do not widen it.
            logratio = torch.clamp(logprobs - batch["logprobs"][idx], -20.0, 20.0)
            ratio = torch.exp(logratio)
            a = adv[idx]
            unclipped = ratio * a
            clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * a
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = torch.nn.functional.mse_loss(values, batch["returns"][idx])
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy.mean()
            extra_info: dict[str, float] = {}
            if extra_loss_fn is not None:
                extra_loss, extra_info = extra_loss_fn(idx)
                loss = loss + extra_loss
            if not torch.isfinite(loss):
                # never let a nonfinite loss reach backward(): one NaN gradient
                # poisons the whole net through the shared grad-norm clip
                optimizer.zero_grad()
                stats["skipped"] += 1
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
            optimizer.step()
            stats["policy_loss"] += policy_loss.item()
            stats["value_loss"] += value_loss.item()
            stats["entropy"] += entropy.mean().item()
            stats["clip_frac"] += ((ratio - 1).abs() > clip).float().mean().item()
            for k, v in extra_info.items():
                stats[k] = stats.get(k, 0.0) + v
                extra_keys.add(k)
            stats["updates"] += 1
    for k in ("policy_loss", "value_loss", "entropy", "clip_frac", *extra_keys):
        stats[k] /= max(stats["updates"], 1)
    return stats
