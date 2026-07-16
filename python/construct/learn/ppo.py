import torch


def ppo_update(
    net,
    optimizer,
    # obs, actions, logprobs, advantages, returns, values (flat tensors).
    # `batch["obs"]` is EITHER a flat tensor [n, obs_size] (v0 MLP path —
    # net.evaluate(obs, actions), byte-identical to the historical behavior)
    # OR a dict of flat tensors sharing leading dim n (v1 entity path: keys
    # ents/mask/query/prev). In the dict case every tensor is indexed by the
    # same minibatch permutation and passed to net.evaluate as keyword
    # arguments — the dict keys are a CONTRACT with the net's evaluate
    # signature (model_v1.EntityPolicyNet.evaluate(ents, mask, query, prev,
    # actions)); train.py's collect() builds them to match.
    batch: dict,

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
    obs = batch["obs"]
    obs_is_dict = isinstance(obs, dict)
    # `first` stands in for the v0 obs tensor when sizing/locating the batch —
    # all dict tensors share the leading dim and device by construction.
    first = next(iter(obs.values())) if obs_is_dict else obs
    n = first.shape[0]
    adv = batch["advantages"]
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0,
             "updates": 0, "skipped": 0}
    extra_keys: set[str] = set()
    for _ in range(epochs):
        perm = torch.randperm(n, device=first.device)
        for s in range(0, n, minibatch_size):
            idx = perm[s : s + minibatch_size]
            if obs_is_dict:
                logprobs, entropy, values = net.evaluate(
                    **{k: v[idx] for k, v in obs.items()}, actions=batch["actions"][idx]
                )
            else:
                logprobs, entropy, values = net.evaluate(obs[idx], batch["actions"][idx])
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
