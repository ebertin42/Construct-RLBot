import torch


def standardise_advantages(adv, *, adv_norm="global", group=None,
                           min_group_rows=2048, group_std_floor=0.05):
    """Return (standardised advantages, info dict).

    WHY THIS EXISTS (v9 D6). Until 2026-07-26 this was one line:

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    ONE scalar for the whole mixed batch, which PRESERVES the measured 9.48x
    1v1:3v3 magnitude ratio exactly (mean|TD advantage| on ck_001283829760:
    1v1 0.2275 / 2v2 0.0322 / 3v3 0.0240 -- six cars share one ball, so per-car
    reward events are ~3x rarer and team_spirit averages what is left). The
    entropy bonus, meanwhile, contributes a per-row gradient of CONSTANT
    magnitude `entropy_coef`. So a 3v3 row's useful signal sat ~2.4x the entropy
    pull while a 1v1 row's sat ~22.7x, and one `entropy_coef` meant two different
    things.

    Giving 3v3 more arenas cannot fix that: the entropy pull and the useful
    signal BOTH scale linearly with a group's row count, so their RATIO is
    share-independent -- collapse is decided purely by per-row |A|/entropy_coef.
    That is why v8 had to skew the arena mix to [0.85, 0.1, 0.05] instead, and
    why the skew then collapsed 1v1 (element p4 0.469 -> 0.417 -> 0.03-0.11 on
    two seeds, with 1v1 entropy falling 3.06 -> 2.27).

    Standardising WITHIN each team size puts mean|A_norm| at the same value in
    every group by construction, so one entropy_coef finally means one thing.

    Notes that are easy to get wrong:
      * RETURNS ARE NOT TOUCHED. Only the policy-loss advantage is rescaled, so
        the critic target and `value_loss` are unchanged.
      * The per-group sigma is FLOORED at `group_std_floor * sigma_batch`.
        0.05, not 0.1: measured sigma per group (Gaussian approx from mean|A|)
        is 0.285 / 0.0404 / 0.0301 against sigma_batch ~= 0.167 under the v9
        equal-row mix, so 3v3 sits at 0.18*sigma_batch -- a 0.1 floor would be
        only 1.8x below it and could BIND, defeating the entire purpose. 0.05
        gives 3.6x clearance and still catches a genuinely degenerate group (no
        reward events at all -> pure critic noise amplified to unit scale).
        `adv_group_floor_hits` in the returned info is the tripwire; it must be 0.
      * A group with fewer than `min_group_rows` rows is left in the global pool
        rather than standardised on a handful of samples.
      * `adv_norm="global"` reproduces the historical single line EXACTLY, and
        it is the default, so every pre-v9 config is byte-identical.
    """
    if adv_norm not in ("global", "group"):
        raise ValueError(f"adv_norm must be 'global' or 'group', got {adv_norm!r}")
    if adv_norm == "global" or group is None:
        return (adv - adv.mean()) / (adv.std() + 1e-8), {"adv_group_floor_hits": 0.0,
                                                         "adv_groups": 0.0}
    if group.shape[0] != adv.shape[0]:
        raise ValueError(
            f"group label length {group.shape[0]} != advantage length {adv.shape[0]}; "
            "the label must come from the engine's collect() output, one entry per "
            "learner row, tiled over T -- see Trainer.collect"
        )
    adv = adv.clone()
    mu_b, sigma_b = adv.mean(), adv.std()
    floor = group_std_floor * sigma_b
    done = torch.zeros_like(adv, dtype=torch.bool)
    floor_hits, n_groups = 0, 0
    for g in torch.unique(group).tolist():
        if g <= 0:
            continue                      # unlabelled (v0 path) -> global pool
        m = group == g
        if int(m.sum()) < min_group_rows:
            continue                      # too few samples to standardise on
        s = adv[m].std()
        if s < floor:
            floor_hits += 1
            s = floor
        adv[m] = (adv[m] - adv[m].mean()) / (s + 1e-8)
        done |= m
        n_groups += 1
    rest = ~done
    if bool(rest.any()):
        # Whatever did not form a big enough group keeps the historical
        # whole-batch statistics, so it is never left unnormalised.
        adv[rest] = (adv[rest] - mu_b) / (sigma_b + 1e-8)
    return adv, {"adv_group_floor_hits": float(floor_hits), "adv_groups": float(n_groups)}


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
    # Weight on the clipped policy-gradient term. 1.0 is every run to date and takes an
    # identity branch below, so the default path is unchanged. 0.0 makes the update purely
    # supervised, which is what a distillation gate needs.
    policy_coef: float = 1.0,
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
    # Advantage standardisation. "global" is the historical single-scalar
    # `(adv - mu)/sigma` over the whole mixed batch and is BYTE-IDENTICAL to
    # pre-v9 behaviour; "group" standardises within each team size. See
    # `standardise_advantages` for why that distinction is the load-bearing one.
    adv_norm: str = "global",
    # Per-row group label (team size 1/2/3), same length as the batch. Required
    # by adv_norm="group"; ignored otherwise. Rows labelled <= 0 fall into the
    # global pool.
    group: "torch.Tensor | None" = None,
    min_group_rows: int = 2048,
    group_std_floor: float = 0.05,
) -> dict:
    obs = batch["obs"]
    obs_is_dict = isinstance(obs, dict)
    # `first` stands in for the v0 obs tensor when sizing/locating the batch —
    # all dict tensors share the leading dim and device by construction.
    first = next(iter(obs.values())) if obs_is_dict else obs
    n = first.shape[0]
    adv, norm_info = standardise_advantages(
        batch["advantages"], adv_norm=adv_norm, group=group,
        min_group_rows=min_group_rows, group_std_floor=group_std_floor,
    )
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "clip_frac": 0.0,
             "updates": 0, "skipped": 0}
    stats.update(norm_info)
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
            # `policy_coef` exists so the policy-gradient term can be switched OFF for a pure
            # supervised run (distillation from a foreign teacher, see foreign_distill.py):
            # extra_loss_fn only ADDS, so without this the PPO gradient fights the teacher for
            # the whole run and the result is a mixture, not the experiment. value_coef and
            # entropy_coef were already zeroable from config; this was the one term that was
            # not. Identity-branch rather than an unconditional `1.0 *` so the default path is
            # the exact same sequence of ops it has always been.
            pg = policy_loss if policy_coef == 1.0 else policy_coef * policy_loss
            loss = pg + value_coef * value_loss - entropy_coef * entropy.mean()
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
