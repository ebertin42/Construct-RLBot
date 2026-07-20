#!/usr/bin/env python3
"""How differently does a candidate ACT from the champion, on identical states?

Why this exists: on 2026-07-20 I measured parameter-space drift between each
arm and the champion and found every arm moved the same distance (~22% of the
champion's weight norm) with no arm structure. I nearly read that as "armH at
lambda 1.0 is not frozen after all". That inference is invalid: `kl_prior`
constrains the OUTPUT DISTRIBUTION, not the parameters, so a high-lambda run is
free to move weights along directions that leave the action distribution
untouched -- and in a 488k-parameter net there are enormously many. Weight
distance simply does not bear on the question.

This measures the quantity lambda actually constrains. Both policies are
evaluated on ONE SHARED batch of states, so any difference is behaviour and not
a difference in the states each happened to visit.

    KL(candidate || champion)  mean nats per state -- the divergence the
                               kl_prior term penalises
    agreement                  fraction of states where the two argmax to the
                               same discrete action
    H(candidate), H(champion)  entropies, to tell "moved" apart from
                               "collapsed" or "went uniform"

Usage:
    scripts/behavior_distance.py --candidates checkpoints_hillclimb/*.pt
    scripts/behavior_distance.py --states 4096 --candidates a.pt b.pt

The state batch is collected by running the CHAMPION against itself under a
fixed seed, so it is reproducible and identical for every candidate compared in
one invocation.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "python"))

CHAMPION_DEFAULT = "checkpoints_entity/ck_000320471040.pt"


# ---------------------------------------------------------------------------
# pure metrics (unit tested; no engine, no checkpoints)
# ---------------------------------------------------------------------------

def kl_divergence(logits_p, logits_q) -> torch.Tensor:
    """Mean KL(P || Q) in nats, over a batch of logit rows.

    Argument order matters and is the usual source of sign confusion: this is
    the divergence FROM q TO p in the sense of "extra nats to encode p's
    samples using q", i.e. what the trainer penalises when p=student,
    q=prior. Computed in log space; never exponentiate logits directly, since
    the gate-era logits reach magnitudes where that overflows f32.
    """
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    return (logp.exp() * (logp - logq)).sum(-1).mean()


def agreement_rate(logits_a, logits_b) -> float:
    """Fraction of states where both policies pick the same argmax action.

    Reported alongside KL because the two answer different questions: KL can be
    large while the ranking is untouched (both still pick the same action, just
    with different confidence), and that distinction is exactly what separates
    "changed its mind" from "changed its certainty"."""
    return float((logits_a.argmax(-1) == logits_b.argmax(-1)).float().mean())


def entropy(logits) -> torch.Tensor:
    """Mean policy entropy in nats. A candidate that has collapsed to a single
    action and one that has gone uniform can show a similar KL from the
    champion; entropy tells them apart."""
    logp = F.log_softmax(logits, dim=-1)
    return -(logp.exp() * logp).sum(-1).mean()


def summarize_pair(logits_cand, logits_champ) -> dict:
    return {
        "kl_cand_champ": float(kl_divergence(logits_cand, logits_champ)),
        "kl_champ_cand": float(kl_divergence(logits_champ, logits_cand)),
        "agreement": agreement_rate(logits_cand, logits_champ),
        "h_cand": float(entropy(logits_cand)),
        "h_champ": float(entropy(logits_champ)),
    }


# ---------------------------------------------------------------------------
# checkpoint -> torch net
# ---------------------------------------------------------------------------

def build_net(ck_path, device="cpu"):
    """Rebuild EntityPolicyNet from a checkpoint's own recorded config.

    Reads dims from ck["config"]["net"] rather than assuming the current
    default: a checkpoint whose dims have drifted from configs/ must either
    load correctly or fail loudly, never be silently rebuilt at the wrong size.
    """
    from construct.learn.model_v1 import EntityPolicyNet

    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sd = ck["model"]
    net_cfg = dict(ck.get("config", {}).get("net", {}))
    action_table = sd["action_table"].numpy()
    net = EntityPolicyNet(
        d_model=int(net_cfg.get("d_model", 128)),
        layers=int(net_cfg.get("layers", 2)),
        heads=int(net_cfg.get("heads", 4)),
        ff=int(net_cfg.get("ff", 512)),
        action_table=action_table,
        aux=bool(net_cfg.get("aux", False)),
    )
    missing, unexpected = net.load_state_dict(sd, strict=False)
    if missing:
        raise SystemExit(f"{ck_path}: state dict is missing {len(missing)} keys "
                         f"(first: {missing[:3]}) -- refusing to score a half-loaded net")
    net.to(device).eval()
    return net


@torch.no_grad()
def logits_on(net, batch) -> torch.Tensor:
    logits, _ = net(batch["ents"], batch["mask"], batch["query"], batch["prev"])
    return logits


# ---------------------------------------------------------------------------
# shared state batch (engine-touching)
# ---------------------------------------------------------------------------

def collect_states(champion_ck, n_states, arenas=8, seed=11, device="cpu") -> dict:
    """Run the champion against itself and keep the states it visits.

    Champion-driven on purpose: the comparison asks how each candidate would
    act IN THE CHAMPION'S OWN DISTRIBUTION. Letting each candidate generate its
    own states would confound behavioural difference with distribution shift --
    the exact confound that made the replay arm worth checking in the first
    place.
    """
    from construct.league.matches import MatchRunner, load_sd

    sd = load_sd(champion_ck)
    mr = MatchRunner(num_arenas=arenas, seed=seed, mode=1, schema_version=1,
                     net_heads=4, reward_config="configs/reward_v3.toml")
    mr.eng.set_weights(sd)
    mr.eng.set_opponents([sd])
    steps = max(1, n_states // arenas + 1)
    out = mr.eng.collect(steps, arena_opponents=mr.assignment)

    def flat(key):
        a = np.asarray(out[key])
        return a.reshape(a.shape[0] * a.shape[1], *a.shape[2:])

    ents, mask, query, prev = flat("ents"), flat("mask"), flat("query"), flat("prev")
    keep = min(n_states, ents.shape[0])
    return {
        "ents": torch.as_tensor(ents[:keep], device=device),
        "mask": torch.as_tensor(mask[:keep], device=device),
        "query": torch.as_tensor(query[:keep], device=device),
        "prev": torch.as_tensor(prev[:keep], device=device).long(),
        "n": keep,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", default=CHAMPION_DEFAULT)
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--states", type=int, default=4096)
    ap.add_argument("--arenas", type=int, default=8)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args(argv)

    batch = collect_states(args.champion, args.states, args.arenas, args.seed)
    print(f"shared state batch: {batch['n']} states "
          f"(champion self-play, seed {args.seed}, {args.arenas} arenas)\n")

    champ_net = build_net(args.champion)
    champ_logits = logits_on(champ_net, batch)

    print(f"{'candidate':38s} {'KL(c||ch)':>10s} {'KL(ch||c)':>10s} "
          f"{'agree':>7s} {'H(c)':>6s} {'H(ch)':>6s}")
    for path in args.candidates:
        net = build_net(path)
        s = summarize_pair(logits_on(net, batch), champ_logits)
        print(f"{Path(path).name[:38]:38s} {s['kl_cand_champ']:10.4f} "
              f"{s['kl_champ_cand']:10.4f} {s['agreement']:7.3f} "
              f"{s['h_cand']:6.3f} {s['h_champ']:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
