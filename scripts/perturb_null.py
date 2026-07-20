#!/usr/bin/env python3
"""Null control: move the champion the SAME distance PPO moves it, in a RANDOM direction.

The finding this exists to test (journal 2026-07-20 ~17:00): the policy is
already ~4 points below the champion after ONE PPO iteration, and it stays
there flat for at least 180 more. Two very different stories fit that:

  (a) PPO's updates are harmful -- it is walking somewhere worse.
  (b) ANY movement is harmful -- the champion sits at a local optimum of the
      gate metric, and every perturbation of it loses, whatever the direction.

Under (b) there is nothing wrong with PPO at all and the whole "PPO destroys
the policy" framing is a misreading; under (a) the update direction is the
problem. A random perturbation MATCHED IN MAGNITUDE separates them: if random
noise costs the same ~4 points, the direction carries no blame.

**Matching is per-tensor, not global.** For each parameter tensor the noise is
scaled to that tensor's own ||delta|| from the real PPO step, so the control
reproduces how the update distributes magnitude across layers (embeddings vs
attention vs heads) and differs ONLY in direction. A single global scale would
confound direction with a different per-layer profile, which is a second
variable.

    scripts/perturb_null.py --reference checkpoints_fine_named/fine_i01_*.pt \\
        --out checkpoints_null/null_seed1.pt --seed 1
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
CHAMPION_DEFAULT = "checkpoints_entity/ck_000320471040.pt"


def float_keys(sd) -> list:
    return [k for k, v in sd.items()
            if torch.is_tensor(v) and v.dtype.is_floating_point]


def per_tensor_deltas(champ, reference) -> dict:
    """||delta|| per tensor between the champion and a real post-update
    checkpoint. This is the magnitude profile the control must reproduce."""
    return {k: float((reference[k] - champ[k]).norm()) for k in float_keys(champ)}


def perturb(champ, norms, generator) -> dict:
    """Champion + random noise, scaled per tensor to the given norms.

    Zero-norm tensors are left exactly alone: if the real update did not move a
    tensor, the control must not move it either, or the comparison is not
    matched.
    """
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in champ.items()}
    for k, target in norms.items():
        if target <= 0:
            continue
        noise = torch.randn(champ[k].shape, generator=generator, dtype=torch.float32)
        n = float(noise.norm())
        if n == 0:
            continue
        out[k] = champ[k] + noise * (target / n)
    return out


def total_norm(sd_a, sd_b) -> float:
    return math.sqrt(sum(float(((sd_a[k] - sd_b[k]) ** 2).sum())
                         for k in float_keys(sd_a)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", default=CHAMPION_DEFAULT)
    ap.add_argument("--reference", required=True,
                    help="a real post-update checkpoint whose per-tensor step "
                         "magnitudes the control should match")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    ck = torch.load(args.champion, map_location="cpu", weights_only=False)
    champ = ck["model"]
    ref = torch.load(args.reference, map_location="cpu", weights_only=False)["model"]

    norms = per_tensor_deltas(champ, ref)
    gen = torch.Generator().manual_seed(args.seed)
    perturbed = perturb(champ, norms, gen)

    d_ref = total_norm(ref, champ)
    d_new = total_norm(perturbed, champ)
    print(f"reference moved ||d||={d_ref:.4f}; control moved ||d||={d_new:.4f} "
          f"(ratio {d_new / d_ref:.4f})")
    if abs(d_new / d_ref - 1.0) > 0.01:
        raise SystemExit("control magnitude does not match the reference -- "
                         "refusing to write an unmatched null control")

    ck["model"] = perturbed
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
