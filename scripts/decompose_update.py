#!/usr/bin/env python3
"""Split a PPO update into its seed-SHARED direction and its seed-SPECIFIC residual.

Established 2026-07-20 (journal f875617): moving the champion a fixed distance
in a random direction costs ~1.5 points; moving it the same distance in the
direction PPO chooses costs ~4. And independently-seeded single-iteration
updates share ~15% of their direction (cosine 0.14-0.18, against ~0.001 for
random vectors in this 488k-dim space). So the harm is systematic, not per-run
noise -- which points at the SHARED component as the culprit.

This builds the two probes that test that directly. Given N update vectors
u_i = (checkpoint_i - champion):

    shared    m    = mean(u_i)            -- what every seed agrees on
    residual  r_i  = u_i - m              -- what only seed i did

Both probes are rescaled to the SAME magnitude as a real update, so the
comparison is direction-only, exactly as in perturb_null.py. Gate them against
the champion and the arithmetic is:

    shared probe ~0.46 (like PPO) and residual ~0.485 (like noise)
        -> the damage lives in the shared component; it is one specific,
           inspectable direction and can be read against behaviour.
    both ~0.485
        -> the harm is not captured by this decomposition; the shared
           component is not the mechanism.

HONEST LIMITS, since they bound what the result can mean:
  * With only N=3 runs the mean is a noisy estimate of the true shared
    direction, and the residuals still contain part of it. Expect attenuation
    toward the middle in BOTH probes; a clean split would need more runs.
  * ||m|| is naturally smaller than ||u_i|| (the u_i are only ~15% aligned).
    Rescaling to match magnitude is what makes the probes comparable, but it
    also means the shared probe is an EXTRAPOLATION along m, not a step any
    real run took.

    scripts/decompose_update.py --updates a.pt b.pt c.pt --outdir checkpoints_decomp
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

CHAMPION_DEFAULT = "checkpoints_entity/ck_000320471040.pt"


def float_keys(sd) -> list:
    return [k for k, v in sd.items()
            if torch.is_tensor(v) and v.dtype.is_floating_point]


def delta(sd, champ, keys) -> dict:
    return {k: (sd[k] - champ[k]) for k in keys}


def mean_delta(deltas, keys) -> dict:
    n = len(deltas)
    return {k: sum(d[k] for d in deltas) / n for k in keys}


def norm_of(d, keys) -> float:
    return math.sqrt(sum(float((d[k] ** 2).sum()) for k in keys))


def rescale(d, keys, target) -> dict:
    """Scale a direction to a target total L2 norm.

    Global rather than per-tensor here, unlike perturb_null: the point is to
    preserve the DIRECTION exactly (including how it distributes across layers,
    which is part of what makes it PPO's direction) and change only its length.
    """
    cur = norm_of(d, keys)
    if cur == 0:
        raise SystemExit("cannot rescale a zero direction")
    s = target / cur
    return {k: d[k] * s for k in keys}


def apply_direction(champ, direction, keys) -> dict:
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in champ.items()}
    for k in keys:
        out[k] = champ[k] + direction[k]
    return out


def cosine(a, b, keys) -> float:
    dot = sum(float((a[k] * b[k]).sum()) for k in keys)
    na, nb = norm_of(a, keys), norm_of(b, keys)
    return dot / (na * nb) if na and nb else 0.0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion", default=CHAMPION_DEFAULT)
    ap.add_argument("--updates", nargs="+", required=True,
                    help="post-update checkpoints from independently seeded runs")
    ap.add_argument("--outdir", default="checkpoints_decomp")
    args = ap.parse_args(argv)

    if len(args.updates) < 2:
        raise SystemExit("need at least 2 independently seeded updates to separate "
                         "a shared component from residuals")

    ck = torch.load(args.champion, map_location="cpu", weights_only=False)
    champ = ck["model"]
    keys = float_keys(champ)

    deltas, norms = [], []
    for p in args.updates:
        sd = torch.load(p, map_location="cpu", weights_only=False)["model"]
        d = delta(sd, champ, keys)
        deltas.append(d)
        norms.append(norm_of(d, keys))
    target = sum(norms) / len(norms)

    m = mean_delta(deltas, keys)
    print(f"{len(deltas)} updates, mean ||u||={target:.4f}, ||shared mean||={norm_of(m, keys):.4f}")
    for i, d in enumerate(deltas):
        print(f"  cos(u{i + 1}, shared) = {cosine(d, m, keys):+.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ck["model"] = apply_direction(champ, rescale(m, keys, target), keys)
    torch.save(ck, outdir / "decomp_shared.pt")
    print(f"wrote {outdir / 'decomp_shared.pt'}")

    for i, d in enumerate(deltas):
        resid = {k: d[k] - m[k] for k in keys}
        ck["model"] = apply_direction(champ, rescale(resid, keys, target), keys)
        torch.save(ck, outdir / f"decomp_resid{i + 1}.pt")
        print(f"wrote {outdir / f'decomp_resid{i + 1}.pt'}  "
              f"(cos with shared = {cosine(resid, m, keys):+.3f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
