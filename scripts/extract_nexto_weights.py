"""Extract Nexto (or Necto) weights from the shipped TorchScript into an npz
keyed by parameter name, for loading as a foreign in-engine opponent.

LICENSE: Nexto/Necto are CC BY-NC-SA 4.0 (Rolv-Arild/Necto). Non-commercial use
with attribution and share-alike. We still keep the extracted weights OUT of git
(binary blobs, and the licence carries obligations) -- they live in
~/.cache/construct/ and are pinned by SHA256 in external_weights.manifest.

Verified structure (nexto-model.pt):
    net.earl.query_preprocess.{0,2}      Linear 32->128, 128->128
    net.earl.key_value_preprocess.{0,2}  Linear 24->128, 128->128
    net.earl.blocks.{0,1}                attention.in_proj_{weight,bias} [384,128]/[384],
                                         attention.out_proj, linear1/2, norm1/2/3
    net.output.net.{0,2,4}               Linear 8->32, 32->32, 32->32
    net.output.emb_convertor             Linear 128->32
"""
import argparse
import hashlib
import pathlib

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_pt", help="deploy/external/nexto/nexto-model.pt")
    ap.add_argument("--out", default=None,
                    help="default: ~/.cache/construct/<stem>_weights.npz")
    args = ap.parse_args()

    m = torch.jit.load(args.model_pt)
    m.eval()
    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in m.named_parameters()}
    if not sd:
        raise SystemExit("no parameters found")

    n_blocks = len({k.split(".")[3] for k in sd if k.startswith("net.earl.blocks.")})
    print(f"parameters: {len(sd)}, earl blocks: {n_blocks}")
    for req in ("net.earl.query_preprocess.0.weight",
                "net.earl.key_value_preprocess.0.weight"):
        if req not in sd:
            raise SystemExit(f"missing expected key {req}")
    # Nexto has a ControlsPredictorDot head (emb_convertor + net); Necto has a
    # single output.linear split into [3,3,2,2,2]. Accept either.
    if "net.output.emb_convertor.weight" in sd:
        print("head: ControlsPredictorDot (Nexto-style, 90-action table)")
    elif "net.output.linear.weight" in sd:
        print("head: multi-discrete (Necto-style, [3,3,2,2,2])")
    else:
        raise SystemExit("no recognised output head")

    stem = pathlib.Path(args.model_pt).stem.replace("-model", "")
    out = pathlib.Path(args.out) if args.out else (
        pathlib.Path.home() / f".cache/construct/{stem}_weights.npz")
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **sd)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out}")
    print(f"sha256 {sha}")
    print("Record in external_weights.manifest. Do NOT commit the npz.")


if __name__ == "__main__":
    main()
