"""Extract Immortal's MLP weights from its TorchScript jit.pt into an npz keyed by
state_dict name, for loading as a foreign in-engine opponent.

LICENSE: Immortal's weights come from RLMarlbot, which publishes NO license. The
output of this script is NOT redistributable and must NEVER be committed to this
repo. It is written outside the repo tree by default (~/.cache/construct/) and the
printed SHA256 goes in external_weights.manifest for tamper-evidence.

Source pin: RLMarlbot @ 9963b3bd328267d424ebac4a63429422df053e9c,
rlmarlbot/immortal/jit.pt

Verified structure (7 Linear + 6 LeakyReLU(0.01), single 126-wide head):
    net.0.weight  (512, 107)   net.0.bias  (512,)
    net.2/4/6/8/10.weight (512, 512)  + bias (512,)
    net.12.weight (126, 512)   net.12.bias (126,)
"""
import argparse, hashlib, pathlib

import numpy as np
import torch

EXPECT_IDX = (0, 2, 4, 6, 8, 10, 12)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jit_pt", help="path to rlmarlbot/immortal/jit.pt")
    ap.add_argument("--out",
                    default=str(pathlib.Path.home() / ".cache/construct/immortal_weights.npz"))
    args = ap.parse_args()

    actor = torch.jit.load(args.jit_pt)
    actor.eval()
    sd = {k: v.detach().cpu().numpy().astype(np.float32)
          for k, v in actor.named_parameters()}

    expect = [f"net.{i}.{p}" for i in EXPECT_IDX for p in ("weight", "bias")]
    if sorted(sd) != sorted(expect):
        raise SystemExit(f"unexpected keys: {sorted(sd)}")
    if sd["net.0.weight"].shape != (512, 107):
        raise SystemExit(f"net.0.weight {sd['net.0.weight'].shape} != (512, 107)")
    if sd["net.12.weight"].shape != (126, 512):
        raise SystemExit(f"net.12.weight {sd['net.12.weight'].shape} != (126, 512)")

    out = pathlib.Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **sd)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out}")
    print(f"sha256 {sha}")
    print("Record the sha in external_weights.manifest. DO NOT commit the npz "
          "(RLMarlbot publishes no license).")


if __name__ == "__main__":
    main()
