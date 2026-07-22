"""Extract Element's weights from RLMarlbot's pickled state_dict into an npz.

LICENSE: RLMarlbot publishes no license -- private research use only, never
committed. Output goes to ~/.cache/construct/ and is SHA-pinned in
external_weights.manifest.

model.p is a plain OrderedDict of tensors (NOT a pickled module), so it loads
without Element's Actor class on the path. Structure:
    fc1 (256,107), fc2..fc5 (256,256), cat_heads (15,256), ber_heads (6,256)
i.e. a 5x256 ReLU MLP with 5 categorical heads (3-way) and 3 Bernoulli heads.
"""
import argparse, hashlib, pathlib, pickle

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_p", help="rlmarlbot/element/model.p")
    ap.add_argument("--out",
                    default=str(pathlib.Path.home() / ".cache/construct/element_weights.npz"))
    args = ap.parse_args()

    with open(args.model_p, "rb") as f:
        sd = pickle.load(f)
    out_sd = {}
    for k, v in sd.items():
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        out_sd[k] = arr.astype(np.float32)

    expect = [f"fc{i}.{p}" for i in range(1, 6) for p in ("weight", "bias")] + \
             [f"{h}.{p}" for h in ("cat_heads", "ber_heads") for p in ("weight", "bias")]
    if sorted(out_sd) != sorted(expect):
        raise SystemExit(f"unexpected keys: {sorted(out_sd)}")
    if out_sd["fc1.weight"].shape != (256, 107):
        raise SystemExit(f"fc1.weight {out_sd['fc1.weight'].shape} != (256, 107)")
    if out_sd["cat_heads.weight"].shape != (15, 256):
        raise SystemExit("cat_heads must be (15, 256) -- 5 categorical heads of 3")
    if out_sd["ber_heads.weight"].shape != (6, 256):
        raise SystemExit("ber_heads must be (6, 256) -- 3 Bernoulli heads of 2")

    out = pathlib.Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **out_sd)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"wrote {out}")
    print(f"sha256 {sha}")
    print("Record in external_weights.manifest. Do NOT commit the npz.")


if __name__ == "__main__":
    main()
