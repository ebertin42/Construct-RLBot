# scripts/bench_collect.py
"""Collection throughput: rust in-engine path, by arena count.

--schema v0|v1 (default v0) selects the obs/policy path. v0 (default) is the
original behavior, byte-identical: a random PolicyValueNet(94, 90, (512,512))
against schema/v0.toml. v1 builds an EntityPolicyNet against schema/v1.toml
-- by default with random weights at the T1-gate launch dims (128/2/4/512,
see .superpowers/sdd/transformer-bench.md), or with a checkpoint's own dims
and trained weights if --checkpoint is given. This is the tool the deploy
runs on the remote to sanity a v1 rebuild's collection throughput before
swapping it in.
"""
import argparse
import time

import numpy as np
import torch

from construct._engine import Engine, action_table_v1
from construct.learn.model import PolicyValueNet
from construct.learn.model_v1 import EntityPolicyNet

p = argparse.ArgumentParser()
p.add_argument("--schema", choices=["v0", "v1"], default="v0")
p.add_argument("--checkpoint", default=None,
               help="v1 only: load net dims + weights from a checkpoint instead of random init")
args = p.parse_args()

torch.manual_seed(0)

heads = 4
if args.schema == "v1":
    schema_path = "schema/v1.toml"
    if args.checkpoint:
        ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        net_cfg = ck["config"]["net"]
        heads = int(net_cfg["heads"])
        try:
            table = action_table_v1()
        except ImportError:
            table = ck["model"]["action_table"].numpy()
        net = EntityPolicyNet(
            d_model=int(net_cfg["d_model"]), layers=int(net_cfg["layers"]),
            heads=heads, ff=int(net_cfg["ff"]), action_table=table,
        )
        net.load_state_dict(ck["model"])
    else:
        net = EntityPolicyNet(d_model=128, layers=2, heads=heads, ff=512, action_table=action_table_v1())
else:
    schema_path = "schema/v0.toml"
    net = PolicyValueNet(94, 90, (512, 512))

sd = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}

for arenas in (64, 96, 192, 256):
    eng_kwargs = dict(num_arenas=arenas, blue=1, orange=1, schema_path=schema_path,
                       reward_config_path="configs/reward_v0.toml", seed=0)
    if args.schema == "v1":
        eng_kwargs["net_heads"] = heads
    eng = Engine(**eng_kwargs)
    eng.set_weights(sd)
    eng.collect(16)  # warmup
    t0 = time.perf_counter()
    T = 128
    eng.collect(T)
    dt = time.perf_counter() - t0
    print(f"arenas={arenas:4d} agents={eng.num_agents:4d}: {T * arenas / dt:>10,.0f} env-steps/s")
