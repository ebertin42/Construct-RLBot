"""Watch a checkpoint play in RLViser.
1) Download rlviser binary (github.com/VirxEC/rlviser/releases) into repo root
2) ./rlviser   (Linux/WSLg)  — or run rlviser on Windows and adjust target IP
3) python scripts/watch.py checkpoints/ck_XXXX.pt [--argmax] [--mode 2v2]

Actions are sampled by default (matches training behavior; argmax play
deadlocks in mirror-symmetric standoffs). Pass --argmax for deterministic.
--mode 1v1|2v2|3v3 sets the match size (default 1v1).

Dispatches on the checkpoint's `schema_version` (v0, the default when the key
is absent, or v1). v0 is the original path, byte-identical: an action-driven
RenderSession(schema/v0.toml), obs -> PolicyValueNet -> sample/argmax ->
step() from Python. v1 obs are entity tensors built and consumed entirely
in-engine (see policy_v1.rs / RenderSession's v1 methods in lib.rs), so
there's nothing for Python to feed step() -- instead set_weights() loads the
EntityPolicyNet state dict into the engine once, then step_policy() drives
one full env step (obs build, forward, sample, physics step, stream to
RLViser) per call. step_policy() always samples in-engine (no argmax hook),
so --argmax is a no-op for v1 (a note is printed, and the run proceeds
sampled).
"""
import sys

import numpy as np
import torch

from construct._engine import RenderSession
from construct.learn.model import PolicyValueNet

deterministic = "--argmax" in sys.argv
size = 1
if "--mode" in sys.argv:
    size = int(sys.argv[sys.argv.index("--mode") + 1][0])
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
is_v1 = int(ck.get("schema_version", 0)) == 1

if is_v1:
    from construct.learn.model_v1 import EntityPolicyNet

    if deterministic:
        print("note: --argmax is not supported on the v1 engine-side sampling "
              "path (step_policy always samples); proceeding sampled.")

    net_cfg = ck["config"]["net"]
    heads = int(net_cfg["heads"])
    try:
        from construct._engine import action_table_v1
        table = action_table_v1()
    except ImportError:
        table = ck["model"]["action_table"].numpy()
    net = EntityPolicyNet(
        d_model=int(net_cfg["d_model"]), layers=int(net_cfg["layers"]),
        heads=heads, ff=int(net_cfg["ff"]), action_table=table,
    )
    net.load_state_dict(ck["model"])
    net.eval()

    sess = RenderSession(blue=size, orange=size, schema_path="schema/v1.toml",
                         reward_config_path="configs/reward_v0.toml", seed=42,
                         net_heads=heads)
    sess.set_weights(
        {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    )
    try:
        while True:
            sess.step_policy()
    except KeyboardInterrupt:
        sess.close()
else:
    sess = RenderSession(blue=size, orange=size, schema_path="schema/v0.toml",
                         reward_config_path="configs/reward_v0.toml", seed=42)
    net = PolicyValueNet(sess.obs_size, sess.action_count, tuple(ck["config"]["net"]["hidden"]))
    net.load_state_dict(ck["model"])
    net.eval()

    obs = torch.as_tensor(sess.reset())
    try:
        while True:
            with torch.no_grad():
                logits = net(obs)[0]
                if deterministic:
                    actions = logits.argmax(-1)
                else:
                    actions = torch.distributions.Categorical(logits=logits).sample()
            nobs, *_ = sess.step(actions.numpy().astype(np.int64))
            obs = torch.as_tensor(nobs)
    except KeyboardInterrupt:
        sess.close()
