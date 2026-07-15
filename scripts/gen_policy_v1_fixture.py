# scripts/gen_policy_v1_fixture.py
"""Golden fixture for Rust (candle) <-> PyTorch parity of the obs-v1 entity
transformer (Task T4, docs/superpowers/plans/2026-07-16-entity-transformer-obs-v1.md).

Builds EntityPolicyNet(128, 2, 4, 512) with a fixed random action table
(seed 0, matching tests/python/test_model_v1.py's `_action_table` helper) and
fixed torch seed 0, forwards a batch of B=6 random inputs -- including a
realistic entity mask pattern (some rows fully unmasked / 1v1-3v3-ish, some
with several masked entities) and random prev-action indices in [0, 92) --
and saves everything (weights, inputs, expected outputs) as one float32 npz
so the Rust test (`engine/tests/policy_v1_test.rs`) can load it with no
PyTorch dependency.
"""
import numpy as np
import torch
from pathlib import Path

from construct.learn.model_v1 import EntityPolicyNet, ENT_FEAT, MAX_ENT, Q_FEAT, PREV_ACTIONS

TABLE_SIZE = 92
D_MODEL, LAYERS, HEADS, FF = 128, 2, 4, 512
B = 6


def _action_table(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(TABLE_SIZE, 8)).astype(np.float32)


def _mask_row(rng: np.random.Generator, pattern: str) -> np.ndarray:
    """Entity layout (obs v1 order): self(0), mates(1-2), opps(3-5), ball(6),
    pads(7-12), ball-pred(13-16). True = masked/ignored."""
    m = np.zeros(MAX_ENT, dtype=bool)
    if pattern == "full":  # 3v3: nothing masked
        pass
    elif pattern == "1v1":  # only self+opp0 present -> mates 1,2 and opps 4,5 masked
        m[1] = True
        m[2] = True
        m[4] = True
        m[5] = True
    elif pattern == "2v1":  # mate 1 present, mate 2 absent; opp 4,5 absent
        m[2] = True
        m[4] = True
        m[5] = True
    elif pattern == "2v2":  # mates 1,2 present; opp 5 absent
        m[5] = True
    elif pattern == "random_extra":  # full roster + a couple of random pad rows masked
        extra = rng.choice(np.arange(7, 13), size=2, replace=False)
        m[extra] = True
    return m


def build_batch(seed: int = 1):
    rng = np.random.default_rng(seed)
    ents = rng.standard_normal((B, MAX_ENT, ENT_FEAT)).astype(np.float32)
    patterns = ["full", "1v1", "2v1", "2v2", "random_extra", "full"]
    mask = np.stack([_mask_row(rng, p) for p in patterns], axis=0)
    # zero out masked entity rows the way the real obs builder does (absent
    # cars -> zero row + mask=1) so the fixture also implicitly exercises
    # "masked content shouldn't matter" without relying on it.
    ents = ents * (~mask)[:, :, None]
    query = rng.standard_normal((B, Q_FEAT)).astype(np.float32)
    prev = rng.integers(0, TABLE_SIZE, size=(B, PREV_ACTIONS)).astype(np.int64)
    return ents, mask, query, prev


def main():
    torch.manual_seed(0)
    table = _action_table(seed=0)
    net = EntityPolicyNet(d_model=D_MODEL, layers=LAYERS, heads=HEADS, ff=FF, action_table=table).eval()

    ents, mask, query, prev = build_batch(seed=1)
    t_ents = torch.as_tensor(ents)
    t_mask = torch.as_tensor(mask)
    t_query = torch.as_tensor(query)
    t_prev = torch.as_tensor(prev)

    with torch.no_grad():
        logits, value = net(t_ents, t_mask, t_query, t_prev)

    out: dict[str, np.ndarray] = {}
    for k, v in net.state_dict().items():
        out[k] = v.detach().cpu().numpy().astype(np.float32)

    out["ents"] = ents
    out["mask"] = mask.astype(np.float32)  # bool -> f32 (Rust side thresholds != 0)
    out["query"] = query
    out["prev"] = prev.astype(np.int64)
    out["expected_logits"] = logits.numpy().astype(np.float32)
    out["expected_value"] = value.numpy().astype(np.float32)
    out["dims"] = np.array([D_MODEL, LAYERS, HEADS, FF, B], dtype=np.int64)

    out_path = Path("engine/tests/fixtures/policy_v1_golden.npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out)
    size = out_path.stat().st_size
    print(f"wrote {out_path} ({size} bytes, {size / 1e6:.2f} MB)")
    print(f"d_model={D_MODEL} layers={LAYERS} heads={HEADS} ff={FF} batch={B}")
    print("state_dict keys:", list(net.state_dict().keys()))


if __name__ == "__main__":
    main()
