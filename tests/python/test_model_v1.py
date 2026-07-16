import numpy as np
import torch

from construct.learn.model_v1 import EntityPolicyNet, ENT_FEAT, MAX_ENT, Q_FEAT, PREV_ACTIONS

TABLE_SIZE = 92

# Entity layout used by these tests (obs v1 order): self(0), mates(1-2),
# opps(3-5), ball(6), pads(7-12), ball-pred(13-16) -- matches
# docs/superpowers/plans/2026-07-16-entity-transformer-obs-v1.md.
SELF_IDX = 0
MATE_IDXS = (1, 2)
OPP_IDXS = (3, 4, 5)


def _action_table(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1, 1, size=(TABLE_SIZE, 8)).astype(np.float32)


def _batch(B: int, seed: int = 0, mask_extra_opp: bool = True):
    """Build a random obs-v1 batch: 2v2 (mates 1,2 unmasked; opp 5 masked)."""
    rng = np.random.default_rng(seed)
    ents = rng.standard_normal((B, MAX_ENT, ENT_FEAT)).astype(np.float32)
    mask = np.zeros((B, MAX_ENT), dtype=bool)
    if mask_extra_opp:
        mask[:, 5] = True  # 2v2: only opp slots 3,4 present, 5 absent
    query = rng.standard_normal((B, Q_FEAT)).astype(np.float32)
    prev = rng.integers(0, TABLE_SIZE, size=(B, PREV_ACTIONS)).astype(np.int64)
    return (
        torch.as_tensor(ents),
        torch.as_tensor(mask),
        torch.as_tensor(query),
        torch.as_tensor(prev),
    )


def _net(seed: int = 0, **kwargs) -> EntityPolicyNet:
    torch.manual_seed(seed)
    return EntityPolicyNet(d_model=32, layers=2, heads=4, ff=64, action_table=_action_table(), **kwargs)


def test_forward_shapes():
    B = 7
    net = _net()
    ents, mask, query, prev = _batch(B)
    logits, value = net(ents, mask, query, prev)
    assert logits.shape == (B, TABLE_SIZE)
    assert value.shape == (B, 1)


def test_mask_invariance():
    """Randomizing a masked entity's features must not change logits at all."""
    B = 5
    net = _net()
    ents, mask, query, prev = _batch(B)
    with torch.no_grad():
        logits_before, value_before = net(ents, mask, query, prev)

    ents2 = ents.clone()
    rng = np.random.default_rng(123)
    ents2[:, 5, :] = torch.as_tensor(rng.standard_normal((B, ENT_FEAT)).astype(np.float32))
    assert mask[:, 5].all(), "index 5 must be masked for this test to be meaningful"

    with torch.no_grad():
        logits_after, value_after = net(ents2, mask, query, prev)

    torch.testing.assert_close(logits_before, logits_after, atol=0, rtol=0)
    torch.testing.assert_close(value_before, value_after, atol=0, rtol=0)


def test_mate_permutation_equivariance():
    """Swapping the two (unmasked) mate rows must leave logits identical --
    no positional encoding, self-attention + pooling are permutation-invariant
    over the entity set."""
    B = 5
    net = _net()
    ents, mask, query, prev = _batch(B)
    assert not mask[:, MATE_IDXS[0]].any() and not mask[:, MATE_IDXS[1]].any()

    ents_swapped = ents.clone()
    ents_swapped[:, MATE_IDXS[0], :], ents_swapped[:, MATE_IDXS[1], :] = (
        ents[:, MATE_IDXS[1], :].clone(),
        ents[:, MATE_IDXS[0], :].clone(),
    )

    with torch.no_grad():
        logits_orig, _ = net(ents, mask, query, prev)
        logits_swapped, _ = net(ents_swapped, mask, query, prev)

    torch.testing.assert_close(logits_orig, logits_swapped, atol=1e-5, rtol=1e-5)


def test_prev_action_sensitivity():
    """Changing prev-action indices must change logits (prev-action pathway
    actually participates in the forward pass)."""
    B = 5
    net = _net()
    ents, mask, query, prev = _batch(B)
    prev2 = (prev + 1) % TABLE_SIZE

    with torch.no_grad():
        logits1, _ = net(ents, mask, query, prev)
        logits2, _ = net(ents, mask, query, prev2)

    assert not torch.allclose(logits1, logits2), "logits must be sensitive to prev actions"


def test_act_evaluate_api_parity():
    """act()/evaluate() mirror model.py's PolicyValueNet API so ppo.py plugs
    in unchanged: act -> (action, logprob, value[B]); evaluate -> (logprob,
    entropy, value[B])."""
    B = 6
    net = _net()
    ents, mask, query, prev = _batch(B)

    with torch.no_grad():
        action, logprob, value = net.act(ents, mask, query, prev)
    assert action.shape == (B,)
    assert logprob.shape == (B,)
    assert value.shape == (B,)  # squeezed, matching model.py's value.squeeze(-1)

    logprob2, entropy, value2 = net.evaluate(ents, mask, query, prev, action)
    assert logprob2.shape == (B,)
    assert entropy.shape == (B,)
    assert value2.shape == (B,)
    torch.testing.assert_close(logprob, logprob2, atol=1e-5, rtol=1e-5)
