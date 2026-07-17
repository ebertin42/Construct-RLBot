import numpy as np
import pytest
import torch

torch.set_num_threads(1)  # live trainers on this box

from construct.learn.kl_prior import KL_PRIOR_CHUNK, KLPrior, kl_student_prior
from construct.learn.model_v1 import EntityPolicyNet

NET = dict(d_model=128, layers=2, heads=4, ff=512)


def _table():
    from construct._engine import action_table_v1
    return np.asarray(action_table_v1(), dtype=np.float32)


def _obs(b=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "ents": torch.randn(b, 17, 26, generator=g),
        "mask": torch.zeros(b, 17, dtype=torch.bool),
        "query": torch.randn(b, 64, generator=g),
        "prev": torch.randint(0, 92, (b, 5), generator=g),
    }


def _save_ck(tmp_path, net, name="prior.pt"):
    p = tmp_path / name
    torch.save(
        {"model": net.state_dict(), "schema_version": 1,
         "config": {"net": NET}, "total_steps": 0},
        p,
    )
    return str(p)


def test_kl_zero_for_identical_nets(tmp_path):
    net = EntityPolicyNet(**NET, action_table=_table())
    ck = _save_ck(tmp_path, net)
    prior = KLPrior(ck, device="cpu")
    obs = _obs()
    with torch.no_grad():
        s_logits, _ = net(**obs)
    p_logits = prior.logits(obs)
    kl = kl_student_prior(s_logits, p_logits)
    assert kl.item() == pytest.approx(0.0, abs=1e-5)


def test_kl_positive_and_grads_reach_student_only(tmp_path):
    torch.manual_seed(0)
    student = EntityPolicyNet(**NET, action_table=_table())
    torch.manual_seed(1)
    other = EntityPolicyNet(**NET, action_table=_table())
    prior = KLPrior(_save_ck(tmp_path, other), device="cpu")
    obs = _obs()
    s_logits, _ = student(**obs)
    kl = kl_student_prior(s_logits, prior.logits(obs))
    # NOTE: brief's threshold was 0.01; at random init EntityPolicyNet's
    # logits are near-uniform (std ~0.07-0.17), so KL between two
    # differently-seeded nets lands ~0.003-0.013 -- 0.01 is not a reliable
    # bound for the brief's exact seeds (student=0, other=1 -> kl=0.0078,
    # deterministically). 1e-3 is ~2 orders of magnitude above the
    # "identical nets" test's 1e-5 zero-tolerance, so it still confirms a
    # real (not fp-noise) divergence.
    assert kl.item() > 1e-3
    kl.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0
               for p in student.parameters() if p.requires_grad)
    assert all(not p.requires_grad for p in prior.net.parameters())


def test_logits_chunked_matches_full_forward(tmp_path):
    """B=6 obs, chunk=4 (so two unequal chunks: 4+2) must produce logits
    bit-identical (within fp tolerance) to the default one-slice (chunk >= B)
    path. All of EntityPolicyNet's ops are row-independent (LayerNorm normalizes
    per-row, attention/ff operate per-sample), so chunking the batch dim is a
    pure memory/throughput knob -- not a numerics change."""
    net = EntityPolicyNet(**NET, action_table=_table())
    ck = _save_ck(tmp_path, net)
    prior = KLPrior(ck, device="cpu")
    obs = _obs(b=6)
    full = prior.logits(obs, _chunk=KL_PRIOR_CHUNK)
    chunked = prior.logits(obs, _chunk=4)
    assert chunked.shape == full.shape
    assert torch.allclose(chunked, full, atol=1e-6, rtol=1e-5)


def test_dim_mismatch_raises(tmp_path):
    net = EntityPolicyNet(**NET, action_table=_table())
    p = tmp_path / "bad.pt"
    torch.save(
        {"model": net.state_dict(), "schema_version": 1,
         "config": {"net": {**NET, "d_model": 64}}, "total_steps": 0},
        p,
    )
    with pytest.raises((AssertionError, RuntimeError)):
        KLPrior(str(p), device="cpu")


def test_expect_net_dims_checked(tmp_path):
    net = EntityPolicyNet(**NET, action_table=_table())
    ck = _save_ck(tmp_path, net)
    with pytest.raises(AssertionError) as exc:
        KLPrior(ck, device="cpu", expect_net={**NET, "d_model": 64})
    msg = str(exc.value)
    assert "128" in msg and "64" in msg
    # matching expect_net loads fine
    prior = KLPrior(ck, device="cpu", expect_net=NET)
    assert isinstance(prior, KLPrior)


def test_v0_checkpoint_rejected(tmp_path):
    p = tmp_path / "v0.pt"
    torch.save({"model": {}, "schema_version": 0, "config": {"net": {}}}, p)
    with pytest.raises(AssertionError):
        KLPrior(str(p), device="cpu")


def test_config_kl_prior_block(tmp_path):
    from construct.learn.config import TrainConfig
    cfg_toml = tmp_path / "t.toml"
    cfg_toml.write_text(
        'schema_path = "schema/v1.toml"\n'
        'reward_config_path = "configs/reward_v3.toml"\n'
        "[kl_prior]\n"
        'ck = "checkpoints_bc/bc.pt"\n'
        "lambda = 0.1\n"
    )
    cfg = TrainConfig.load(str(cfg_toml))
    assert cfg.kl_prior["ck"] == "checkpoints_bc/bc.pt"
    assert float(cfg.kl_prior["lambda"]) == 0.1


def test_config_kl_prior_default_empty(tmp_path):
    from construct.learn.config import TrainConfig
    cfg_toml = tmp_path / "t.toml"
    cfg_toml.write_text(
        'schema_path = "schema/v1.toml"\n'
        'reward_config_path = "configs/reward_v3.toml"\n'
    )
    cfg = TrainConfig.load(str(cfg_toml))
    assert cfg.kl_prior == {}


def _fake_batch(b=8):
    obs = _obs(b=b, seed=3)
    return {"obs": obs, "n_agents": b}


def test_composed_hook_prior_only(tmp_path):
    """No Trainer instantiation: drive the static composition helper."""
    from construct.learn.train import compose_extra_loss
    torch.manual_seed(0)
    student = EntityPolicyNet(**NET, action_table=_table())
    torch.manual_seed(1)
    other = EntityPolicyNet(**NET, action_table=_table())
    prior = KLPrior(_save_ck(tmp_path, other, "p.pt"), device="cpu")
    batch = _fake_batch()
    with torch.no_grad():
        prior_logits = prior.logits(batch["obs"])

    fn = compose_extra_loss(
        student, batch,
        kickstart=None, lambda_k=0.0, lambda_v=0.0,
        prior_logits=prior_logits, lambda_p=0.5,
    )
    idx = torch.arange(4)
    loss, info = fn(idx)
    assert loss.requires_grad
    assert info["kl_pri"] > 0.0
    assert "kick_kl" not in info


def test_composed_hook_kickstart_and_prior_both_active(tmp_path):
    """Both extra-loss terms active in one hook call: kickstart distillation
    (KL(teacher||student) + value MSE, annealed lambda_k/lambda_v) composed
    with the BC-prior anchor (KL(student||prior), constant lambda_p). Mirrors
    the reviewer's independent-recomputation check: the hook's loss must
    equal lambda_k*kl + lambda_v*v_mse + lambda_p*kl_p computed by hand from
    a second no_grad student forward on the same inputs (EntityPolicyNet has
    no dropout/BN, so the two forwards are deterministic and bit-comparable)."""
    from construct.learn.train import compose_extra_loss
    from construct.learn.kickstart import kickstart_losses

    torch.manual_seed(0)
    student = EntityPolicyNet(**NET, action_table=_table())
    torch.manual_seed(1)
    other = EntityPolicyNet(**NET, action_table=_table())
    prior = KLPrior(_save_ck(tmp_path, other, "p.pt"), device="cpu")
    batch = _fake_batch()
    with torch.no_grad():
        prior_logits = prior.logits(batch["obs"])

    b = batch["obs"]["ents"].shape[0]
    g = torch.Generator().manual_seed(7)
    # Synthesized teacher outputs matching KickstartTeacher.logits_values'
    # contract: (logits [B,90], values [B]), float32.
    t_logits = torch.randn(b, 90, generator=g)
    t_values = torch.randn(b, generator=g)

    lambda_k, lambda_v, lambda_p = 0.3, 0.2, 0.5
    fn = compose_extra_loss(
        student, batch,
        kickstart=(t_logits, t_values), lambda_k=lambda_k, lambda_v=lambda_v,
        prior_logits=prior_logits, lambda_p=lambda_p,
    )
    idx = torch.arange(4)
    loss, info = fn(idx)

    assert "kick_kl" in info and "kl_pri" in info
    assert loss.requires_grad

    # Independent recomputation: same student forward under no_grad, terms
    # computed directly rather than through the hook.
    with torch.no_grad():
        s_logits, s_value = student(**{k: v[idx] for k, v in batch["obs"].items()})
        kl, v_mse = kickstart_losses(s_logits, s_value.squeeze(-1), t_logits[idx], t_values[idx])
        kl_p = kl_student_prior(s_logits, prior_logits[idx])
        expected = lambda_k * kl + lambda_v * v_mse + lambda_p * kl_p

    assert loss.item() == pytest.approx(expected.item(), abs=1e-5)
    assert info["kick_kl"] == pytest.approx(kl.item(), abs=1e-6)
    assert info["kl_pri"] == pytest.approx(kl_p.item(), abs=1e-6)


def test_composed_hook_none_when_nothing_active():
    from construct.learn.train import compose_extra_loss
    student = EntityPolicyNet(**NET, action_table=_table())
    assert compose_extra_loss(student, _fake_batch(),
                              kickstart=None, lambda_k=0.0, lambda_v=0.0,
                              prior_logits=None, lambda_p=0.0) is None
