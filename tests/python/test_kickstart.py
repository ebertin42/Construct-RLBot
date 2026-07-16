import numpy as np
import pytest
import torch

from construct.learn.kickstart import (
    KickstartSchedule,
    KickstartTeacher,
    kickstart_losses,
    pad_teacher_logits,
)
from construct.learn.model import PolicyValueNet
from construct.learn.model_v1 import ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT, EntityPolicyNet

TABLE_SIZE_V1 = 92


def test_pad_teacher_logits_shape_and_padding_value():
    logits90 = torch.randn(4, 90)
    padded = pad_teacher_logits(logits90, out_dim=92)
    assert padded.shape == (4, 92)
    torch.testing.assert_close(padded[:, :90], logits90)
    assert torch.all(padded[:, 90:] == -1e9)


def test_pad_teacher_logits_softmax_and_kl_finite_no_nan():
    torch.manual_seed(0)
    logits90 = torch.randn(8, 90) * 10  # exaggerate to stress-test the padding
    padded = pad_teacher_logits(logits90, out_dim=92)
    probs = torch.softmax(padded, dim=-1)
    assert torch.isfinite(probs).all()
    assert not torch.isnan(probs).any()
    # the padded (stall) slots should carry ~0 probability
    assert torch.all(probs[:, 90:] == 0.0)

    student_logits = torch.randn(8, 92) * 10
    student_values = torch.randn(8)
    teacher_values = torch.randn(8)
    kl, v_mse = kickstart_losses(student_logits, student_values, logits90, teacher_values)
    assert torch.isfinite(kl)
    assert not torch.isnan(kl)
    assert torch.isfinite(v_mse)


def test_kickstart_schedule_anneal_points():
    sched = KickstartSchedule(lambda_k0=1.0, kickstart_steps=500_000_000, lambda_v=0.5)
    assert sched.coef(0) == (1.0, 0.5)

    lk, lv = sched.coef(250_000_000)
    assert lk == pytest.approx(0.5)
    assert lv == pytest.approx(0.5)

    assert sched.coef(500_000_000) == (0.0, 0.0)
    assert sched.coef(10**12) == (0.0, 0.0)


def test_kickstart_schedule_never_negative_past_horizon():
    sched = KickstartSchedule(lambda_k0=1.0, kickstart_steps=100, lambda_v=0.5)
    lk, lv = sched.coef(1000)
    assert lk == 0.0
    assert lv == 0.0


def test_distill_pull_kl_decreases_toward_teacher():
    """Tiny synthetic teacher/student on a fixed random batch: optimizing
    only the KL term must strictly pull the student toward the teacher's
    distribution, proving gradient direction + padding correctness
    end-to-end (Rust-independent)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)

    teacher = PolicyValueNet(94, 90, (32,))
    teacher.eval()
    teacher.requires_grad_(False)

    action_table = rng.uniform(-1, 1, size=(TABLE_SIZE_V1, 8)).astype(np.float32)
    student = EntityPolicyNet(32, 1, 2, 64, action_table=action_table)

    B = 16
    obs_v0 = torch.as_tensor(rng.standard_normal((B, 94)).astype(np.float32))
    ents = torch.as_tensor(rng.standard_normal((B, MAX_ENT, ENT_FEAT)).astype(np.float32))
    mask = torch.zeros((B, MAX_ENT), dtype=torch.bool)
    query = torch.as_tensor(rng.standard_normal((B, Q_FEAT)).astype(np.float32))
    prev = torch.as_tensor(rng.integers(0, TABLE_SIZE_V1, size=(B, PREV_ACTIONS)).astype(np.int64))

    with torch.no_grad():
        t_logits, t_values = teacher(obs_v0)

    opt = torch.optim.Adam(student.parameters(), lr=1e-2)

    with torch.no_grad():
        s_logits0, s_value0 = student(ents, mask, query, prev)
        initial_kl, _ = kickstart_losses(s_logits0, s_value0.squeeze(-1), t_logits, t_values)
    initial_kl = initial_kl.item()

    final_kl = initial_kl
    for _ in range(100):
        s_logits, s_value = student(ents, mask, query, prev)
        kl, _ = kickstart_losses(s_logits, s_value.squeeze(-1), t_logits, t_values)
        opt.zero_grad()
        kl.backward()
        opt.step()
        final_kl = kl.item()

    assert final_kl < 0.5 * initial_kl, f"initial={initial_kl} final={final_kl}"


def test_teacher_loads_synthetic_checkpoint_and_infers(tmp_path):
    torch.manual_seed(0)
    ref = PolicyValueNet(94, 90, (32,))
    ck = {"model": ref.state_dict(), "config": {"net": {"hidden": [32]}}}
    ck_path = tmp_path / "teacher.pt"
    torch.save(ck, ck_path)

    teacher = KickstartTeacher(str(ck_path))

    for p in teacher.net.parameters():
        assert not p.requires_grad
    assert not teacher.net.training

    obs = torch.randn(5, 94)
    logits, values = teacher.logits_values(obs)
    assert logits.shape == (5, 90)
    assert values.shape == (5,)
