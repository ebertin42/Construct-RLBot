"""Step-keyed learning-rate anneal (v9 D8, train.lr_at + Trainer._apply_lr).

WHY THIS EXISTS. The learning rate was frozen at launch for the life of a
lineage, twice over: resume_train.py takes cfg.ppo FROM the checkpoint, and
Adam's restored param_groups carry `lr` as well, so even editing the dict lost
to `opt.load_state_dict`. The only escape was --reset-optimizer, which also
throws away the moment estimates. Meanwhile the RLGym-PPO-Guide stages lr by
what the policy can do (2e-4 cannot score / 1e-4 attempting to score / <=0.8e-4
complex mechanics) and v9 is ONE run that crosses all three stages.

The tests below pin the four properties the anneal has to have:
  1. absent `lr_final` it is an EXACT no-op -- every pre-v9 config unchanged;
  2. the hold really holds, so the launch window is identical to a flat config;
  3. it is keyed on total_steps, not on the iteration counter, so a resume
     cannot rewind it to the launch rate;
  4. cfg.ppo["lr"] beats Adam's restored param_groups, which is the whole point.
"""
import pytest
import torch

from construct.learn.train import lr_at


def test_absent_lr_final_is_an_exact_no_op():
    """Every config in configs/ except v9 has no lr_final; they must not move."""
    ppo = {"lr": 3e-4}
    for step in (0, 1, 10**6, 10**9, 10**12):
        assert lr_at(step, ppo) == 3e-4
    # ...and the other keys alone are still inert without lr_final, so a
    # half-configured schedule cannot silently anneal to zero.
    ppo2 = {"lr": 3e-4, "lr_hold_steps": 10, "lr_anneal_steps": 10}
    assert lr_at(10**9, ppo2) == 3e-4


def test_the_hold_holds_which_is_what_makes_the_launch_window_identical():
    """v9's hold is 200M steps: the G-gates and the whole "cannot score" ->
    "attempting to score" arc (v8 finished it by ~114M) must run at the launch
    rate, so the anneal is never competing with v9's other five changes for the
    explanation of what happened."""
    ppo = {"lr": 3e-4, "lr_final": 1e-4,
           "lr_hold_steps": 200_000_000, "lr_anneal_steps": 400_000_000}
    for step in (0, 20_000_000, 114_000_000, 200_000_000):
        assert lr_at(step, ppo) == pytest.approx(3e-4)
    assert lr_at(200_000_001, ppo) < 3e-4


def test_linear_ramp_then_flat():
    ppo = {"lr": 3e-4, "lr_final": 1e-4,
           "lr_hold_steps": 200_000_000, "lr_anneal_steps": 400_000_000}
    assert lr_at(400_000_000, ppo) == pytest.approx(2e-4)     # halfway
    assert lr_at(600_000_000, ppo) == pytest.approx(1e-4)     # end of ramp
    assert lr_at(10**12, ppo) == pytest.approx(1e-4)          # flat after
    # monotone, so a mis-read of the ramp can never speed the run up
    prev = 1e9
    for step in range(0, 800_000_000, 25_000_000):
        cur = lr_at(step, ppo)
        assert cur <= prev + 1e-12
        prev = cur


def test_zero_span_is_a_step_change_not_a_divide_by_zero():
    ppo = {"lr": 3e-4, "lr_final": 1e-4, "lr_hold_steps": 100, "lr_anneal_steps": 0}
    assert lr_at(100, ppo) == pytest.approx(3e-4)
    assert lr_at(101, ppo) == pytest.approx(1e-4)


def test_keyed_on_total_steps_not_on_the_iteration_counter():
    """`it` restarts at 0 on every resume; total_steps does not. A schedule keyed
    on `it` would rewind to the launch rate every time the box is restarted --
    silently, since nothing in the log line would look wrong."""
    ppo = {"lr": 3e-4, "lr_final": 1e-4,
           "lr_hold_steps": 200_000_000, "lr_anneal_steps": 400_000_000}
    # a run resumed at 500M steps and on iteration 1 is 3/4 of the way down the
    # ramp, not back at the launch rate
    assert lr_at(500_000_000, ppo) == pytest.approx(1.5e-4)


def test_apply_lr_beats_adams_restored_param_groups():
    """The trap this closes: opt.load_state_dict restores `lr` into param_groups,
    so before 2026-07-26 a resume could not change the rate at all without also
    discarding the moment estimates (--reset-optimizer)."""
    net = torch.nn.Linear(4, 4)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    net(torch.zeros(2, 4)).sum().backward()
    opt.step()                                   # populate Adam moments
    saved = opt.state_dict()
    assert saved["param_groups"][0]["lr"] == 3e-4

    fresh = torch.optim.Adam(net.parameters(), lr=1e-4)
    fresh.load_state_dict(saved)
    assert fresh.param_groups[0]["lr"] == 3e-4, (
        "if this ever fails torch changed its restore semantics and the trap "
        "this test documents is gone"
    )
    # what Trainer does after the restore, and what the anneal does every iter
    for g in fresh.param_groups:
        g["lr"] = lr_at(500_000_000, {"lr": 3e-4, "lr_final": 1e-4,
                                      "lr_hold_steps": 200_000_000,
                                      "lr_anneal_steps": 400_000_000})
    assert fresh.param_groups[0]["lr"] == pytest.approx(1.5e-4)
    # ...and the moments survived, which --reset-optimizer would not have done
    assert fresh.state_dict()["state"], "Adam moment estimates must be preserved"


def _small_cfg(tmp_path):
    from construct.learn.config import TrainConfig
    cfg = TrainConfig.load("configs/train_v0.toml")
    cfg.env.update(num_arenas=4)
    cfg.ppo.update(rollout_steps=16, minibatch_size=128)
    cfg.run.update(device="cpu", checkpoint_dir=str(tmp_path), save_every_iters=1)
    return cfg


def test_the_wiring_actually_moves_adams_lr_and_survives_a_resume(tmp_path):
    """End-to-end: lr_at is useless if run() never writes it into param_groups,
    and the resume half is the part that was broken before -- so the assertion
    that matters is the one AFTER load_checkpoint."""
    from construct.learn.train import Trainer

    cfg = _small_cfg(tmp_path)
    # one iteration is 16 * 8 = 128 steps; hold for 0 and ramp over 128 so the
    # schedule is observably mid-ramp inside a two-iteration test
    cfg.ppo.update(lr=3e-4, lr_final=1e-4, lr_hold_steps=0, lr_anneal_steps=256)
    t = Trainer(cfg)
    assert t.opt.param_groups[0]["lr"] == pytest.approx(3e-4)   # step 0
    t.run(max_iterations=1)
    assert t.total_steps == 128
    # run() sets the rate BEFORE the update, i.e. from the pre-iteration step
    # count, so after one iteration param_groups still reads the step-0 value
    assert t.opt.param_groups[0]["lr"] == pytest.approx(3e-4)
    t.run(max_iterations=1)
    assert t.opt.param_groups[0]["lr"] == pytest.approx(2e-4)   # step 128, halfway

    p = f"{tmp_path}/ck.pt"
    t.save_checkpoint(p)
    t2 = Trainer.load_checkpoint(p)
    assert t2.total_steps == 256
    # the schedule rode along in the checkpoint's ppo block and is applied on top
    # of the restored optimizer state -- 256/256 of the ramp = lr_final
    assert t2.opt.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_a_config_without_the_schedule_is_untouched_end_to_end(tmp_path):
    from construct.learn.train import Trainer

    cfg = _small_cfg(tmp_path)
    base = float(cfg.ppo["lr"])
    t = Trainer(cfg)
    t.run(max_iterations=1)
    assert t.opt.param_groups[0]["lr"] == base


def test_v9_config_declares_the_anneal_and_keeps_minibatch_8192():
    """Guards the two decisions this file exists for against a silent edit."""
    from construct.learn.config import TrainConfig
    cfg = TrainConfig.load("configs/train_v9_fromscratch.toml")
    p = cfg.ppo
    assert p["lr"] == 3e-4, "launch rate is the champion's, measured; see the toml"
    assert p["lr_final"] == 1e-4
    # the hold must clear the window v9 is judged in, or the anneal starts
    # competing with v9's other changes for the explanation
    assert p["lr_hold_steps"] >= 150_000_000
    assert p["minibatch_size"] == 8192
    # 36 Adam steps/iter -- the champion's regime. If rows/iter or the minibatch
    # move, this is the number to re-derive.
    rows = 365 * p["rollout_steps"]
    n_mb = -(-rows // p["minibatch_size"])
    assert n_mb * p["epochs"] == 36
