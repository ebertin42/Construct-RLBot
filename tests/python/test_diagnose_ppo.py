"""Pure-function tests for scripts/diagnose_ppo.py.

Everything under test here is arithmetic on synthetic arrays: no engine, no
checkpoint, no GPU, no rollout. The I/O half of the diagnostic (checkpoint ->
Engine -> collect) is deliberately not covered — it needs a real engine build
and a multi-second rollout, and its correctness is "does it do what
Trainer.collect does", which is a code-reading question, not a unit-test one.
"""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _load():
    """scripts/ is not a package — load diagnose_ppo.py by path."""
    path = REPO / "scripts" / "diagnose_ppo.py"
    spec = importlib.util.spec_from_file_location("diagnose_ppo", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["diagnose_ppo"] = mod
    spec.loader.exec_module(mod)
    return mod


dp = _load()


# ---------------------------------------------------------------- explained var

def test_ev_perfect_prediction_is_one():
    rng = np.random.default_rng(0)
    returns = rng.normal(size=1000)
    assert dp.explained_variance(returns, returns) == pytest.approx(1.0)


def test_ev_mean_only_prediction_is_zero():
    rng = np.random.default_rng(1)
    returns = rng.normal(loc=3.0, scale=2.0, size=1000)
    values = np.full_like(returns, returns.mean())
    assert dp.explained_variance(returns, values) == pytest.approx(0.0, abs=1e-12)


def test_ev_anticorrelated_prediction_is_negative():
    rng = np.random.default_rng(2)
    returns = rng.normal(size=1000)
    # values = -returns => residual = 2*returns => ev = 1 - 4 = -3
    assert dp.explained_variance(returns, -returns) == pytest.approx(-3.0)


def test_ev_ignores_a_constant_bias_which_is_why_check_5_exists():
    # ev uses Var(residual), not mean(residual^2) (the SB3 convention), so a
    # value head that is perfectly shaped but uniformly offset still scores
    # ev = 1.0. That blind spot is exactly what check [5] (value prediction
    # scale) covers: it reports the bias in return-std units.
    rng = np.random.default_rng(3)
    returns = rng.normal(scale=1.0, size=5000)
    assert dp.explained_variance(returns, returns + 0.5) == pytest.approx(1.0)
    biased = dp.value_scale_stats(returns + 0.5, returns)
    assert biased["bias_in_ret_std"] == pytest.approx(0.5, rel=0.05)


def test_ev_half_scale_prediction_loses_variance():
    # values = 0.5 * returns => residual = 0.5 * returns => ev = 1 - 0.25 = 0.75
    rng = np.random.default_rng(31)
    returns = rng.normal(scale=1.0, size=5000)
    assert dp.explained_variance(returns, 0.5 * returns) == pytest.approx(0.75)


def test_ev_constant_returns_is_nan():
    assert math.isnan(dp.explained_variance(np.ones(50), np.zeros(50)))


def test_ev_rejects_shape_mismatch():
    with pytest.raises(AssertionError):
        dp.explained_variance(np.zeros(10), np.zeros(11))


# ------------------------------------------------------------------ ratio stats

def test_ratio_stats_identical_logprobs_are_exactly_one():
    lp = np.log(np.full(500, 1.0 / 92.0))
    s = dp.ratio_delta_stats(lp, lp)
    assert s["mean_delta"] == 0.0
    assert s["max_abs_delta"] == 0.0
    assert s["ratio_mean"] == pytest.approx(1.0)
    assert s["frac_outside"] == 0.0
    assert s["n"] == 500


def test_ratio_stats_tiny_float_noise_stays_inside_band():
    rng = np.random.default_rng(4)
    eng = rng.normal(loc=-4.0, size=2000)
    tor = eng + rng.normal(scale=1e-7, size=2000)  # the known candle/torch gemm delta
    s = dp.ratio_delta_stats(eng, tor)
    assert abs(s["mean_delta"]) < 1e-7
    assert s["max_abs_delta"] < 1e-5
    assert s["frac_outside"] == 0.0
    assert dp.verdict_ratio(s)[0] == "PASS"


def test_ratio_stats_systematic_offset_is_detected():
    eng = np.full(1000, -4.0)
    tor = eng + 0.5  # every recomputed logprob shifted: ratio = exp(0.5) = 1.649
    s = dp.ratio_delta_stats(eng, tor)
    assert s["mean_delta"] == pytest.approx(0.5)
    assert s["ratio_mean"] == pytest.approx(math.exp(0.5))
    assert s["frac_outside"] == 1.0
    assert dp.verdict_ratio(s)[0] == "FAIL"


def test_ratio_stats_percentiles_bracket_the_spread():
    rng = np.random.default_rng(5)
    eng = np.zeros(10_000)
    tor = rng.normal(scale=0.2, size=10_000)
    s = dp.ratio_delta_stats(eng, tor)
    assert s["ratio_p1"] < 1.0 < s["ratio_p99"]
    assert 0.0 < s["frac_outside"] < 1.0


def test_ratio_stats_rejects_shape_mismatch():
    with pytest.raises(AssertionError):
        dp.ratio_delta_stats(np.zeros(4), np.zeros(5))


# -------------------------------------------------------------- advantage stats

def test_advantage_stats_basic_moments_and_percentiles():
    adv = np.arange(-50, 51, dtype=np.float64)  # symmetric, median 0
    ret = adv + 10.0
    s = dp.advantage_stats(adv, ret)
    assert s["n"] == 101
    assert s["adv_mean"] == pytest.approx(0.0)
    assert s["adv_pct"][50] == pytest.approx(0.0)
    assert s["adv_max_abs"] == pytest.approx(50.0)
    assert s["adv_frac_zero"] == pytest.approx(1 / 101)
    assert s["ret_mean"] == pytest.approx(10.0)


def test_advantage_stats_all_zero_is_degenerate():
    s = dp.advantage_stats(np.zeros(200), np.zeros(200))
    assert s["adv_frac_zero"] == 1.0
    assert s["adv_std"] == 0.0
    assert dp.verdict_adv(s)[0] == "FAIL"


def test_advantage_stats_healthy_spread_passes():
    rng = np.random.default_rng(6)
    adv = rng.normal(size=5000)
    s = dp.advantage_stats(adv, adv + 1.0)
    assert s["adv_frac_zero"] == 0.0
    assert dp.verdict_adv(s)[0] == "PASS"


def test_verdict_adv_partial_sparsity_is_weak():
    adv = np.concatenate([np.zeros(300), np.ones(700)])  # 30% exactly zero
    s = dp.advantage_stats(adv, adv)
    assert s["adv_frac_zero"] == pytest.approx(0.3)
    assert dp.verdict_adv(s)[0] == "WEAK"


# ---------------------------------------------------------------- sparsity

def test_reward_sparsity_dense_signal():
    r = np.full((4, 5), 0.01)
    s = dp.reward_sparsity(r, event_scale=1.0)
    assert s["n"] == 20
    assert s["frac_nonzero"] == 1.0
    assert s["frac_event"] == 0.0
    assert s["mean_abs"] == pytest.approx(0.01)


def test_reward_sparsity_counts_goal_scale_events_separately():
    r = np.zeros(1000)
    r[:10] = 0.05          # shaping
    r[10:13] = [10.0, -8.0, 10.0]  # goal-scale
    s = dp.reward_sparsity(r, event_scale=1.0)
    assert s["frac_nonzero"] == pytest.approx(0.013)
    assert s["frac_event"] == pytest.approx(0.003)
    assert s["max_abs"] == pytest.approx(10.0)


def test_reward_sparsity_all_zero():
    s = dp.reward_sparsity(np.zeros(64))
    assert s["frac_nonzero"] == 0.0
    assert s["frac_event"] == 0.0
    assert s["mean"] == 0.0


# ------------------------------------------------------------ action distribution

def test_action_stats_uniform_hits_max_entropy():
    n_actions = 92
    actions = np.tile(np.arange(n_actions), 100)
    s = dp.action_stats(actions, n_actions)
    assert s["entropy"] == pytest.approx(math.log(n_actions))
    assert s["entropy_ratio"] == pytest.approx(1.0)
    assert s["n_never"] == 0
    assert s["n_cover"] == pytest.approx(83, abs=1)  # ceil(0.9 * 92)
    assert dp.verdict_actions(s)[0] == "FAIL"        # ~uniform == not a policy


def test_action_stats_collapsed_policy():
    actions = np.zeros(1000, dtype=np.int64)
    s = dp.action_stats(actions, 92)
    assert s["entropy"] == pytest.approx(0.0)
    assert s["top1_share"] == pytest.approx(1.0)
    assert s["n_cover"] == 1
    assert s["n_never"] == 91
    assert dp.verdict_actions(s)[0] == "FAIL"


def test_action_stats_concentrated_policy_passes():
    # 4 actions with ~equal mass out of 92 -> H = ln(4) = 1.386, ratio 0.31
    actions = np.tile(np.arange(4), 250)
    s = dp.action_stats(actions, 92)
    assert s["entropy"] == pytest.approx(math.log(4))
    assert s["n_cover"] == 4
    assert s["n_never"] == 88
    assert s["top_k_share"] == pytest.approx(1.0)
    assert dp.verdict_actions(s)[0] == "PASS"


def test_action_stats_near_uniform_is_weak_before_it_fails():
    # 92 actions, one of them slightly favored: ratio lands in the 0.85-0.95 band
    rng = np.random.default_rng(7)
    p = np.full(92, 1.0 / 92.0)
    p[0] = 0.20
    p[1:] = 0.80 / 91.0
    actions = rng.choice(92, size=100_000, p=p)
    s = dp.action_stats(actions, 92)
    assert dp.ENT_HIGH_WEAK < s["entropy_ratio"] <= dp.ENT_HIGH_FAIL
    assert dp.verdict_actions(s)[0] == "WEAK"


def test_action_stats_rejects_out_of_range_action_id():
    with pytest.raises(AssertionError):
        dp.action_stats(np.array([0, 1, 200]), 92)


def test_action_stats_without_per_state_entropy_leaves_mi_unset():
    s = dp.action_stats(np.tile(np.arange(4), 100), 92)
    assert s["per_state_entropy"] is None
    assert s["mi_ratio"] is None


def test_state_dependence_is_marginal_minus_per_state_entropy():
    actions = np.tile(np.arange(92), 100)          # marginal H = ln(92)
    s = dp.action_stats(actions, 92, per_state_entropy=math.log(92) - 1.0)
    assert s["state_dependence"] == pytest.approx(1.0)
    assert s["mi_ratio"] == pytest.approx(1.0 / math.log(92))
    assert s["per_state_ratio"] == pytest.approx((math.log(92) - 1.0) / math.log(92))


def test_state_blind_policy_fails_even_at_moderate_entropy():
    # marginal == per-state => I(S;A) = 0: the policy samples the same
    # distribution in every state. Entropy alone (60% of uniform) looks fine.
    actions = np.tile(np.arange(20), 500)  # H = ln(20) = 3.00 = 0.66 * ln(92)
    s = dp.action_stats(actions, 92, per_state_entropy=math.log(20))
    assert s["state_dependence"] == pytest.approx(0.0)
    assert 0.15 < s["per_state_ratio"] < 0.85    # entropy band alone would PASS
    assert dp.verdict_actions(s)[0] == "FAIL"


def test_partially_state_dependent_policy_is_weak():
    actions = np.tile(np.arange(92), 100)
    # I(S;A) = 0.6 nats of ln(92)=4.52 -> 13%: above the FAIL floor, below PASS
    s = dp.action_stats(actions, 92, per_state_entropy=math.log(92) - 0.6)
    assert dp.MI_FAIL < s["mi_ratio"] < dp.MI_PASS
    assert dp.verdict_actions(s)[0] == "WEAK"


def test_sharp_state_dependent_policy_passes():
    actions = np.tile(np.arange(92), 100)          # marginal H = ln(92) = 4.52
    s = dp.action_stats(actions, 92, per_state_entropy=2.0)  # per-state much sharper
    assert s["mi_ratio"] > dp.MI_PASS
    assert dp.verdict_actions(s)[0] == "PASS"


def test_verdict_actions_prefers_per_state_entropy_over_marginal():
    # marginal is ~uniform (would FAIL on its own) but each state is sharp and
    # I(S;A) is large: this is a healthy policy visiting diverse states.
    actions = np.tile(np.arange(92), 100)
    s = dp.action_stats(actions, 92, per_state_entropy=1.5)
    assert s["entropy_ratio"] == pytest.approx(1.0)   # marginal alone: FAIL
    assert dp.verdict_actions(s)[0] == "PASS"


def test_verdict_actions_flags_an_over_sharp_policy_as_weak():
    # per-state entropy 0.5 nats of 4.52 = 11%: below the low-entropy band even
    # though I(S;A) is huge. Near-deterministic is its own failure mode.
    s = dp.action_stats(np.tile(np.arange(92), 100), 92, per_state_entropy=0.5)
    assert s["per_state_ratio"] < dp.ENT_LOW_WEAK
    assert dp.verdict_actions(s)[0] == "WEAK"


# ------------------------------------------------------------------ value scale

def test_value_scale_perfect_match():
    rng = np.random.default_rng(8)
    ret = rng.normal(loc=2.0, scale=3.0, size=2000)
    s = dp.value_scale_stats(ret, ret)
    assert s["std_ratio"] == pytest.approx(1.0)
    assert s["bias_in_ret_std"] == pytest.approx(0.0)
    assert s["corr"] == pytest.approx(1.0)
    assert s["rmse"] == pytest.approx(0.0)
    assert dp.verdict_value_scale(s)[0] == "PASS"


def test_value_scale_wrong_magnitude_fails():
    rng = np.random.default_rng(9)
    ret = rng.normal(scale=1.0, size=2000)
    s = dp.value_scale_stats(ret * 10.0, ret)  # value head stuck on an old reward scale
    assert s["std_ratio"] == pytest.approx(10.0, rel=0.05)
    assert dp.verdict_value_scale(s)[0] == "FAIL"


def test_value_scale_biased_but_right_shape_is_weak():
    rng = np.random.default_rng(10)
    ret = rng.normal(scale=1.0, size=4000)
    s = dp.value_scale_stats(ret + 1.0, ret)  # 1.0 return-std of pure offset
    assert s["std_ratio"] == pytest.approx(1.0, rel=0.05)
    assert s["bias_in_ret_std"] == pytest.approx(1.0, rel=0.05)
    assert dp.verdict_value_scale(s)[0] == "WEAK"


def test_value_scale_rejects_shape_mismatch():
    with pytest.raises(AssertionError):
        dp.value_scale_stats(np.zeros(3), np.zeros(4))


# -------------------------------------------------------------------- verdicts

@pytest.mark.parametrize("ev,expected", [
    (0.95, "PASS"), (0.80, "PASS"), (0.79, "WEAK"), (0.30, "WEAK"),
    (0.29, "FAIL"), (0.05, "FAIL"), (-1.0, "FAIL"), (float("nan"), "FAIL"),
])
def test_verdict_ev_bands(ev, expected):
    label, thr = dp.verdict_ev(ev)
    assert label == expected
    assert "ev" in thr  # the threshold text is printed with the verdict


def test_verdict_ratio_band_edges():
    inside = {"mean_delta": 1e-5, "frac_outside": 1e-4}
    weak = {"mean_delta": 1e-3, "frac_outside": 1e-3}
    bad = {"mean_delta": 0.2, "frac_outside": 0.5}
    assert dp.verdict_ratio(inside)[0] == "PASS"
    assert dp.verdict_ratio(weak)[0] == "WEAK"
    assert dp.verdict_ratio(bad)[0] == "FAIL"
    # a tail-only failure (no mean offset) must not be excused by the mean test
    assert dp.verdict_ratio({"mean_delta": 0.0, "frac_outside": 0.4})[0] == "FAIL"


def test_verdict_actions_bands():
    def s(ratio):
        # no per-state entropy available -> falls back to the marginal, and the
        # state-dependence floor is not applied
        return {"entropy_ratio": ratio}
    assert dp.verdict_actions(s(0.99))[0] == "FAIL"
    assert dp.verdict_actions(s(0.90))[0] == "WEAK"
    assert dp.verdict_actions(s(0.50))[0] == "PASS"
    assert dp.verdict_actions(s(0.10))[0] == "WEAK"
    assert dp.verdict_actions(s(0.01))[0] == "FAIL"
    assert dp.verdict_actions(s(float("nan")))[0] == "FAIL"


def test_overall_verdict_is_worst_of():
    assert dp.overall_verdict(["PASS", "PASS"]) == "PASS"
    assert dp.overall_verdict(["PASS", "WEAK"]) == "WEAK"
    assert dp.overall_verdict(["WEAK", "FAIL", "PASS"]) == "FAIL"


# ---------------------------------------------------- rendering (no engine needed)

def _fake_result(ck="ck_a.pt", ev=0.5):
    rng = np.random.default_rng(11)
    ret = rng.normal(size=500)
    val = ret * 0.7
    adv = rng.normal(size=500)
    acts = rng.integers(0, 92, size=500)
    ratio = dp.ratio_delta_stats(np.zeros(500), np.zeros(500))
    d = {
        "ck": ck, "total_steps": 320_471_040, "reward_config": "configs/reward_v3.toml",
        "curriculum_config": "configs/curriculum_v1.toml",
        "ck_reward_config": "configs/reward_v3.toml",
        "T": 4, "N": 125, "rows": 500, "arenas": 8, "gamma": 0.99, "lam": 0.95,
        "device": "cpu", "collect_s": 1.0, "fwd_s": 0.5, "wall_s": 2.0,
        "episodes": 7, "ep_reward_mean": 0.5, "ev": ev, "ev_mc": ev - 0.2,
        "ratio": ratio,
        "adv": dp.advantage_stats(adv, ret),
        "sparsity": dp.reward_sparsity(rng.normal(size=500)),
        "actions": dp.action_stats(acts, 92, per_state_entropy=3.5),
        "vscale": dp.value_scale_stats(val, ret),
        "vparity": ratio, "torch_entropy": 3.5,
    }
    d["labels"] = {"ev": dp.verdict_ev(ev)[0]}
    d["overall"] = d["labels"]["ev"]
    return d


def test_render_emits_every_check_and_a_verdict_per_check():
    out = dp.render(_fake_result())
    for header in ("[1] VALUE-HEAD EXPLAINED VARIANCE", "[2] IMPORTANCE-RATIO",
                   "[3] ADVANTAGE", "[4] ACTION DISTRIBUTION",
                   "[5] VALUE PREDICTION SCALE", "OVERALL:", "ev_mc"):
        assert header in out
    assert out.count("verdict:") == 5


def test_render_compare_lines_up_both_checkpoints():
    out = dp.render_compare(_fake_result("ck_a.pt", 0.9), _fake_result("ck_b.pt", 0.01))
    assert "COMPARISON" in out
    assert "ck_a.pt" in out and "ck_b.pt" in out
    assert "explained variance" in out
