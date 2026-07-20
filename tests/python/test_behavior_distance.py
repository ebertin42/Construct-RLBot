"""Tests for scripts/behavior_distance.py.

The engine-touching parts (collect_states, build_net) are exercised by the
smoke test at the bottom, which is skipped when the compiled engine or the
champion checkpoint isn't present. Everything else is pure and tested here --
these metrics are the ones that will be quoted in the journal, so their sign
and argument order need pinning.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import behavior_distance as bd  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CHAMPION = REPO / "checkpoints_entity" / "ck_000320471040.pt"


# --- KL ---------------------------------------------------------------------

def test_kl_of_a_distribution_with_itself_is_zero():
    logits = torch.randn(64, 92)
    assert bd.kl_divergence(logits, logits).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_is_non_negative():
    for _ in range(5):
        a, b = torch.randn(32, 92), torch.randn(32, 92)
        assert bd.kl_divergence(a, b).item() >= -1e-6


def test_kl_is_asymmetric_and_the_argument_order_is_the_documented_one():
    """KL(P||Q) != KL(Q||P). If these ever come out equal the implementation
    has been symmetrised by accident and the reported number stops meaning
    what the kl_prior term penalises."""
    # NOT mirror images: [[10,0,0]] vs [[0,0,10]] are symmetric under swapping
    # the two arguments, so they give EQUAL KLs in both directions and would
    # pass this test against a wrongly-symmetrised implementation.
    p = torch.tensor([[10.0, 0.0, 0.0]])
    q = torch.tensor([[0.0, 1.0, 2.0]])
    assert bd.kl_divergence(p, q).item() != pytest.approx(bd.kl_divergence(q, p).item())


def test_kl_grows_as_the_distributions_separate():
    base = torch.zeros(1, 3)
    near = torch.tensor([[1.0, 0.0, 0.0]])
    far = torch.tensor([[20.0, 0.0, 0.0]])
    assert bd.kl_divergence(far, base) > bd.kl_divergence(near, base)


def test_kl_is_shift_invariant_because_logits_are():
    """Adding a constant to a row of logits is the same distribution. A KL that
    moves under that shift would be reading raw logits, not the softmax."""
    a, b = torch.randn(16, 92), torch.randn(16, 92)
    shifted = a + 7.5
    assert bd.kl_divergence(shifted, b).item() == pytest.approx(
        bd.kl_divergence(a, b).item(), abs=1e-5)


def test_kl_survives_logits_large_enough_to_overflow_a_naive_exp():
    """exp(800) is inf in f32/f64; a log-space implementation is fine."""
    big = torch.tensor([[800.0, 799.0, 0.0]])
    other = torch.tensor([[0.0, 1.0, 2.0]])
    v = bd.kl_divergence(big, other).item()
    assert v == v and v != float("inf"), "KL must not be NaN/inf on extreme logits"


# --- agreement --------------------------------------------------------------

def test_agreement_is_one_for_identical_policies():
    logits = torch.randn(128, 92)
    assert bd.agreement_rate(logits, logits) == 1.0


def test_agreement_is_zero_when_argmaxes_never_coincide():
    a = torch.tensor([[5.0, 0.0], [5.0, 0.0]])
    b = torch.tensor([[0.0, 5.0], [0.0, 5.0]])
    assert bd.agreement_rate(a, b) == 0.0


def test_agreement_ignores_confidence_while_kl_does_not():
    """The whole reason both metrics are reported: a policy that keeps its
    ranking but sharpens has agreement 1.0 and KL > 0. Conflating the two would
    read 'changed its certainty' as 'changed its mind'."""
    a = torch.tensor([[1.0, 0.0, 0.0]])
    sharper = torch.tensor([[8.0, 0.0, 0.0]])
    assert bd.agreement_rate(a, sharper) == 1.0
    assert bd.kl_divergence(sharper, a).item() > 0.1


# --- entropy ----------------------------------------------------------------

def test_entropy_of_a_uniform_policy_is_log_n():
    import math
    logits = torch.zeros(4, 92)
    assert bd.entropy(logits).item() == pytest.approx(math.log(92), abs=1e-5)


def test_entropy_of_a_collapsed_policy_is_near_zero():
    logits = torch.full((4, 92), -50.0)
    logits[:, 0] = 50.0
    assert bd.entropy(logits).item() == pytest.approx(0.0, abs=1e-5)


def test_entropy_separates_collapse_from_uniform_at_similar_kl():
    """Both a collapsed and a uniform candidate can sit far from a mid-entropy
    champion; entropy is what tells the two apart in the report."""
    champ = torch.randn(1, 92) * 0.5
    collapsed = torch.full((1, 92), -50.0)
    collapsed[:, 0] = 50.0
    uniform = torch.zeros(1, 92)
    assert bd.entropy(collapsed) < bd.entropy(champ) < bd.entropy(uniform)


# --- summary ----------------------------------------------------------------

def test_summarize_pair_reports_both_kl_directions():
    a, b = torch.randn(32, 92), torch.randn(32, 92)
    s = bd.summarize_pair(a, b)
    assert set(s) == {"kl_cand_champ", "kl_champ_cand", "agreement", "h_cand", "h_champ"}
    assert s["kl_cand_champ"] != pytest.approx(s["kl_champ_cand"])


def test_summarize_pair_of_identical_policies_is_the_null_result():
    logits = torch.randn(32, 92)
    s = bd.summarize_pair(logits, logits)
    assert s["kl_cand_champ"] == pytest.approx(0.0, abs=1e-6)
    assert s["agreement"] == 1.0
    assert s["h_cand"] == pytest.approx(s["h_champ"])


# --- engine smoke -----------------------------------------------------------

@pytest.mark.skipif(not CHAMPION.exists(), reason="champion checkpoint not present")
def test_build_net_round_trips_the_champion():
    net = bd.build_net(str(CHAMPION))
    assert next(net.parameters()).requires_grad is True
    assert not net.training, "net must be in eval mode before scoring"


@pytest.mark.skipif(not CHAMPION.exists(), reason="champion checkpoint not present")
def test_champion_against_itself_is_exactly_zero_divergence():
    """End-to-end null control: the same checkpoint loaded twice must produce
    identical logits on the same states. If this ever fails, something is
    non-deterministic and every number this tool prints is suspect."""
    pytest.importorskip("construct._engine")
    batch = bd.collect_states(str(CHAMPION), n_states=256, arenas=4, seed=11)
    net_a, net_b = bd.build_net(str(CHAMPION)), bd.build_net(str(CHAMPION))
    s = bd.summarize_pair(bd.logits_on(net_a, batch), bd.logits_on(net_b, batch))
    assert s["kl_cand_champ"] == pytest.approx(0.0, abs=1e-6)
    assert s["agreement"] == 1.0
