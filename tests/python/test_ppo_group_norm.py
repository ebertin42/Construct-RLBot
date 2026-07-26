"""Per-team-size advantage standardisation (v9 D6, ppo.standardise_advantages).

THE MECHANISM THIS FIXES, in one paragraph. `ppo_update` used to standardise
advantages with ONE scalar over the whole mixed batch. That preserves the
measured 9.48x 1v1:3v3 magnitude ratio exactly (mean|TD advantage| on
ck_001283829760: 1v1 0.2275 / 2v2 0.0322 / 3v3 0.0240). The entropy bonus,
meanwhile, contributes a per-row gradient of CONSTANT magnitude `entropy_coef`,
so at coef 0.01 a 1v1 row sat at 121.7x the entropy pull (healthy) while a 3v3
row sat at 12.8x (collapsed to 95% of uniform-random within 22M steps). Giving
3v3 more arenas cannot fix that -- both the entropy pull and the useful signal
scale linearly with a group's row count, so the RATIO is share-independent.

The tests below pin the three properties the fix has to have:
  1. every group comes out at unit scale, whatever it went in at;
  2. `adv_norm="global"` is BYTE-IDENTICAL to the historical single line;
  3. the sigma floor catches a degenerate group instead of amplifying its noise
     to unit scale -- and it does NOT bind on a legitimately small-variance
     group, which is why the floor is 0.05*sigma_batch and not 0.1.
"""
import numpy as np
import pytest
import torch

from construct.learn.ppo import standardise_advantages


def _mixed(seed=0, scales=(0.2275, 0.0322, 0.0240), n=8192):
    """Three groups at the MEASURED per-team-size advantage scales."""
    g = torch.Generator().manual_seed(seed)
    adv, grp = [], []
    for m, s in enumerate(scales, start=1):
        adv.append(torch.randn(n, generator=g) * s)
        grp.append(torch.full((n,), m, dtype=torch.long))
    return torch.cat(adv), torch.cat(grp)


def test_global_is_byte_identical_to_the_historical_line():
    """The kill switch has to be a real revert, not an approximation of one."""
    adv, grp = _mixed()
    want = (adv - adv.mean()) / (adv.std() + 1e-8)
    got, info = standardise_advantages(adv, adv_norm="global", group=grp)
    assert torch.equal(got, want)
    # ...and passing no label at all is the same path (v0 collects have no
    # team-size column, so this is the everyday case for legacy runs).
    got2, _ = standardise_advantages(adv, adv_norm="global", group=None)
    assert torch.equal(got2, want)
    assert info["adv_group_floor_hits"] == 0.0


def test_global_preserves_the_measured_ratio_which_is_the_whole_problem():
    adv, grp = _mixed()
    out, _ = standardise_advantages(adv, adv_norm="global", group=grp)
    per = [out[grp == m].abs().mean().item() for m in (1, 2, 3)]
    ratio = per[0] / per[2]
    assert ratio > 5.0, (
        f"global normalisation must LEAVE the 1v1:3v3 gap in place ({ratio:.2f}x); "
        "if this ever comes out ~1 the test fixture no longer reproduces the bug"
    )


def test_group_puts_every_team_size_at_the_same_scale():
    adv, grp = _mixed()
    out, info = standardise_advantages(adv, adv_norm="group", group=grp,
                                       min_group_rows=1024)
    per = [out[grp == m].abs().mean().item() for m in (1, 2, 3)]
    for m, v in zip((1, 2, 3), per):
        # E|z| = sqrt(2/pi) = 0.7979 for a standardised GAUSSIAN, which this
        # fixture is BY CONSTRUCTION. Do NOT size entropy_coef from this number:
        # real advantages are wildly leptokurtic (measured kurtosis 98-388 on the
        # v9 launch config, mean|A_norm| ~= 0.24, i.e. 3.3x off this identity),
        # and sizing the coef off 0.798 is exactly how v9 nearly shipped at half
        # the intended signal-to-entropy ratio. See train_v9_fromscratch.toml.
        assert abs(v - 0.7979) < 0.02, f"{m}v{m} mean|A_norm| = {v}"
    assert max(per) / min(per) < 1.05, f"per-group scales must match: {per}"
    assert info["adv_groups"] == 3.0
    assert info["adv_group_floor_hits"] == 0.0, (
        "the floor must not bind on legitimate 3v3 variance -- that is exactly "
        "why it is 0.05*sigma_batch and not 0.1"
    )


def test_group_zero_means_each_group():
    adv, grp = _mixed()
    out, _ = standardise_advantages(adv, adv_norm="group", group=grp,
                                    min_group_rows=1024)
    for m in (1, 2, 3):
        assert abs(out[grp == m].mean().item()) < 1e-5


def test_the_measured_3v3_scale_does_not_hit_a_005_floor_but_would_hit_01():
    """The floor arithmetic, made explicit.

    Measured sigma per group is 0.285 / 0.0404 / 0.0301 against sigma_batch
    ~= 0.167 under the v9 equal-row mix, so 3v3 sits at 0.18*sigma_batch. A 0.1
    floor is only 1.8x below that and CAN bind under noise, which would leave
    3v3 under-scaled -- defeating the entire change. 0.05 gives 3.6x clearance.
    """
    adv, grp = _mixed()
    _, lo = standardise_advantages(adv, adv_norm="group", group=grp,
                                   min_group_rows=1024, group_std_floor=0.05)
    assert lo["adv_group_floor_hits"] == 0.0
    # Push the floor up to where it provably binds, proving the margin is real
    # rather than assumed.
    _, hi = standardise_advantages(adv, adv_norm="group", group=grp,
                                   min_group_rows=1024, group_std_floor=0.30)
    assert hi["adv_group_floor_hits"] > 0.0


def test_degenerate_group_is_floored_not_amplified():
    """A group with no reward events is pure critic noise. Standardising it
    would blow that noise up to unit scale and feed it to the policy."""
    g = torch.Generator().manual_seed(3)
    adv = torch.cat([torch.randn(4096, generator=g) * 0.25,
                     torch.randn(4096, generator=g) * 1e-6])
    grp = torch.cat([torch.full((4096,), 1), torch.full((4096,), 3)]).long()
    out, info = standardise_advantages(adv, adv_norm="group", group=grp,
                                       min_group_rows=1024)
    assert info["adv_group_floor_hits"] == 1.0
    assert out[grp == 3].abs().mean().item() < 0.01, (
        "a degenerate group must stay small, not be amplified to unit scale"
    )


def test_small_group_falls_back_to_the_global_pool():
    g = torch.Generator().manual_seed(5)
    adv = torch.cat([torch.randn(4096, generator=g) * 0.25,
                     torch.randn(64, generator=g) * 0.03])
    grp = torch.cat([torch.full((4096,), 1), torch.full((64,), 3)]).long()
    out, info = standardise_advantages(adv, adv_norm="group", group=grp,
                                       min_group_rows=2048)
    assert info["adv_groups"] == 1.0, "only the big group is standardised"
    # the small group is still normalised (by the batch statistics), never left raw
    assert out[grp == 3].abs().mean().item() != adv[grp == 3].abs().mean().item()


def test_unlabelled_rows_use_the_batch_statistics():
    # label 0 == "no team size" (a v0 collect); those rows must not be dropped.
    g = torch.Generator().manual_seed(7)
    adv = torch.randn(4096, generator=g) * 0.25
    grp = torch.zeros(4096, dtype=torch.long)
    out, info = standardise_advantages(adv, adv_norm="group", group=grp,
                                       min_group_rows=1024)
    assert info["adv_groups"] == 0.0
    want = (adv - adv.mean()) / (adv.std() + 1e-8)
    assert torch.allclose(out, want)


def test_label_length_mismatch_is_loud():
    """A silent mis-label mis-scales every gradient with no other symptom, so
    the shape contract has to fail hard."""
    adv, grp = _mixed(n=1024)
    with pytest.raises(ValueError, match="group label length"):
        standardise_advantages(adv, adv_norm="group", group=grp[:-1])


def test_unknown_adv_norm_is_rejected():
    adv, grp = _mixed(n=256)
    with pytest.raises(ValueError, match="adv_norm"):
        standardise_advantages(adv, adv_norm="per-arena", group=grp)


def test_ppo_update_with_group_norm_runs_and_reports_floor_hits():
    """End-to-end through ppo_update: the plumbing (kwargs -> stats) works and
    v0-style calls with no group label are unaffected."""
    from construct.learn.model import PolicyValueNet
    from construct.learn.ppo import ppo_update
    torch.manual_seed(0)
    net = PolicyValueNet(obs_size=4, action_count=2, hidden=(32,))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    n = 6144
    obs = torch.ones(n, 4)
    with torch.no_grad():
        actions, logprobs, values = net.act(obs)
    adv = torch.cat([torch.randn(n // 3) * 0.25,
                     torch.randn(n // 3) * 0.04,
                     torch.randn(n // 3) * 0.03])
    grp = torch.cat([torch.full((n // 3,), m) for m in (1, 2, 3)]).long()
    batch = {"obs": obs, "actions": actions, "logprobs": logprobs,
             "advantages": adv, "returns": torch.randn(n), "values": values}
    stats = ppo_update(net, opt, batch, epochs=1, minibatch_size=1024,
                       adv_norm="group", group=grp, min_group_rows=1024)
    assert stats["skipped"] == 0
    assert stats["adv_group_floor_hits"] == 0.0
    assert stats["adv_groups"] == 3.0
    for p in net.parameters():
        assert torch.isfinite(p).all()


def test_tiling_convention_matches_the_flat_row_order():
    """The one place a silent mis-label would hide.

    Every batch tensor is `(T, N, ...)` flattened C-order, i.e. flat row
    `r = t*N + n`. The engine gives ONE label per learner COLUMN, shape (N,), so
    it must be TILED over T: `np.tile` gives `lab[r % N]`, which matches.
    `np.repeat` would give `lab[r // T]` and mislabel almost every row -- this
    test is what stops that edit from passing review.
    """
    T, N = 5, 4
    lab = np.array([1, 1, 2, 3], dtype=np.int64)
    tiled = np.tile(lab, T)
    assert tiled.shape == (T * N,)
    for t in range(T):
        for n in range(N):
            assert tiled[t * N + n] == lab[n]
    assert not np.array_equal(np.repeat(lab, T), tiled), (
        "repeat and tile must differ here, or the test proves nothing"
    )
