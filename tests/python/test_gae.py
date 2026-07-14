import numpy as np
from construct.learn.gae import compute_gae


def test_gae_matches_hand_computation():
    # T=3, N=1, gamma=0.5, lam=1.0 (plain discounted advantage), no dones
    rewards = np.array([[1.0], [1.0], [1.0]], dtype=np.float32)
    values = np.array([[0.0], [0.0], [0.0]], dtype=np.float32)
    final_values = np.zeros((3, 1), dtype=np.float32)
    term = np.zeros((3, 1), dtype=bool)
    trunc = np.zeros((3, 1), dtype=bool)
    # bootstrap value after last step = 0 (values of step T would be needed;
    # convention: caller appends next_value row to values -> shape (T+1, N))
    values_ext = np.vstack([values, np.zeros((1, 1), dtype=np.float32)])
    adv, ret = compute_gae(rewards, values_ext, final_values, term, trunc, gamma=0.5, lam=1.0)
    # deltas: r + 0.5*V' - V = [1,1,1]; adv_2=1, adv_1=1+0.5*1=1.5, adv_0=1+0.5*1.5=1.75
    np.testing.assert_allclose(adv[:, 0], [1.75, 1.5, 1.0], atol=1e-6)
    np.testing.assert_allclose(ret[:, 0], adv[:, 0] + values[:, 0], atol=1e-6)


def test_truncation_bootstraps_final_obs_value_and_blocks_next_episode():
    rewards = np.array([[1.0], [1.0]], dtype=np.float32)
    values_ext = np.array([[0.0], [0.0], [9.0]], dtype=np.float32)  # V(post-reset obs) = 9
    final_values = np.array([[0.0], [5.0]], dtype=np.float32)       # V(final obs of step 1) = 5
    term = np.array([[False], [False]])
    trunc = np.array([[False], [True]])
    adv, _ = compute_gae(rewards, values_ext, final_values, term, trunc, gamma=1.0, lam=1.0)
    # step 1 (truncated): delta_1 = 1 + V(final_obs)=5 - 0 = 6; done_1 blocks flow
    #   from the NEXT episode (post-reset V=9 must not leak in) -> adv_1 = 6
    # step 0 (same episode as step 1): delta_0 = 1 + values[1]=0 - 0 = 1;
    #   done_0=False so adv_1 flows back: adv_0 = 1 + 1*1*6 = 7
    np.testing.assert_allclose(adv[:, 0], [7.0, 6.0], atol=1e-6)
