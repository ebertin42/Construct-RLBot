import numpy as np


def compute_gae(
    rewards: np.ndarray,      # (T, N)
    values: np.ndarray,       # (T+1, N) — includes V(s_{T}) bootstrap row
    final_values: np.ndarray, # (T, N) — V(final_obs) rows valid where truncated
    terminated: np.ndarray,   # (T, N) bool
    truncated: np.ndarray,    # (T, N) bool
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        done = terminated[t] | truncated[t]
        # value after this transition:
        #  - terminated: 0
        #  - truncated: V(final_obs) (pre-reset state)
        #  - otherwise: V(next obs) = values[t+1]
        next_v = np.where(terminated[t], 0.0, np.where(truncated[t], final_values[t], values[t + 1]))
        delta = rewards[t] + gamma * next_v - values[t]
        last = delta + gamma * lam * (~done) * last
        adv[t] = last
    returns = adv + values[:T]
    return adv, returns
