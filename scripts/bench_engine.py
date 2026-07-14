import time
import numpy as np
from construct._engine import Engine

eng = Engine(num_arenas=64, blue=1, orange=1)
eng.reset()
acts = np.random.default_rng(0).integers(0, 90, size=eng.num_agents).astype(np.int64)
t0 = time.perf_counter()
N = 2000
for _ in range(N):
    eng.step(acts)
dt = time.perf_counter() - t0
print(f"{N * eng.num_agents / dt:,.0f} agent-steps/sec ({N * 64 / dt:,.0f} env-steps/sec)")
