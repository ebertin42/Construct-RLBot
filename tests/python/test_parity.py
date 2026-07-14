import json
import sys

import numpy as np

sys.path.insert(0, "deploy")
from obs import build_obs  # deploy/obs.py
from construct._engine import Engine, schema_dict


def test_deploy_obs_matches_engine_obs_exactly():
    eng = Engine(num_arenas=2, blue=1, orange=1, schema_path="schema/v0.toml",
                 reward_config_path="configs/reward_v0.toml", seed=99)
    eng.reset()
    rng = np.random.default_rng(1)
    s = schema_dict("schema/v0.toml")
    for _ in range(30):  # step to varied states
        eng.step(rng.integers(0, 90, size=eng.num_agents).astype(np.int64))
    for arena in range(2):
        state_json, rust_obs = eng.debug_state_and_obs(arena)
        state = json.loads(state_json)
        for i, car in enumerate(state["cars"]):
            py_obs = build_obs(state, car["id"], s["pos_norm"], s["vel_norm"], s["ang_vel_norm"])
            diff = np.max(np.abs(py_obs - rust_obs[i]))
            assert diff < 1e-5, f"arena {arena} car {car['id']}: max diff {diff}"


def test_action_tables_match():
    from actions import make_lookup_table  # deploy/actions.py
    from construct._engine import action_table
    t = make_lookup_table()
    assert t.shape == (90, 8)
    # spot-check against the Rust-side contract rows (see Task 3 tests)
    np.testing.assert_array_equal(t[0], [-1, -1, 0, -1, 0, 0, 0, 0])
    # full parity: every row of deploy's table matches the Rust engine's table exactly
    np.testing.assert_array_equal(action_table(), t)
