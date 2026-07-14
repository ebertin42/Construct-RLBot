import sys

import numpy as np

sys.path.insert(0, "deploy")
from actions import make_lookup_table


def test_action_row_to_controller_semantics():
    t = make_lookup_table()
    # row 0: full reverse + left, no boost/jump/handbrake
    row = t[0]
    assert row[0] == -1 and row[1] == -1
    assert not any(row[5:8])
    # every row: bounded controls
    assert np.all(np.abs(t[:, :5]) <= 1.0)
    assert set(np.unique(t[:, 5:8])) <= {0.0, 1.0}


def test_state_dict_obs_smoke():
    from obs import build_obs
    state = {
        "ball": {"pos": [0, 0, 93.15], "vel": [0, 0, 0], "ang_vel": [0, 0, 0]},
        "cars": [
            {"id": 1, "team": 0, "pos": [0, -4608, 17], "vel": [0, 0, 0], "ang_vel": [0, 0, 0],
             "forward": [0, 1, 0], "up": [0, 0, 1], "boost": 33.3,
             "is_on_ground": True, "has_flip": True, "is_demoed": False},
            {"id": 2, "team": 1, "pos": [0, 4608, 17], "vel": [0, 0, 0], "ang_vel": [0, 0, 0],
             "forward": [0, -1, 0], "up": [0, 0, 1], "boost": 33.3,
             "is_on_ground": True, "has_flip": True, "is_demoed": False},
        ],
    }
    o1 = build_obs(state, 1, 1 / 2300, 1 / 2300, 1 / 5.5)
    o2 = build_obs(state, 2, 1 / 2300, 1 / 2300, 1 / 5.5)
    assert o1.shape == (94,)
    np.testing.assert_allclose(o1, o2, atol=1e-6)  # mirrored symmetric state
