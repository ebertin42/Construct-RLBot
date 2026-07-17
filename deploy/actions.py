import numpy as np


def make_lookup_table() -> np.ndarray:
    actions = []
    # Ground
    for throttle in (-1, 0, 1):
        for steer in (-1, 0, 1):
            for boost in (0, 1):
                for handbrake in (0, 1):
                    if boost == 1 and throttle != 1:
                        continue
                    actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
    # Aerial
    for pitch in (-1, 0, 1):
        for yaw in (-1, 0, 1):
            for roll in (-1, 0, 1):
                for jump in (0, 1):
                    for boost in (0, 1):
                        if jump == 1 and yaw != 0:
                            continue
                        if pitch == roll == jump == 0:
                            continue
                        handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                        actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
    return np.array(actions, dtype=np.float32)


def make_lookup_table_v1() -> np.ndarray:
    """v1.1 action table: the 90 v0 rows APPENDED with 2 stall rows at 90/91.

    Mirrors engine/src/actions.rs make_lookup_table_v1 exactly (parity-tested
    in tests/python/test_deploy_v1.py). Stall = jump with yaw = -roll, so the
    dodge direction (-pitch, yaw + roll) is zero -> no flip impulse.
    Row layout: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
    """
    stalls = np.array(
        [
            [0.0, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.vstack([make_lookup_table(), stalls])
