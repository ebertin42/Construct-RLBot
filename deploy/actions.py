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


def make_lookup_table_v1_air() -> np.ndarray:
    """v1-air action table: v1.1's 92 rows APPENDED with a 12-row clean-air block.

    Mirrors engine/src/actions.rs make_lookup_table_v1_air exactly
    (parity-tested in tests/python/test_deploy_v1.py). Append-only, so indices
    0..92 keep their v1.1 meaning byte-for-byte.

    The appended rows are the clean-air grid -- pitch in {-1,0,+1} x boost in
    {0,1}, jump=1, steer=yaw=roll=HANDBRAKE=0 -- emitted twice. v1.1 cannot
    express "jump without air roll" at any index, because its handbrake is
    derived from rotation (`handbrake = jump and (pitch or yaw or roll)` in
    make_lookup_table above) and its aerial loop also skips `jump and yaw`.
    That leaves 2/92 clean-jump rows and makes a sustained takeoff effectively
    unsamplable; the block takes clean jumps to 14/104 and the probability of
    holding one for three consecutive decisions from 1.03e-5 to 2.44e-3.
    Doubled per ZealanL's RLGym-PPO-Guide ("doubling jump actions in discrete
    action parsers"); duplicate rows are a free prior because EntityPolicyNet
    computes logits from the action-table ROW, not a per-slot parameter.
    Row layout: [throttle, steer, pitch, yaw, roll, jump, boost, handbrake].
    """
    air = []
    for _ in range(2):
        for pitch in (-1, 0, 1):
            for boost in (0, 1):
                #          throttle steer  pitch  yaw roll jump boost handbrake
                air.append([boost, 0, pitch, 0, 0, 1, boost, 0])
    return np.vstack([make_lookup_table_v1(), np.array(air, dtype=np.float32)])
