import numpy as np

OBS_SIZE = 94
MAX_OTHERS = 5


def _mir(v: np.ndarray, mirror: bool) -> np.ndarray:
    return np.array([-v[0], -v[1], v[2]], dtype=np.float32) if mirror else v.astype(np.float32)


def build_obs(state: dict, car_id: int, pos_norm: float, vel_norm: float, ang_vel_norm: float) -> np.ndarray:
    pk, vk, ak = np.float32(pos_norm), np.float32(vel_norm), np.float32(ang_vel_norm)
    out = np.zeros(OBS_SIZE, dtype=np.float32)
    cars = {c["id"]: c for c in state["cars"]}
    me = cars[car_id]
    mirror = me["team"] == 1  # orange
    i = 0

    def put3(v, k):
        nonlocal i
        out[i : i + 3] = _mir(np.asarray(v, dtype=np.float32), mirror) * k
        i += 3

    def put(x):
        nonlocal i
        out[i] = np.float32(x)
        i += 1

    put3(me["pos"], pk); put3(me["forward"], np.float32(1.0)); put3(me["up"], np.float32(1.0))
    put3(me["vel"], vk); put3(me["ang_vel"], ak)
    put(me["boost"] / 100.0); put(float(me["is_on_ground"]))
    put(float(me["has_flip"])); put(float(me["is_demoed"]))

    b = state["ball"]
    put3(b["pos"], pk); put3(b["vel"], vk); put3(b["ang_vel"], ak)
    rel_p = np.asarray(b["pos"], np.float32) - np.asarray(me["pos"], np.float32)
    rel_v = np.asarray(b["vel"], np.float32) - np.asarray(me["vel"], np.float32)
    put3(rel_p, pk); put3(rel_v, vk)

    others = [c for c in state["cars"] if c["id"] != car_id]
    others.sort(key=lambda c: (c["team"] != me["team"], c["id"]))
    for c in others[:MAX_OTHERS]:
        put3(c["pos"], pk); put3(c["vel"], vk); put3(c["forward"], np.float32(1.0))
        put(c["boost"] / 100.0); put(float(c["team"] == me["team"])); put(float(not c["is_demoed"]))
    return out
