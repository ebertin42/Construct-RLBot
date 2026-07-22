"""Generate (state, expected_107_obs) fixtures for the Rust AdvancedObs golden
test. Reproduces rlmarlbot/immortal/obs/advanced_obs.py math, INCLUDING the exact
boost-pad reorder (engine canonical order -> rlgym BOOST_LOCATIONS order) and the
orange inversion (rlgym order is reverse-antisymmetric, so inverted = reverse).
Weights-free -- safe to commit.

Pad handling (the parity-critical part):
  obs_pad[j] = canonical_active[ rlgym_to_canon[ (33-j) if orange else j ] ]
where rlgym_to_canon[j] = canonical pad nearest BOOST_LOCATIONS[j]. The Rust
builder derives the SAME permutation from the live pad positions + the vendored
BOOST_LOCATIONS, so this fixture validates the full 107 floats incl. pads.

Normalizers: POS_STD=2300 (pos & vel), ANG_STD=pi (ang vel). Orange inversion of
vectors = negate x,y keep z (matches engine obs::mir)."""
import json, math, pathlib, random

POS_STD, ANG_STD = 2300.0, math.pi

# rlgym BOOST_LOCATIONS (rlgym_compat.common_values, verified) -- the order
# Immortal's obs `pads` are in.
BOOST_LOCATIONS = [
    [0.0, -4240.0, 70.0], [-1792.0, -4184.0, 70.0], [1792.0, -4184.0, 70.0],
    [-3072.0, -4096.0, 73.0], [3072.0, -4096.0, 73.0], [-940.0, -3308.0, 70.0],
    [940.0, -3308.0, 70.0], [0.0, -2816.0, 70.0], [-3584.0, -2484.0, 70.0],
    [3584.0, -2484.0, 70.0], [-1788.0, -2300.0, 70.0], [1788.0, -2300.0, 70.0],
    [-2048.0, -1036.0, 70.0], [0.0, -1024.0, 70.0], [2048.0, -1036.0, 70.0],
    [-3584.0, 0.0, 73.0], [-1024.0, 0.0, 70.0], [1024.0, 0.0, 70.0],
    [3584.0, 0.0, 73.0], [-2048.0, 1036.0, 70.0], [0.0, 1024.0, 70.0],
    [2048.0, 1036.0, 70.0], [-1788.0, 2300.0, 70.0], [1788.0, 2300.0, 70.0],
    [-3584.0, 2484.0, 70.0], [3584.0, 2484.0, 70.0], [0.0, 2816.0, 70.0],
    [-940.0, 3310.0, 70.0], [940.0, 3308.0, 70.0], [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0], [-1792.0, 4184.0, 70.0], [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
]

# Canonical (RocketSim arena) pad order: 6 big (LOCS_BIG_SOCCAR) then 28 small.
# The order engine `state.pads` is in. Vendored from deploy/obs.py.
CANONICAL_PAD_LOCATIONS = [
    [-3584.0, 0.0, 73.0], [3584.0, 0.0, 73.0], [-3072.0, 4096.0, 73.0],
    [3072.0, 4096.0, 73.0], [-3072.0, -4096.0, 73.0], [3072.0, -4096.0, 73.0],
    [0.0, -4240.0, 70.0], [-1792.0, -4184.0, 70.0], [1792.0, -4184.0, 70.0],
    [-940.0, -3308.0, 70.0], [940.0, -3308.0, 70.0], [0.0, -2816.0, 70.0],
    [-3584.0, -2484.0, 70.0], [3584.0, -2484.0, 70.0], [-1788.0, -2300.0, 70.0],
    [1788.0, -2300.0, 70.0], [-2048.0, -1036.0, 70.0], [0.0, -1024.0, 70.0],
    [2048.0, -1036.0, 70.0], [-1024.0, 0.0, 70.0], [1024.0, 0.0, 70.0],
    [-2048.0, 1036.0, 70.0], [0.0, 1024.0, 70.0], [2048.0, 1036.0, 70.0],
    [-1788.0, 2300.0, 70.0], [1788.0, 2300.0, 70.0], [-3584.0, 2484.0, 70.0],
    [3584.0, 2484.0, 70.0], [0.0, 2816.0, 70.0], [-940.0, 3308.0, 70.0],
    [940.0, 3308.0, 70.0], [-1792.0, 4184.0, 70.0], [1792.0, 4184.0, 70.0],
    [0.0, 4240.0, 70.0],
]


def rlgym_to_canon():
    perm = []
    for bx, by, _ in BOOST_LOCATIONS:
        best, bd = 0, 1e30
        for i, (cx, cy, _) in enumerate(CANONICAL_PAD_LOCATIONS):
            d = (cx - bx) ** 2 + (cy - by) ** 2
            if d < bd:
                bd, best = d, i
        assert bd < 50.0 ** 2, f"pad match too far: {bd**0.5:.1f}uu"
        perm.append(best)
    assert len(set(perm)) == 34, "not a bijection"
    return perm


R2C = rlgym_to_canon()


def mir(v, m):
    return [-v[0], -v[1], v[2]] if m else [v[0], v[1], v[2]]


def div(v, k):
    return [v[0] / k, v[1] / k, v[2] / k]


def sub(a, b):
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]


def add_player(obs, car, ball_pos, ball_vel, m):
    pos = mir(car["pos"], m); vel = mir(car["vel"], m)
    fwd = mir(car["forward"], m); up = mir(car["up"], m); av = mir(car["ang_vel"], m)
    bp = mir(ball_pos, m); bv = mir(ball_vel, m)
    obs += div(sub(bp, pos), POS_STD)
    obs += div(sub(bv, vel), POS_STD)
    obs += div(pos, POS_STD)
    obs += fwd
    obs += up
    obs += div(vel, POS_STD)
    obs += div(av, ANG_STD)
    obs += [car["boost"], float(car["on_ground"]), float(car["has_flip"]), float(car["is_demoed"])]
    return pos, vel


def build_obs(state, self_idx):
    self_car = state["cars"][self_idx]
    m = self_car["team"] == 1
    obs = []
    obs += div(mir(state["ball"]["pos"], m), POS_STD)
    obs += div(mir(state["ball"]["vel"], m), POS_STD)
    obs += div(mir(state["ball"]["ang_vel"], m), ANG_STD)
    obs += state["prev_action"]
    # pads: reorder canonical -> rlgym; orange = reverse (inverted_boost_pads)
    canon_active = state["pad_active"]
    for j in range(34):
        rj = (33 - j) if m else j
        obs.append(float(canon_active[R2C[rj]]))
    self_pos, self_vel = add_player(obs, self_car, state["ball"]["pos"], state["ball"]["vel"], m)
    for i, car in enumerate(state["cars"]):
        if i == self_idx:
            continue
        opos, ovel = add_player(obs, car, state["ball"]["pos"], state["ball"]["vel"], m)
        obs += div(sub(mir(car["pos"], m), self_pos), POS_STD)
        obs += div(sub(mir(car["vel"], m), self_vel), POS_STD)
    return obs


def rnd_vec(r, s=1.0):
    return [r.uniform(-s, s), r.uniform(-s, s), r.uniform(-s, s)]


def rnd_car(r, team):
    on_ground = r.random() > 0.5
    # has_flip_or_jump() = on_ground || (!has_flipped && !has_double_jumped), so it
    # is forced True whenever on_ground. The fixture must respect that constraint
    # (state_from_fixture sets the underlying fields to reproduce this value).
    has_flip = True if on_ground else (r.random() > 0.5)
    return {"pos": rnd_vec(r, 4000), "vel": rnd_vec(r, 2000), "ang_vel": rnd_vec(r, 5),
            "forward": rnd_vec(r, 1), "up": rnd_vec(r, 1),
            "boost": r.uniform(0, 1), "on_ground": on_ground,
            "has_flip": has_flip, "is_demoed": r.random() > 0.5, "team": team}


def main():
    r = random.Random(1234)
    cases = []
    for _ in range(16):
        for self_idx, _ in ((0, 0), (1, 1)):   # blue + orange perspective
            state = {
                "cars": [rnd_car(r, 0), rnd_car(r, 1)],
                "ball": {"pos": rnd_vec(r, 4000), "vel": rnd_vec(r, 2000), "ang_vel": rnd_vec(r, 5)},
                "pad_active": [1 if r.random() > 0.5 else 0 for _ in range(34)],
                "prev_action": [r.uniform(-1, 1) for _ in range(8)],
            }
            obs = build_obs(state, self_idx)
            assert len(obs) == 107, len(obs)
            cases.append({"state": state, "self_idx": self_idx, "obs": obs})
    out = pathlib.Path("engine/tests/fixtures/immortal_obs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"pad_positions": CANONICAL_PAD_LOCATIONS, "cases": cases}))
    print(f"wrote {out} ({len(cases)} cases, 107 floats each)")


if __name__ == "__main__":
    main()
