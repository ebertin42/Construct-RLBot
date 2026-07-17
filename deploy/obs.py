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


# --------------------------------------------------------------------------
# obs v1 (entity transformer). Source of truth: engine/src/obs_v1.rs; the
# semantics of shard->obs mapping are mirrored from replay/src/bc_obs.rs.
# Parity-tested row-for-row against real bc-export goldens in
# tests/python/test_deploy_v1.py.
# --------------------------------------------------------------------------

MAX_ENT = 17
ENT_FEAT = 26
Q_FEAT = 64
PREV_ACTIONS = 5
PAD_COUNT = 34
NUM_PRED = 4
BALL_PRED_HORIZONS_SEC = (0.5, 1.0, 1.5, 2.0)

# Normalization constants, vendored from schema/v1.toml [normalization]
# (pos_norm/vel_norm = 1/2300, ang_vel_norm = 1/5.5). Keep in sync manually;
# the golden parity test catches drift.
POS_NORM_V1 = np.float32(1.0 / 2300.0)
VEL_NORM_V1 = np.float32(1.0 / 2300.0)
ANG_VEL_NORM_V1 = np.float32(1.0 / 5.5)

# Entity slot layout (engine/src/obs_v1.rs):
# [0] self | [1..3) mates asc id | [3..6) opps asc id | [6] ball
# [7..13) big pads (canonical arena order) | [13..17) ball-pred asc horizon
SELF_IDX = 0
MATES_START = 1
MAX_MATES = 2
OPPS_START = 3
MAX_OPPS = 3
BALL_IDX = 6
PADS_START = 7
BIG_PAD_COUNT = 6
PRED_START = 13

# Canonical 34 boost-pad positions in RocketSim's fixed arena-construction
# order: the 6 big pads (RLConst::BoostPads::LOCS_BIG_SOCCAR) followed by the
# 28 small pads (LOCS_SMALL_SOCCAR). Vendored from rocketsim_rs 0.37.0,
# RocketSim/src/RLConst.h + Arena.cpp's boost-pad init loop (isBig = i < 6).
# This is the order shard `pads` rows are stored in and the order obs_v1's
# pad entities / query pad-timers use.
CANONICAL_PAD_LOCATIONS = np.array(
    [
        # big (LOCS_BIG_SOCCAR)
        [-3584.0, 0.0, 73.0],
        [3584.0, 0.0, 73.0],
        [-3072.0, 4096.0, 73.0],
        [3072.0, 4096.0, 73.0],
        [-3072.0, -4096.0, 73.0],
        [3072.0, -4096.0, 73.0],
        # small (LOCS_SMALL_SOCCAR)
        [0.0, -4240.0, 70.0],
        [-1792.0, -4184.0, 70.0],
        [1792.0, -4184.0, 70.0],
        [-940.0, -3308.0, 70.0],
        [940.0, -3308.0, 70.0],
        [0.0, -2816.0, 70.0],
        [-3584.0, -2484.0, 70.0],
        [3584.0, -2484.0, 70.0],
        [-1788.0, -2300.0, 70.0],
        [1788.0, -2300.0, 70.0],
        [-2048.0, -1036.0, 70.0],
        [0.0, -1024.0, 70.0],
        [2048.0, -1036.0, 70.0],
        [-1024.0, 0.0, 70.0],
        [1024.0, 0.0, 70.0],
        [-2048.0, 1036.0, 70.0],
        [0.0, 1024.0, 70.0],
        [2048.0, 1036.0, 70.0],
        [-1788.0, 2300.0, 70.0],
        [1788.0, 2300.0, 70.0],
        [-3584.0, 2484.0, 70.0],
        [3584.0, 2484.0, 70.0],
        [0.0, 2816.0, 70.0],
        [-940.0, 3308.0, 70.0],
        [940.0, 3308.0, 70.0],
        [-1792.0, 4184.0, 70.0],
        [1792.0, 4184.0, 70.0],
        [0.0, 4240.0, 70.0],
    ],
    dtype=np.float32,
)

# Per-canonical-index respawn time: 10 s big, 4 s small (RLConst.h
# COOLDOWN_BIG/COOLDOWN_SMALL). Used by the bot adapter to convert RLBot's
# "seconds SINCE pickup" timer into RocketSim's "seconds UNTIL respawn".
PAD_RECHARGE_SECONDS = np.where(np.arange(PAD_COUNT) < BIG_PAD_COUNT, 10.0, 4.0).astype(np.float32)


def _pad_mirror_perm() -> np.ndarray:
    """perm[i] = canonical index of the pad nearest (-x_i, -y_i) — the same
    involution engine/src/obs_v1.rs::mirror_pad_perm computes (xy distance
    only). Static because pad positions never move."""
    xy = CANONICAL_PAD_LOCATIONS[:, :2]
    d2 = ((xy[:, None, :] * -1.0 - xy[None, :, :]) ** 2).sum(-1)  # d2[i,j] = |(-p_i) - p_j|^2
    return np.argmin(d2, axis=1).astype(np.int64)


PAD_MIRROR_PERM = _pad_mirror_perm()


def pad_order_mapping(pad_positions: np.ndarray) -> np.ndarray:
    """Maps an external pad ordering (e.g. RLBot field_info.boost_pads) to
    canonical indices by nearest xy position: mapping[i] = canonical index of
    external pad i. RLBot/rlgym_compat pad order is NOT arena order, and one
    standard-map pad differs by ~2uu between conventions ((-940, 3310) vs
    (-940, 3308)), hence nearest-match with a tolerance instead of equality."""
    pos = np.asarray(pad_positions, dtype=np.float32)
    assert pos.ndim == 2 and pos.shape[1] >= 2, f"pad_positions shape {pos.shape}"
    d2 = ((pos[:, None, :2] - CANONICAL_PAD_LOCATIONS[None, :, :2]) ** 2).sum(-1)
    mapping = np.argmin(d2, axis=1).astype(np.int64)
    worst = float(np.sqrt(d2[np.arange(len(pos)), mapping].max()))
    assert worst < 50.0, f"pad position match too far off ({worst:.1f}uu) — non-standard map?"
    assert len(set(mapping.tolist())) == len(pos), "pad mapping is not a bijection"
    return mapping


def _write_car_row(row: np.ndarray, onehot, car: dict, mirror: bool):
    """One 26-float car entity row (engine/src/obs_v1.rs EntityRow::write)."""
    row[0:5] = onehot
    row[5:8] = _mir(np.asarray(car["pos"], np.float32), mirror) * POS_NORM_V1
    row[8:11] = _mir(np.asarray(car["vel"], np.float32), mirror) * VEL_NORM_V1
    row[11:14] = _mir(np.asarray(car["ang_vel"], np.float32), mirror) * ANG_VEL_NORM_V1
    row[14:17] = _mir(np.asarray(car["forward"], np.float32), mirror)
    row[17:20] = _mir(np.asarray(car["up"], np.float32), mirror)
    row[20] = np.float32(car["boost"]) / np.float32(100.0)
    row[21] = float(car["is_on_ground"])
    row[22] = float(car["has_flip"])  # HasFlipOrJump semantics — see bot.py adapter
    row[23] = float(car["is_demoed"])
    # row[24] horizon, row[25] reserved: stay 0


def _timer_norm(cooldown: float, active: bool) -> np.float32:
    """0.0 if active, else cooldown/10 — always the BIG pad constant, even for
    small pads (engine/src/obs_v1.rs timer_norm, a deliberate simplification)."""
    return np.float32(0.0) if active else np.float32(cooldown) / np.float32(10.0)


def build_obs_v1(state: dict, car_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Builds (ents [17,26] f32, mask [17] bool True=absent, query [64] f32)
    for one car — the exact layout of engine/src/obs_v1.rs::build.

    `state` is a plain dict (testable without rlbot):
      ball: {pos, vel, ang_vel}                      raw world units
      cars: [{id, team (0 blue/1 orange), pos, vel, ang_vel, forward, up,
              boost (0..100), is_on_ground, has_flip, is_demoed}]
      pads_cooldown: [34] f32 seconds until respawn (0 while active),
                     canonical arena order (CANONICAL_PAD_LOCATIONS)
      pads_active:   [34] bool, canonical arena order
      ball_pred:     [4,6] f32 raw pos3+vel3 at +0.5/1/1.5/2 s

    prev-5 action indices ride separately (they feed the net directly, not
    these tensors) — see update_prev_ring.
    """
    ents = np.zeros((MAX_ENT, ENT_FEAT), dtype=np.float32)
    mask = np.ones(MAX_ENT, dtype=bool)  # True = absent/masked
    query = np.zeros(Q_FEAT, dtype=np.float32)

    cars = {c["id"]: c for c in state["cars"]}
    me = cars[car_id]
    mirror = me["team"] == 1  # orange plays as blue

    # --- self ---
    _write_car_row(ents[SELF_IDX], (1.0, 0.0, 0.0, 0.0, 0.0), me, mirror)
    mask[SELF_IDX] = False

    # --- mates / opps, ascending car id ---
    mates = sorted((c for c in state["cars"] if c["id"] != car_id and c["team"] == me["team"]),
                   key=lambda c: c["id"])
    for slot, c in enumerate(mates[:MAX_MATES]):
        _write_car_row(ents[MATES_START + slot], (0.0, 1.0, 0.0, 0.0, 0.0), c, mirror)
        mask[MATES_START + slot] = False
    opps = sorted((c for c in state["cars"] if c["team"] != me["team"]), key=lambda c: c["id"])
    for slot, c in enumerate(opps[:MAX_OPPS]):
        _write_car_row(ents[OPPS_START + slot], (0.0, 0.0, 1.0, 0.0, 0.0), c, mirror)
        mask[OPPS_START + slot] = False

    # --- ball ---
    b = state["ball"]
    row = ents[BALL_IDX]
    row[0:5] = (0.0, 0.0, 0.0, 1.0, 0.0)
    row[5:8] = _mir(np.asarray(b["pos"], np.float32), mirror) * POS_NORM_V1
    row[8:11] = _mir(np.asarray(b["vel"], np.float32), mirror) * VEL_NORM_V1
    row[11:14] = _mir(np.asarray(b["ang_vel"], np.float32), mirror) * ANG_VEL_NORM_V1
    mask[BALL_IDX] = False

    # --- big pads (canonical arena order 0..6) ---
    # When mirroring, slot k's SOURCE pad is PAD_MIRROR_PERM[k]: the pad that
    # physically sits at the mirrored location. Position AND timer/is_active
    # all come from that one source pad (obs_v1.rs "Mirror-permutation").
    cooldown = np.asarray(state["pads_cooldown"], dtype=np.float32)
    active = np.asarray(state["pads_active"], dtype=bool)
    assert cooldown.shape == (PAD_COUNT,) and active.shape == (PAD_COUNT,)
    for slot in range(BIG_PAD_COUNT):
        src = int(PAD_MIRROR_PERM[slot]) if mirror else slot
        row = ents[PADS_START + slot]
        row[0:5] = (0.0, 0.0, 0.0, 0.0, 1.0)
        row[5:8] = _mir(CANONICAL_PAD_LOCATIONS[src], mirror) * POS_NORM_V1
        row[20] = _timer_norm(cooldown[src], bool(active[src]))
        row[21] = float(active[src])
        mask[PADS_START + slot] = False

    # --- ball prediction (ascending horizon; feature = tau/2 -> .25/.5/.75/1) ---
    pred = np.asarray(state["ball_pred"], dtype=np.float32)
    assert pred.shape == (NUM_PRED, 6), f"ball_pred shape {pred.shape}"
    for slot in range(NUM_PRED):
        row = ents[PRED_START + slot]
        row[5:8] = _mir(pred[slot, 0:3], mirror) * POS_NORM_V1
        row[8:11] = _mir(pred[slot, 3:6], mirror) * VEL_NORM_V1
        row[24] = (slot + 1) * 0.25
        mask[PRED_START + slot] = False

    # --- query: self row (26) + 34 pad timers + 3 scoreboard + 1 reserved ---
    # Scoreboard floats stay 0.0: the net was trained (engine + bc-export)
    # with no score/clock state, so writing real values here would be
    # out-of-distribution. See engine/src/obs_v1.rs "Scoreboard decision".
    query[0:ENT_FEAT] = ents[SELF_IDX]
    for k in range(PAD_COUNT):
        src = int(PAD_MIRROR_PERM[k]) if mirror else k
        query[ENT_FEAT + k] = _timer_norm(cooldown[src], bool(active[src]))
    return ents, mask, query


def update_prev_ring(prev: np.ndarray, action_idx: int) -> np.ndarray:
    """Shifts the executed action into the prev-5 ring IN PLACE, newest-first
    (engine/src/episode.rs semantics: prev[0] = last executed action). Call
    AFTER building the obs for the current step — the obs at t must only see
    actions from t-1 back. Reset to zeros at episode boundaries."""
    assert prev.shape == (PREV_ACTIONS,)
    prev[1:] = prev[:-1]
    prev[0] = action_idx
    return prev
