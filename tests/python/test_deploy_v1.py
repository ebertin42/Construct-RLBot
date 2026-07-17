"""Golden parity for the v1 (entity-transformer) deploy path.

Verifies deploy/{actions,obs,model}.py against the Rust engine WITHOUT rlbot:
  (a) deploy's 92-row action table == construct._engine.action_table_v1()
  (b) deploy's build_obs_v1 reproduces a real bc-export golden
      (replay/src/bc_obs.rs output) row-for-row, including the orange-POV
      mirror path and the prev-5 ring semantics
  (c) a real v1 checkpoint loads through deploy/model.load_policy and runs
      forward on the parity obs

deploy/bot.py (the rlbot/rlgym_compat adapter) is deliberately NOT imported —
rlbot is Windows-only; those seams are covered by the README's
live-verification checklist.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_num_threads(1)

sys.path.insert(0, "deploy")
from actions import make_lookup_table, make_lookup_table_v1  # deploy/actions.py
from model import load_policy  # deploy/model.py
from obs import (  # deploy/obs.py
    BIG_PAD_COUNT,
    CANONICAL_PAD_LOCATIONS,
    ENT_FEAT,
    MAX_ENT,
    PAD_COUNT,
    PAD_MIRROR_PERM,
    PREV_ACTIONS,
    Q_FEAT,
    build_obs_v1,
    pad_order_mapping,
    update_prev_ring,
)

SHARD_DIR = Path("data/shards_v4")
BC_DIR = Path("data/bc")
CK_DIR = Path("checkpoints_entity")


def _find_pair():
    """First shard stem present in both data/shards_v4 and data/bc."""
    if not (SHARD_DIR.is_dir() and BC_DIR.is_dir()):
        return None
    for p in sorted(SHARD_DIR.glob("*.npz")):
        bc = BC_DIR / f"bc_{p.stem}.npz"
        if bc.exists() and p.with_suffix(".json").exists():
            return p, bc
    return None


PAIR = _find_pair()
CKS = sorted(CK_DIR.glob("ck_*.pt")) if CK_DIR.is_dir() else []
needs_pair = pytest.mark.skipif(PAIR is None, reason="no shard+bc-export pair on disk")


# --------------------------------------------------------------- (a) actions

def test_action_table_v1_matches_engine():
    from construct._engine import action_table_v1

    t = make_lookup_table_v1()
    assert t.shape == (92, 8)
    np.testing.assert_array_equal(t[:90], make_lookup_table())  # append-only
    np.testing.assert_array_equal(t[90], [0, 0, 0, 1, -1, 1, 0, 1])  # stall: yaw=-roll
    np.testing.assert_array_equal(t[91], [0, 0, 0, -1, 1, 1, 0, 1])
    np.testing.assert_array_equal(np.asarray(action_table_v1(), dtype=np.float32), t)


def test_pad_constants_are_sane():
    # canonical order: 6 big (z=73) then 28 small (z=70)
    assert CANONICAL_PAD_LOCATIONS.shape == (PAD_COUNT, 3)
    assert (CANONICAL_PAD_LOCATIONS[:BIG_PAD_COUNT, 2] == 73.0).all()
    assert (CANONICAL_PAD_LOCATIONS[BIG_PAD_COUNT:, 2] == 70.0).all()
    # mirror perm is an involution that exactly negates xy
    perm = PAD_MIRROR_PERM
    assert (perm[perm] == np.arange(PAD_COUNT)).all()
    np.testing.assert_allclose(
        CANONICAL_PAD_LOCATIONS[perm][:, :2], -CANONICAL_PAD_LOCATIONS[:, :2], atol=1e-6
    )
    # rlgym/rlbot pad ordering (rlgym_compat common_values.BOOST_LOCATIONS,
    # incl. its (-940, 3310) 2uu quirk) maps to a bijection with the 6 big
    # pads landing on canonical 0..6
    rlgym_order = np.array(
        [(0.0, -4240.0, 70.0), (-1792.0, -4184.0, 70.0), (1792.0, -4184.0, 70.0),
         (-3072.0, -4096.0, 73.0), (3072.0, -4096.0, 73.0), (-940.0, -3308.0, 70.0),
         (940.0, -3308.0, 70.0), (0.0, -2816.0, 70.0), (-3584.0, -2484.0, 70.0),
         (3584.0, -2484.0, 70.0), (-1788.0, -2300.0, 70.0), (1788.0, -2300.0, 70.0),
         (-2048.0, -1036.0, 70.0), (0.0, -1024.0, 70.0), (2048.0, -1036.0, 70.0),
         (-3584.0, 0.0, 73.0), (-1024.0, 0.0, 70.0), (1024.0, 0.0, 70.0),
         (3584.0, 0.0, 73.0), (-2048.0, 1036.0, 70.0), (0.0, 1024.0, 70.0),
         (2048.0, 1036.0, 70.0), (-1788.0, 2300.0, 70.0), (1788.0, 2300.0, 70.0),
         (-3584.0, 2484.0, 70.0), (3584.0, 2484.0, 70.0), (0.0, 2816.0, 70.0),
         (-940.0, 3310.0, 70.0), (940.0, 3308.0, 70.0), (-3072.0, 4096.0, 73.0),
         (3072.0, 4096.0, 73.0), (-1792.0, 4184.0, 70.0), (1792.0, 4184.0, 70.0),
         (0.0, 4240.0, 70.0)],
        dtype=np.float32,
    )
    mapping = pad_order_mapping(rlgym_order)
    assert sorted(mapping.tolist()) == list(range(PAD_COUNT))
    big = {3, 4, 15, 18, 29, 30}  # rlgym LARGE_BOOST_INDICES
    assert {int(mapping[i]) for i in big} == set(range(BIG_PAD_COUNT))


# ------------------------------------------------------------------ (b) obs

def _quat_rotate(q: np.ndarray, v) -> np.ndarray:
    """v' = q * v * q^-1 for q = [x,y,z,w] — the exact cross-product expansion
    of replay/src/reconstruct.rs::rotate_vector, in f32."""
    q = np.asarray(q, dtype=np.float32)
    v = np.asarray(v, dtype=np.float32)
    qv, qw = q[:3], q[3]
    t = np.float32(2.0) * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def _load_shard():
    npz = np.load(PAIR[0])
    sidecar = json.loads(PAIR[0].with_suffix(".json").read_text())
    return npz, sidecar


def _state_for_tick(sh, t: int) -> dict:
    """Rebuild the plain-dict state deploy/obs.py consumes from shard columns,
    mirroring replay/src/bc_obs.rs's GameState reconstruction (car id = p+1,
    quat -> forward/up, boost 0..1 -> 0..100, has_flip -> HasFlipOrJump,
    pad timer_norm -> cooldown seconds)."""
    cars = []
    for p in range(sh["cars_state"].shape[1]):
        row = sh["cars_state"][t, p]
        on_ground = bool(row[14] != 0.0)
        cars.append({
            "id": p + 1,
            "team": int(sh["player_teams"][p]),
            "pos": row[0:3], "vel": row[3:6], "ang_vel": row[6:9],
            "forward": _quat_rotate(row[9:13], [1.0, 0.0, 0.0]),
            "up": _quat_rotate(row[9:13], [0.0, 0.0, 1.0]),
            "boost": float(np.clip(row[13] * np.float32(100.0), 0.0, 100.0)),
            "is_on_ground": on_ground,
            # engine f22 = CarState::has_flip_or_jump() = on_ground || stored flag
            "has_flip": bool(on_ground or row[16] != 0.0),
            "is_demoed": bool(row[15] != 0.0),
        })
    ball = sh["ball"][t]
    pads = sh["pads"][t]  # [34,2] = (timer_norm = cooldown/10, is_active)
    active = pads[:, 1] != 0.0
    cooldown = np.where(active, np.float32(0.0), pads[:, 0] * np.float32(10.0))
    return {
        "ball": {"pos": ball[0:3], "vel": ball[3:6], "ang_vel": ball[6:9]},
        "cars": cars,
        "pads_cooldown": cooldown.astype(np.float32),
        "pads_active": active,
        "ball_pred": sh["ball_pred"][t],
    }


@needs_pair
def test_obs_v1_matches_bc_export_golden():
    sh, sidecar = _load_shard()
    bc = np.load(PAIR[1])
    t_count, p_count = sh["cars_state"].shape[:2]
    assert bc["ents"].shape == (t_count * p_count, MAX_ENT, ENT_FEAT)
    teams = set(sh["player_teams"].tolist())
    assert teams == {0, 1}, "need a blue AND an orange car to cover the mirror path"

    # spread of ticks + the first rows + last row
    ticks = sorted(set([0, 1, 2, 3] + list(range(0, t_count, max(1, t_count // 40))) + [t_count - 1]))
    for t in ticks:
        state = _state_for_tick(sh, t)
        for p in range(p_count):
            s = t * p_count + p
            ents, mask, query = build_obs_v1(state, car_id=p + 1)
            np.testing.assert_array_equal(
                mask.astype(np.uint8), bc["mask"][s], err_msg=f"mask t={t} p={p}"
            )
            np.testing.assert_allclose(
                ents, bc["ents"][s], atol=1e-5, err_msg=f"ents t={t} p={p}"
            )
            np.testing.assert_allclose(
                query, bc["query"][s], atol=1e-5, err_msg=f"query t={t} p={p}"
            )


@needs_pair
def test_prev_ring_matches_bc_export_golden():
    """deploy's update_prev_ring, driven with bc_obs.rs's reset rule (zeros at
    t=0 and on tick_index gaps), reproduces the exported prev array exactly."""
    sh, sidecar = _load_shard()
    bc = np.load(PAIR[1])
    stride = int(sidecar["stride"])
    t_count, p_count = sh["cars_action_idx"].shape
    tick_index = sh["tick_index"]
    rings = np.zeros((p_count, PREV_ACTIONS), dtype=np.int64)
    expect = np.zeros((t_count * p_count, PREV_ACTIONS), dtype=np.int64)
    for t in range(t_count):
        if t > 0 and tick_index[t] - tick_index[t - 1] != stride:
            rings[:] = 0
        expect[t * p_count:(t + 1) * p_count] = rings
        for p in range(p_count):
            update_prev_ring(rings[p], int(sh["cars_action_idx"][t, p]))
    np.testing.assert_array_equal(expect, bc["prev"])


# ---------------------------------------------------------------- (c) model

@pytest.mark.skipif(not CKS, reason="no v1 checkpoint in checkpoints_entity")
@needs_pair
def test_v1_checkpoint_runs_on_parity_obs():
    net, version = load_policy(str(CKS[-1]))
    assert version == 1
    assert tuple(net.action_table.shape) == (92, 8)

    bc = np.load(PAIR[1])
    n = 64
    with torch.no_grad():
        logits, value = net(
            torch.from_numpy(bc["ents"][:n]),
            torch.from_numpy(bc["mask"][:n].astype(bool)),
            torch.from_numpy(bc["query"][:n]),
            torch.from_numpy(bc["prev"][:n]),
        )
    assert logits.shape == (n, 92) and value.shape == (n, 1)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    picks = logits.argmax(-1)
    assert ((picks >= 0) & (picks < 92)).all()
    # deterministic: same input -> same argmax (deploy acts greedily)
    with torch.no_grad():
        logits2, _ = net(
            torch.from_numpy(bc["ents"][:n]),
            torch.from_numpy(bc["mask"][:n].astype(bool)),
            torch.from_numpy(bc["query"][:n]),
            torch.from_numpy(bc["prev"][:n]),
        )
    assert torch.equal(picks, logits2.argmax(-1))
