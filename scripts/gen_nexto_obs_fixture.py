"""Generate golden fixtures for the Rust Nexto/Necto obs builder.

Reference is the REAL NextoObsBuilder from deploy/external/nexto/nexto_obs.py --
not a re-derivation. (Re-deriving is how you get a silently wrong port; see the
boost-pad ordering trap in docs/foreign-opponents.md.) Its rlgym_compat imports
need `rlbot`, which we don't have, so they are stubbed: the builder only uses
those names for type hints on paths we don't call.

We synthesise `encoded_states` directly (the packed rlgym-compat state array),
since batched_build_obs consumes that rather than a GameState. Layout used:
    [3:37]         34 boost pad values
    [37:37+9]      ball POS(3), LIN_VEL(3), ANG_VEL(3)
    [55 + 38*i]    player i, with
                     +1 team, +2:5 pos, +5:9 quat, +9:12 lin_vel,
                     +12:15 ang_vel, +33 demo, +34 on_ground,
                     +36 has_flip, +37 boost

The fixture also stores each car's forward/up vectors as derived by the
reference's own quaternion->rotation conversion, because our engine carries a
rotation matrix directly and never sees quaternions.

Output: engine/tests/fixtures/nexto_obs.json  (weights-free, safe to commit)
"""
import json
import pathlib
import sys
import types

import numpy as np
import torch

BALL_START = 37
PLAYER_START = 55
PLAYER_LEN = 38


def _stub_rlgym_compat():
    """Insert a minimal fake rlgym_compat so nexto_obs imports."""
    pkg = types.ModuleType("rlgym_compat")
    v1 = types.ModuleType("rlgym_compat.v1_game_state")
    class _Stub:  # noqa: D401 - placeholder types only used in annotations
        pass
    v1.V1GameState = _Stub
    v1.V1PlayerData = _Stub
    pkg.v1_game_state = v1
    sys.modules.setdefault("rlgym_compat", pkg)
    sys.modules.setdefault("rlgym_compat.v1_game_state", v1)


def load_builder():
    _stub_rlgym_compat()
    sys.path.insert(0, "deploy/external/nexto")
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "nexto_obs", "deploy/external/nexto/nexto_obs.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


_MODEL = None


def _model(path="deploy/external/nexto/nexto-model.pt"):
    global _MODEL
    if _MODEL is None:
        _MODEL = {}
    if path not in _MODEL:
        m = torch.jit.load(path)
        m.eval()
        _MODEL[path] = m
    return _MODEL[path]


def rand_quat(rng):
    q = rng.standard_normal(4)
    return q / np.linalg.norm(q)


def main():
    m = load_builder()
    rng = np.random.default_rng(20260722)
    n_players = 2
    width = PLAYER_START + PLAYER_LEN * n_players

    cases = []
    for _ in range(8):
        enc = np.zeros((1, width), dtype=np.float64)
        pads = (rng.random(34) > 0.5).astype(np.float64)
        enc[0, 3:37] = pads
        ball_pos = rng.uniform(-4000, 4000, 3)
        ball_vel = rng.uniform(-2000, 2000, 3)
        ball_ang = rng.uniform(-5, 5, 3)
        enc[0, BALL_START:BALL_START + 9] = np.concatenate([ball_pos, ball_vel, ball_ang])

        cars = []
        for i in range(n_players):
            base = PLAYER_START + PLAYER_LEN * i
            team = float(i)                      # car 0 blue, car 1 orange
            pos = rng.uniform(-4000, 4000, 3)
            quat = rand_quat(rng)
            lin = rng.uniform(-2000, 2000, 3)
            ang = rng.uniform(-5, 5, 3)
            boost = float(rng.uniform(0, 1))   # rlgym convention: fraction in [0,1]
            demo = 0.0
            on_ground = float(rng.random() > 0.5)
            # Physically constrained: a car on the ground always has its jump.
            # The engine derives has_flip_or_jump() = on_ground || (...), so an
            # (on_ground=1, has_flip=0) state is unreachable and must not be
            # synthesised or the golden test compares an impossible case.
            has_flip = 1.0 if on_ground else float(rng.random() > 0.5)
            enc[0, base + 1] = team
            enc[0, base + 2:base + 5] = pos
            enc[0, base + 5:base + 9] = quat
            enc[0, base + 9:base + 12] = lin
            enc[0, base + 12:base + 15] = ang
            enc[0, base + 33] = demo
            enc[0, base + 34] = on_ground
            enc[0, base + 36] = has_flip
            enc[0, base + 37] = boost
            # forward/up as the reference derives them from the quaternion --
            # our engine has these directly off the rotation matrix.
            rot = m.NextoObsBuilder._quats_to_rot_mtx(quat[None, :])
            cars.append({
                "team": team, "pos": pos.tolist(), "lin_vel": lin.tolist(),
                "ang_vel": ang.tolist(), "boost": boost, "demo": demo,
                "on_ground": on_ground, "has_flip": has_flip,
                "forward": rot[0, :, 0].tolist(), "up": rot[0, :, 2].tolist(),
            })

        builder = m.NextoObsBuilder(n_players=n_players)
        obs = builder.batched_build_obs(enc.copy())
        prev = rng.uniform(-1, 1, (n_players, 8))
        for i in range(n_players):
            builder.add_actions(obs, prev[i], player_index=i)

        # expected logits from the REAL torch model, so the candle port has a
        # net-level golden target as well as an obs-level one
        model = _model()
        for i in range(n_players):
            q, kv, mask = obs[i]
            with torch.no_grad():
                out = model((torch.as_tensor(q, dtype=torch.float32),
                             torch.as_tensor(kv, dtype=torch.float32),
                             torch.as_tensor(mask, dtype=torch.bool)))
            logits = out[0] if isinstance(out, tuple) else out
            logits = np.asarray(logits).reshape(-1).tolist()
            # Necto shares the obs contract but has 1 block, no LayerNorms and a
            # multi-discrete head -- capture its per-head logits too.
            nec = _model("deploy/external/necto/necto-model.pt")
            with torch.no_grad():
                nout = nec((torch.as_tensor(q, dtype=torch.float32),
                            torch.as_tensor(kv, dtype=torch.float32),
                            torch.as_tensor(mask, dtype=torch.bool)))
            nheads = [np.asarray(h).reshape(-1).tolist() for h in nout[0]]
            # Reference decode, copied from deploy/external/necto/agent.py, so the
            # Rust decode is checked against the real thing rather than only
            # producing "finite" numbers.
            acts = np.array([int(np.argmax(h)) for h in nheads], dtype=np.int64)
            a0, a1 = acts[0] - 1, acts[1] - 1
            a2, a3, a4 = acts[2], acts[3], acts[4]
            nparsed = [float(a0), float(a1), float(a0), float(a1 * (1 - a4)),
                       float(a1 * a4), float(a2), float(a3), float(a4)]
            cases.append({
                "logits": logits,
                "necto_heads": nheads,
                "necto_controls": nparsed,
                "self_idx": i,
                "pads": pads.tolist(),
                "ball": {"pos": ball_pos.tolist(), "lin_vel": ball_vel.tolist(),
                          "ang_vel": ball_ang.tolist()},
                "cars": cars,
                "prev_action": prev[i].tolist(),
                "q": np.asarray(q[0, 0]).tolist(),
                "kv": np.asarray(kv[0]).tolist(),
                "mask": np.asarray(mask[0]).tolist(),
            })

    out = pathlib.Path("engine/tests/fixtures/nexto_obs.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"cases": cases}))
    k0 = np.asarray(cases[0]["kv"])
    print(f"wrote {out}: {len(cases)} cases, q={len(cases[0]['q'])}, "
          f"kv={k0.shape}, mask={len(cases[0]['mask'])}")


if __name__ == "__main__":
    main()
