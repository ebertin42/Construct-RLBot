import os

import numpy as np
import torch

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot
from rlgym_compat import GameState
from rlgym_compat.sim_extra_info import SimExtraInfo

from actions import make_lookup_table, make_lookup_table_v1
from model import load_policy
from obs import (
    BALL_PRED_HORIZONS_SEC,
    NUM_PRED,
    PAD_COUNT,
    PAD_RECHARGE_SECONDS,
    PREV_ACTIONS,
    build_obs,
    build_obs_v1,
    pad_order_mapping,
    update_prev_ring,
)

TICK_SKIP = 8  # act every 8 physics ticks — matches the engine (schema tick_skip)
POS_NORM = 1.0 / 2300.0
VEL_NORM = 1.0 / 2300.0
ANG_VEL_NORM = 1.0 / 5.5
GRAVITY_Z = -650.0  # uu/s^2, for the ball-prediction ballistic fallback
BALL_REST_Z = 93.15  # resting ball center height, fallback floor clamp


def compat_to_state_dict(game_state: GameState) -> dict:
    """Adapt rlgym-compat GameState to the deploy/obs.py dict contract.

    UNVERIFIABLE HERE: rlbot + rlgym_compat are not installed in this dev
    environment (Windows-side deps for the RLBot v5 game host), so this
    field mapping cannot be exercised against a live GameState. Field names
    have drifted across rlgym-compat versions in the past. Verify against a
    real match on the Windows host (Task 14) before trusting this bot:
      - car.physics.forward / car.physics.up: are these rotation-matrix
        column accessors on this installed rlgym-compat version, or is the
        forward vector obtained differently (e.g. via a quaternion or an
        explicit rotation matrix indexing)?
      - car.boost_amount: VERIFIED 0..100 (rlgym_compat 2.3.6 car.py:48);
        passed through unscaled. The original 0..1 assumption froze the bot
        (boost=33 in obs instead of 0.33 = out-of-distribution).
      - car.can_flip: confirm this is the correct attribute name for "has
        a flip/double-jump available" (vs. e.g. has_flip, has_jump).
      - car.on_ground, car.is_demoed, car.team_num: confirm attribute names
        and types (bool vs int/enum) match.
      - agent_id (dict key of game_state.cars) vs self.player_id: confirm
        both are the same int-convertible identifier space so that
        `int(agent_id) == int(self.player_id)` lines up in build_obs.
    """
    cars = []
    for agent_id, car in game_state.cars.items():
        phys = car.physics
        cars.append({
            "id": int(agent_id),
            "team": int(car.team_num),
            "pos": phys.position.tolist(),
            "vel": phys.linear_velocity.tolist(),
            "ang_vel": phys.angular_velocity.tolist(),
            "forward": phys.forward.tolist(),
            "up": phys.up.tolist(),
            "boost": float(car.boost_amount),  # compat stores 0..100 (verified car.py:48) — same scale as engine
            "is_on_ground": bool(car.on_ground),
            "has_flip": bool(car.can_flip),
            "is_demoed": bool(car.is_demoed),
        })
    ball = game_state.ball
    return {
        "ball": {
            "pos": ball.position.tolist(),
            "vel": ball.linear_velocity.tolist(),
            "ang_vel": ball.angular_velocity.tolist(),
        },
        "cars": cars,
    }


def ball_pred_rows(ball_prediction, ball_pos, ball_vel) -> np.ndarray:
    """[4,6] raw pos3+vel3 at +0.5/1/1.5/2 s for obs_v1's ball-pred entities.

    Primary source: RLBot v5's BallPrediction (Bot.ball_prediction) — 120 Hz
    slices, slices[0] = now, car-collision-free, same semantics as the
    engine's car-less RocketSim tracker (engine/src/ballpred.rs). Horizon
    tau maps to slice index tau*120 (60/120/180/240).

    Fallback (prediction missing/short): ballistic extrapolation under
    gravity, no wall/floor bounces beyond a floor clamp — APPROXIMATE, see
    README. Matches the engine's "ball keeps doing what it's doing" spirit.
    """
    out = np.zeros((NUM_PRED, 6), dtype=np.float32)
    slices = list(ball_prediction.slices) if ball_prediction is not None else []
    if len(slices) > 240:
        for h, tau in enumerate(BALL_PRED_HORIZONS_SEC):
            phys = slices[int(round(tau * 120))].physics
            out[h, 0:3] = (phys.location.x, phys.location.y, phys.location.z)
            out[h, 3:6] = (phys.velocity.x, phys.velocity.y, phys.velocity.z)
        return out
    p = np.asarray(ball_pos, dtype=np.float32)
    v = np.asarray(ball_vel, dtype=np.float32)
    for h, tau in enumerate(BALL_PRED_HORIZONS_SEC):
        pos = p + v * tau
        pos[2] = max(pos[2] + 0.5 * GRAVITY_Z * tau * tau, BALL_REST_Z)
        vel = v.copy()
        vel[2] = v[2] + GRAVITY_Z * tau
        out[h, 0:3] = pos
        out[h, 3:6] = vel
    return out


def pads_canonical(packet: GamePacket, pad_map) -> tuple[np.ndarray, np.ndarray]:
    """(cooldown[34] seconds-until-respawn, active[34] bool) in canonical
    RocketSim arena order.

    RLBot's BoostPadState.timer counts seconds SINCE pickup (0 while active);
    RocketSim's cooldown counts seconds UNTIL respawn — hence
    `recharge - timer` (10 s big / 4 s small, clamped at 0). packet.boost_pads
    is in field_info.boost_pads order (the rlbot contract rlgym_compat also
    relies on); `pad_map` translates that to canonical order.

    Degraded fallback (no map / unexpected pad count): all pads reported
    active with zero timers — keeps the obs in-distribution rather than
    inventing state.
    """
    cooldown = np.zeros(PAD_COUNT, dtype=np.float32)
    active = np.ones(PAD_COUNT, dtype=bool)
    if pad_map is not None and len(packet.boost_pads) == len(pad_map):
        for i, pad in enumerate(packet.boost_pads):
            k = int(pad_map[i])
            active[k] = bool(pad.is_active)
            if not pad.is_active:
                cooldown[k] = max(0.0, float(PAD_RECHARGE_SECONDS[k]) - float(pad.timer))
    return cooldown, active


def compat_to_state_dict_v1(game_state: GameState, packet: GamePacket, ball_prediction, pad_map) -> dict:
    """Extends the v0 dict with obs_v1's pads + ball prediction, and swaps the
    per-car flip flag to RocketSim HasFlipOrJump semantics (the engine's f22:
    `is_on_ground || has_flip`, Car.cpp:307) — `car.can_flip` (v0) excludes
    grounded cars, which always have a jump. Same live-verification caveats
    as compat_to_state_dict."""
    state = compat_to_state_dict(game_state)
    for c, car in zip(state["cars"], game_state.cars.values()):
        c["has_flip"] = bool(car.on_ground or car.has_flip)
    cooldown, active = pads_canonical(packet, pad_map)
    state["pads_cooldown"] = cooldown
    state["pads_active"] = active
    state["ball_pred"] = ball_pred_rows(
        ball_prediction, state["ball"]["pos"], state["ball"]["vel"]
    )
    return state


class ConstructBot(Bot):
    def initialize(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.net, self.schema_version = load_policy(os.path.join(here, "checkpoint.pt"))
        self.table = make_lookup_table_v1() if self.schema_version == 1 else make_lookup_table()
        # rlgym_compat >= 2.x dropped the tick_skip param (we do our own
        # tick-skip accounting via frame_num deltas in get_output)
        self.extra_info = SimExtraInfo(self.field_info)
        self.game_state = GameState.create_compat_game_state(self.field_info)
        self.ticks = TICK_SKIP  # act on first packet
        self.prev_control = ControllerState()
        self.prev_frame = -1
        self._debug_budget = 30  # log first N acting frames / exceptions
        # v1: prev-5 executed-action ring (newest-first, zeros at match start;
        # re-zeroed on goals — see _get_output) + field-info pad order mapping
        self.prev_ring = np.zeros(PREV_ACTIONS, dtype=np.int64)
        self.pad_map = None
        if self.schema_version == 1:
            positions = np.array(
                [[p.location.x, p.location.y, p.location.z] for p in self.field_info.boost_pads],
                dtype=np.float32,
            ).reshape(-1, 3)
            if len(positions) == PAD_COUNT:
                self.pad_map = pad_order_mapping(positions)
            else:
                self._dbg(f"non-standard map: {len(positions)} boost pads (expected {PAD_COUNT}); "
                          "pad entities will read all-active")

    def get_output(self, packet: GamePacket) -> ControllerState:
        try:
            return self._get_output(packet)
        except Exception:
            if self._debug_budget > 0:
                self._debug_budget -= 1
                import traceback

                self._dbg("EXCEPTION:\n" + traceback.format_exc())
            return self.prev_control

    def _dbg(self, msg: str):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bot_debug.log"), "a") as f:
            f.write(msg + "\n")

    def _get_output(self, packet: GamePacket) -> ControllerState:
        if not packet.balls:
            return self.prev_control  # replay/kickoff countdown frames
        frame = packet.match_info.frame_num
        self.ticks += max(0, frame - self.prev_frame) if self.prev_frame >= 0 else TICK_SKIP
        self.prev_frame = frame
        if self.ticks < TICK_SKIP:
            return self.prev_control
        self.ticks = 0

        extra = self.extra_info.get_extra_info(packet)
        self.game_state.update(packet, extra_info=extra)

        if self.schema_version == 1:
            index, dbg = self._act_v1(packet)
        else:
            index, dbg = self._act_v0()
        row = self.table[index]

        if self._debug_budget > 0:
            self._debug_budget -= 1
            self._dbg(f"frame={frame} player_id={self.player_id!r} index={self.index} "
                      f"{dbg} action_idx={index} action_row={row.tolist()}")

        c = ControllerState()
        c.throttle, c.steer = float(row[0]), float(row[1])
        c.pitch, c.yaw, c.roll = float(row[2]), float(row[3]), float(row[4])
        c.jump, c.boost, c.handbrake = bool(row[5]), bool(row[6]), bool(row[7])
        self.prev_control = c
        return c

    def _act_v0(self) -> tuple[int, str]:
        state = compat_to_state_dict(self.game_state)
        obs = build_obs(state, int(self.player_id), POS_NORM, VEL_NORM, ANG_VEL_NORM)
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(obs).unsqueeze(0))
        dbg = (f"cars={[(c['id'], c['team']) for c in state['cars']]} "
               f"obs[:6]={obs[:6].tolist()} obs_absmax={float(abs(obs).max()):.3f}")
        return int(logits.argmax(-1)), dbg

    def _act_v1(self, packet: GamePacket) -> tuple[int, str]:
        # Fresh action history at the post-goal kickoff (the engine zeroes its
        # ring on episode reset; goal replay/kickoff is the only in-match
        # analogue we can detect). goal_scored stays True through the
        # GoalScored phase, so the ring is held at zero until play resumes.
        if self.game_state.goal_scored:
            self.prev_ring[:] = 0
        state = compat_to_state_dict_v1(self.game_state, packet, self.ball_prediction, self.pad_map)
        ents, mask, query = build_obs_v1(state, int(self.player_id))
        with torch.no_grad():
            logits, _ = self.net(
                torch.from_numpy(ents).unsqueeze(0),
                torch.from_numpy(mask).unsqueeze(0),
                torch.from_numpy(query).unsqueeze(0),
                torch.from_numpy(self.prev_ring).unsqueeze(0),
            )
        index = int(logits.argmax(-1))
        update_prev_ring(self.prev_ring, index)  # obs at t sees actions t-1 back
        dbg = (f"cars={[(c['id'], c['team']) for c in state['cars']]} "
               f"ents_absmax={float(np.abs(ents).max()):.3f} "
               f"mask={mask.astype(int).tolist()} prev={self.prev_ring.tolist()}")
        return index, dbg


if __name__ == "__main__":
    ConstructBot("construct/construct_v0").run()
