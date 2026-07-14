import os

import numpy as np
import torch

from rlbot.flat import ControllerState, GamePacket
from rlbot.managers import Bot
from rlgym_compat import GameState
from rlgym_compat.sim_extra_info import SimExtraInfo

from actions import make_lookup_table
from model import load_policy
from obs import build_obs

TICK_SKIP = 8
POS_NORM = 1.0 / 2300.0
VEL_NORM = 1.0 / 2300.0
ANG_VEL_NORM = 1.0 / 5.5


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
      - car.boost_amount: confirm it is 0..1 scaled (as assumed here, hence
        the *100.0 below) and not already 0..100.
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
            "boost": float(car.boost_amount * 100.0),   # compat stores 0..1; engine uses 0..100
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


class ConstructBot(Bot):
    def initialize(self):
        here = os.path.dirname(os.path.abspath(__file__))
        self.table = make_lookup_table()
        self.net = load_policy(os.path.join(here, "checkpoint.pt"), obs_size=94, action_count=90)
        # rlgym_compat >= 2.x dropped the tick_skip param (we do our own
        # tick-skip accounting via frame_num deltas in get_output)
        self.extra_info = SimExtraInfo(self.field_info)
        self.game_state = GameState.create_compat_game_state(self.field_info)
        self.ticks = TICK_SKIP  # act on first packet
        self.prev_control = ControllerState()
        self.prev_frame = -1

    def get_output(self, packet: GamePacket) -> ControllerState:
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
        state = compat_to_state_dict(self.game_state)
        obs = build_obs(state, int(self.player_id), POS_NORM, VEL_NORM, ANG_VEL_NORM)
        with torch.no_grad():
            logits, _ = self.net(torch.from_numpy(obs).unsqueeze(0))
        row = self.table[int(logits.argmax(-1))]

        c = ControllerState()
        c.throttle, c.steer = float(row[0]), float(row[1])
        c.pitch, c.yaw, c.roll = float(row[2]), float(row[3]), float(row[4])
        c.jump, c.boost, c.handbrake = bool(row[5]), bool(row[6]), bool(row[7])
        self.prev_control = c
        return c


if __name__ == "__main__":
    ConstructBot("construct/construct_v0").run()
