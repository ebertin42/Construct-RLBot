import socket

import numpy as np
import pytest

from construct._engine import RenderSession

# RenderSession's constructor binds a UDP socket on this port to stream state
# to RLViser (see engine/src/viser.rs: UdpSocket::bind(("0.0.0.0", 34254))).
# A viewer/relay already holding it (a known, unrelated nuisance -- see
# rlviser-viewer-setup notes) makes construction fail outright, not just the
# render itself, so probe first and skip cleanly instead of failing.
_VISER_PORT = 34254


def _viser_port_busy() -> bool:
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.bind(("0.0.0.0", _VISER_PORT))
    except OSError:
        return True
    finally:
        probe.close()
    return False


def test_render_session_smoke():
    if _viser_port_busy():
        pytest.skip(f"UDP port {_VISER_PORT} is in use (viewer/relay likely running)")
    sess = RenderSession(blue=1, orange=1, schema_path="schema/v0.toml",
                         reward_config_path="configs/reward_v0.toml", seed=3)
    assert sess.num_agents == 2 and sess.obs_size == 94 and sess.action_count == 90
    obs = sess.reset()
    assert obs.shape == (2, 94) and obs.dtype == np.float32
    for _ in range(3):  # keep short: Pacer sleeps ~66ms/step
        obs, rew, term, trunc, final_obs = sess.step(np.zeros(2, dtype=np.int64))
    assert obs.shape == (2, 94) and rew.shape == (2,)
    assert term.dtype == np.bool_ and trunc.dtype == np.bool_
    assert final_obs.shape == (2, 94)
    sess.close()  # must not raise
