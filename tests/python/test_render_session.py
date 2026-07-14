import numpy as np
from construct._engine import RenderSession


def test_render_session_smoke():
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
