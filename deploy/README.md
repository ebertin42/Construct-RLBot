# Deploying Construct to Rocket League (Windows)

1. Install RLBot v5 launcher: https://rlbot.org/v5/
2. Install Python 3.11 (python.org, add to PATH)
3. In this folder (copied to Windows, e.g. from \\wsl$\...\Construct-RLBot\deploy):
   py -3.11 -m venv ..\venv
   ..\venv\Scripts\pip install -r requirements.txt
4. Copy a trained checkpoint here as checkpoint.pt
5. RLBot GUI -> Add -> Load Folder -> select this folder -> start a match
   (or: server headless route — run RLBotServer, then `python bot.py` with
   RLBOT_AGENT_ID=construct/construct_v0 set)
RLBot only works in local/offline matches; it launches the game with -rlbot.

## Live-verification checklist

`deploy/bot.py` was written against the rlgym-compat API surface described in
the Task 13 brief, but `rlbot` and `rlgym_compat` are Windows-only
dependencies that could not be installed or imported in the WSL/Linux dev
environment used to build this package. The field mapping in
`compat_to_state_dict()` is therefore **unverified** — it type-checks and
compiles, but has never been run against a live `GameState`. Before trusting
this bot in a real match, run it in a live RLBot v5 match on the Windows host
and check the following, e.g. by temporarily adding a `print(state)` /
`print(obs)` in `get_output()` and eyeballing one packet's worth of output:

- [ ] `car.physics.forward` and `car.physics.up` — confirm these are the
      correct accessor names on the installed `rlgym_compat` version for the
      car's forward/up unit vectors (some versions expose a rotation matrix
      instead, requiring e.g. `car.physics.rotation_mtx[:, 0]`). Sanity check:
      each should be a unit-length 3-vector.
- [ ] `car.boost_amount` scale — confirm it is `0..1` (this code multiplies
      by `100.0` to match the engine's `0..100` convention used by
      `deploy/obs.py`). Sanity check: a freshly spawned car's boost value
      after the `*100.0` conversion should read ~33.3, not ~0.333 or ~3330.
- [ ] `car.can_flip` — confirm this is the right attribute for "has a
      flip/double-jump available" (vs. `has_flip`, `has_jump`, or similar) on
      the installed rlgym-compat version.
- [ ] `car.on_ground`, `car.is_demoed`, `car.team_num` — confirm attribute
      names and types (bool vs. int/enum) match what's assumed here.
- [ ] `agent_id` (the dict key of `game_state.cars`) vs. `self.player_id` —
      confirm both live in the same integer identifier space so that
      `int(agent_id) == int(self.player_id)` correctly identifies "my" car
      inside `build_obs`.
- [ ] End-to-end sanity: with the bot loaded and the ball stationary at
      kickoff, confirm the printed `obs` vector's ball-position entries
      (indices 15-17, before the `pos_norm` scale) are near `[0, 0, 93.15]`
      and that `obs.shape == (94,)`.

If any of the above don't match, update `compat_to_state_dict()` in
`deploy/bot.py` accordingly — `deploy/obs.py` and `deploy/actions.py` are
already parity-tested against the Rust engine (see
`tests/python/test_parity.py`) and should not need changes.
