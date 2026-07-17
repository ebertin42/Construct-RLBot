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

## Checkpoint schema dispatch (v0 MLP / v1 entity transformer)

`deploy/model.py::load_policy` dispatches on `checkpoint.pt`'s
`schema_version`:

- `0` — the original 94-obs MLP (`PolicyValueNet`) with the 90-row action
  table. Behavior unchanged.
- `1` — the entity-transformer (`EntityPolicyNet`, 128/2/4/512, vendored from
  `python/construct/learn/model_v1.py`) with the 92-row v1.1 action table
  (90 v0 rows + 2 stalls). Obs is the obs-v1 entity set
  (`engine/src/obs_v1.rs`): 17 entities x 26 features + mask + 64-float query
  + prev-5 action indices.

To switch models just replace `checkpoint.pt` (e.g. with any
`checkpoints_entity/ck_*.pt`) — the bot picks the right net, obs builder,
and action table at startup. Both paths act greedily (argmax) every 8
physics ticks.

### v1: verified offline (WSL, tests/python/test_deploy_v1.py)

- Action table: deploy's 92 rows equal `construct._engine.action_table_v1()`
  row-for-row.
- Obs: `deploy/obs.py::build_obs_v1` reproduces a real bc-export golden
  (`replay/src/bc_obs.rs`, the exact engine `obs_v1::build` output) to 1e-5
  on ents/mask/query — including the orange-POV mirror and the boost-pad
  mirror permutation.
- prev-5 ring: `update_prev_ring` (newest-first) reproduces the bc-export
  `prev` array exactly over a full shard.
- Model: a real `checkpoints_entity` checkpoint loads through `load_policy`
  and its logits/value are bit-identical to the training-side
  `construct.learn.model_v1.EntityPolicyNet` on the same inputs.
- Pad order: the rlgym/rlbot pad ordering (rlgym_compat `BOOST_LOCATIONS`,
  including its (-940, 3310) 2uu quirk) maps bijectively onto the canonical
  RocketSim arena order via nearest-position matching.

### v1: needs live checking (Windows, real match)

- Ball prediction mapping: the bot reads RLBot v5's `self.ball_prediction`
  (120 Hz slices, `slices[0]` = now) at indices 60/120/180/240 for the
  +0.5/1/1.5/2 s horizons. Confirm slices are present and the sampled
  positions look sane. If the prediction is missing the bot falls back to a
  ballistic extrapolation (gravity -650 uu/s^2, **no wall/floor bounces** —
  approximate by design; the trained net saw RocketSim rollouts).
- Pad matching: `field_info.boost_pads` positions must be the standard 34
  soccar pads (mapping is built once in `initialize`; a non-standard count
  degrades to "all pads active"). Confirm `packet.boost_pads` order matches
  `field_info.boost_pads` order and that `BoostPadState.timer` counts
  seconds SINCE pickup (deploy converts to seconds-until-respawn via
  10 s big / 4 s small).
- Compat accessors: everything in the v0 checklist below, plus the v1 flip
  flag — obs f22 uses RocketSim `HasFlipOrJump` semantics, mapped as
  `car.on_ground or car.has_flip` (NOT v0's `can_flip`).
- prev-action ring: zeroed at bot init and while `goal_scored` is set (the
  post-goal replay/kickoff); otherwise continuous. Mid-match hot-joins keep
  a stale ring for ~5 acting steps (accepted, mirrors bc-export's known
  goal-gap behavior).
- Scoreboard: the 3 score/clock floats in the query row are deliberately
  ALWAYS 0.0 — the net was trained with zeros (engine has no scoreboard);
  writing real values would be out-of-distribution.
- Car ordering: mate/opp entity slots sort ascending by rlbot `player_id`;
  the engine sorts by RocketSim car id (spawn order). Confirm the id space
  rlbot hands out is stable within a match.

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
      (indices 19-21; index 15 is `boost/100`, not part of the ball block)
      are near `[0, 0, 0.0405]` — this is the ball's z-position (~93.15) after
      the `pos_norm` (1/2300) scale is applied — and that `obs.shape == (94,)`.

If any of the above don't match, update `compat_to_state_dict()` in
`deploy/bot.py` accordingly — `deploy/obs.py` and `deploy/actions.py` are
already parity-tested against the Rust engine (see
`tests/python/test_parity.py`) and should not need changes.

## Collision meshes (required)

`SimExtraInfo` runs RocketSim internally and needs the game's collision
meshes at `collision_meshes/soccar/*.cmf` inside THIS folder (16 files).
They are game-derived assets and never committed — copy them from the
training box: `assets/collision_meshes/soccar/` (fetched there by
`scripts/fetch_meshes.sh`).
