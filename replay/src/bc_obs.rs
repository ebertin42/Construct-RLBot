//! BC-export core (Task B3, BC-pretrain plan): converts schema-v4 shards
//! (`shard.rs`) into obs-v1 training tensors by rebuilding a minimal
//! `rocketsim_rs::GameState` per stored tick — a pure struct fill, NO physics
//! stepping — and calling `construct_engine::obs_v1::build` for every car,
//! i.e. the exact same function the live engine's collect loop uses. Train/
//! deploy obs consistency holds by construction: there is no reimplemented
//! obs logic here to drift.
//!
//! ## Sample layout
//! One input shard (`T` ticks, `P` players) becomes `S = T * P` samples in
//! fixed row-major `(t, p)` order: sample `s = t * P + p` is car `p`'s own
//! mirrored POV of tick `t` (`obs_v1::build` handles the orange-mirror given
//! `car_idx`). Output arrays (see `BcTensors`): `ents [S,17,26] f32`,
//! `mask [S,17] u8` (1 = absent/masked, matching the engine's `true = masked`
//! bool convention), `query [S,64] f32`, `prev [S,5] i64`, `action [S] i64`
//! (= the shard's stored `cars_action_idx`, Task B2's projected 92-table
//! index).
//!
//! ## GameState reconstruction
//! - Cars: `CarInfo { id: p + 1, .. }` in shard player order — ascending id
//!   therefore equals shard order, matching `reconstruct.rs`'s arena (which
//!   `add_car`s in `player_teams` order, getting increasing ids), so
//!   `obs_v1::build`'s asc-id mate/opp slotting sees the same ordering the
//!   live engine would. Rotation comes from the stored quaternion via
//!   `reconstruct::quat_to_rotmat` (the exact inverse of the
//!   `rotmat_to_quat` used at shard-write time). `has_flip` is encoded as
//!   `has_flipped = !has_flip` (with default `has_double_jumped = false`,
//!   `air_time_since_jump = 0`), which makes `CarState::has_flip_or_jump()`
//!   — the very call `obs_v1::car_row` makes — return the stored value for
//!   every state the sim can actually produce (grounded cars always have
//!   flip, exactly as `HasFlipOrJump`'s `isOnGround ||` short-circuit says).
//! - Pads: positions/is_big come from a one-time `Arena::default_standard()`
//!   template (`pad_template`, the arena's fixed construction order — the
//!   same order shard `pads` rows are stored in); per-tick dynamic state is
//!   filled from the shard: `is_active`, and `cooldown = timer * 10.0`
//!   (inverting `reconstruct::pad_timer_norm` / `obs_v1::timer_norm`'s
//!   always-big-pad normalization).
//! - Ball prediction: the stored `ball_pred [T,4,6]` rows are handed to
//!   `build` as raw-world `BallSnap`s; mirroring/normalization happen inside
//!   `build` like everywhere else.
//!
//! ## prev-5 window
//! Same most-recent-first ring semantics as the live engine's per-agent
//! history (`engine/src/episode.rs`): `prev[0]` = the action executed on the
//! previous stored row, `prev[4]` = five rows back; fresh windows are all
//! zeros. The window resets (zeros) at t=0 and whenever
//! `tick_index[t] - tick_index[t-1] != stride` — a dropped-tick gap, the
//! shard's only genuine history discontinuity. `is_boundary` is deliberately
//! NOT a reset trigger: it marks routine 30 Hz re-snaps to authoritative
//! replay frames (~22% of stored rows on a real stride-8 shard), where the
//! match timeline — and therefore the player's action history — is
//! continuous; a drop-adjacent boundary is already covered by its
//! tick_index gap.
//!
//! Known accepted gap: goal→kickoff resets leave NO tick_index gap — the
//! stored counter stays contiguous across re-snaps — so the first ~5 stored
//! rows after a goal carry pre-goal prev actions, whereas the live engine
//! zeroes its ring at episode reset. Accepted as-is: it affects <1% of rows,
//! and the kickoff countdown mostly flushes the ring before play resumes. A
//! proper goal/kickoff marker is deferred to a future shard schema v5.
//!
//! ## Working directory requirement
//! `pad_template` builds one `Arena::default_standard()`, which needs
//! RocketSim's collision meshes to resolve from the current working
//! directory (see `sim_init`) — run callers from the repo root or `replay/`,
//! same as `replay-parse`.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3};
use ndarray_npy::{NpzReader, NpzWriter};
use rocketsim_rs::{
    math::Vec3,
    sim::{Arena, BallState, CarConfig, CarState, Team},
    BoostPad, CarInfo, GameState,
};

use construct_engine::{
    ballpred::BallSnap,
    obs_v1::{self, ENT_FEAT, MAX_ENT, PREV_ACTIONS, Q_FEAT},
    schema::Normalization,
};

use crate::reconstruct::quat_to_rotmat;
use crate::shard::SHARD_SCHEMA_VERSION;

/// A loaded, validated schema-v4 shard — just the arrays `build_tensors`
/// needs (`cars_action`/`is_boundary` are deliberately not loaded; see the
/// module doc's "prev-5 window" for why `is_boundary` plays no role here).
#[derive(Debug)]
pub struct ShardV4 {
    /// `[T, 13]` — see `shard::BALL_COLUMNS`.
    pub ball: Array2<f32>,
    /// `[T, P, 17]` — see `shard::CARS_STATE_COLUMNS`.
    pub cars_state: Array3<f32>,
    /// `[T, P]` — Task B2's projected 92-table action index.
    pub cars_action_idx: Array2<i64>,
    /// `[T, 34, 2]` — (timer, is_active) in fixed arena order.
    pub pads: Array3<f32>,
    /// `[T, 4, 6]` — pos3+vel3 at +0.5/1/1.5/2s.
    pub ball_pred: Array3<f32>,
    /// `[P]` — 0=blue / 1=orange.
    pub player_teams: Array1<i64>,
    /// `[T]` — original 120 Hz tick positions (not renumbered by stride).
    pub tick_index: Array1<i64>,
    /// Sidecar `stride`: contiguous stored rows differ by exactly this much
    /// in `tick_index`; any larger delta is a dropped-tick gap.
    pub stride: usize,
    /// Sidecar `action_table_size` (92 for the v1 table).
    pub action_table_size: usize,
}

/// Obs-v1 training tensors for one shard, `S = T * P` samples in row-major
/// `(t, p)` order — see the module doc's "Sample layout".
pub struct BcTensors {
    pub ents: Array3<f32>,
    pub mask: Array2<u8>,
    pub query: Array2<f32>,
    pub prev: Array2<i64>,
    pub action: Array1<i64>,
}

/// Loads `<stem>.npz` + its `<stem>.json` sidecar, failing loud (with the
/// found vs required version) on anything but `SHARD_SCHEMA_VERSION == 4` —
/// v3 shards lack `pads`/`ball_pred`/`has_flip`/`cars_action_idx` and must
/// be re-parsed, not silently mis-read.
pub fn load_shard_v4(npz_path: &Path) -> Result<ShardV4, String> {
    let sidecar_path = npz_path.with_extension("json");
    let sidecar_file = std::fs::File::open(&sidecar_path)
        .map_err(|e| format!("open sidecar {}: {e}", sidecar_path.display()))?;
    let sidecar: serde_json::Value = serde_json::from_reader(sidecar_file)
        .map_err(|e| format!("parse sidecar {}: {e}", sidecar_path.display()))?;

    let version = sidecar["schema_version"].as_u64().unwrap_or(0) as u32;
    if version != SHARD_SCHEMA_VERSION {
        return Err(format!(
            "{}: shard schema_version {version} unsupported — bc-export requires v{SHARD_SCHEMA_VERSION} \
             (pads/has_flip/ball_pred/cars_action_idx); re-parse the replay with the current replay-parse",
            npz_path.display()
        ));
    }
    let cars_state_cols = sidecar["cars_state_columns"]
        .as_array()
        .map(|a| a.len())
        .ok_or_else(|| format!("{}: sidecar missing cars_state_columns", sidecar_path.display()))?;
    if cars_state_cols != 17 {
        return Err(format!(
            "{}: expected 17 cars_state columns (v4), sidecar documents {cars_state_cols}",
            npz_path.display()
        ));
    }
    let stride = sidecar["stride"]
        .as_u64()
        .ok_or_else(|| format!("{}: sidecar missing stride", sidecar_path.display()))?
        as usize;
    let action_table_size = sidecar["action_table_size"]
        .as_u64()
        .ok_or_else(|| format!("{}: sidecar missing action_table_size", sidecar_path.display()))?
        as usize;

    let file = std::fs::File::open(npz_path)
        .map_err(|e| format!("open {}: {e}", npz_path.display()))?;
    let mut npz = NpzReader::new(file).map_err(|e| format!("{}: {e}", npz_path.display()))?;
    let arr = |e: ndarray_npy::ReadNpzError| format!("{}: {e}", npz_path.display());
    let ball: Array2<f32> = npz.by_name("ball.npy").map_err(arr)?;
    if ball.shape()[1] != 13 {
        return Err(format!(
            "{}: expected 13 ball columns (v4, see shard::BALL_COLUMNS), got {}",
            npz_path.display(),
            ball.shape()[1]
        ));
    }
    Ok(ShardV4 {
        ball,
        cars_state: npz.by_name("cars_state.npy").map_err(arr)?,
        cars_action_idx: npz.by_name("cars_action_idx.npy").map_err(arr)?,
        pads: npz.by_name("pads.npy").map_err(arr)?,
        ball_pred: npz.by_name("ball_pred.npy").map_err(arr)?,
        player_teams: npz.by_name("player_teams.npy").map_err(arr)?,
        tick_index: npz.by_name("tick_index.npy").map_err(arr)?,
        stride,
        action_table_size,
    })
}

/// One-time pad template: a standard soccar arena's 34 `BoostPad`s in fixed
/// construction order, carrying the static `config` (position, is_big) that
/// shards don't store. Dynamic `state` fields are overwritten per tick by
/// `build_tensors`. Runs `sim_init::ensure_init` itself (mesh loading — see
/// the module doc's working-directory note).
pub fn pad_template() -> Vec<BoostPad> {
    crate::sim_init::ensure_init(None);
    let mut arena = Arena::default_standard();
    arena.pin_mut().get_game_state().pads
}

fn vec3(x: f32, y: f32, z: f32) -> Vec3 {
    Vec3::new(x, y, z)
}

/// Builds the obs-v1 tensors for every `(tick, car)` of `shard` — see the
/// module doc for the sample layout, GameState reconstruction, and prev
/// window semantics. Deterministic: pure function of its inputs, fixed
/// iteration order.
pub fn build_tensors(
    shard: &ShardV4,
    pad_template: &[BoostPad],
    norm: &Normalization,
) -> Result<BcTensors, String> {
    let t_count = shard.ball.shape()[0];
    let p_count = shard.cars_state.shape()[1];
    if t_count == 0 || p_count == 0 {
        return Err(format!("empty shard (T={t_count}, P={p_count})"));
    }
    if shard.cars_state.shape() != [t_count, p_count, 17] {
        return Err(format!("cars_state shape {:?} != [T, P, 17]", shard.cars_state.shape()));
    }
    if shard.cars_action_idx.shape() != [t_count, p_count]
        || shard.pads.shape() != [t_count, 34, 2]
        || shard.ball_pred.shape() != [t_count, 4, 6]
        || shard.player_teams.len() != p_count
        || shard.tick_index.len() != t_count
    {
        return Err("shard array shapes are mutually inconsistent".to_string());
    }
    if pad_template.len() != 34 {
        return Err(format!("pad template must have 34 pads, got {}", pad_template.len()));
    }

    let s_count = t_count * p_count;
    let mut ents = Array3::<f32>::zeros((s_count, MAX_ENT, ENT_FEAT));
    let mut mask = Array2::<u8>::zeros((s_count, MAX_ENT));
    let mut query = Array2::<f32>::zeros((s_count, Q_FEAT));
    let mut prev = Array2::<i64>::zeros((s_count, PREV_ACTIONS));
    let mut action = Array1::<i64>::zeros(s_count);

    // Reused per-tick buffers for obs_v1::build's out params.
    let mut ents_buf = vec![0.0f32; MAX_ENT * ENT_FEAT];
    let mut mask_buf = vec![false; MAX_ENT];
    let mut query_buf = vec![0.0f32; Q_FEAT];

    // Per-car prev-action rings, most-recent-first (engine/src/episode.rs
    // semantics) — see the module doc's "prev-5 window".
    let mut rings: Vec<[i64; PREV_ACTIONS]> = vec![[0; PREV_ACTIONS]; p_count];

    // The GameState skeleton is built once and mutated per tick: pads keep
    // their template config, cars keep their id/team/config.
    let mut gs = GameState {
        pads: pad_template.to_vec(),
        ..Default::default()
    };
    for p in 0..p_count {
        gs.cars.push(CarInfo {
            id: p as u32 + 1,
            team: if shard.player_teams[p] == 0 { Team::Blue } else { Team::Orange },
            state: CarState::default(),
            config: *CarConfig::octane(),
        });
    }

    for t in 0..t_count {
        // Discontinuity check BEFORE emitting tick t's samples: a gap between
        // t-1 and t means t must start from a fresh (zero) history. Note
        // goal→kickoff resets are tick-contiguous and deliberately NOT caught
        // here — see the module doc's "prev-5 window" for why that's accepted.
        if t > 0 && shard.tick_index[t] - shard.tick_index[t - 1] != shard.stride as i64 {
            for ring in rings.iter_mut() {
                *ring = [0; PREV_ACTIONS];
            }
        }

        // --- ball ---
        let b = &mut gs.ball;
        *b = BallState::default();
        b.pos = vec3(shard.ball[[t, 0]], shard.ball[[t, 1]], shard.ball[[t, 2]]);
        b.vel = vec3(shard.ball[[t, 3]], shard.ball[[t, 4]], shard.ball[[t, 5]]);
        b.ang_vel = vec3(shard.ball[[t, 6]], shard.ball[[t, 7]], shard.ball[[t, 8]]);
        b.rot_mat = quat_to_rotmat([
            shard.ball[[t, 9]],
            shard.ball[[t, 10]],
            shard.ball[[t, 11]],
            shard.ball[[t, 12]],
        ]);

        // --- cars (see module doc: has_flip encoding, boost 0..1 -> 0..100) ---
        for p in 0..p_count {
            let cs = &mut gs.cars[p].state;
            *cs = CarState::default();
            cs.pos = vec3(
                shard.cars_state[[t, p, 0]],
                shard.cars_state[[t, p, 1]],
                shard.cars_state[[t, p, 2]],
            );
            cs.vel = vec3(
                shard.cars_state[[t, p, 3]],
                shard.cars_state[[t, p, 4]],
                shard.cars_state[[t, p, 5]],
            );
            cs.ang_vel = vec3(
                shard.cars_state[[t, p, 6]],
                shard.cars_state[[t, p, 7]],
                shard.cars_state[[t, p, 8]],
            );
            cs.rot_mat = quat_to_rotmat([
                shard.cars_state[[t, p, 9]],
                shard.cars_state[[t, p, 10]],
                shard.cars_state[[t, p, 11]],
                shard.cars_state[[t, p, 12]],
            ]);
            cs.boost = (shard.cars_state[[t, p, 13]] * 100.0).clamp(0.0, 100.0);
            cs.is_on_ground = shard.cars_state[[t, p, 14]] != 0.0;
            cs.is_demoed = shard.cars_state[[t, p, 15]] != 0.0;
            cs.has_flipped = shard.cars_state[[t, p, 16]] == 0.0;
        }

        // --- pads: template config + this tick's dynamic state ---
        for (i, pad) in gs.pads.iter_mut().enumerate() {
            let active = shard.pads[[t, i, 1]] != 0.0;
            pad.state.is_active = active;
            pad.state.cooldown = if active { 0.0 } else { shard.pads[[t, i, 0]] * 10.0 };
        }

        // --- ball prediction snapshots (raw world units; build mirrors/scales) ---
        let mut pred = [BallSnap::default(); 4];
        for (h, snap) in pred.iter_mut().enumerate() {
            snap.pos = vec3(
                shard.ball_pred[[t, h, 0]],
                shard.ball_pred[[t, h, 1]],
                shard.ball_pred[[t, h, 2]],
            );
            snap.vel = vec3(
                shard.ball_pred[[t, h, 3]],
                shard.ball_pred[[t, h, 4]],
                shard.ball_pred[[t, h, 5]],
            );
        }

        for p in 0..p_count {
            let s = t * p_count + p;
            obs_v1::build(&gs, p, &pred, norm, &mut ents_buf, &mut mask_buf, &mut query_buf);
            for e in 0..MAX_ENT {
                for k in 0..ENT_FEAT {
                    ents[[s, e, k]] = ents_buf[e * ENT_FEAT + k];
                }
                mask[[s, e]] = mask_buf[e] as u8;
            }
            for (k, &v) in query_buf.iter().enumerate() {
                query[[s, k]] = v;
            }
            prev.row_mut(s).assign(&ndarray::ArrayView1::from(&rings[p][..]));

            let a = shard.cars_action_idx[[t, p]];
            if a < 0 || a >= shard.action_table_size as i64 {
                return Err(format!(
                    "cars_action_idx[[{t},{p}]] = {a} out of [0, {})",
                    shard.action_table_size
                ));
            }
            action[s] = a;
        }

        // Shift this tick's executed actions into the rings AFTER emitting
        // its samples — the obs at t must only see actions from t-1 back.
        for (p, ring) in rings.iter_mut().enumerate() {
            for i in (1..PREV_ACTIONS).rev() {
                ring[i] = ring[i - 1];
            }
            ring[0] = shard.cars_action_idx[[t, p]];
        }
    }

    Ok(BcTensors { ents, mask, query, prev, action })
}

/// Exports one shard file to `<out_dir>/bc_<stem>.npz`. Returns
/// `Ok(None)` without touching anything if the output already exists
/// (resumability), else `Ok(Some((path, num_samples)))`. Writes to a `.tmp`
/// sibling first and renames into place, so an interrupted export never
/// leaves a partial file that a resumed run would wrongly skip.
pub fn export_shard_file(
    shard_npz: &Path,
    out_dir: &Path,
    pad_template: &[BoostPad],
    norm: &Normalization,
) -> Result<Option<(PathBuf, usize)>, String> {
    let stem = shard_npz
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| format!("unreadable file stem: {}", shard_npz.display()))?;
    let out_path = out_dir.join(format!("bc_{stem}.npz"));
    if out_path.exists() {
        return Ok(None);
    }

    let shard = load_shard_v4(shard_npz)?;
    let bc = build_tensors(&shard, pad_template, norm)?;
    let samples = bc.action.len();

    let tmp_path = out_dir.join(format!("bc_{stem}.npz.tmp"));
    let file = std::fs::File::create(&tmp_path)
        .map_err(|e| format!("create {}: {e}", tmp_path.display()))?;
    let mut npz = NpzWriter::new(file);
    npz.add_array("ents", &bc.ents).map_err(|e| e.to_string())?;
    npz.add_array("mask", &bc.mask).map_err(|e| e.to_string())?;
    npz.add_array("query", &bc.query).map_err(|e| e.to_string())?;
    npz.add_array("prev", &bc.prev).map_err(|e| e.to_string())?;
    npz.add_array("action", &bc.action).map_err(|e| e.to_string())?;
    npz.finish().map_err(|e| e.to_string())?;
    std::fs::rename(&tmp_path, &out_path)
        .map_err(|e| format!("rename {} -> {}: {e}", tmp_path.display(), out_path.display()))?;

    Ok(Some((out_path, samples)))
}
