//! Shard writer: serializes a [`Reconstructed`] 120 Hz tick stream (Task 4)
//! into one `.npz` per replay (via `ndarray-npy`) plus a JSON sidecar
//! documenting every column, so downstream IDM/BC loaders can validate shape
//! and schema without guessing.
//!
//! ## Array layout
//! - `ball` `[T, 13]`: pos3, vel3, ang_vel3, quat4 (`[x,y,z,w]`).
//! - `cars_state` `[T, P, 16]`: pos3, vel3, ang_vel3, quat4, boost (0..1),
//!   on_ground (0/1), demoed (0/1). `P` = `player_teams.len()`, same car
//!   order as `player_teams` and `cars_action`.
//! - `cars_action` `[T, P, 8]`: throttle, steer, pitch, yaw, roll, jump,
//!   boost, handbrake (jump/boost/handbrake as 0.0/1.0) — the applied
//!   `CarControls` for that tick's 30 Hz interval (see `Tick::actions`).
//! - `player_teams` `[P]` (`i64`, 0=blue/1=orange).
//! - `tick_index` `[T]` (`i64`): global monotonic 120 Hz tick position in the
//!   ORIGINAL undropped sub-step sequence (see `reconstruct`'s module doc).
//!   A gap `>1` between consecutive entries means one or more ticks were
//!   dropped between them for insanity.
//! - `is_boundary` `[T]` (`i64`, 0/1): 1 iff this tick is the first
//!   simulated sub-step right after an authoritative snap to the replay's
//!   frame data (a 30 Hz interval boundary), 0 otherwise. Consumers building
//!   `(state[i], state[i+1]) -> action[i]` IDM pairs must drop the pair
//!   whenever `is_boundary[i+1] == 1` — that transition is a fresh step from
//!   a re-snapped arena, not `action[i]` applied to `state[i]`.
//!
//! `cars_state`'s column count is 16, not the 15 a naive plan sketch assumed
//! (pos3+vel3+ang_vel3+quat4 = 13, +boost+on_ground+demoed = 16) — every
//! field is kept because IDM/BC need `demoed` to mask dead-car ticks and
//! `on_ground` for the aerial-vs-grounded action split. The sidecar's
//! `cars_state_columns` is the source of truth; consumers should assert
//! against its length rather than hardcoding 15 or 16.
//!
//! `T`/`P` == 0 (empty reconstruction, e.g. a replay with no players or no
//! usable ticks) is treated as a hard error rather than writing an empty
//! shard — callers (the `replay-parse` CLI) should count that as a failed
//! replay, not a zero-row shard some downstream loader could accidentally
//! ingest.

use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2, Array3};
use ndarray_npy::NpzWriter;
use serde::Serialize;

use crate::{meta::ReplayMeta, reconstruct::Reconstructed};

/// Bump whenever the array layout or column semantics below change.
/// Downstream loaders should fail loud on a version they don't recognize.
///
/// v2: added `tick_index` `[T]` i64 and `is_boundary` `[T]` i64 (0/1) arrays
/// (see module doc) so a downstream IDM can detect dropped-tick gaps and
/// 30 Hz snap boundaries instead of silently mispairing across them.
pub const SHARD_SCHEMA_VERSION: u32 = 2;

pub const BALL_COLUMNS: [&str; 13] = [
    "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "quat_x", "quat_y", "quat_z", "quat_w",
];

pub const CARS_STATE_COLUMNS: [&str; 16] = [
    "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "ang_vel_x", "ang_vel_y", "ang_vel_z",
    "quat_x", "quat_y", "quat_z", "quat_w", "boost", "on_ground", "demoed",
];

pub const CARS_ACTION_COLUMNS: [&str; 8] =
    ["throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake"];

#[derive(Debug, Serialize)]
struct ShardSidecar {
    schema_version: u32,
    num_ticks: usize,
    num_players: usize,
    team_size: u8,
    /// Source replay frame rate the reconstruction was built from (i.e. the
    /// `--fps` the CLI passed to `extract_frames`), not the 120 Hz tick rate
    /// of the arrays themselves.
    fps: u32,
    playlist: String,
    ball_columns: Vec<String>,
    cars_state_columns: Vec<String>,
    cars_action_columns: Vec<String>,
}

/// Writes `rec` to `<out_dir>/<replay_id>.npz` (+ `<replay_id>.json`
/// sidecar). Returns the `.npz` path.
///
/// Errors (and writes nothing) if `rec` has zero ticks or zero players —
/// callers should treat that as a failed replay, not an empty shard.
pub fn write_shard(
    out_dir: &Path,
    replay_id: &str,
    meta: &ReplayMeta,
    rec: &Reconstructed,
) -> Result<PathBuf, String> {
    let num_ticks = rec.ticks.len();
    let num_players = rec.player_teams.len();

    if num_ticks == 0 || num_players == 0 {
        return Err(format!(
            "refusing to write empty shard for '{replay_id}' (num_ticks={num_ticks}, num_players={num_players})"
        ));
    }

    std::fs::create_dir_all(out_dir)
        .map_err(|e| format!("create_dir_all {}: {e}", out_dir.display()))?;

    let mut ball = Array2::<f32>::zeros((num_ticks, 13));
    let mut cars_state = Array3::<f32>::zeros((num_ticks, num_players, 16));
    let mut cars_action = Array3::<f32>::zeros((num_ticks, num_players, 8));
    let mut tick_index = Array1::<i64>::zeros(num_ticks);
    let mut is_boundary = Array1::<i64>::zeros(num_ticks);

    for (t, tick) in rec.ticks.iter().enumerate() {
        if tick.cars.len() != num_players || tick.actions.len() != num_players {
            return Err(format!(
                "tick {t}: expected {num_players} cars, got {} cars / {} actions",
                tick.cars.len(),
                tick.actions.len()
            ));
        }

        tick_index[t] = tick.tick_index;
        is_boundary[t] = if tick.is_boundary { 1 } else { 0 };

        let b = &tick.ball;
        ball[[t, 0]] = b.pos[0];
        ball[[t, 1]] = b.pos[1];
        ball[[t, 2]] = b.pos[2];
        ball[[t, 3]] = b.vel[0];
        ball[[t, 4]] = b.vel[1];
        ball[[t, 5]] = b.vel[2];
        ball[[t, 6]] = b.ang_vel[0];
        ball[[t, 7]] = b.ang_vel[1];
        ball[[t, 8]] = b.ang_vel[2];
        ball[[t, 9]] = b.quat[0];
        ball[[t, 10]] = b.quat[1];
        ball[[t, 11]] = b.quat[2];
        ball[[t, 12]] = b.quat[3];

        for (p, car) in tick.cars.iter().enumerate() {
            let rb = &car.rb;
            cars_state[[t, p, 0]] = rb.pos[0];
            cars_state[[t, p, 1]] = rb.pos[1];
            cars_state[[t, p, 2]] = rb.pos[2];
            cars_state[[t, p, 3]] = rb.vel[0];
            cars_state[[t, p, 4]] = rb.vel[1];
            cars_state[[t, p, 5]] = rb.vel[2];
            cars_state[[t, p, 6]] = rb.ang_vel[0];
            cars_state[[t, p, 7]] = rb.ang_vel[1];
            cars_state[[t, p, 8]] = rb.ang_vel[2];
            cars_state[[t, p, 9]] = rb.quat[0];
            cars_state[[t, p, 10]] = rb.quat[1];
            cars_state[[t, p, 11]] = rb.quat[2];
            cars_state[[t, p, 12]] = rb.quat[3];
            cars_state[[t, p, 13]] = car.boost;
            cars_state[[t, p, 14]] = if car.on_ground { 1.0 } else { 0.0 };
            cars_state[[t, p, 15]] = if car.demoed { 1.0 } else { 0.0 };

            let action = tick.actions[p];
            for (k, &v) in action.iter().enumerate() {
                cars_action[[t, p, k]] = v;
            }
        }
    }

    let mut player_teams = Array1::<i64>::zeros(num_players);
    for (p, &team) in rec.player_teams.iter().enumerate() {
        player_teams[p] = team as i64;
    }

    let npz_path = out_dir.join(format!("{replay_id}.npz"));
    let file = std::fs::File::create(&npz_path)
        .map_err(|e| format!("create {}: {e}", npz_path.display()))?;
    let mut npz = NpzWriter::new(file);
    npz.add_array("ball", &ball).map_err(|e| e.to_string())?;
    npz.add_array("cars_state", &cars_state).map_err(|e| e.to_string())?;
    npz.add_array("cars_action", &cars_action).map_err(|e| e.to_string())?;
    npz.add_array("player_teams", &player_teams).map_err(|e| e.to_string())?;
    npz.add_array("tick_index", &tick_index).map_err(|e| e.to_string())?;
    npz.add_array("is_boundary", &is_boundary).map_err(|e| e.to_string())?;
    npz.finish().map_err(|e| e.to_string())?;

    let sidecar = ShardSidecar {
        schema_version: SHARD_SCHEMA_VERSION,
        num_ticks,
        num_players,
        team_size: meta.team_size,
        fps: rec.fps,
        playlist: meta.playlist.clone(),
        ball_columns: BALL_COLUMNS.iter().map(|s| s.to_string()).collect(),
        cars_state_columns: CARS_STATE_COLUMNS.iter().map(|s| s.to_string()).collect(),
        cars_action_columns: CARS_ACTION_COLUMNS.iter().map(|s| s.to_string()).collect(),
    };
    let sidecar_path = out_dir.join(format!("{replay_id}.json"));
    let sidecar_file = std::fs::File::create(&sidecar_path)
        .map_err(|e| format!("create {}: {e}", sidecar_path.display()))?;
    serde_json::to_writer_pretty(sidecar_file, &sidecar).map_err(|e| e.to_string())?;

    Ok(npz_path)
}
