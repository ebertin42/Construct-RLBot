//! ReplayMutator reset-state pool: samples standalone ball+car "reset
//! states" out of a [`Reconstructed`] 120 Hz tick stream (Task 4), so a
//! future engine curriculum task can mix real human-replay states into
//! training resets via `Arena::set_ball`/`set_car` (spec §4, "replay-state
//! resets 0.7"). See `engine/src/curriculum.rs:98-134` for the exact
//! get-mutate-set idiom + field list this schema mirrors.
//!
//! ## jsonl schema (one [`ResetState`] per line, UTF-8 JSON, `serde_json`)
//! ```json
//! {"ball":{"pos":[x,y,z],"vel":[x,y,z],"ang_vel":[x,y,z]},
//!  "cars":[{"pos":[x,y,z],"vel":[x,y,z],"ang_vel":[x,y,z],
//!           "quat":[x,y,z,w],"boost":0.37,"team":0,"on_ground":true}, ...]}
//! ```
//! - Units: positions in uu, velocities in uu/s, `ang_vel` in rad/s — same
//!   raw RocketSim-native units as the rest of this crate (see
//!   `reconstruct.rs`'s header comment: no scale correction needed).
//! - Ball has **no rotation field**: `engine::curriculum::random_reset`
//!   never sets `BallState.rot_mat` either (the ball is a rotationally
//!   symmetric sphere; only `pos`/`vel`/`ang_vel` matter for `set_ball`).
//! - Car `quat` is `[x, y, z, w]` (matches `RigidFrame::quat`'s field
//!   order elsewhere in this crate). `CarState.rot_mat` is a `RotMat`, not a
//!   quaternion — the engine curriculum task that consumes this pool is
//!   expected to convert via the same `quat_to_rotmat` construction
//!   `reconstruct.rs` uses before calling `set_car`.
//! - `boost` is **0..1** here (matches `CarFrame::boost`), NOT the engine's
//!   `CarState.boost` range of `0..100`. The engine multiplies by 100 at
//!   load time, same as `reconstruct::set_car_state` already does.
//! - `team` is `0` = blue, `1` = orange (same convention as `player_teams`
//!   elsewhere in this crate).
//! - `on_ground` maps directly to `CarState.is_on_ground`.

use std::{
    io::Write,
    path::Path,
};

use serde::{Deserialize, Serialize};

use crate::reconstruct::Reconstructed;

/// Ball reset state: exactly the fields `Arena::set_ball` needs
/// (`engine/src/curriculum.rs:96-102`) — no rotation (see module doc).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BallSpawn {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
}

/// One car's reset state: exactly the fields `Arena::set_car` needs
/// (`engine/src/curriculum.rs:114-134`) — pos/vel/ang_vel, rotation (as a
/// quaternion; see module doc), boost (`0..1`, see module doc), and
/// `on_ground`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CarSpawn {
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    pub ang_vel: [f32; 3],
    /// `[x, y, z, w]` quaternion — engine converts to `RotMat` at load time.
    pub quat: [f32; 4],
    /// `0..1` — NOT `0..100` (see module doc).
    pub boost: f32,
    /// `0` = blue, `1` = orange.
    pub team: u8,
    pub on_ground: bool,
}

/// One full-arena reset state: ball + every car, snapshotted from a single
/// reconstructed 120 Hz tick.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ResetState {
    pub ball: BallSpawn,
    pub cars: Vec<CarSpawn>,
}

/// Minimal PCG32 (O'Neill) — deterministic, no external rng dependency.
/// Mirrors `engine::sampler::Pcg32`'s construction (duplicated rather than
/// imported: the `replay` crate has no dependency on `engine`).
struct Pcg32 {
    state: u64,
    inc: u64,
}

impl Pcg32 {
    fn new(seed: u64) -> Self {
        let mut s = Self { state: 0, inc: (seed << 1) | 1 };
        s.next_u32();
        s.state = s.state.wrapping_add(seed);
        s.next_u32();
        s
    }

    fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old.wrapping_mul(6364136223846793005).wrapping_add(self.inc);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform integer in `0..bound` (Lemire's multiply-high method; slight
    /// bias is immaterial at this sample size and there's no external rng
    /// dependency available to do better).
    fn next_below(&mut self, bound: usize) -> usize {
        if bound == 0 {
            return 0;
        }
        ((self.next_u32() as u64 * bound as u64) >> 32) as usize
    }
}

/// Deterministic index sample of `min(k, eligible_end - eligible_start)`
/// unique values from `eligible_start..eligible_end`, via a seeded partial
/// Fisher-Yates shuffle. Returned in ascending order (order is otherwise
/// arbitrary; sorting just makes the pool file's per-replay block readable).
fn sample_indices(eligible_start: usize, eligible_end: usize, k: usize, seed: u64) -> Vec<usize> {
    let mut pool: Vec<usize> = (eligible_start..eligible_end).collect();
    let n = pool.len();
    let take = k.min(n);
    let mut rng = Pcg32::new(seed);
    for i in 0..take {
        let j = i + rng.next_below(n - i);
        pool.swap(i, j);
    }
    let mut sampled: Vec<usize> = pool[..take].to_vec();
    sampled.sort_unstable();
    sampled
}

/// Ticks to skip at the start of the eligible range: ~2s @ 120 Hz, kickoff.
const SKIP_START_TICKS: usize = 240;
/// Ticks to skip at the end of the eligible range: ~0.5s @ 120 Hz, avoids
/// near-goal / terminal frames.
const SKIP_END_TICKS: usize = 60;

/// Samples up to `k` reset states from `rec`, uniformly over tick indices in
/// `[SKIP_START_TICKS, len - SKIP_END_TICKS)` (skipping kickoff and
/// near-terminal frames). If `rec` is shorter than the skip window, samples
/// from the whole tick range instead (still up to `k`, may be fewer if `rec`
/// has fewer ticks than `k`).
///
/// Deterministic: identical `(rec, k, seed)` always produces an identical
/// `Vec<ResetState>` (same seeded PCG32, same index order pre-sort).
pub fn sample_reset_states(rec: &Reconstructed, k: usize, seed: u64) -> Vec<ResetState> {
    let len = rec.ticks.len();
    if k == 0 || len == 0 {
        return Vec::new();
    }

    let (eligible_start, eligible_end) = if len > SKIP_START_TICKS + SKIP_END_TICKS {
        (SKIP_START_TICKS, len - SKIP_END_TICKS)
    } else {
        (0, len)
    };

    sample_indices(eligible_start, eligible_end, k, seed)
        .into_iter()
        .map(|i| {
            let tick = &rec.ticks[i];
            ResetState {
                ball: BallSpawn {
                    pos: tick.ball.pos,
                    vel: tick.ball.vel,
                    ang_vel: tick.ball.ang_vel,
                },
                cars: tick
                    .cars
                    .iter()
                    .map(|c| CarSpawn {
                        pos: c.rb.pos,
                        vel: c.rb.vel,
                        ang_vel: c.rb.ang_vel,
                        quat: c.rb.quat,
                        boost: c.boost,
                        team: c.team,
                        on_ground: c.on_ground,
                    })
                    .collect(),
            }
        })
        .collect()
}

/// Writes `states` to `path` as jsonl (one [`ResetState`] JSON object per
/// line). `append == true` appends to an existing file (creating it if
/// absent); `append == false` truncates/creates fresh. Creates `path`'s
/// parent directory if missing.
pub fn write_pool_jsonl(path: &Path, states: &[ResetState], append: bool) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
        }
    }

    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(append)
        .write(true)
        .truncate(!append)
        .open(path)
        .map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(file);
    for state in states {
        let line = serde_json::to_string(state).map_err(|e| format!("serialize: {e}"))?;
        writeln!(w, "{line}").map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    w.flush().map_err(|e| format!("flush {}: {e}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_indices_are_unique_and_sorted() {
        let idx = sample_indices(240, 1000, 20, 7);
        assert_eq!(idx.len(), 20);
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(idx, sorted, "already sorted ascending");
        let mut uniq = idx.clone();
        uniq.dedup();
        assert_eq!(uniq.len(), idx.len(), "no duplicate indices");
        assert!(idx.iter().all(|&i| (240..1000).contains(&i)));
    }

    #[test]
    fn sample_indices_caps_at_available_range() {
        let idx = sample_indices(0, 5, 20, 1);
        assert_eq!(idx.len(), 5, "fewer than k available -> return what's there");
    }

    #[test]
    fn write_pool_jsonl_roundtrips() {
        let dir = std::env::temp_dir().join(format!("reset_pool_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pool.jsonl");
        let states = vec![ResetState {
            ball: BallSpawn { pos: [1.0, 2.0, 3.0], vel: [0.0; 3], ang_vel: [0.0; 3] },
            cars: vec![CarSpawn {
                pos: [0.0; 3],
                vel: [0.0; 3],
                ang_vel: [0.0; 3],
                quat: [0.0, 0.0, 0.0, 1.0],
                boost: 0.5,
                team: 0,
                on_ground: true,
            }],
        }];
        write_pool_jsonl(&path, &states, false).unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text.lines().count(), 1);
        let parsed: ResetState = serde_json::from_str(text.lines().next().unwrap()).unwrap();
        assert_eq!(parsed, states[0]);

        // append mode adds a second line rather than truncating
        write_pool_jsonl(&path, &states, true).unwrap();
        let text2 = std::fs::read_to_string(&path).unwrap();
        assert_eq!(text2.lines().count(), 2);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
