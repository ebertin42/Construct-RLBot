use std::pin::Pin;
use std::sync::Arc;

use rocketsim_rs::{
    math::{Angle, Vec3},
    sim::Arena,
};
use serde::Deserialize;

use crate::reset_pool::{self, ResetState};
use crate::sampler::Pcg32;

#[derive(Debug, Clone, Deserialize)]
pub struct RandomStateBounds {
    pub car_speed_max: f32,
    pub ball_speed_max: f32,
    pub z_max: f32,
    pub min_separation: f32,
}

impl Default for RandomStateBounds {
    fn default() -> Self {
        Self { car_speed_max: 1800.0, ball_speed_max: 2500.0, z_max: 1700.0, min_separation: 300.0 }
    }
}

/// Where to find the replay-state reset pool, and how much of it to keep.
#[derive(Debug, Clone, Deserialize)]
pub struct ReplayPoolConfig {
    pub path: String,
    /// `0` = unlimited (load every clean state). Anything else caps the pool via
    /// reservoir sampling — see `reset_pool::load_or_empty`.
    #[serde(default)]
    pub max_states: usize,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct CurriculumConfig {
    pub kickoff_weight: f32,
    pub random_weight: f32,
    /// Defaults to 0.0 so `curriculum_v1.toml` (which has no such key) keeps
    /// its exact pre-replay-pool behavior — see `episode::reset_episode`, where
    /// a zero replay weight is proven bit-identical to the legacy two-way coin.
    #[serde(default)]
    pub replay_weight: f32,
    #[serde(default)]
    pub random: RandomStateBounds,
    #[serde(default)]
    pub replay_pool: Option<ReplayPoolConfig>,
    /// Task #56 Phase 1: run FULL MATCHES instead of single episodes. A goal
    /// updates the score and kicks off again; only clock expiry terminates.
    ///
    /// Defaults to false, and that default is load-bearing: with match_mode on,
    /// goals stop terminating episodes, which changes reset dynamics and would
    /// make the goal-share gate incomparable with every historical entry in
    /// logs/champion_history.jsonl. The screen must keep running legacy mode.
    #[serde(default)]
    pub match_mode: bool,
    /// Populated by `load()` after the TOML parse. The `Arc` is load-bearing:
    /// `MultiEngine::new` clones this config once per worker AND
    /// `EpisodeArena::new_full` clones it once per arena. By value the full
    /// pool is ~169 MB x 192 arenas = ~32 GB; as an `Arc` it is a refcount bump.
    #[serde(skip)]
    pub pool: Arc<Vec<ResetState>>,
}

impl CurriculumConfig {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let mut c: Self = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        if c.replay_weight < 0.0 {
            return Err(format!("{path}: replay_weight must be nonnegative, got {}", c.replay_weight));
        }
        if c.kickoff_weight < 0.0
            || c.random_weight < 0.0
            || c.kickoff_weight + c.random_weight + c.replay_weight <= 0.0
        {
            return Err(format!(
                "{path}: kickoff_weight, random_weight and replay_weight must be nonnegative and sum > 0"
            ));
        }
        // A declared replay share with no pool section is a config typo, not a
        // runtime condition — catch it before the run starts. (A *missing pool
        // file* is the opposite: see the load below, which never errors.)
        if c.replay_weight > 0.0 && c.replay_pool.is_none() {
            return Err(format!("{path}: replay_weight > 0 requires a [replay_pool] section"));
        }
        let b = &c.random;
        if b.z_max <= 93.15 {
            return Err(format!("{path}: z_max must be > 93.15 (ball resting height), got {}", b.z_max));
        }
        if b.car_speed_max <= 0.0 {
            return Err(format!("{path}: car_speed_max must be > 0, got {}", b.car_speed_max));
        }
        if b.ball_speed_max <= 0.0 {
            return Err(format!("{path}: ball_speed_max must be > 0, got {}", b.ball_speed_max));
        }
        if b.min_separation < 0.0 {
            return Err(format!("{path}: min_separation must be >= 0, got {}", b.min_separation));
        }
        // Loaded here, not in `MultiEngine::new`: this is the one existing call
        // site that already does IO + validation, it runs once on the caller
        // thread before any `thread::spawn`, and it leaves every constructor
        // signature in episode.rs/engine.rs/lib.rs untouched. Path resolves
        // relative to CWD, same as every other config path in the repo.
        // Gated on the weight, not merely on the section being present: the
        // natural rollback for this lever is `replay_weight = 0`, and paying a
        // ~2.1 s / 133 MB load (measured on the v5 corpus) for a pool no draw
        // can ever reach makes that rollback needlessly expensive — the Arc is
        // then held by every one of 192 arenas for the life of the run.
        if c.replay_weight > 0.0 {
            if let Some(rp) = &c.replay_pool {
                c.pool = Arc::new(reset_pool::load_or_empty(&rp.path, rp.max_states));
                // Only `load()` knows both the pool size and the weights, so the
                // effective-mix warning has to live here rather than in the loader.
                if c.pool.is_empty() {
                    let rest = c.kickoff_weight + c.random_weight;
                    let (k, r) = if rest > 0.0 {
                        (c.kickoff_weight / rest, c.random_weight / rest)
                    } else {
                        (1.0, 0.0)
                    };
                    eprintln!(
                        "[curriculum] WARNING: replay pool is EMPTY; replay resets DISABLED, \
                         effective mix kickoff {k:.3} / random {r:.3}"
                    );
                }
            }
        }
        // Unconditional, and stated as the REALIZED mixture rather than the
        // configured one. `replay_weight` is `#[serde(default)]` (it has to be,
        // so curriculum_v1 keeps working), and there is no
        // `deny_unknown_fields`, so a typo like `replay_weigth = 0.7` parses
        // fine, silently yields 0.0, and — because the empty-pool warning above
        // is itself gated on the weight — leaves an operator looking at a clean
        // startup, concluding the v2 switch took, while the run trains on the v1
        // distribution. This line is the one place that cannot lie about it.
        let total = c.replay_weight + c.kickoff_weight + c.random_weight;
        let eff = if c.pool.is_empty() { 0.0 } else { c.replay_weight };
        // `> 0.0` is validated for the CONFIGURED sum, but zeroing an empty
        // pool's share can still leave nothing behind (replay 1.0 / 0 / 0 with
        // no pool file), which would print NaN. That case degrades to kickoff.
        let t = if eff + c.kickoff_weight + c.random_weight > 0.0 {
            eff + c.kickoff_weight + c.random_weight
        } else {
            1.0
        };
        eprintln!(
            "[curriculum] {path}: effective mix replay {:.3} / kickoff {:.3} / random {:.3} \
             (pool {} states, configured replay_weight {:.3} of {:.3})",
            eff / t,
            c.kickoff_weight / t,
            c.random_weight / t,
            c.pool.len(),
            c.replay_weight,
            total,
        );
        Ok(c)
    }
}

const TAU: f32 = std::f32::consts::TAU;

fn rand_range(rng: &mut Pcg32, lo: f32, hi: f32) -> f32 {
    lo + rng.next_f32() * (hi - lo)
}

fn rand_vel(rng: &mut Pcg32, max: f32) -> Vec3 {
    // random direction (uniform-ish over sphere via yaw+pitch), random magnitude
    let yaw = rand_range(rng, 0.0, TAU);
    let pitch = rand_range(rng, -1.2, 1.2);
    let mag = rng.next_f32() * max;
    Vec3::new(mag * pitch.cos() * yaw.cos(), mag * pitch.cos() * yaw.sin(), mag * pitch.sin())
}

/// Overwrite ball + all car states with a bounded random scenario. Positions are
/// resampled up to 10 times to keep `min_separation` from already-placed bodies;
/// after 10 attempts the last sample is accepted (best-effort, never spins).
pub fn random_reset(mut arena: Pin<&mut Arena>, rng: &mut Pcg32, b: &RandomStateBounds) {
    let mut placed: Vec<Vec3> = Vec::new();
    let sample_pos = |rng: &mut Pcg32, z_lo: f32, z_hi: f32, placed: &[Vec3]| -> Vec3 {
        let mut p = Vec3::new(0.0, 0.0, z_lo);
        for _ in 0..10 {
            p = Vec3::new(
                rand_range(rng, -3500.0, 3500.0),
                rand_range(rng, -4500.0, 4500.0),
                rand_range(rng, z_lo, z_hi),
            );
            let ok = placed.iter().all(|q| {
                let d2 = (p.x - q.x).powi(2) + (p.y - q.y).powi(2) + (p.z - q.z).powi(2);
                d2 >= b.min_separation * b.min_separation
            });
            if ok {
                break;
            }
        }
        p
    };

    // ball first
    let ball_pos = sample_pos(rng, 93.15, b.z_max, &placed);
    placed.push(ball_pos);
    let mut ball = arena.as_mut().get_ball();
    ball.pos = ball_pos;
    ball.vel = rand_vel(rng, b.ball_speed_max);
    ball.ang_vel = Vec3::new(0.0, 0.0, 0.0);
    arena.as_mut().set_ball(ball);

    // cars
    let ids: Vec<u32> = {
        let gs = arena.as_mut().get_game_state();
        gs.cars.iter().map(|c| c.id).collect()
    };
    for id in ids {
        let grounded = rng.next_f32() < 0.5;
        let (z_lo, z_hi) = if grounded { (17.0, 17.0) } else { (100.0, b.z_max) };
        let pos = sample_pos(rng, z_lo, z_hi, &placed);
        placed.push(pos);
        let mut cs = arena.as_mut().get_car(id);
        cs.pos = pos;
        cs.vel = if grounded {
            let v = rand_vel(rng, b.car_speed_max);
            Vec3::new(v.x, v.y, 0.0)
        } else {
            rand_vel(rng, b.car_speed_max)
        };
        cs.ang_vel = Vec3::new(0.0, 0.0, 0.0);
        let ang = Angle {
            yaw: rand_range(rng, 0.0, TAU),
            // Airborne pitch/roll deliberately restricted to ±1.0 rad (not full range):
            // near-vertical spawns are degenerate — floor/wall clipping, unrecoverable
            // states. See the plan's contract amendment.
            pitch: if grounded { 0.0 } else { rand_range(rng, -1.0, 1.0) },
            roll: if grounded { 0.0 } else { rand_range(rng, -1.0, 1.0) },
        };
        cs.rot_mat = ang.to_rotmat();
        cs.boost = rand_range(rng, 0.0, 100.0);
        cs.is_on_ground = grounded;
        arena.as_mut().set_car(id, cs).expect("car exists");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_bounds_match_brief() {
        let b = RandomStateBounds::default();
        assert_eq!(b.car_speed_max, 1800.0);
        assert_eq!(b.ball_speed_max, 2500.0);
        assert_eq!(b.z_max, 1700.0);
        assert_eq!(b.min_separation, 300.0);
    }

    #[test]
    fn load_rejects_zero_weights() {
        let dir = std::env::temp_dir();
        let path = dir.join("construct_curriculum_zero_weights_test.toml");
        std::fs::write(&path, "kickoff_weight = 0.0\nrandom_weight = 0.0\n").unwrap();
        let err = CurriculumConfig::load(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("sum > 0"), "{err}");
        let _ = std::fs::remove_file(&path);
    }

    /// Writes a temp curriculum TOML and returns its path.
    fn tmp_toml(name: &str, body: &str) -> String {
        let p = std::env::temp_dir().join(format!("construct_curriculum_{name}.toml"));
        std::fs::write(&p, body).unwrap();
        p.to_str().unwrap().to_string()
    }

    // ---- T13 ----
    #[test]
    fn v1_config_still_loads_with_default_replay_weight() {
        let c = CurriculumConfig::load("../configs/curriculum_v1.toml").unwrap();
        assert_eq!(c.replay_weight, 0.0, "v1 must default to no replay resets");
        assert!(c.replay_pool.is_none());
        assert!(c.pool.is_empty());
    }

    // ---- T14 ----
    #[test]
    fn rejects_negative_replay_weight() {
        let p = tmp_toml(
            "neg_replay",
            "kickoff_weight = 0.5\nrandom_weight = 0.5\nreplay_weight = -0.1\n",
        );
        let err = CurriculumConfig::load(&p).unwrap_err();
        assert!(err.contains("replay_weight"), "error must name the field: {err}");
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn rejects_all_zero_weights() {
        let p = tmp_toml(
            "all_zero",
            "kickoff_weight = 0.0\nrandom_weight = 0.0\nreplay_weight = 0.0\n",
        );
        let err = CurriculumConfig::load(&p).unwrap_err();
        assert!(err.contains("sum > 0"), "{err}");
        let _ = std::fs::remove_file(&p);

        // ...but replay alone is a legitimate (test/debug) mixture.
        let q = tmp_toml(
            "replay_only",
            "kickoff_weight = 0.0\nrandom_weight = 0.0\nreplay_weight = 1.0\n\n[replay_pool]\npath = \"/nope.jsonl\"\n",
        );
        let c = CurriculumConfig::load(&q).unwrap();
        assert_eq!(c.replay_weight, 1.0);
        let _ = std::fs::remove_file(&q);
    }

    // ---- T15 ----
    #[test]
    fn replay_weight_without_pool_section_is_an_error() {
        let p = tmp_toml(
            "no_pool_section",
            "kickoff_weight = 0.1\nrandom_weight = 0.2\nreplay_weight = 0.7\n",
        );
        let err = CurriculumConfig::load(&p).unwrap_err();
        assert!(
            err.contains("requires a [replay_pool] section"),
            "a missing section is a config typo, not a runtime condition: {err}"
        );
        let _ = std::fs::remove_file(&p);
    }

    // ---- T16: the graceful-degradation gate ----
    #[test]
    fn missing_pool_file_is_not_a_load_error() {
        let p = tmp_toml(
            "missing_pool_file",
            "kickoff_weight = 0.1\nrandom_weight = 0.2\nreplay_weight = 0.7\n\n[replay_pool]\npath = \"/nope/absent.jsonl\"\n",
        );
        let c = CurriculumConfig::load(&p).expect("an absent pool file must NOT fail the load");
        assert!(c.pool.is_empty());
        assert_eq!(c.replay_weight, 0.7);
        let _ = std::fs::remove_file(&p);
    }

    #[test]
    fn load_rejects_bad_z_max() {
        let dir = std::env::temp_dir();
        let path = dir.join("construct_curriculum_bad_z_max_test.toml");
        std::fs::write(
            &path,
            "kickoff_weight = 0.5\nrandom_weight = 0.5\n\n[random]\ncar_speed_max = 1800.0\nball_speed_max = 2500.0\nz_max = 50.0\nmin_separation = 300.0\n",
        )
        .unwrap();
        let err = CurriculumConfig::load(path.to_str().unwrap()).unwrap_err();
        assert!(err.contains("z_max"), "error must name the field: {err}");
        let _ = std::fs::remove_file(&path);
    }
}
