use std::pin::Pin;

use rocketsim_rs::{
    math::{Angle, Vec3},
    sim::Arena,
};
use serde::Deserialize;

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

#[derive(Debug, Clone, Deserialize)]
pub struct CurriculumConfig {
    pub kickoff_weight: f32,
    pub random_weight: f32,
    #[serde(default)]
    pub random: RandomStateBounds,
}

impl CurriculumConfig {
    pub fn load(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
        let c: Self = toml::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
        if c.kickoff_weight < 0.0 || c.random_weight < 0.0 || c.kickoff_weight + c.random_weight <= 0.0 {
            return Err(format!("{path}: weights must be nonnegative and sum > 0"));
        }
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
}
