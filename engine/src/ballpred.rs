//! Ball-prediction tracker (T3, entity-transformer obs v1).
//!
//! Owns a persistent, car-less standard arena used purely for ball physics
//! roll-forward: no cars are ever added to it, so every `step` only advances
//! ball/ground/wall/goal-post collision — never car-ball or car-car contact.
//! This intentionally does NOT predict deflections off cars we can't see the
//! future actions of; it's a "what does the ball do if nothing touches it"
//! prior, which is what the plan's obs v1 spec asks for (ball-pred entities
//! 13-16, see `obs_v1`).
//!
//! `Arena::clone(bool)` in rocketsim_rs 0.37 takes a `copy_callbacks: bool`
//! parameter, not a "keep cars" flag, so it isn't the right tool to get a
//! car-less arena — a fresh `Arena::default_standard()` with zero `add_car`
//! calls is simpler and is exactly "car-less" by construction.
use cxx::UniquePtr;
use rocketsim_rs::{
    math::Vec3,
    sim::{Arena, BallState},
};

/// A single predicted-ball snapshot (position + velocity only — no angular
/// velocity; the obs v1 entity row for ball-pred entities doesn't carry one,
/// see `obs_v1`'s `EntityRow`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct BallSnap {
    pub pos: Vec3,
    pub vel: Vec3,
}

/// Persistent car-less arena used to roll the ball forward without
/// reallocating per call. One `Tracker` per env arena (owned by
/// `EpisodeArena` starting T6) — no global/shared state.
pub struct Tracker {
    arena: UniquePtr<Arena>,
}

impl Tracker {
    /// Builds a fresh standard arena with no cars. Requires
    /// `sim_init::ensure_init` to have already run (same precondition as
    /// every other `Arena::default_standard()` call site in this crate).
    pub fn new() -> Self {
        Self { arena: Arena::default_standard() }
    }

    /// Copies `ball` into the tracker arena, then steps it 60 ticks (0.5s at
    /// the arena's 120Hz tick rate) four times, snapshotting (pos, vel) after
    /// each step. Returns snapshots at +60/120/180/240 ticks == +0.5/1/1.5/2s.
    /// Allocation-free: `set_ball`/`step`/`get_ball` all operate on the
    /// already-owned `UniquePtr<Arena>` and return/take `Copy` structs.
    pub fn predict(&mut self, ball: &BallState) -> [BallSnap; 4] {
        self.arena.pin_mut().set_ball(*ball);
        let mut out = [BallSnap::default(); 4];
        for snap in out.iter_mut() {
            self.arena.pin_mut().step(60);
            let b = self.arena.pin_mut().get_ball();
            *snap = BallSnap { pos: b.pos, vel: b.vel };
        }
        out
    }
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim_init::ensure_init;

    fn stationary_ball(pos: [f32; 3]) -> BallState {
        let mut b = BallState::default();
        b.pos = Vec3::new(pos[0], pos[1], pos[2]);
        b
    }

    #[test]
    fn kickoff_ball_stays_near_center() {
        ensure_init(None);
        let mut t = Tracker::new();
        let ball = stationary_ball([0.0, 0.0, 93.15]);
        let snaps = t.predict(&ball);
        for (i, s) in snaps.iter().enumerate() {
            assert!(s.pos.x.is_finite() && s.pos.y.is_finite() && s.pos.z.is_finite(), "snap {i} pos finite");
            assert!(s.vel.x.is_finite() && s.vel.y.is_finite() && s.vel.z.is_finite(), "snap {i} vel finite");
            let dx = s.pos.x - 0.0;
            let dy = s.pos.y - 0.0;
            let horiz = (dx * dx + dy * dy).sqrt();
            assert!(horiz < 50.0, "snap {i} drifted horizontally too far: {horiz}");
        }
    }

    #[test]
    fn moving_ball_y_increases_across_horizons() {
        ensure_init(None);
        let mut t = Tracker::new();
        let mut ball = stationary_ball([0.0, 0.0, 93.15]);
        ball.vel = Vec3::new(0.0, 2000.0, 0.0);
        let snaps = t.predict(&ball);
        assert!(snaps.iter().all(|s| s.pos.y.is_finite() && s.pos.x.is_finite() && s.pos.z.is_finite()));
        for w in snaps.windows(2) {
            assert!(w[1].pos.y > w[0].pos.y, "y must strictly increase: {:?}", snaps);
        }
        assert!(snaps[0].pos.y > 0.0);
    }

    #[test]
    fn predict_is_deterministic() {
        ensure_init(None);
        let mut t1 = Tracker::new();
        let mut t2 = Tracker::new();
        let mut ball = stationary_ball([500.0, -1200.0, 300.0]);
        ball.vel = Vec3::new(300.0, -800.0, 200.0);
        let s1 = t1.predict(&ball);
        let s2 = t2.predict(&ball);
        assert_eq!(s1, s2, "same input ball state must produce identical snapshots");
    }
}
