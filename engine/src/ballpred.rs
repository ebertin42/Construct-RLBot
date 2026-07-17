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
        Self { arena: Self::build_arena() }
    }

    /// Copies `ball` into the tracker arena, then steps it 60 ticks (0.5s at
    /// the arena's 120Hz tick rate) four times, snapshotting (pos, vel) after
    /// each step. Returns snapshots at +60/120/180/240 ticks == +0.5/1/1.5/2s.
    /// Allocation-free on the healthy path: `set_ball`/`step`/`get_ball` all
    /// operate on the already-owned `UniquePtr<Arena>` and return/take `Copy`
    /// structs.
    ///
    /// NaN containment (same philosophy as `EpisodeArena`'s `state_is_finite`
    /// guard, see the physics-NaN playbook): Bullet's solver output for a
    /// freshly constructed arena is sensitive to heap-allocation history —
    /// observed 2026-07-16, a fresh tracker's *first* predict returned all-NaN
    /// snapshots in one binary layout and ~0.01uu-perturbed (finite) ones in
    /// another. Nonfinite snapshots must never reach the obs tensor (they'd
    /// flow straight into the candle net), so they collapse to the "no
    /// prediction" fallback: the input ball state at every horizon. If the
    /// *input* ball is itself nonfinite the arena's own containment
    /// (episode.rs) owns that case — the real ball entity row would carry the
    /// same NaN regardless of what we return here.
    ///
    /// The arena does NOT self-heal on the next `set_ball` (an earlier
    /// version of this doc claimed it did — empirically false, verified over
    /// a 13,120-tick run 2026-07-17: 69% echo predictions in the v4 replay
    /// corpus). Bullet's `updateSingleAabb` latches `DISABLE_SIMULATION` on
    /// the ball body the instant its AABB goes nonfinite, and RocketSim never
    /// clears that latch (`Ball::SetState` only attempts activation when
    /// velocity != 0, and is refused anyway; RocketSim never calls
    /// `forceActivationState`) — same one-way latch `EpisodeArena` hit (see
    /// `episode.rs`'s `rebuild_arena`, commit a1c33e0). Left in place, the
    /// poisoned body silently produces nonfinite snapshots on every
    /// subsequent call forever, so `contain_nonfinite` keeps tripping and
    /// `predict` returns nothing but an echo of whatever ball state it was
    /// just given — never a genuine rollout — for the rest of the process.
    /// The fix mirrors `rebuild_arena`: on containment, discard the poisoned
    /// arena and swap in a fresh one (same construction as `new`/`build_arena`)
    /// before returning, so the *next* `predict` gets a clean body instead of
    /// the latched one. Rare event (containment trips), so the allocation
    /// cost of a rebuild is irrelevant; the healthy path is unaffected.
    pub fn predict(&mut self, ball: &BallState) -> [BallSnap; 4] {
        self.arena.pin_mut().set_ball(*ball);
        let mut out = [BallSnap::default(); 4];
        for snap in out.iter_mut() {
            self.arena.pin_mut().step(60);
            let b = self.arena.pin_mut().get_ball();
            *snap = BallSnap { pos: b.pos, vel: b.vel };
        }
        if !all_finite(&out) {
            // Poisoned: rebuild before returning so the arena we hand back
            // control with is never the DISABLE_SIMULATION-latched one.
            self.rebuild_arena();
        }
        contain_nonfinite(out, ball)
    }

    fn build_arena() -> UniquePtr<Arena> {
        Arena::default_standard()
    }

    /// Discards the (possibly poisoned) arena and replaces it with a fresh
    /// one built the same way the constructor does. See `predict`'s doc for
    /// why resetting the poisoned arena in place is not an option.
    fn rebuild_arena(&mut self) {
        self.arena = Self::build_arena();
    }
}

fn vec3_finite(v: &Vec3) -> bool {
    v.x.is_finite() && v.y.is_finite() && v.z.is_finite()
}

fn all_finite(out: &[BallSnap; 4]) -> bool {
    out.iter().all(|s| vec3_finite(&s.pos) && vec3_finite(&s.vel))
}

/// If any snapshot went nonfinite, replace the whole prediction with the
/// input ball state repeated at every horizon ("ball keeps doing what it's
/// doing now" — finite whenever the input is, which episode.rs guarantees).
fn contain_nonfinite(out: [BallSnap; 4], ball: &BallState) -> [BallSnap; 4] {
    if all_finite(&out) {
        out
    } else {
        [BallSnap { pos: ball.pos, vel: ball.vel }; 4]
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
    fn nonfinite_snapshots_collapse_to_input_fallback() {
        let ball = {
            let mut b = stationary_ball([100.0, -200.0, 300.0]);
            b.vel = Vec3::new(1.0, 2.0, 3.0);
            b
        };
        let good = [BallSnap { pos: Vec3::new(1.0, 2.0, 3.0), vel: Vec3::new(4.0, 5.0, 6.0) }; 4];
        assert_eq!(contain_nonfinite(good, &ball), good, "finite snaps pass through untouched");

        let mut bad = good;
        bad[2].pos.y = f32::NAN;
        let contained = contain_nonfinite(bad, &ball);
        let fb = BallSnap { pos: ball.pos, vel: ball.vel };
        assert_eq!(contained, [fb; 4], "any NaN collapses all horizons to the input ball");

        let mut inf = good;
        inf[0].vel.z = f32::INFINITY;
        assert_eq!(contain_nonfinite(inf, &ball), [fb; 4]);
    }

    #[test]
    fn poisoned_arena_rebuilds_and_next_prediction_evolves() {
        // Deterministic poison: feed a NaN ball state straight through the
        // public `predict` API, mirroring episode.rs's `debug_place_ball`
        // NaN-position poisoning technique (see
        // `containment_rebuilds_arena_leaving_live_physics`,
        // engine/tests/episode_test.rs) instead of relying on a natural
        // Bullet blowup — an injected NaN reproduces the
        // DISABLE_SIMULATION latch deterministically every run.
        ensure_init(None);
        let mut t = Tracker::new();

        let mut poison = BallState::default();
        poison.pos = Vec3::new(f32::NAN, f32::NAN, f32::NAN);
        t.predict(&poison);

        // Hand the SAME tracker a perfectly healthy, airborne, moving ball.
        // Pre-fix: Bullet's `updateSingleAabb` latched DISABLE_SIMULATION on
        // the ball body the instant its AABB went nonfinite above, and that
        // latch never clears (`Ball::SetState` only attempts activation for
        // nonzero velocity, and is refused anyway). The observed mechanism is
        // even more specific than "nonfinite forever": `set_ball` still
        // writes whatever position we hand it, but a disabled body's
        // position no longer gets integrated by `step` at all, so it stays
        // pinned bit-exactly at that just-set input for every one of the 4
        // horizons, on every subsequent call, forever — velocity alone keeps
        // drifting (Bullet still applies gravity/damping to the stored
        // velocity of a disabled body without ever moving it) which is why
        // this is checked via *position*, not full snapshot equality.
        // Empirically verified live over a 13,120-tick / ~54-call run.
        let mut healthy = BallState::default();
        // High and slow enough to stay airborne (never touch the ground or
        // ceiling) for the full 2s horizon -- a ground bounce is a legitimate
        // but non-monotonic trajectory (and, near the docs' "insane-but-
        // finite blowup precursor," a needlessly fragile one to assert on),
        // so free-fall keeps the gravity check simple and robust.
        healthy.pos = Vec3::new(0.0, 0.0, 1800.0); // airborne, ~500uu clear of the ground after 2s of free-fall
        healthy.vel = Vec3::new(800.0, 0.0, 0.0); // moving sideways, well clear of walls/goals

        // Retry a bounded number of times rather than asserting on the very
        // first post-poison call: a rebuilt arena's first step is subject to
        // the SAME (unrelated, already-documented) allocation-history-
        // sensitive Bullet flake that a brand-new `Tracker::new()` can hit
        // (see module doc, and `ballpred_stationary_and_moving` in
        // engine/tests/obs_v1_test.rs) — that flake self-heals within a call
        // or two. A permanent DISABLE_SIMULATION latch cannot self-heal no
        // matter how many retries (position stays pinned at the input
        // forever), so this loop cleanly discriminates "still poisoned
        // forever" (pre-fix, must FAIL) from "rebuilt and healthy" (post-fix,
        // must PASS) without being sensitive to the unrelated transient
        // flake.
        let moved_from_input = |s: &BallSnap| {
            (s.pos.x - healthy.pos.x).abs() > 5.0 || (s.pos.z - healthy.pos.z).abs() > 5.0
        };
        let mut healed = None;
        for _ in 0..10 {
            let snaps = t.predict(&healthy);
            let all_finite = snaps.iter().all(|s| s.pos.x.is_finite() && s.pos.y.is_finite() && s.pos.z.is_finite());
            if all_finite && moved_from_input(&snaps[3]) {
                healed = Some(snaps);
                break;
            }
        }
        let snaps = healed.expect(
            "position must move away from the input within 10 retries -- a permanent \
             DISABLE_SIMULATION latch pins position at the input forever (this is the bug under test)",
        );

        assert_ne!(
            snaps[0], snaps[3],
            "snapshots must differ across horizons — a frozen/echoed prediction repeats the same snap 4x"
        );

        // Physically-motivated evolution, not just "some" numerical drift:
        // gravity must pull the airborne ball down, and it must have made
        // forward progress consistent with its own speed.
        assert!(
            snaps[3].pos.z < healthy.pos.z - 200.0,
            "gravity must act on the airborne ball, not echo the input: {:?}",
            snaps
        );
        assert!(
            snaps[3].pos.x > healthy.pos.x + 100.0,
            "prediction must evolve forward in x, not echo the input: {:?}",
            snaps
        );
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
