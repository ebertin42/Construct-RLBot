use construct_engine::ballpred::{BallSnap, Tracker};
use construct_engine::obs_v1::{self, ENT_FEAT, MAX_ENT, Q_FEAT};
use construct_engine::{schema::Schema, sim_init::ensure_init};
use rocketsim_rs::sim::{Arena, CarConfig, Team};

fn norm() -> construct_engine::schema::Normalization {
    Schema::load("../schema/v1.toml").unwrap().normalization
}

fn kickoff_1v1(seed: u32) -> rocketsim_rs::GameState {
    ensure_init(None);
    let mut arena = Arena::default_standard();
    arena.pin_mut().add_car(Team::Blue, CarConfig::octane());
    arena.pin_mut().add_car(Team::Orange, CarConfig::octane());
    arena.pin_mut().reset_to_random_kickoff(Some(seed));
    arena.pin_mut().get_game_state()
}

#[test]
fn layout_golden_1v1_kickoff() {
    let gs = kickoff_1v1(7);
    let pred = [BallSnap::default(); 4];
    let nrm = norm();
    let mut ents = [0.0f32; MAX_ENT * ENT_FEAT];
    let mut mask = [false; MAX_ENT];
    let mut query = [0.0f32; Q_FEAT];
    obs_v1::build(&gs, 0, &pred, &nrm, &mut ents, &mut mask, &mut query);

    // self present, IS_SELF one-hot
    assert!(!mask[0]);
    assert_eq!(&ents[0..5], &[1.0, 0.0, 0.0, 0.0, 0.0]);

    // 1v1: both mate slots masked, one opponent present + one masked opp slot,
    // and one further masked opp slot (mates start at 1, opps at 3)
    assert!(mask[1] && mask[2], "no mates in 1v1");
    assert!(!mask[3], "single opponent present");
    assert_eq!(&ents[3 * ENT_FEAT..3 * ENT_FEAT + 5], &[0.0, 0.0, 1.0, 0.0, 0.0]);
    assert!(mask[4] && mask[5], "remaining opp slots masked in 1v1");

    // ball: present, IS_BALL, position matches normalized arena ball
    assert!(!mask[6]);
    let ball_row = &ents[6 * ENT_FEAT..7 * ENT_FEAT];
    assert_eq!(&ball_row[0..5], &[0.0, 0.0, 0.0, 1.0, 0.0]);
    let pk = nrm.pos_norm as f32;
    assert!((ball_row[5] - gs.ball.pos.x * pk).abs() < 1e-5);
    assert!((ball_row[6] - gs.ball.pos.y * pk).abs() < 1e-5);
    assert!((ball_row[7] - gs.ball.pos.z * pk).abs() < 1e-5);

    // 6 big pads present with IS_PAD flag
    for slot in 0..6 {
        let idx = 7 + slot;
        assert!(!mask[idx], "pad slot {slot}");
        let row = &ents[idx * ENT_FEAT..idx * ENT_FEAT + 5];
        assert_eq!(row, &[0.0, 0.0, 0.0, 0.0, 1.0], "pad slot {slot} one-hot");
    }

    // 4 ball-pred entities present, horizon feature 0.25/0.5/0.75/1.0
    let horizons = [0.25f32, 0.5, 0.75, 1.0];
    for slot in 0..4 {
        let idx = 13 + slot;
        assert!(!mask[idx], "pred slot {slot}");
        let row = &ents[idx * ENT_FEAT..idx * ENT_FEAT + ENT_FEAT];
        assert!((row[24] - horizons[slot]).abs() < 1e-6, "pred slot {slot} horizon = {}", row[24]);
    }

    assert!(ents.iter().all(|x| x.is_finite()), "ents must be finite");
    assert!(query.iter().all(|x| x.is_finite()), "query must be finite");
    assert_eq!(mask.len(), MAX_ENT);
}

#[test]
fn orange_mirrors_blue_at_kickoff() {
    let gs = kickoff_1v1(7);
    let pred = [BallSnap::default(); 4];
    let nrm = norm();
    let (mut eb, mut eo) = ([0.0f32; MAX_ENT * ENT_FEAT], [0.0f32; MAX_ENT * ENT_FEAT]);
    let (mut mb, mut mo) = ([false; MAX_ENT], [false; MAX_ENT]);
    let (mut qb, mut qo) = ([0.0f32; Q_FEAT], [0.0f32; Q_FEAT]);
    obs_v1::build(&gs, 0, &pred, &nrm, &mut eb, &mut mb, &mut qb);
    obs_v1::build(&gs, 1, &pred, &nrm, &mut eo, &mut mo, &mut qo);
    for i in 0..eb.len() {
        assert!((eb[i] - eo[i]).abs() < 1e-4, "ents idx {i}: blue {} vs orange {}", eb[i], eo[i]);
    }
    assert_eq!(mb, mo, "mask pattern must match (kickoff is symmetric)");
    for i in 0..qb.len() {
        assert!((qb[i] - qo[i]).abs() < 1e-4, "query idx {i}: blue {} vs orange {}", qb[i], qo[i]);
    }
}

#[test]
fn ballpred_stationary_and_moving() {
    ensure_init(None);
    let mut t = Tracker::new();
    let mut ball = rocketsim_rs::sim::BallState::default();
    ball.pos = rocketsim_rs::math::Vec3::new(0.0, 0.0, 93.15);
    let snaps = t.predict(&ball);
    for s in &snaps {
        assert!(s.pos.x.is_finite() && s.pos.y.is_finite() && s.pos.z.is_finite());
        assert!(s.vel.x.is_finite() && s.vel.y.is_finite() && s.vel.z.is_finite());
        let horiz = (s.pos.x * s.pos.x + s.pos.y * s.pos.y).sqrt();
        assert!(horiz < 50.0, "stationary ball drifted horizontally: {horiz}");
    }

    let mut t2 = Tracker::new();
    ball.vel = rocketsim_rs::math::Vec3::new(0.0, 2000.0, 0.0);
    let moving = t2.predict(&ball);
    assert!(moving.iter().all(|s| s.pos.is_finite_all()));
    for w in moving.windows(2) {
        assert!(w[1].pos.y > w[0].pos.y, "y must increase across horizons: {:?}", moving);
    }

    // Cross-instance agreement: same input -> ~same output within a physical
    // tolerance. Bit-equality across separately-constructed trackers does NOT
    // hold: Bullet contact resolution is sensitive to heap-allocation history
    // (verified 2026-07-16 — trackers constructed after another arena had
    // stepped differed by ~0.01-0.3uu after 60-240 ticks, while the same
    // binary reproduces its exact numbers run after run, and up-front
    // constructed trackers match bitwise). Per-tracker re-predict IS
    // bit-deterministic (ballpred.rs::predict_is_deterministic), and the
    // whole-process fixed-config determinism contract is covered at the
    // Engine level (engine_v1_test / test_collect_deterministic_fixed_config).
    let mut t3 = Tracker::new();
    let again = t3.predict(&ball);
    for (i, (a, b)) in moving.iter().zip(again.iter()).enumerate() {
        for (pa, pb) in [(a.pos.x, b.pos.x), (a.pos.y, b.pos.y), (a.pos.z, b.pos.z)] {
            assert!((pa - pb).abs() < 2.0, "snap {i} pos drift too large: {moving:?} vs {again:?}");
        }
        for (va, vb) in [(a.vel.x, b.vel.x), (a.vel.y, b.vel.y), (a.vel.z, b.vel.z)] {
            assert!((va - vb).abs() < 5.0, "snap {i} vel drift too large: {moving:?} vs {again:?}");
        }
    }
}

trait FiniteVec3 {
    fn is_finite_all(&self) -> bool;
}
impl FiniteVec3 for rocketsim_rs::math::Vec3 {
    fn is_finite_all(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}
