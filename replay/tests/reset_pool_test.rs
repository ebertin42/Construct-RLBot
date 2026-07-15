use construct_replay::{frames::extract_frames, reconstruct::reconstruct_120hz, reset_pool::sample_reset_states};

#[test]
fn samples_k_finite_reset_states_deterministically() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap()).unwrap();
    let a = sample_reset_states(&rec, 8, 42);
    let b = sample_reset_states(&rec, 8, 42);
    assert_eq!(a.len(), 8);
    assert_eq!(a, b, "same seed -> identical sample");
    assert!(a.iter().all(|s| s.ball.pos.iter().all(|x| x.is_finite() && x.abs() < 12_000.0)));
}

#[test]
fn different_seeds_can_differ() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap()).unwrap();
    let a = sample_reset_states(&rec, 8, 1);
    let b = sample_reset_states(&rec, 8, 2);
    assert_eq!(a.len(), 8);
    assert_eq!(b.len(), 8);
    assert_ne!(a, b, "different seeds should (almost certainly) sample different states");
}

#[test]
fn cars_carry_all_engine_set_car_fields() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap()).unwrap();
    let states = sample_reset_states(&rec, 4, 99);
    for s in &states {
        assert!(!s.cars.is_empty());
        for c in &s.cars {
            assert!(c.pos.iter().all(|x| x.is_finite()));
            assert!(c.vel.iter().all(|x| x.is_finite()));
            assert!(c.ang_vel.iter().all(|x| x.is_finite()));
            assert!(c.quat.iter().all(|x| x.is_finite()));
            assert!((0.0..=1.0).contains(&c.boost), "boost must be 0..1, got {}", c.boost);
            assert!(c.team == 0 || c.team == 1);
        }
    }
}
