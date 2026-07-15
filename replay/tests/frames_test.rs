use construct_replay::frames::extract_frames;

#[test]
fn extracts_resampled_frames() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let f = extract_frames(&bytes, 30).unwrap();
    assert_eq!(f.fps, 30);
    assert!(f.ball.len() > 100);
    assert_eq!(f.cars.len(), f.ball.len(), "one car-set per ball frame");
    assert!(!f.player_teams.is_empty());
    // ball must move (not all-zero) and stay finite
    let moved = f.ball.iter().any(|r| r.pos[0].abs() + r.pos[1].abs() > 1.0);
    assert!(moved && f.ball.iter().all(|r| r.pos.iter().all(|x| x.is_finite())));

    // Raw replay input bytes map to -1..1 via (byte - 128) / 127, which
    // slightly overshoots at byte 0 (-1.008) unless clamped at the mapping
    // site in frames.rs.
    for row in &f.cars {
        for car in row {
            assert!(
                car.throttle >= -1.0 && car.throttle <= 1.0,
                "throttle out of [-1,1]: {}",
                car.throttle
            );
            assert!(car.steer >= -1.0 && car.steer <= 1.0, "steer out of [-1,1]: {}", car.steer);
        }
    }
}
