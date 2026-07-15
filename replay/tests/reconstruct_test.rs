use construct_replay::{frames::extract_frames, reconstruct::reconstruct_120hz};

#[test]
fn reconstructs_dense_120hz_finite() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let f = extract_frames(&bytes, 30).unwrap();
    let r = reconstruct_120hz(&f).unwrap();
    // ~4 ticks per 30 Hz frame
    assert!(r.ticks.len() >= f.ball.len() * 3, "expected ~4x densification");
    assert!(r.ticks.iter().all(|t| t.ball.pos.iter().all(|x| x.is_finite())));
    assert!(
        r.ticks.iter().all(|t| t.ball.pos.iter().all(|x| x.abs() < 20_000.0)),
        "reconstructed states must be physically sane (matches engine state_is_sane bounds)"
    );
    // ball must actually move across the reconstruction, not sit frozen.
    let first = r.ticks.first().unwrap().ball.pos;
    let moved = r
        .ticks
        .iter()
        .any(|t| (t.ball.pos[0] - first[0]).abs() + (t.ball.pos[1] - first[1]).abs() > 1.0);
    assert!(moved, "ball should move across the reconstructed 120Hz stream");
}
