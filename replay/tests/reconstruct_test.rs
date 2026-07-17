use construct_replay::{frames::extract_frames, reconstruct::reconstruct_120hz};

#[test]
fn reconstructs_dense_120hz_finite() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let f = extract_frames(&bytes, 30).unwrap();
    let r = reconstruct_120hz(&f, 1).unwrap();
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

#[test]
fn tick_index_is_monotonic_with_gaps_on_drops_and_has_boundaries() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let f = extract_frames(&bytes, 30).unwrap();
    let r = reconstruct_120hz(&f, 1).unwrap();

    // tick_index must be strictly increasing across surviving ticks -- it's
    // the global monotonic 120 Hz sub-step counter in the ORIGINAL
    // undropped sequence, so even with drops the survivors stay ordered.
    for w in r.ticks.windows(2) {
        assert!(
            w[1].tick_index > w[0].tick_index,
            "tick_index must be strictly increasing: {} -> {}",
            w[0].tick_index,
            w[1].tick_index
        );
    }

    // At least one authoritative snap-boundary tick must exist (the first
    // interval's first sub-step, at minimum).
    assert!(r.ticks.iter().any(|t| t.is_boundary), "expected at least one boundary tick");

    // Total sub-steps RocketSim was asked to attempt across the whole
    // replay (before any sanity-drop) -- if fewer ticks survived than were
    // attempted, some were dropped, and a gap >1 must be observable in the
    // surviving tick_index sequence.
    let num_frames = f.ball.len();
    let dt = 1.0 / f.fps as f32;
    let n_substeps = (120.0 * dt).round().max(1.0) as u32;
    let attempted = (num_frames as u64 - 1) * n_substeps as u64;
    if (r.ticks.len() as u64) < attempted {
        let has_gap = r.ticks.windows(2).any(|w| w[1].tick_index - w[0].tick_index > 1);
        assert!(has_gap, "expected a tick_index gap given {} dropped tick(s)", attempted - r.ticks.len() as u64);
    }
}
