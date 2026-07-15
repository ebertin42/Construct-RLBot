use construct_replay::meta::parse_meta;

#[test]
fn parses_header_of_fixture() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let m = parse_meta(&bytes).unwrap();
    assert!(m.num_frames > 100, "expected real network frames, got {}", m.num_frames);
    assert!(m.team_size >= 1 && m.team_size <= 3);
    assert!(m.duration_secs > 0);
}
