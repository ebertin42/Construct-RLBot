use construct_replay::{
    frames::extract_frames,
    meta::parse_meta,
    reconstruct::reconstruct_120hz,
    shard::{write_shard, SHARD_SCHEMA_VERSION},
};
use ndarray::Array3;
use ndarray_npy::NpzReader;

#[test]
fn writes_loadable_shard_with_schema() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let meta = parse_meta(&bytes).unwrap();
    let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap()).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let p = write_shard(dir.path(), "sample", &meta, &rec).unwrap();
    assert!(p.exists());

    let sidecar: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir.path().join("sample.json")).unwrap()).unwrap();
    assert_eq!(sidecar["schema_version"], SHARD_SCHEMA_VERSION);
    assert!(sidecar["num_ticks"].as_u64().unwrap() > 100);

    let num_players = sidecar["num_players"].as_u64().unwrap() as usize;
    let cars_state_columns = sidecar["cars_state_columns"].as_array().unwrap();
    let cars_action_columns = sidecar["cars_action_columns"].as_array().unwrap();
    let ball_columns = sidecar["ball_columns"].as_array().unwrap();
    assert_eq!(ball_columns.len(), 13);
    assert_eq!(cars_action_columns.len(), 8, "cars_action last-dim must be 8 (throttle..handbrake)");
    // pos3+vel3+ang_vel3+quat4 (13) + boost + on_ground + demoed = 16 columns.
    // (An earlier plan sketch guessed 15; the explicit column list the task
    // requires — including both on_ground and demoed as separate channels —
    // sums to 16, and the sidecar's column arrays are the schema's source of
    // truth, so the .npz array widths are asserted against them directly.)
    assert_eq!(cars_state_columns.len(), 16, "cars_state last-dim must match its documented column list");

    // The .npz must actually load, and its array shapes must match the
    // sidecar-documented column counts exactly.
    let mut npz = NpzReader::new(std::fs::File::open(&p).unwrap()).unwrap();
    let ball: ndarray::Array2<f32> = npz.by_name("ball.npy").unwrap();
    assert_eq!(ball.shape()[1], ball_columns.len());

    let cars_state: Array3<f32> = npz.by_name("cars_state.npy").unwrap();
    assert_eq!(cars_state.shape()[1], num_players);
    assert_eq!(cars_state.shape()[2], cars_state_columns.len());
    assert_eq!(cars_state.shape()[2], 15 + 1, "cars_state last-dim has 16 columns, not the plan sketch's 15");

    let cars_action: Array3<f32> = npz.by_name("cars_action.npy").unwrap();
    assert_eq!(cars_action.shape()[1], num_players);
    assert_eq!(cars_action.shape()[2], 8);
    assert_eq!(cars_action.shape()[2], cars_action_columns.len());

    let player_teams: ndarray::Array1<i64> = npz.by_name("player_teams.npy").unwrap();
    assert_eq!(player_teams.len(), num_players);

    let num_ticks = sidecar["num_ticks"].as_u64().unwrap() as usize;
    let tick_index: ndarray::Array1<i64> = npz.by_name("tick_index.npy").unwrap();
    assert_eq!(tick_index.len(), num_ticks, "tick_index must have length T");

    let is_boundary: ndarray::Array1<i64> = npz.by_name("is_boundary.npy").unwrap();
    assert_eq!(is_boundary.len(), num_ticks, "is_boundary must have length T");
    assert!(
        is_boundary.iter().all(|&v| v == 0 || v == 1),
        "is_boundary must be 0/1-valued"
    );
}

#[test]
fn empty_reconstruction_is_an_error_not_an_empty_shard() {
    use construct_replay::reconstruct::Reconstructed;
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let meta = parse_meta(&bytes).unwrap();
    let empty = Reconstructed { ticks: Vec::new(), player_teams: vec![0, 1], fps: 30 };
    let dir = tempfile::tempdir().unwrap();
    let err = write_shard(dir.path(), "empty", &meta, &empty);
    assert!(err.is_err(), "T=0 must error rather than write an empty shard");
    assert!(!dir.path().join("empty.npz").exists());
}
