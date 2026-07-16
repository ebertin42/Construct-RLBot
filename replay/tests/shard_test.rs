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
    let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap(), 1).unwrap();
    let dir = tempfile::tempdir().unwrap();
    let p = write_shard(dir.path(), "sample", &meta, &rec, 1).unwrap();
    assert!(p.exists());

    let sidecar: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir.path().join("sample.json")).unwrap()).unwrap();
    assert_eq!(sidecar["schema_version"], SHARD_SCHEMA_VERSION);
    assert_eq!(SHARD_SCHEMA_VERSION, 4, "schema must be bumped for pads/has_flip/ball_pred");
    assert!(sidecar["num_ticks"].as_u64().unwrap() > 100);

    let num_players = sidecar["num_players"].as_u64().unwrap() as usize;
    let cars_state_columns = sidecar["cars_state_columns"].as_array().unwrap();
    let cars_action_columns = sidecar["cars_action_columns"].as_array().unwrap();
    let ball_columns = sidecar["ball_columns"].as_array().unwrap();
    let pad_columns = sidecar["pad_columns"].as_array().unwrap();
    let ball_pred_columns = sidecar["ball_pred_columns"].as_array().unwrap();
    assert_eq!(ball_columns.len(), 13);
    assert_eq!(cars_action_columns.len(), 8, "cars_action last-dim must be 8 (throttle..handbrake)");
    // pos3+vel3+ang_vel3+quat4 (13) + boost + on_ground + demoed + has_flip = 17
    // columns (v4 adds has_flip to the v3 16-column layout).
    assert_eq!(cars_state_columns.len(), 17, "cars_state last-dim must match its documented column list");
    assert!(
        cars_state_columns.iter().any(|c| c == "has_flip"),
        "cars_state_columns must document the new has_flip column"
    );
    assert_eq!(pad_columns.len(), 2, "pad columns: timer, is_active");
    assert_eq!(ball_pred_columns.len(), 6, "ball_pred columns: pos3+vel3");

    // The .npz must actually load, and its array shapes must match the
    // sidecar-documented column counts exactly.
    let mut npz = NpzReader::new(std::fs::File::open(&p).unwrap()).unwrap();
    let ball: ndarray::Array2<f32> = npz.by_name("ball.npy").unwrap();
    assert_eq!(ball.shape()[1], ball_columns.len());

    let cars_state: Array3<f32> = npz.by_name("cars_state.npy").unwrap();
    assert_eq!(cars_state.shape()[1], num_players);
    assert_eq!(cars_state.shape()[2], cars_state_columns.len());
    assert_eq!(cars_state.shape()[2], 17, "cars_state last-dim has 17 columns in schema v4");
    let has_flip_idx = cars_state_columns.iter().position(|c| c == "has_flip").unwrap();
    for p in 0..num_players {
        for t in 0..cars_state.shape()[0] {
            let v = cars_state[[t, p, has_flip_idx]];
            assert!(v == 0.0 || v == 1.0, "has_flip must be 0/1-valued, got {v}");
        }
    }

    let cars_action: Array3<f32> = npz.by_name("cars_action.npy").unwrap();
    assert_eq!(cars_action.shape()[1], num_players);
    assert_eq!(cars_action.shape()[2], 8);
    assert_eq!(cars_action.shape()[2], cars_action_columns.len());

    // --- cars_action_idx [T, P] i64: projected 92-table action index (Task B2) ---
    let action_table_size = sidecar["action_table_size"].as_u64().unwrap() as usize;
    assert_eq!(action_table_size, 92, "v1 action table has 92 rows");
    let cars_action_idx: ndarray::Array2<i64> = npz.by_name("cars_action_idx.npy").unwrap();
    assert_eq!(
        cars_action_idx.shape(),
        &[cars_action.shape()[0], num_players],
        "cars_action_idx must be [T, P]"
    );
    assert!(
        cars_action_idx.iter().all(|&v| (0..action_table_size as i64).contains(&v)),
        "every cars_action_idx value must be in [0, {action_table_size})"
    );

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

    // --- pads [T, 34, 2]: timer in [0,1], is_active in {0,1} ---
    let pads: Array3<f32> = npz.by_name("pads.npy").unwrap();
    assert_eq!(pads.shape(), &[num_ticks, 34, 2], "pads must be [T, 34, 2]");
    for t in 0..num_ticks {
        for pad in 0..34 {
            let timer = pads[[t, pad, 0]];
            let active = pads[[t, pad, 1]];
            assert!((0.0..=1.0).contains(&timer), "pad timer must be in [0,1], got {timer}");
            assert!(active == 0.0 || active == 1.0, "pad is_active must be 0/1-valued, got {active}");
        }
    }

    // --- ball_pred [T, 4, 6]: finite, and evolving across horizons for a moving ball ---
    let ball_pred: Array3<f32> = npz.by_name("ball_pred.npy").unwrap();
    assert_eq!(ball_pred.shape(), &[num_ticks, 4, 6], "ball_pred must be [T, 4, 6]");
    assert!(ball_pred.iter().all(|x| x.is_finite()), "ball_pred must be all finite");
    // The fixture's ball is in motion for at least some ticks; on those ticks
    // the +2s horizon prediction must differ from the ball's current position
    // (otherwise ball-pred is a no-op copy, not an actual roll-forward).
    let mut any_diverged = false;
    for t in 0..num_ticks {
        let cur = [ball[[t, 0]], ball[[t, 1]], ball[[t, 2]]];
        let far = [ball_pred[[t, 3, 0]], ball_pred[[t, 3, 1]], ball_pred[[t, 3, 2]]];
        let dist = ((cur[0] - far[0]).powi(2) + (cur[1] - far[1]).powi(2) + (cur[2] - far[2]).powi(2)).sqrt();
        if dist > 5.0 {
            any_diverged = true;
            break;
        }
    }
    assert!(any_diverged, "expected at least one tick where the +2s ball prediction diverges from current position");
}

#[test]
fn empty_reconstruction_is_an_error_not_an_empty_shard() {
    use construct_replay::reconstruct::Reconstructed;
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let meta = parse_meta(&bytes).unwrap();
    let empty = Reconstructed { ticks: Vec::new(), player_teams: vec![0, 1], fps: 30 };
    let dir = tempfile::tempdir().unwrap();
    let err = write_shard(dir.path(), "empty", &meta, &empty, 1);
    assert!(err.is_err(), "T=0 must error rather than write an empty shard");
    assert!(!dir.path().join("empty.npz").exists());
}

#[test]
fn stride_subsamples_ticks_and_preserves_tick_index_semantics() {
    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let meta = parse_meta(&bytes).unwrap();
    // stride=1 at reconstruct time so every tick gets a real (non-fallback)
    // ball-pred, regardless of which write_shard stride is exercised below
    // (write_shard's selection is always a subset of a stride=1 reconstruct's).
    let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap(), 1).unwrap();

    // stride=1 baseline: current (pre-stride) behavior, full 120 Hz.
    let dir1 = tempfile::tempdir().unwrap();
    write_shard(dir1.path(), "sample", &meta, &rec, 1).unwrap();
    let sidecar1: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir1.path().join("sample.json")).unwrap()).unwrap();
    let num_ticks_1 = sidecar1["num_ticks"].as_u64().unwrap() as usize;

    // stride=8: 15 Hz decision-rate storage.
    let dir8 = tempfile::tempdir().unwrap();
    let p8 = write_shard(dir8.path(), "sample", &meta, &rec, 8).unwrap();
    let sidecar8: serde_json::Value =
        serde_json::from_reader(std::fs::File::open(dir8.path().join("sample.json")).unwrap()).unwrap();
    let num_ticks_8 = sidecar8["num_ticks"].as_u64().unwrap() as usize;

    assert_eq!(sidecar8["schema_version"], SHARD_SCHEMA_VERSION);
    assert_eq!(SHARD_SCHEMA_VERSION, 4, "schema must be bumped for pads/has_flip/ball_pred");
    assert_eq!(sidecar8["stride"], 8);
    assert_eq!(sidecar8["effective_hz"], 15.0);

    // Every stride-th surviving tick is kept: indices 0, 8, 16, ... from the
    // full-rate array, so this is a ceil-division of the stride=1 count.
    let expected_num_ticks_8 = num_ticks_1.div_ceil(8);
    assert_eq!(
        num_ticks_8, expected_num_ticks_8,
        "stride=8 must keep ~1/8 the ticks of stride=1 (indices 0, 8, 16, ...)"
    );

    let mut npz = NpzReader::new(std::fs::File::open(&p8).unwrap()).unwrap();
    let tick_index: ndarray::Array1<i64> = npz.by_name("tick_index.npy").unwrap();
    assert_eq!(tick_index.len(), num_ticks_8, "tick_index length must match stored (subsampled) tick count");

    // tick_index values are the ORIGINAL 120 Hz indices (not renumbered), and
    // stay strictly increasing so drop-gaps are still detectable downstream.
    let tick_index_vec: Vec<i64> = tick_index.to_vec();
    for w in tick_index_vec.windows(2) {
        assert!(w[1] > w[0], "tick_index must remain strictly increasing after subsampling");
    }
    let expected_tick_index: Vec<i64> =
        rec.ticks.iter().step_by(8).map(|t| t.tick_index).collect();
    assert_eq!(tick_index_vec, expected_tick_index, "subsample must take indices 0, stride, 2*stride, ... from the surviving tick array");

    let ball: ndarray::Array2<f32> = npz.by_name("ball.npy").unwrap();
    assert_eq!(ball.shape()[0], num_ticks_8);
    let cars_state: Array3<f32> = npz.by_name("cars_state.npy").unwrap();
    assert_eq!(cars_state.shape()[0], num_ticks_8);
    let cars_action: Array3<f32> = npz.by_name("cars_action.npy").unwrap();
    assert_eq!(cars_action.shape()[0], num_ticks_8);
    let cars_action_idx: ndarray::Array2<i64> = npz.by_name("cars_action_idx.npy").unwrap();
    assert_eq!(cars_action_idx.shape()[0], num_ticks_8);
    let is_boundary: ndarray::Array1<i64> = npz.by_name("is_boundary.npy").unwrap();
    assert_eq!(is_boundary.len(), num_ticks_8);
}
