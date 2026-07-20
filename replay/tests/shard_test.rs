use construct_replay::{
    frames::{extract_frames, CarFrame, RigidFrame},
    meta::parse_meta,
    reconstruct::{reconstruct_120hz, Reconstructed, Tick},
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
    assert_eq!(SHARD_SCHEMA_VERSION, 5, "schema must be bumped for is_demoed/episode_marker");
    assert!(sidecar["num_ticks"].as_u64().unwrap() > 100);
    assert!(
        sidecar["episode_marker"].as_str().unwrap().contains("goal-reset/kickoff"),
        "v5 sidecar must document the episode_marker array's semantics"
    );

    let num_players = sidecar["num_players"].as_u64().unwrap() as usize;
    let cars_state_columns = sidecar["cars_state_columns"].as_array().unwrap();
    let cars_action_columns = sidecar["cars_action_columns"].as_array().unwrap();
    let ball_columns = sidecar["ball_columns"].as_array().unwrap();
    let pad_columns = sidecar["pad_columns"].as_array().unwrap();
    let ball_pred_columns = sidecar["ball_pred_columns"].as_array().unwrap();
    assert_eq!(ball_columns.len(), 13);
    assert_eq!(cars_action_columns.len(), 8, "cars_action last-dim must be 8 (throttle..handbrake)");
    // pos3+vel3+ang_vel3+quat4 (13) + boost + on_ground + demoed + has_flip
    // + is_demoed = 18 columns (v5 adds is_demoed to the v4 17-column layout).
    assert_eq!(cars_state_columns.len(), 18, "cars_state last-dim must match its documented column list");
    assert!(
        cars_state_columns.iter().any(|c| c == "has_flip"),
        "cars_state_columns must document the has_flip column"
    );
    assert!(
        cars_state_columns.iter().any(|c| c == "is_demoed"),
        "cars_state_columns must document the new v5 is_demoed column"
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
    assert_eq!(cars_state.shape()[2], 18, "cars_state last-dim has 18 columns in schema v5");
    let has_flip_idx = cars_state_columns.iter().position(|c| c == "has_flip").unwrap();
    let is_demoed_idx = cars_state_columns.iter().position(|c| c == "is_demoed").unwrap();
    let demoed_idx = cars_state_columns.iter().position(|c| c == "demoed").unwrap();
    let mut demoed_rows = 0usize;
    for p in 0..num_players {
        for t in 0..cars_state.shape()[0] {
            let v = cars_state[[t, p, has_flip_idx]];
            assert!(v == 0.0 || v == 1.0, "has_flip must be 0/1-valued, got {v}");
            let d = cars_state[[t, p, is_demoed_idx]];
            assert!(d == 0.0 || d == 1.0, "is_demoed must be 0/1-valued, got {d}");
            if d == 1.0 {
                demoed_rows += 1;
                // The replay-frame demoed flag was snapped into the sim
                // (reconstruct::set_car_state, v5 fix), so the post-step sim
                // `demoed` column must agree whenever the replay says demoed.
                assert_eq!(
                    cars_state[[t, p, demoed_idx]],
                    1.0,
                    "t={t} p={p}: replay is_demoed=1 must imply sim demoed=1 (set_car_state must propagate)"
                );
            }
        }
    }
    // The fixture's goal reset + trailing post-goal segment leave both
    // players absent (carried forward) for hundreds of frames — the v5
    // is_demoed column must actually capture that, not sit at all-zeros
    // (which was exactly v4's frozen-ghost gap).
    assert!(
        demoed_rows > 100,
        "fixture must yield demoed car-ticks (goal reset + trailing segment), got {demoed_rows}"
    );
    assert!(
        demoed_rows < num_players * cars_state.shape()[0] / 2,
        "most car-ticks must be live, got {demoed_rows} demoed"
    );

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

    // --- episode_marker [T] u8 (v5): goal-reset/kickoff boundaries ---
    // The fixture (probed at 30 fps) has exactly 3 dead-ball run starts:
    // the opening kickoff (frame 0), one mid-match goal→kickoff reset
    // (frames 1279..=1565: post-goal ball despawn, respawned ball settling
    // at center, frozen countdown), and the post-goal segment the recording
    // ends on (frame 3182..). Each must yield exactly one marker.
    let episode_marker: ndarray::Array1<u8> = npz.by_name("episode_marker.npy").unwrap();
    assert_eq!(episode_marker.len(), num_ticks, "episode_marker must have length T");
    assert!(
        episode_marker.iter().all(|&v| v == 0 || v == 1),
        "episode_marker must be 0/1-valued"
    );
    assert_eq!(
        episode_marker[0], 1,
        "fixture opens on its kickoff — the first stored tick must carry a marker"
    );
    let marker_count: usize = episode_marker.iter().map(|&v| v as usize).sum();
    assert_eq!(
        marker_count, 3,
        "fixture has exactly 3 dead-ball run starts (opening kickoff, one goal reset, trailing post-goal segment)"
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

    // --- ball_pred [T, 4, 6]: finite, and a faithful write of Tick::ball_pred ---
    // `write_shard` must persist whatever `Tick::ball_pred` the
    // reconstruction computed — whether a genuine roll-forward or a
    // contained fallback — with no reordering/truncation/corruption. Assert
    // that directly against `rec` (already held in this test, stride=1 so
    // `write_shard`'s `step_by(1)` selection is the identity).
    let ball_pred: Array3<f32> = npz.by_name("ball_pred.npy").unwrap();
    assert_eq!(ball_pred.shape(), &[num_ticks, 4, 6], "ball_pred must be [T, 4, 6]");
    assert!(ball_pred.iter().all(|x| x.is_finite()), "ball_pred must be all finite");
    assert_eq!(num_ticks, rec.ticks.len(), "stride=1 write must keep every reconstructed tick");
    for (t, tick) in rec.ticks.iter().enumerate() {
        for (h, row) in tick.ball_pred.iter().enumerate() {
            for (k, &v) in row.iter().enumerate() {
                assert_eq!(
                    ball_pred[[t, h, k]], v,
                    "ball_pred[{t},{h},{k}] must exactly match the reconstruction's Tick::ball_pred \
                     (shard write must be a lossless copy, not a re-derivation)"
                );
            }
        }
    }

    // --- ball_pred divergence (restored, task #45 → schema v5) ---
    // Removed in d798ed8 because `ballpred::Tracker`'s poisoned Bullet arena
    // latched DISABLE_SIMULATION forever after its first containment trip,
    // collapsing EVERY subsequent prediction to an echo of the input ball —
    // whether that latch engaged depended on binary allocation history, so
    // any divergence assertion was testing per-build solver luck. Engine
    // commit 0928e59 fixed the latch (predict() rebuilds its arena on
    // containment before returning), so at most the rare individual
    // containment-tripping call can still echo; a moving ball's +2s
    // prediction must otherwise genuinely diverge from its current position.
    // Assert the overwhelming majority of moving-ball ticks diverge — an
    // all-echo (or mostly-echo) shard means the poisoned-latch regressed.
    // (Measured on this box: 10714/10719 moving ticks diverge; the 5 echoes
    // are individual containment trips, i.e. the expected transient.)
    let mut moving = 0usize;
    let mut diverged = 0usize;
    for t in 0..num_ticks {
        let speed = (ball[[t, 3]].powi(2) + ball[[t, 4]].powi(2) + ball[[t, 5]].powi(2)).sqrt();
        if speed < 300.0 {
            continue; // near-rest ball legitimately predicts near itself
        }
        moving += 1;
        let d2: f32 = (0..3).map(|k| (ball_pred[[t, 3, k]] - ball[[t, k]]).powi(2)).sum();
        if d2.sqrt() > 5.0 {
            diverged += 1;
        }
    }
    assert!(moving > 1000, "fixture must have plenty of moving-ball ticks, got {moving}");
    assert!(
        diverged as f64 >= moving as f64 * 0.9,
        "+2s ball_pred must diverge >5uu from the current ball position on at least 90% of \
         moving-ball (speed>300) ticks; got {diverged}/{moving} — echo predictions at this rate \
         mean the ballpred poisoned-arena latch (fixed in engine 0928e59) is back"
    );
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
    assert_eq!(SHARD_SCHEMA_VERSION, 5, "schema must be bumped for is_demoed/episode_marker");
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

    // episode_marker survives subsampling: markers on stride-skipped ticks
    // carry forward, so the total marker count is stride-invariant (the
    // fixture's 3 boundaries are far enough apart to never collapse into
    // one stored row at stride 8).
    let mut npz1 = NpzReader::new(std::fs::File::open(dir1.path().join("sample.npz")).unwrap()).unwrap();
    let marker1: ndarray::Array1<u8> = npz1.by_name("episode_marker.npy").unwrap();
    let marker8: ndarray::Array1<u8> = npz.by_name("episode_marker.npy").unwrap();
    assert_eq!(marker8.len(), num_ticks_8);
    assert_eq!(
        marker8.iter().map(|&v| v as usize).sum::<usize>(),
        marker1.iter().map(|&v| v as usize).sum::<usize>(),
        "stride subsampling must not lose (or duplicate) episode markers"
    );
}

/// Synthetic marker/demoed coverage at the `write_shard` unit level: markers
/// on stride-skipped ticks must carry forward to the next stored row (the
/// "after" case), markers on stored ticks must land on that row (the "at"
/// case), and per-car `replay_demoed` must come out as the `is_demoed`
/// column of exactly the stored rows.
#[test]
fn synthetic_markers_and_demoed_survive_stride_subsampling() {
    fn synth_tick(tick_index: i64, marker: bool, demoed: bool) -> Tick {
        let rb = RigidFrame {
            pos: [100.0, 200.0, 17.0],
            vel: [0.0; 3],
            ang_vel: [0.0; 3],
            quat: [0.0, 0.0, 0.0, 1.0],
        };
        Tick {
            ball: RigidFrame {
                pos: [0.0, 0.0, 93.0],
                vel: [0.0; 3],
                ang_vel: [0.0; 3],
                quat: [0.0, 0.0, 0.0, 1.0],
            },
            cars: vec![CarFrame {
                rb,
                boost: 0.5,
                team: 0,
                throttle: 0.0,
                steer: 0.0,
                handbrake: false,
                jump_active: false,
                dodge_active: false,
                on_ground: true,
                demoed,
            }],
            actions: vec![[0.0; 8]],
            tick_index,
            is_boundary: tick_index % 4 == 0,
            pads: vec![(0.0, true); 34],
            has_flip: vec![true],
            replay_demoed: vec![demoed],
            episode_marker: marker,
            ball_pred: [[0.0; 6]; 4],
        }
    }

    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let meta = parse_meta(&bytes).unwrap();

    // 24 ticks, stride 8 -> stored rows are original indices 0, 8, 16.
    // Marker at tick 3 (skipped) must carry onto stored row 1 (tick 8);
    // marker at tick 16 (stored) must land on stored row 2 directly.
    // Ticks 8..16 are demoed -> stored row 1 must have is_demoed = 1.
    let ticks: Vec<Tick> = (0..24)
        .map(|i| synth_tick(i as i64, i == 3 || i == 16, (8..16).contains(&i)))
        .collect();
    let rec = Reconstructed { ticks, player_teams: vec![0], fps: 30 };

    let dir = tempfile::tempdir().unwrap();
    let p = write_shard(dir.path(), "synth", &meta, &rec, 8).unwrap();
    let mut npz = NpzReader::new(std::fs::File::open(&p).unwrap()).unwrap();

    let marker: ndarray::Array1<u8> = npz.by_name("episode_marker.npy").unwrap();
    assert_eq!(
        marker.to_vec(),
        vec![0, 1, 1],
        "marker at skipped tick 3 must carry to stored row 1; marker at stored tick 16 lands on row 2"
    );

    let cars_state: Array3<f32> = npz.by_name("cars_state.npy").unwrap();
    assert_eq!(cars_state.shape(), &[3, 1, 18]);
    let demoed_col: Vec<f32> = (0..3).map(|t| cars_state[[t, 0, 17]]).collect();
    assert_eq!(demoed_col, vec![0.0, 1.0, 0.0], "is_demoed must reflect stored rows' replay_demoed");
}
