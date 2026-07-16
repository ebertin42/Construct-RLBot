//! Task B3: `bc_obs` — v4 shards -> obs-v1 training tensors.
//!
//! The fixture shard is built ONCE per test binary (a `OnceLock`-guarded
//! parse of `tests/fixtures/sample.replay`, the same fixture every other
//! integration test here uses) because reconstruction is by far the
//! slowest step and every test below only needs read access to the same
//! deterministic shard.

use std::path::PathBuf;
use std::sync::OnceLock;

use construct_engine::schema::Schema;
use construct_replay::{
    bc_obs::{build_tensors, export_shard_file, load_shard_v4, pad_template, BcTensors},
    frames::extract_frames,
    meta::parse_meta,
    reconstruct::reconstruct_120hz,
    shard::write_shard,
};
use ndarray_npy::NpzReader;

const STRIDE: usize = 8;

/// Parses the fixture replay into a v4 shard exactly once for the whole test
/// binary; the `TempDir` is stored alongside the path so the directory
/// outlives every test.
fn fixture_shard() -> &'static PathBuf {
    static SHARD: OnceLock<(tempfile::TempDir, PathBuf)> = OnceLock::new();
    let (_dir, path) = SHARD.get_or_init(|| {
        let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
        let meta = parse_meta(&bytes).unwrap();
        let rec = reconstruct_120hz(&extract_frames(&bytes, 30).unwrap(), STRIDE).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let p = write_shard(dir.path(), "sample", &meta, &rec, STRIDE).unwrap();
        (dir, p)
    });
    path
}

fn norm() -> construct_engine::schema::Normalization {
    // Tests run with cwd = `replay/`, same relative path obs_v1's own tests use.
    Schema::load("../schema/v1.toml").unwrap().normalization
}

fn fixture_tensors() -> &'static (BcTensors, construct_replay::bc_obs::ShardV4) {
    static TENSORS: OnceLock<(BcTensors, construct_replay::bc_obs::ShardV4)> = OnceLock::new();
    TENSORS.get_or_init(|| {
        let shard = load_shard_v4(fixture_shard()).unwrap();
        let pads = pad_template();
        let out = build_tensors(&shard, &pads, &norm()).unwrap();
        (out, shard)
    })
}

#[test]
fn shapes_are_s_by_obs_v1_dims() {
    let (bc, shard) = fixture_tensors();
    let t = shard.tick_index.len();
    let p = shard.player_teams.len();
    let s = t * p;
    assert!(t > 100, "fixture shard must have a real tick count, got {t}");
    assert_eq!(p, 2, "fixture is a 1v1 replay");
    assert_eq!(bc.ents.shape(), &[s, 17, 26], "ents must be [S, MAX_ENT, ENT_FEAT]");
    assert_eq!(bc.mask.shape(), &[s, 17], "mask must be [S, MAX_ENT]");
    assert_eq!(bc.query.shape(), &[s, 64], "query must be [S, Q_FEAT]");
    assert_eq!(bc.prev.shape(), &[s, 5], "prev must be [S, PREV_ACTIONS]");
    assert_eq!(bc.action.shape(), &[s], "action must be [S]");
}

#[test]
fn self_one_hot_set_and_unmasked_per_sample() {
    let (bc, _) = fixture_tensors();
    let s = bc.ents.shape()[0];
    for i in 0..s {
        // slot 0 = self: IS_SELF one-hot, present (mask 0).
        assert_eq!(bc.ents[[i, 0, 0]], 1.0, "sample {i}: self IS_SELF flag");
        for k in 1..5 {
            assert_eq!(bc.ents[[i, 0, k]], 0.0, "sample {i}: self one-hot col {k} must be 0");
        }
        assert_eq!(bc.mask[[i, 0]], 0, "sample {i}: self slot must be unmasked (present)");
        // ball (slot 6) is always present too.
        assert_eq!(bc.mask[[i, 6]], 0, "sample {i}: ball slot must be present");
        // mask is 0/1-valued.
        for e in 0..17 {
            let m = bc.mask[[i, e]];
            assert!(m == 0 || m == 1, "sample {i} slot {e}: mask must be 0/1, got {m}");
        }
    }
}

#[test]
fn all_values_finite() {
    let (bc, _) = fixture_tensors();
    assert!(bc.ents.iter().all(|x| x.is_finite()), "ents must be all finite");
    assert!(bc.query.iter().all(|x| x.is_finite()), "query must be all finite");
}

#[test]
fn action_indices_in_92_table_range() {
    let (bc, shard) = fixture_tensors();
    assert_eq!(shard.action_table_size, 92);
    assert!(
        bc.action.iter().all(|&a| (0..92).contains(&a)),
        "every action index must be in [0, 92)"
    );
    assert!(
        bc.prev.iter().all(|&a| (0..92).contains(&a)),
        "every prev index must be in [0, 92)"
    );
    // Label column is exactly the shard's stored cars_action_idx in (t, p)
    // row-major order.
    let t_count = shard.tick_index.len();
    let p_count = shard.player_teams.len();
    for t in 0..t_count {
        for p in 0..p_count {
            assert_eq!(
                bc.action[t * p_count + p],
                shard.cars_action_idx[[t, p]],
                "action[{t}*{p_count}+{p}] must equal cars_action_idx[[{t},{p}]]"
            );
        }
    }
}

#[test]
fn prev_window_is_most_recent_first_and_resets_on_gap() {
    let (bc, shard) = fixture_tensors();
    let t_count = shard.tick_index.len();
    let p_count = shard.player_teams.len();

    // t=0: episode start, prev all zeros.
    for p in 0..p_count {
        for k in 0..5 {
            assert_eq!(bc.prev[[p, k]], 0, "t=0 sample must have all-zero prev");
        }
    }

    let mut saw_contiguous = false;
    let mut saw_gap = false;
    for t in 1..t_count {
        let delta = shard.tick_index[t] - shard.tick_index[t - 1];
        for p in 0..p_count {
            let s = t * p_count + p;
            let s_prev = (t - 1) * p_count + p;
            if delta == STRIDE as i64 {
                saw_contiguous = true;
                // Most-recent-first, same ring semantics as the live engine
                // (engine/src/episode.rs): prev[0] = last executed action,
                // prev[1..] = the previous sample's prev[..4] shifted down.
                assert_eq!(
                    bc.prev[[s, 0]],
                    bc.action[s_prev],
                    "t={t} p={p}: prev[0] must be the previous row's action"
                );
                for k in 1..5 {
                    assert_eq!(
                        bc.prev[[s, k]],
                        bc.prev[[s_prev, k - 1]],
                        "t={t} p={p}: prev[{k}] must shift from previous sample's prev[{}]",
                        k - 1
                    );
                }
            } else {
                saw_gap = true;
                // Dropped-tick gap: history is discontinuous, window resets.
                for k in 0..5 {
                    assert_eq!(
                        bc.prev[[s, k]],
                        0,
                        "t={t} p={p}: prev must reset to zeros across a tick_index gap (delta={delta})"
                    );
                }
            }
        }
    }
    assert!(saw_contiguous, "fixture must contain contiguous rows to exercise the shift path");
    // The fixture reconstruction is known to drop a handful of insane ticks,
    // so at least one gap should exist; if this ever fails the fixture became
    // perfectly clean and the reset path needs a synthetic-gap test instead.
    assert!(saw_gap, "fixture must contain at least one tick_index gap to exercise the reset path");
}

#[test]
fn export_file_writes_npz_and_resumes_by_skipping_existing() {
    let out_dir = tempfile::tempdir().unwrap();
    let pads = pad_template();
    let nrm = norm();

    let first = export_shard_file(fixture_shard(), out_dir.path(), &pads, &nrm).unwrap();
    let (out_path, samples) = first.expect("first export must write");
    assert_eq!(out_path, out_dir.path().join("bc_sample.npz"));
    assert!(out_path.exists());
    assert!(samples > 0);

    // Written npz round-trips with the documented names, dtypes, and shapes.
    let mut npz = NpzReader::new(std::fs::File::open(&out_path).unwrap()).unwrap();
    let ents: ndarray::Array3<f32> = npz.by_name("ents.npy").unwrap();
    let mask: ndarray::Array2<u8> = npz.by_name("mask.npy").unwrap();
    let query: ndarray::Array2<f32> = npz.by_name("query.npy").unwrap();
    let prev: ndarray::Array2<i64> = npz.by_name("prev.npy").unwrap();
    let action: ndarray::Array1<i64> = npz.by_name("action.npy").unwrap();
    assert_eq!(ents.shape(), &[samples, 17, 26]);
    assert_eq!(mask.shape(), &[samples, 17]);
    assert_eq!(query.shape(), &[samples, 64]);
    assert_eq!(prev.shape(), &[samples, 5]);
    assert_eq!(action.shape(), &[samples]);

    // Resume: an existing output is skipped, not rewritten.
    let mtime = std::fs::metadata(&out_path).unwrap().modified().unwrap();
    let second = export_shard_file(fixture_shard(), out_dir.path(), &pads, &nrm).unwrap();
    assert!(second.is_none(), "existing output must be skipped (resumable)");
    assert_eq!(
        std::fs::metadata(&out_path).unwrap().modified().unwrap(),
        mtime,
        "skipped output must not be touched"
    );
}

#[test]
fn export_is_deterministic() {
    let (bc, shard) = fixture_tensors();
    let pads = pad_template();
    let again = build_tensors(shard, &pads, &norm()).unwrap();
    assert_eq!(bc.ents, again.ents);
    assert_eq!(bc.mask, again.mask);
    assert_eq!(bc.query, again.query);
    assert_eq!(bc.prev, again.prev);
    assert_eq!(bc.action, again.action);
}

#[test]
fn v3_schema_is_rejected_with_clear_error() {
    // Copy the fixture shard, then downgrade the sidecar's schema_version.
    let dir = tempfile::tempdir().unwrap();
    let src = fixture_shard();
    let npz = dir.path().join("old.npz");
    std::fs::copy(src, &npz).unwrap();
    let mut sidecar: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(src.with_extension("json")).unwrap(),
    )
    .unwrap();
    sidecar["schema_version"] = serde_json::json!(3);
    serde_json::to_writer(
        std::fs::File::create(dir.path().join("old.json")).unwrap(),
        &sidecar,
    )
    .unwrap();

    let err = load_shard_v4(&npz).unwrap_err();
    assert!(
        err.contains("schema_version") && err.contains('3') && err.contains('4'),
        "error must clearly name the found and required schema versions, got: {err}"
    );
}

#[test]
fn missing_sidecar_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let npz = dir.path().join("nosidecar.npz");
    std::fs::copy(fixture_shard(), &npz).unwrap();
    let err = load_shard_v4(&npz).unwrap_err();
    assert!(err.contains("nosidecar.json"), "error must name the missing sidecar, got: {err}");
}
