//! Task B3 + schema v5: `bc_obs` — v4/v5 shards -> obs-v1 training tensors.
//!
//! The fixture shard is built ONCE per test binary (a `OnceLock`-guarded
//! parse of `tests/fixtures/sample.replay`, the same fixture every other
//! integration test here uses) because reconstruction is by far the
//! slowest step and every test below only needs read access to the same
//! deterministic shard.
//!
//! The fixture (probed at 30 fps) carries every v5 signal natively: one
//! mid-match goal→kickoff reset plus a trailing post-goal segment (3
//! episode markers total, counting the opening kickoff) and two per-player
//! demoed/absent stretches (the goal-reset actor teardown and the trailing
//! segment) — so marker resets, demoed-sample skipping, and the
//! demoed→live respawn reset are all exercised on real data, with a
//! synthetic shard covering the exact per-row semantics at the unit level.

use std::path::PathBuf;
use std::sync::OnceLock;

use construct_engine::schema::Schema;
use construct_replay::{
    bc_obs::{build_tensors, export_shard_file, load_shard, pad_template, BcTensors, Shard},
    frames::{extract_frames, CarFrame, RigidFrame},
    meta::parse_meta,
    reconstruct::{reconstruct_120hz, Reconstructed, Tick},
    shard::write_shard,
};
use ndarray::s;
use ndarray_npy::{NpzReader, NpzWriter};

const STRIDE: usize = 8;

/// Parses the fixture replay into a v5 shard exactly once for the whole test
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

fn fixture_tensors() -> &'static (BcTensors, Shard) {
    static TENSORS: OnceLock<(BcTensors, Shard)> = OnceLock::new();
    TENSORS.get_or_init(|| {
        let shard = load_shard(fixture_shard()).unwrap();
        let pads = pad_template();
        let out = build_tensors(&shard, &pads, &norm()).unwrap();
        (out, shard)
    })
}

/// Whether car `p` is demolished/absent on stored row `t` (v5 `is_demoed`
/// column; v4 shards have no such column and every row counts as live).
fn is_demoed(shard: &Shard, t: usize, p: usize) -> bool {
    shard.cars_state.shape()[2] >= 18 && shard.cars_state[[t, p, 17]] != 0.0
}

/// The documented emission order: row-major `(t, p)` with demoed rows
/// absent. `live_map(shard)[s]` is the `(t, p)` behind sample `s`.
fn live_map(shard: &Shard) -> Vec<(usize, usize)> {
    let t_count = shard.tick_index.len();
    let p_count = shard.player_teams.len();
    let mut map = Vec::with_capacity(t_count * p_count);
    for t in 0..t_count {
        for p in 0..p_count {
            if !is_demoed(shard, t, p) {
                map.push((t, p));
            }
        }
    }
    map
}

#[test]
fn shapes_are_s_by_obs_v1_dims() {
    let (bc, shard) = fixture_tensors();
    let t = shard.tick_index.len();
    let p = shard.player_teams.len();
    let live = live_map(shard);
    let s = live.len();
    assert!(t > 100, "fixture shard must have a real tick count, got {t}");
    assert_eq!(p, 2, "fixture is a 1v1 replay");
    assert_eq!(shard.schema_version, 5, "fixture shard is written at schema v5");
    assert!(
        s < t * p,
        "fixture has demoed rows (goal reset + trailing segment) — some samples must be skipped"
    );
    assert!(s > (t * p) / 2, "most rows are live");
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
fn orange_pov_query_pos_is_mirrored_blue_is_not() {
    // Guards the player_teams -> Team mapping in bc_obs.rs: an inverted
    // mapping (0 -> Orange) would mirror the wrong POVs yet pass every other
    // test in this file. query[0..26] is the self entity row (obs_v1::build /
    // EntityRow::write), whose pos slice is [5..8) = mir(raw_pos) * pos_norm,
    // with mir negating x,y exactly when the car's team is Orange.
    let (bc, shard) = fixture_tensors();
    let pk = norm().pos_norm as f32;
    assert!(shard.player_teams.iter().any(|&t| t == 0), "fixture must have a blue car");
    assert!(shard.player_teams.iter().any(|&t| t == 1), "fixture must have an orange car");

    let mut checked = [0usize; 2]; // [blue, orange] rows actually asserted
    for (s, &(t, p)) in live_map(shard).iter().enumerate() {
        let (x, y, z) = (
            shard.cars_state[[t, p, 0]],
            shard.cars_state[[t, p, 1]],
            shard.cars_state[[t, p, 2]],
        );
        // Mirroring only observably changes x/y — skip rows where either
        // is near zero and the assertion would be vacuous.
        if x.abs() < 100.0 || y.abs() < 100.0 {
            continue;
        }
        let team = shard.player_teams[p];
        let expect = if team == 1 {
            [-x * pk, -y * pk, z * pk] // orange: play-as-blue mirror
        } else {
            [x * pk, y * pk, z * pk] // blue: unmirrored
        };
        for (k, &e) in expect.iter().enumerate() {
            let got = bc.query[[s, 5 + k]];
            assert!(
                (got - e).abs() < 1e-5,
                "t={t} p={p} team={team}: query[{}] = {got}, expected {e}",
                5 + k
            );
        }
        checked[team as usize] += 1;
    }
    assert!(checked[0] > 0, "no blue row with |x|,|y| > 100 was asserted");
    assert!(checked[1] > 0, "no orange row with |x|,|y| > 100 was asserted");
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
    // Label column is exactly the shard's stored cars_action_idx in the
    // documented emission order: row-major (t, p) with demoed rows skipped.
    let live = live_map(shard);
    assert_eq!(bc.action.len(), live.len());
    for (s, &(t, p)) in live.iter().enumerate() {
        assert_eq!(
            bc.action[s],
            shard.cars_action_idx[[t, p]],
            "action[{s}] must equal cars_action_idx[[{t},{p}]]"
        );
    }
}

#[test]
fn no_sample_is_emitted_for_demoed_rows() {
    let (bc, shard) = fixture_tensors();
    let t_count = shard.tick_index.len();
    let p_count = shard.player_teams.len();
    let demoed_rows: usize = (0..t_count)
        .flat_map(|t| (0..p_count).map(move |p| (t, p)))
        .filter(|&(t, p)| is_demoed(shard, t, p))
        .count();
    assert!(demoed_rows > 10, "fixture must actually have demoed rows, got {demoed_rows}");
    assert_eq!(
        bc.action.len() + demoed_rows,
        t_count * p_count,
        "every non-demoed (t, p) emits exactly one sample; every demoed one emits none"
    );
}

#[test]
fn prev_window_semantics_shift_gap_marker_respawn() {
    let (bc, shard) = fixture_tensors();
    let t_count = shard.tick_index.len();
    let p_count = shard.player_teams.len();

    // Walk stored rows in emission order, tracking each car's previously
    // emitted sample, and assert the documented prev-window rule at every
    // sample: zeros after any discontinuity (t=0, dropped-tick gap, v5
    // episode marker, own demoed→live respawn), most-recent-first shift
    // otherwise.
    let mut cursor = 0usize;
    let mut last_sample: Vec<Option<usize>> = vec![None; p_count];
    let (mut saw_shift, mut saw_gap, mut saw_marker, mut saw_respawn) = (false, false, false, false);
    for t in 0..t_count {
        let gap = t > 0 && shard.tick_index[t] - shard.tick_index[t - 1] != STRIDE as i64;
        let marker = shard.episode_marker[t] != 0;
        for p in 0..p_count {
            if is_demoed(shard, t, p) {
                continue;
            }
            let s = cursor;
            cursor += 1;
            let respawn = t > 0 && is_demoed(shard, t - 1, p);
            if t == 0 || gap || marker || respawn {
                if t > 0 {
                    saw_gap |= gap;
                    saw_marker |= marker;
                    saw_respawn |= respawn && !gap && !marker;
                }
                for k in 0..5 {
                    assert_eq!(
                        bc.prev[[s, k]],
                        0,
                        "t={t} p={p}: prev must be zeros after a discontinuity \
                         (gap={gap} marker={marker} respawn={respawn})"
                    );
                }
            } else {
                // No discontinuity and p was live on row t-1 (respawn is
                // false), so its ring advanced there: most-recent-first,
                // same semantics as the live engine (engine/src/episode.rs).
                saw_shift = true;
                let s_prev = last_sample[p].expect("live at t-1 implies an emitted sample");
                assert_eq!(
                    bc.prev[[s, 0]],
                    shard.cars_action_idx[[t - 1, p]],
                    "t={t} p={p}: prev[0] must be the action executed on row t-1"
                );
                for k in 1..5 {
                    assert_eq!(
                        bc.prev[[s, k]],
                        bc.prev[[s_prev, k - 1]],
                        "t={t} p={p}: prev[{k}] must shift from previous sample's prev[{}]",
                        k - 1
                    );
                }
            }
            last_sample[p] = Some(s);
        }
    }
    assert_eq!(cursor, bc.action.len());
    assert!(saw_shift, "fixture must contain contiguous live rows to exercise the shift path");
    // Whether the fixture reconstruction drops any tick at all is Bullet
    // allocation-history luck (schema v5's demoed-car fix removed the demo
    // windows that used to be dropped wholesale; what remains is a ~3-tick
    // blowup that appears in some binary layouts and not others — same
    // sensitivity d798ed8 documented). The per-row walk above still verifies
    // gap-reset behavior whenever a gap IS present; the guaranteed gap-reset
    // coverage lives in `synthetic_v5_skip_and_reset_semantics`.
    let _ = saw_gap;
    assert!(
        saw_marker,
        "fixture must contain a goal→kickoff episode marker with live cars (the mid-match goal)"
    );
    assert!(
        saw_respawn,
        "fixture must contain a demoed→live respawn transition (post-goal actor teardown)"
    );
}

#[test]
fn export_file_writes_npz_and_resumes_by_skipping_existing() {
    let out_dir = tempfile::tempdir().unwrap();
    let pads = pad_template();
    let nrm = norm();

    let first = export_shard_file(fixture_shard(), out_dir.path(), &pads, &nrm, false).unwrap();
    let (out_path, samples, existed) = first.expect("first export must write");
    assert_eq!(out_path, out_dir.path().join("bc_sample.npz"));
    assert!(out_path.exists());
    assert!(samples > 0);
    assert!(!existed, "first export must not report a pre-existing output");

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
    let second = export_shard_file(fixture_shard(), out_dir.path(), &pads, &nrm, false).unwrap();
    assert!(second.is_none(), "existing output must be skipped (resumable)");
    assert_eq!(
        std::fs::metadata(&out_path).unwrap().modified().unwrap(),
        mtime,
        "skipped output must not be touched"
    );
}

#[test]
fn force_overwrites_existing_output_instead_of_skipping() {
    // Tomorrow's in-place corpus re-export (v5 re-parse writing back into
    // the same `data/bc` dir) depends on `--force` actually replacing every
    // existing `bc_*.npz`, not silently trusting the skip-existing resume
    // check. Guard: without force, an existing output is left untouched
    // (mtime + content); with force, it's rewritten (mtime advances, the
    // `existed` flag comes back true) and the decoded tensor content matches
    // a fresh export, i.e. the overwrite is a faithful re-export, not a
    // truncated or stale write.
    let out_dir = tempfile::tempdir().unwrap();
    let pads = pad_template();
    let nrm = norm();

    let first = export_shard_file(fixture_shard(), out_dir.path(), &pads, &nrm, false).unwrap();
    let (out_path, samples, existed) = first.expect("first export must write");
    assert!(!existed, "first export must not report a pre-existing output");
    let read_action = |p: &std::path::Path| -> ndarray::Array1<i64> {
        let mut npz = NpzReader::new(std::fs::File::open(p).unwrap()).unwrap();
        npz.by_name("action.npy").unwrap()
    };
    let original_action = read_action(&out_path);
    let mtime_before = std::fs::metadata(&out_path).unwrap().modified().unwrap();

    // Corrupt the existing output in place (as a truncated/stale prior
    // export might be) so a genuine overwrite is unambiguous: a skip would
    // leave the corruption; a real overwrite replaces it with valid data.
    std::fs::write(&out_path, b"stale corpus data from before the re-parse").unwrap();

    // Without --force: still skipped, corruption survives untouched.
    let no_force = export_shard_file(fixture_shard(), out_dir.path(), &pads, &nrm, false).unwrap();
    assert!(no_force.is_none(), "existing output must still be skipped without --force");
    assert_eq!(
        std::fs::read(&out_path).unwrap(),
        b"stale corpus data from before the re-parse",
        "without --force the corrupted existing output must not be touched"
    );

    // Ensure the filesystem mtime clock can actually distinguish the two
    // writes (some filesystems have coarse mtime resolution).
    std::thread::sleep(std::time::Duration::from_millis(1100));

    // With --force: the corrupted output is overwritten, reported as such.
    let forced = export_shard_file(fixture_shard(), out_dir.path(), &pads, &nrm, true).unwrap();
    let (forced_path, forced_samples, forced_existed) = forced.expect("--force must write, not skip");
    assert_eq!(forced_path, out_path);
    assert_eq!(forced_samples, samples, "re-export of the same shard must produce the same sample count");
    assert!(forced_existed, "--force overwrite must report existed=true");

    let mtime_after = std::fs::metadata(&out_path).unwrap().modified().unwrap();
    assert!(mtime_after > mtime_before, "--force must actually rewrite the file (mtime must advance)");

    // The rewritten file must be a valid, correct npz again (not the
    // corrupted stand-in) — decode it, since the npz container embeds a
    // wall-clock timestamp and so isn't byte-identical run to run even for
    // identical tensor content (see `export_is_deterministic`, which checks
    // decoded arrays for the same reason).
    let rewritten_action = read_action(&out_path);
    assert_eq!(
        rewritten_action, original_action,
        "--force re-export of the same shard must produce identical tensor content"
    );
}

#[test]
fn foreign_leftover_tmp_file_does_not_break_export() {
    // Task #45 follow-up: `export_shard_file`'s tmp filename is now
    // pid-suffixed (`bc_<stem>.npz.tmp.<pid>`) specifically so concurrent/
    // retried exporters can't interleave writes to the same tmp path. Guard
    // that a *foreign* leftover tmp file (e.g. from a killed process, a
    // different pid than ours) sitting in `out_dir` is simply ignored --
    // never matched by the `out_path.exists()` skip-check, never read back,
    // never clobbered by our own write+rename.
    let out_dir = tempfile::tempdir().unwrap();
    let pads = pad_template();
    let nrm = norm();

    let foreign_tmp = out_dir.path().join("bc_sample.npz.tmp.999999999");
    std::fs::write(&foreign_tmp, b"stale partial data from a dead process").unwrap();

    let result = export_shard_file(fixture_shard(), out_dir.path(), &pads, &nrm, false).unwrap();
    let (out_path, samples, existed) = result.expect("export must succeed despite a foreign leftover tmp file");
    assert!(!existed, "no real output existed yet — only a foreign tmp file");
    assert_eq!(out_path, out_dir.path().join("bc_sample.npz"));
    assert!(out_path.exists());
    assert!(samples > 0);

    // Foreign tmp file: untouched, not deleted, not mistaken for our output.
    assert!(foreign_tmp.exists(), "foreign .tmp file must not be deleted by export");
    assert_eq!(
        std::fs::read(&foreign_tmp).unwrap(),
        b"stale partial data from a dead process",
        "foreign .tmp file's contents must be untouched"
    );

    // The real output must load correctly (not e.g. accidentally the
    // foreign file renamed into place).
    let mut npz = NpzReader::new(std::fs::File::open(&out_path).unwrap()).unwrap();
    let action: ndarray::Array1<i64> = npz.by_name("action.npy").unwrap();
    assert_eq!(action.shape(), &[samples]);
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

/// Synthesizes a faithful schema-v4 shard from the v5 fixture (drop the
/// `is_demoed` column and the `episode_marker` array, downgrade the
/// sidecar) and asserts (a) it still loads, with the documented v4
/// fallbacks, (b) it exports with v4 semantics — S == T*P, nothing skipped —
/// and (c) per-sample obs/labels are IDENTICAL to the v5 export at the
/// mapped indices: v5 only removes demoed samples and resets prev windows
/// earlier; it must not perturb ents/mask/query/action of surviving rows.
#[test]
fn v4_shard_still_loads_and_exports_identically() {
    let (bc5, shard5) = fixture_tensors();
    let dir = tempfile::tempdir().unwrap();
    let npz_path = dir.path().join("old4.npz");

    // v4 npz: same arrays minus episode_marker, cars_state cut to 17 cols.
    let cars_state17 = shard5.cars_state.slice(s![.., .., ..17]).to_owned();
    let file = std::fs::File::create(&npz_path).unwrap();
    let mut npz = NpzWriter::new(file);
    npz.add_array("ball", &shard5.ball).unwrap();
    npz.add_array("cars_state", &cars_state17).unwrap();
    npz.add_array("cars_action_idx", &shard5.cars_action_idx).unwrap();
    npz.add_array("pads", &shard5.pads).unwrap();
    npz.add_array("ball_pred", &shard5.ball_pred).unwrap();
    npz.add_array("player_teams", &shard5.player_teams).unwrap();
    npz.add_array("tick_index", &shard5.tick_index).unwrap();
    npz.finish().unwrap();

    // v4 sidecar: downgrade version, 17 documented columns, no marker doc.
    let mut sidecar: serde_json::Value = serde_json::from_reader(
        std::fs::File::open(fixture_shard().with_extension("json")).unwrap(),
    )
    .unwrap();
    sidecar["schema_version"] = serde_json::json!(4);
    let cols = sidecar["cars_state_columns"].as_array().unwrap()[..17].to_vec();
    sidecar["cars_state_columns"] = serde_json::Value::Array(cols);
    sidecar.as_object_mut().unwrap().remove("episode_marker");
    serde_json::to_writer(std::fs::File::create(dir.path().join("old4.json")).unwrap(), &sidecar)
        .unwrap();

    let shard4 = load_shard(&npz_path).unwrap();
    assert_eq!(shard4.schema_version, 4);
    assert_eq!(shard4.cars_state.shape()[2], 17, "v4 cars_state has 17 columns");
    assert!(
        shard4.episode_marker.iter().all(|&v| v == 0),
        "v4 shards must load with an all-zero synthesized episode_marker"
    );

    let t_count = shard4.tick_index.len();
    let p_count = shard4.player_teams.len();
    let bc4 = build_tensors(&shard4, &pad_template(), &norm()).unwrap();
    assert_eq!(
        bc4.action.len(),
        t_count * p_count,
        "v4 export must emit every (t, p) sample — no is_demoed column, nothing skipped"
    );
    for t in 0..t_count {
        for p in 0..p_count {
            assert_eq!(
                bc4.action[t * p_count + p],
                shard4.cars_action_idx[[t, p]],
                "v4 keeps the dense s = t*P + p mapping"
            );
        }
    }

    // Cross-version consistency at the mapped indices.
    for (s5, &(t, p)) in live_map(shard5).iter().enumerate() {
        let s4 = t * p_count + p;
        assert_eq!(bc5.action[s5], bc4.action[s4], "action differs at t={t} p={p}");
        assert_eq!(
            bc5.ents.slice(s![s5, .., ..]),
            bc4.ents.slice(s![s4, .., ..]),
            "ents differ at t={t} p={p}"
        );
        assert_eq!(
            bc5.mask.slice(s![s5, ..]),
            bc4.mask.slice(s![s4, ..]),
            "mask differs at t={t} p={p}"
        );
        assert_eq!(
            bc5.query.slice(s![s5, ..]),
            bc4.query.slice(s![s4, ..]),
            "query differs at t={t} p={p}"
        );
    }
}

/// Unit-level v5 semantics on a fully synthetic shard (P=1, 10 stored rows
/// at stride 8): rows 3-4 demoed, marker on row 7, and a 5-tick tick_index
/// gap (dropped-tick simulation) before row 9. Expected per-row behavior:
///   row 0: zeros (start)          row 5: zeros (respawn after demo)
///   row 1: shift of a0            row 6: shift of a5
///   row 2: shift of a1, a0        row 7: zeros (episode marker)
///   rows 3-4: NO samples          row 8: shift of a7
///                                 row 9: zeros (tick_index gap)
#[test]
fn synthetic_v5_skip_and_reset_semantics() {
    fn synth_tick(i: usize, demoed: bool, marker: bool) -> Tick {
        let block = i / 8; // stored-row index this tick belongs to
        let throttle = [(-1.0f32), 0.0, 1.0][block % 3];
        let rb = RigidFrame {
            pos: [500.0, -300.0, 17.0],
            vel: [100.0, 0.0, 0.0],
            ang_vel: [0.0; 3],
            quat: [0.0, 0.0, 0.0, 1.0],
        };
        Tick {
            ball: RigidFrame {
                pos: [0.0, 1000.0, 93.0],
                vel: [0.0; 3],
                ang_vel: [0.0; 3],
                quat: [0.0, 0.0, 0.0, 1.0],
            },
            cars: vec![CarFrame {
                rb,
                boost: 0.5,
                team: 0,
                throttle,
                steer: 0.0,
                handbrake: false,
                jump_active: false,
                dodge_active: false,
                on_ground: true,
                demoed,
            }],
            actions: vec![[throttle, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            // A 5-tick jump before the last stored row (as if ticks were
            // dropped for insanity) — the gap-reset path guaranteed here
            // because the fixture's natural drops are allocation-luck.
            tick_index: if i >= 72 { i as i64 + 5 } else { i as i64 },
            is_boundary: i % 4 == 0,
            pads: vec![(0.0, true); 34],
            has_flip: vec![true],
            replay_demoed: vec![demoed],
            episode_marker: marker,
            ball_pred: [[0.0; 6]; 4],
        }
    }

    let bytes = std::fs::read("tests/fixtures/sample.replay").unwrap();
    let meta = parse_meta(&bytes).unwrap();
    let ticks: Vec<Tick> = (0..80)
        .map(|i| synth_tick(i, (24..40).contains(&i), i == 56))
        .collect();
    let rec = Reconstructed { ticks, player_teams: vec![0], fps: 30 };
    let dir = tempfile::tempdir().unwrap();
    let path = write_shard(dir.path(), "synth", &meta, &rec, 8).unwrap();

    let shard = load_shard(&path).unwrap();
    assert_eq!(shard.tick_index.len(), 10);
    assert_eq!(shard.episode_marker.to_vec(), vec![0, 0, 0, 0, 0, 0, 0, 1, 0, 0]);
    let a = |r: usize| shard.cars_action_idx[[r, 0]];
    assert_ne!(a(0), a(1), "adjacent stored rows must project to distinct action indices");

    let bc = build_tensors(&shard, &pad_template(), &norm()).unwrap();
    assert_eq!(bc.action.len(), 8, "rows 3 and 4 are demoed and must emit no samples");
    let emitted_rows = [0usize, 1, 2, 5, 6, 7, 8, 9];
    for (s, &r) in emitted_rows.iter().enumerate() {
        assert_eq!(bc.action[s], a(r), "sample {s} must be stored row {r}'s action");
    }
    let prev_of = |s: usize| -> [i64; 5] { std::array::from_fn(|k| bc.prev[[s, k]]) };
    assert_eq!(prev_of(0), [0; 5], "row 0: fresh history");
    assert_eq!(prev_of(1), [a(0), 0, 0, 0, 0], "row 1: shift of a0");
    assert_eq!(prev_of(2), [a(1), a(0), 0, 0, 0], "row 2: shift of a1, a0");
    assert_eq!(prev_of(3), [0; 5], "row 5: zeros — demoed→live respawn resets the ring");
    assert_eq!(prev_of(4), [a(5), 0, 0, 0, 0], "row 6: shift of a5");
    assert_eq!(prev_of(5), [0; 5], "row 7: zeros — episode marker resets the ring");
    assert_eq!(prev_of(6), [a(7), 0, 0, 0, 0], "row 8: shift of a7");
    assert_eq!(
        shard.tick_index[9] - shard.tick_index[8],
        13,
        "row 9 must sit across the synthetic dropped-tick gap"
    );
    assert_eq!(prev_of(7), [0; 5], "row 9: zeros — tick_index gap resets the ring");

    // All-demoed shard: loud error, not a silent empty export.
    let ticks: Vec<Tick> = (0..16).map(|i| synth_tick(i, true, false)).collect();
    let rec = Reconstructed { ticks, player_teams: vec![0], fps: 30 };
    let path = write_shard(dir.path(), "alldemo", &meta, &rec, 8).unwrap();
    let shard = load_shard(&path).unwrap();
    let err = match build_tensors(&shard, &pad_template(), &norm()) {
        Err(e) => e,
        Ok(_) => panic!("all-demoed shard must not export"),
    };
    assert!(err.contains("zero samples"), "all-demoed must fail loud, got: {err}");
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

    let err = load_shard(&npz).unwrap_err();
    assert!(
        err.contains("schema_version") && err.contains('3') && err.contains('4') && err.contains('5'),
        "error must clearly name the found and supported schema versions, got: {err}"
    );
}

#[test]
fn missing_sidecar_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let npz = dir.path().join("nosidecar.npz");
    std::fs::copy(fixture_shard(), &npz).unwrap();
    let err = load_shard(&npz).unwrap_err();
    assert!(err.contains("nosidecar.json"), "error must name the missing sidecar, got: {err}");
}
