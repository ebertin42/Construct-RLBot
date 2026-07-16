//! `replay-parse`: batch-parses `.replay` files into 120 Hz physics-state +
//! action shards (`<id>.npz` + `<id>.json`) via `construct_replay`'s frame
//! extraction (Task 2), analytic pitch/yaw/roll (Task 3), RocketSim
//! tick-stepping reconstruction (Task 4), and shard writer (Task 5).
//!
//! Optionally (`--reset-pool-out` + `--reset-samples-per-replay > 0`) also
//! samples a `reset_pool::ResetState` pool from each replay's reconstruction
//! (Task 6) and appends it, in jsonl form, to one pool file for the whole
//! batch.
//!
//! ## Working directory requirement
//! Reconstruction steps a RocketSim `Arena`, which loads collision-mesh
//! assets via `construct_replay::sim_init::ensure_init` the first time it
//! runs. That loader resolves mesh assets *relative to the process's
//! current working directory*, trying `assets/collision_meshes` (workspace
//! root) then `../assets/collision_meshes` (crate root) — it does **not**
//! resolve relative to `--input-dir`/`--output-dir` or the binary's own
//! path. Run this binary from the repository root or from `replay/` (e.g.
//! `cd /path/to/Construct-RLBot && ./target/release/replay-parse ...`), or
//! every replay in the batch will fail with a mesh-loading error.
//!
//! One bad replay never aborts the batch: each replay is parsed inside a
//! `Result`-returning closure, so a parse/reconstruction/write failure is
//! logged to stderr and counted, and the batch continues.

use std::{
    fs,
    path::{Path, PathBuf},
};

use clap::Parser;
use construct_replay::{
    frames::extract_frames,
    meta::parse_meta,
    reconstruct::reconstruct_120hz,
    reset_pool::{sample_reset_states, write_pool_jsonl, ResetState},
    shard::write_shard,
};
use rayon::prelude::*;

/// Batch-parse `.replay` files into 120 Hz physics-state + action shards.
///
/// Reconstruction requires RocketSim's collision-mesh assets to resolve as
/// `assets/collision_meshes` or `../assets/collision_meshes` from the
/// current working directory — run this from the repo root or `replay/`.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Directory to scan (non-recursively) for `*.replay` files.
    #[arg(long)]
    input_dir: PathBuf,

    /// Directory to write `<id>.npz` + `<id>.json` shards into (created if it doesn't exist).
    #[arg(long)]
    output_dir: PathBuf,

    /// Target resample fps for frame extraction, prior to 120 Hz reconstruction.
    #[arg(long, default_value_t = 30)]
    fps: u32,

    /// Rayon thread-pool size for parallel replay parsing. Omit to use
    /// rayon's default (one worker per logical CPU).
    #[arg(long)]
    jobs: Option<usize>,

    /// Replays with a per-side team_size below this are skipped (not parsed
    /// or written), and counted separately from failures in the summary.
    #[arg(long, default_value_t = 1)]
    min_team_size: u8,

    /// Path to a single reset-state-pool jsonl file. Only takes effect when
    /// paired with `--reset-samples-per-replay > 0`; reset states sampled
    /// from every replay in the batch are collected in memory and appended
    /// to this one file after the whole batch finishes (see
    /// `construct_replay::reset_pool`'s module doc for the jsonl schema).
    #[arg(long)]
    reset_pool_out: Option<PathBuf>,

    /// Number of reset states to sample per replay for `--reset-pool-out`.
    /// `0` (default) disables reset-pool sampling entirely.
    #[arg(long, default_value_t = 0)]
    reset_samples_per_replay: usize,

    /// Storage stride over the reconstructed 120 Hz tick stream: only every
    /// `stride`-th tick is written to the shard (see `shard::write_shard`).
    /// Default 8 matches the bot's `tick_skip=8` decision rate, i.e.
    /// 120/8 = 15 Hz storage instead of the full 120 Hz. Use `1` to store
    /// every tick (the old, pre-stride behavior).
    #[arg(long, default_value_t = 8)]
    stride: usize,
}

/// Deterministic per-replay seed for reset-state sampling, derived from the
/// replay's id (file stem) via FNV-1a so re-running the CLI over the same
/// input file always samples the same states regardless of rayon's
/// scheduling order or `--jobs`.
fn seed_for_replay(replay_id: &str) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for b in replay_id.bytes() {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

enum Outcome {
    Parsed,
    Skipped,
    Failed,
}

fn parse_one(
    path: &Path,
    output_dir: &Path,
    fps: u32,
    min_team_size: u8,
    reset_samples_per_replay: usize,
    stride: usize,
) -> (Outcome, Vec<ResetState>) {
    let replay_id = match path.file_stem().and_then(|s| s.to_str()) {
        Some(s) => s.to_string(),
        None => {
            eprintln!("failed {}: unreadable file stem", path.display());
            return (Outcome::Failed, Vec::new());
        }
    };

    let result: Result<(Outcome, Vec<ResetState>), String> = (|| {
        let bytes = fs::read(path).map_err(|e| format!("read: {e}"))?;
        let meta = parse_meta(&bytes)?;
        if meta.team_size < min_team_size {
            return Ok((Outcome::Skipped, Vec::new()));
        }
        let frames = extract_frames(&bytes, fps)?;
        let rec = reconstruct_120hz(&frames, stride)?;
        write_shard(output_dir, &replay_id, &meta, &rec, stride)?;
        let states = if reset_samples_per_replay > 0 {
            sample_reset_states(&rec, reset_samples_per_replay, seed_for_replay(&replay_id))
        } else {
            Vec::new()
        };
        Ok((Outcome::Parsed, states))
    })();

    match result {
        Ok(out) => out,
        Err(e) => {
            eprintln!("failed {}: {e}", path.display());
            (Outcome::Failed, Vec::new())
        }
    }
}

fn main() {
    let args = Args::parse();

    if let Some(jobs) = args.jobs {
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global() {
            eprintln!("failed to configure rayon thread pool with --jobs {jobs}: {e}");
            std::process::exit(1);
        }
    }

    let entries: Vec<PathBuf> = match fs::read_dir(&args.input_dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("replay"))
            .collect(),
        Err(e) => {
            eprintln!("cannot read --input-dir {}: {e}", args.input_dir.display());
            std::process::exit(1);
        }
    };

    if let Err(e) = fs::create_dir_all(&args.output_dir) {
        eprintln!("cannot create --output-dir {}: {e}", args.output_dir.display());
        std::process::exit(1);
    }

    // Collect-then-write-once: each worker's outcome + sampled reset states
    // are gathered by rayon's `collect` (no shared mutable state during the
    // parallel phase), then the whole batch's reset states are appended to
    // `--reset-pool-out` in one single-threaded write after `par_iter`
    // finishes — avoids interleaved/corrupted jsonl lines from concurrent
    // appends without needing a `Mutex`-guarded writer.
    let results: Vec<(Outcome, Vec<ResetState>)> = entries
        .par_iter()
        .map(|path| {
            parse_one(
                path,
                &args.output_dir,
                args.fps,
                args.min_team_size,
                args.reset_samples_per_replay,
                args.stride,
            )
        })
        .collect();

    let mut parsed = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut reset_states: Vec<ResetState> = Vec::new();
    for (outcome, states) in results {
        match outcome {
            Outcome::Parsed => parsed += 1,
            Outcome::Skipped => skipped += 1,
            Outcome::Failed => failed += 1,
        }
        reset_states.extend(states);
    }

    if args.reset_samples_per_replay > 0 {
        if let Some(pool_path) = &args.reset_pool_out {
            if let Err(e) = write_pool_jsonl(pool_path, &reset_states, true) {
                eprintln!("failed to write reset pool {}: {e}", pool_path.display());
                std::process::exit(1);
            }
        }
    }

    println!(
        "parsed={parsed} skipped={skipped} failed={failed} reset_states={}",
        reset_states.len()
    );
}
