//! `replay-parse`: batch-parses `.replay` files into 120 Hz physics-state +
//! action shards (`<id>.npz` + `<id>.json`) via `construct_replay`'s frame
//! extraction (Task 2), analytic pitch/yaw/roll (Task 3), RocketSim
//! tick-stepping reconstruction (Task 4), and shard writer (Task 5).
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
    sync::atomic::{AtomicUsize, Ordering},
};

use clap::Parser;
use construct_replay::{
    frames::extract_frames, meta::parse_meta, reconstruct::reconstruct_120hz, shard::write_shard,
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
}

enum Outcome {
    Parsed,
    Skipped,
    Failed,
}

fn parse_one(path: &Path, output_dir: &Path, fps: u32, min_team_size: u8) -> Outcome {
    let replay_id = match path.file_stem().and_then(|s| s.to_str()) {
        Some(s) => s.to_string(),
        None => {
            eprintln!("failed {}: unreadable file stem", path.display());
            return Outcome::Failed;
        }
    };

    let result: Result<Outcome, String> = (|| {
        let bytes = fs::read(path).map_err(|e| format!("read: {e}"))?;
        let meta = parse_meta(&bytes)?;
        if meta.team_size < min_team_size {
            return Ok(Outcome::Skipped);
        }
        let frames = extract_frames(&bytes, fps)?;
        let rec = reconstruct_120hz(&frames)?;
        write_shard(output_dir, &replay_id, &meta, &rec)?;
        Ok(Outcome::Parsed)
    })();

    match result {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!("failed {}: {e}", path.display());
            Outcome::Failed
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

    let parsed = AtomicUsize::new(0);
    let skipped = AtomicUsize::new(0);
    let failed = AtomicUsize::new(0);

    entries.par_iter().for_each(|path| {
        let outcome = parse_one(path, &args.output_dir, args.fps, args.min_team_size);
        let counter = match outcome {
            Outcome::Parsed => &parsed,
            Outcome::Skipped => &skipped,
            Outcome::Failed => &failed,
        };
        counter.fetch_add(1, Ordering::Relaxed);
    });

    println!(
        "parsed={} skipped={} failed={}",
        parsed.load(Ordering::Relaxed),
        skipped.load(Ordering::Relaxed),
        failed.load(Ordering::Relaxed)
    );
}
