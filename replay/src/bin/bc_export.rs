//! `bc-export`: batch-converts schema-v4 shards (`<id>.npz` + `<id>.json`,
//! from `replay-parse`) into obs-v1 BC training tensors (`bc_<id>.npz`) via
//! `construct_replay::bc_obs` — which rebuilds a minimal `GameState` per
//! stored tick and calls the live engine's own `obs_v1::build` per car (one
//! sample per car per tick, each from its own mirrored POV). See `bc_obs`'s
//! module doc for the exact array layout and prev-window semantics.
//!
//! Resumable: shards whose `bc_<id>.npz` already exists in `--out` are
//! skipped untouched (interrupted writes never count — outputs are written
//! to a `.tmp` and renamed into place). Deterministic: the shard list is
//! sorted by filename and every per-shard export is a pure function of the
//! shard's contents, so sample order is identical across runs regardless of
//! `--jobs`.
//!
//! ## Working directory requirement
//! The one-time pad template builds a RocketSim `Arena`, whose collision
//! meshes resolve relative to the current working directory (same
//! constraint, and same fix, as `replay-parse`'s module doc): run this from
//! the repository root or `replay/`.

use std::path::PathBuf;

use clap::Parser;
use construct_engine::schema::Schema;
use construct_replay::bc_obs::{export_shard_file, pad_template};
use rayon::prelude::*;

/// Batch-convert schema-v4 replay shards into obs-v1 BC training tensors.
///
/// Requires RocketSim's collision-mesh assets to resolve as
/// `assets/collision_meshes` or `../assets/collision_meshes` from the
/// current working directory — run this from the repo root or `replay/`.
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    /// Directory to scan (non-recursively) for `*.npz` v4 shards (each needs
    /// its `<id>.json` sidecar next to it).
    #[arg(long)]
    shards: PathBuf,

    /// Directory to write `bc_<id>.npz` tensor files into (created if it
    /// doesn't exist). Existing outputs are skipped (resume).
    #[arg(long)]
    out: PathBuf,

    /// Rayon thread-pool size. Omit to use rayon's default (one worker per
    /// logical CPU).
    #[arg(long)]
    jobs: Option<usize>,

    /// Obs schema TOML supplying the position/velocity normalization
    /// constants — must be the same schema the trained net deploys with.
    #[arg(long, default_value = "schema/v1.toml")]
    schema: String,
}

enum Outcome {
    Exported(usize),
    Skipped,
    Failed,
}

fn main() {
    let args = Args::parse();

    if let Some(jobs) = args.jobs {
        if let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(jobs).build_global() {
            eprintln!("failed to configure rayon thread pool with --jobs {jobs}: {e}");
            std::process::exit(1);
        }
    }

    let norm = match Schema::load(&args.schema) {
        Ok(s) => s.normalization,
        Err(e) => {
            eprintln!("cannot load --schema {}: {e}", args.schema);
            std::process::exit(1);
        }
    };

    let mut entries: Vec<PathBuf> = match std::fs::read_dir(&args.shards) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("npz"))
            .collect(),
        Err(e) => {
            eprintln!("cannot read --shards {}: {e}", args.shards.display());
            std::process::exit(1);
        }
    };
    // Fixed iteration order -> identical export order (and identical
    // failure/summary attribution) run to run, independent of readdir order.
    entries.sort();

    if let Err(e) = std::fs::create_dir_all(&args.out) {
        eprintln!("cannot create --out {}: {e}", args.out.display());
        std::process::exit(1);
    }

    // One arena construction for the whole run; workers share the immutable
    // template (plain-data `BoostPad`s). Built before the parallel phase so
    // mesh loading happens exactly once, up front.
    let pads = pad_template();

    let results: Vec<Outcome> = entries
        .par_iter()
        .map(|path| match export_shard_file(path, &args.out, &pads, &norm) {
            Ok(Some((_, samples))) => Outcome::Exported(samples),
            Ok(None) => Outcome::Skipped,
            Err(e) => {
                eprintln!("failed {}: {e}", path.display());
                Outcome::Failed
            }
        })
        .collect();

    let mut exported = 0usize;
    let mut skipped = 0usize;
    let mut failed = 0usize;
    let mut samples = 0usize;
    for outcome in results {
        match outcome {
            Outcome::Exported(n) => {
                exported += 1;
                samples += n;
            }
            Outcome::Skipped => skipped += 1,
            Outcome::Failed => failed += 1,
        }
    }

    println!("exported={exported} skipped_existing={skipped} failed={failed} samples={samples}");
}
