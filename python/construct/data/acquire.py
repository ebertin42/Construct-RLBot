"""Bulk replay acquisition from the HuggingFace dataset mirror.

`chrisrca/rocket-league-replays` is a ~410 GB dataset of GC1-3 replays laid
out as `<rank>/<playlist>/*.replay` (verified layout: grand-champion-1/duels,
grand-champion-2/duels, grand-champion-3/doubles — NOT a uniform grid). Each
`<rank>/<playlist>/batch_NNNN/` holds ~10k replays (~7.6 GB). `allow_patterns`
globs scope a pull (e.g. `["grand-champion-2/duels/**"]`).

The batches are tens of thousands of small files, so throughput is dominated
by per-file request overhead, not bandwidth. We counter that with high
`max_workers` (many concurrent small fetches) and `hf_transfer` (Rust
accelerated transport). Both are the load-bearing speedups for this dataset.
"""
from __future__ import annotations

import os

# Enable Xet high-performance transfer before huggingface_hub imports its
# transport (current accelerated path; the older HF_HUB_ENABLE_HF_TRANSFER is
# deprecated). No-op if the `hf_xet` backend isn't installed / the repo isn't
# Xet-backed — in which case raw `max_workers` concurrency carries the speedup.
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "chrisrca/rocket-league-replays"


def pull_hf_subset(
    dest: Path,
    allow_patterns: list[str],
    max_files: int | None = None,
    max_workers: int = 32,
) -> int:
    """Download replay files matching `allow_patterns` into `dest` (resumable
    -- snapshot_download skips files already present). Returns the number of
    `*.replay` files found under `dest` afterward.

    `max_workers` sets concurrent file fetches (default 32): the dataset's
    batches are ~10k small files, so throughput scales with concurrency far
    more than with per-file bandwidth. `max_files` is advisory only:
    `snapshot_download` has no hard file-count limit, so real subsetting is
    done via `allow_patterns`; if set, the returned count is capped.
    """
    dest = Path(dest)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        max_workers=max_workers,
    )
    count = sum(1 for _ in dest.rglob("*.replay"))
    if max_files is not None:
        return min(count, max_files)
    return count
