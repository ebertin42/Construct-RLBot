"""Bulk replay acquisition from the HuggingFace dataset mirror.

`chrisrca/rocket-league-replays` is a ~410 GB dataset of GC1-3 replays laid
out as `<rank>/<playlist>/*.replay`; `allow_patterns` globs scope a pull to a
manageable subset (e.g. `["grand-champion-3/duels/**"]`).
"""
from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

HF_REPO_ID = "chrisrca/rocket-league-replays"


def pull_hf_subset(dest: Path, allow_patterns: list[str], max_files: int | None = None) -> int:
    """Download replay files matching `allow_patterns` into `dest` (resumable
    -- snapshot_download skips files already present). Returns the number of
    `*.replay` files found under `dest` afterward.

    `max_files` is advisory only: `snapshot_download` has no hard file-count
    limit, so real subsetting is done via `allow_patterns`. If set, the
    returned count is capped at `max_files` for convenience.
    """
    dest = Path(dest)
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        local_dir=str(dest),
        allow_patterns=allow_patterns,
    )
    count = sum(1 for _ in dest.rglob("*.replay"))
    if max_files is not None:
        return min(count, max_files)
    return count
