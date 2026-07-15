"""Shard directory indexing: aggregates the `<id>.json` sidecars a
`replay-parse` batch writes (see `replay/src/shard.rs`'s `ShardSidecar`) into
one `manifest.json` summarizing the whole shard directory.

Every sidecar carries a `schema_version` (see `SHARD_SCHEMA_VERSION` in
`replay/src/shard.rs`); `build_index` asserts every shard in the directory
agrees on that version so downstream loaders can trust a single manifest
value rather than re-checking every shard.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

MANIFEST_FILENAME = "manifest.json"


def build_index(shard_dir: Path) -> dict:
    """Scans `shard_dir` for `*.json` sidecars (excluding `manifest.json`
    itself, so re-running is idempotent) and aggregates them into
    `{total_ticks, num_shards, by_team_size, schema_version}`. Writes the
    same dict to `<shard_dir>/manifest.json` and returns it.

    Raises `ValueError` if sidecars disagree on `schema_version`.
    """
    shard_dir = Path(shard_dir)
    manifest_path = shard_dir / MANIFEST_FILENAME

    total_ticks = 0
    num_shards = 0
    by_team_size: Counter[str] = Counter()
    schema_version: int | None = None

    for sidecar_path in sorted(shard_dir.glob("*.json")):
        if sidecar_path.name == MANIFEST_FILENAME:
            continue
        sidecar = json.loads(sidecar_path.read_text())

        sidecar_version = sidecar["schema_version"]
        if schema_version is None:
            schema_version = sidecar_version
        elif sidecar_version != schema_version:
            raise ValueError(
                f"schema_version mismatch in {sidecar_path}: "
                f"expected {schema_version}, got {sidecar_version}"
            )

        total_ticks += sidecar["num_ticks"]
        num_shards += 1
        by_team_size[str(sidecar["team_size"])] += 1

    manifest = {
        "total_ticks": total_ticks,
        "num_shards": num_shards,
        "by_team_size": dict(by_team_size),
        "schema_version": schema_version,
    }

    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest
