"""Replay dataset pipeline orchestration: download -> parse -> index.

Each stage is independently resumable, so re-running the same command line
picks up where a prior (interrupted, or intentionally incremental) run left
off:
  1. ACQUIRE replays into `--dest`.
     - `--source hf`: `construct.data.acquire.pull_hf_subset` wraps
       `huggingface_hub.snapshot_download`, which skips files already
       present on disk (resumable for free).
     - `--source ballchasing`: `construct.data.ballchasing.Client.download`
       skips any `<id>.replay` already present in `--dest`; re-running with
       the same filters re-paginates from the top but only re-downloads
       replays missing from disk.
  2. PARSE: shells out to the `replay-parse` release binary (Task 5/6),
     built separately via `cargo build -p construct-replay --release`. Must
     run with `cwd` = repo root so RocketSim's collision-mesh assets resolve
     (see `replay/src/bin/replay_parse.rs`'s module doc). To avoid
     re-parsing replays that already have a shard, this script only feeds
     the binary replays whose `<id>.json` sidecar is not yet present in
     `--shards` (via a throwaway directory of symlinks — `replay-parse`
     itself has no such skip logic).
  3. INDEX: `construct.data.index.build_index` aggregates every sidecar in
     `--shards` into `manifest.json` and prints a summary. Safe to re-run at
     any time; it rescans the directory from scratch each time.

Usage:
    python scripts/build_replay_dataset.py --source hf --min-rank grand-champion-3 --playlist ranked-duels
    BALLCHASING_TOKEN=... python scripts/build_replay_dataset.py --source ballchasing --min-rank grand-champion-3 --playlist ranked-duels --max 500
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
REPLAY_PARSE_BIN = REPO_ROOT / "target" / "release" / "replay-parse"
RESET_SAMPLES_PER_REPLAY = 16

DEFAULT_HF_ALLOW_PATTERNS = ["grand-champion-3/**"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["hf", "ballchasing"], required=True)
    ap.add_argument("--dest", default="data/replays", help="raw .replay download directory")
    ap.add_argument("--shards", default="data/shards", help="parsed shard output directory")
    ap.add_argument("--reset-pool", default="data/reset_pool.jsonl", help="ReplayMutator reset-state pool jsonl")
    ap.add_argument("--max", type=int, default=None, help="advisory cap on number of replays acquired")
    ap.add_argument("--min-team-size", type=int, default=1, help="skip replays with per-side team_size below this")
    ap.add_argument("--min-rank", default=None, help="ballchasing min-rank filter (also scopes the HF subset)")
    ap.add_argument("--playlist", default=None, help="ballchasing playlist filter (also scopes the HF subset)")
    return ap.parse_args(argv)


def hf_allow_patterns(min_rank: str | None, playlist: str | None) -> list[str]:
    """Best-effort mapping of `--min-rank`/`--playlist` onto the HF dataset's
    `<rank>/<playlist>/*.replay` layout (see `construct.data.acquire`'s
    module doc). `playlist` values follow ballchasing's `ranked-*` naming;
    the HF dataset uses the bare playlist name, so the `ranked-` prefix (if
    any) is stripped. Falls back to `DEFAULT_HF_ALLOW_PATTERNS` when neither
    filter is given.
    """
    if not min_rank and not playlist:
        return list(DEFAULT_HF_ALLOW_PATTERNS)
    rank = min_rank or "grand-champion-3"
    if playlist:
        bare_playlist = playlist.removeprefix("ranked-")
        return [f"{rank}/{bare_playlist}/**"]
    return [f"{rank}/**"]


def check_replay_parse_binary() -> None:
    if not REPLAY_PARSE_BIN.exists():
        print(
            f"error: {REPLAY_PARSE_BIN} not found -- run "
            "`cargo build -p construct-replay --release` first",
            file=sys.stderr,
        )
        sys.exit(1)


def acquire_hf(dest: Path, min_rank: str | None, playlist: str | None, max_files: int | None) -> int:
    from construct.data.acquire import pull_hf_subset

    patterns = hf_allow_patterns(min_rank, playlist)
    print(f"acquiring (hf): allow_patterns={patterns} -> {dest}")
    count = pull_hf_subset(dest, allow_patterns=patterns, max_files=max_files)
    print(f"acquired (hf): {count} replay file(s) present under {dest}")
    return count


def acquire_ballchasing(dest: Path, min_rank: str | None, playlist: str | None, max_n: int | None) -> int:
    from construct.data.ballchasing import Client

    token = os.environ.get("BALLCHASING_TOKEN")
    if not token:
        print(
            "error: BALLCHASING_TOKEN environment variable is not set. "
            "Get a token from https://ballchasing.com/upload (Steam login) "
            "and `export BALLCHASING_TOKEN=...` to use --source ballchasing.",
            file=sys.stderr,
        )
        sys.exit(1)

    dest.mkdir(parents=True, exist_ok=True)
    client = Client(token)

    downloaded = 0
    cursor: str | None = None
    while max_n is None or downloaded < max_n:
        page_count = min(200, max_n - downloaded) if max_n is not None else 200
        rows, cursor = client.search(min_rank=min_rank, playlist=playlist, count=page_count, after=cursor)
        if not rows:
            break
        for row in rows:
            if max_n is not None and downloaded >= max_n:
                break
            client.download(row["id"], dest)
            downloaded += 1
        if cursor is None:
            break

    print(f"acquired (ballchasing): {downloaded} replay(s) downloaded to {dest}")
    return downloaded


def unshredded_replays(dest: Path, shards: Path) -> list[Path]:
    """Replays in `dest` that don't already have a `<id>.json` sidecar in
    `shards` -- skipping these makes the PARSE stage resumable without
    needing skip logic inside `replay-parse` itself."""
    already_shredded = {p.stem for p in shards.glob("*.json")} if shards.exists() else set()
    return sorted(p for p in dest.glob("*.replay") if p.stem not in already_shredded)


def run_parse(dest: Path, shards: Path, reset_pool: Path, min_team_size: int) -> None:
    to_parse = unshredded_replays(dest, shards)
    if not to_parse:
        print("parse: nothing new to parse (all replays already have shards)")
        return

    shards.mkdir(parents=True, exist_ok=True)
    print(f"parse: {len(to_parse)} replay(s) to parse (skipping already-shard'd ones)")

    with tempfile.TemporaryDirectory(prefix="replay_parse_input_") as tmp:
        tmp_dir = Path(tmp)
        for replay_path in to_parse:
            os.symlink(replay_path.resolve(), tmp_dir / replay_path.name)

        subprocess.run(
            [
                str(REPLAY_PARSE_BIN),
                "--input-dir", str(tmp_dir),
                "--output-dir", str(shards),
                "--reset-pool-out", str(reset_pool),
                "--reset-samples-per-replay", str(RESET_SAMPLES_PER_REPLAY),
                "--min-team-size", str(min_team_size),
            ],
            cwd=REPO_ROOT,
            check=True,
        )


def run_index(shards: Path) -> dict:
    from construct.data.index import build_index

    manifest = build_index(shards)
    print(f"index: {json.dumps(manifest)}")
    return manifest


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    check_replay_parse_binary()

    dest = (REPO_ROOT / args.dest) if not Path(args.dest).is_absolute() else Path(args.dest)
    shards = (REPO_ROOT / args.shards) if not Path(args.shards).is_absolute() else Path(args.shards)
    reset_pool = (REPO_ROOT / args.reset_pool) if not Path(args.reset_pool).is_absolute() else Path(args.reset_pool)
    dest.mkdir(parents=True, exist_ok=True)

    if args.source == "hf":
        acquire_hf(dest, args.min_rank, args.playlist, args.max)
    else:
        acquire_ballchasing(dest, args.min_rank, args.playlist, args.max)

    run_parse(dest, shards, reset_pool, args.min_team_size)
    run_index(shards)


if __name__ == "__main__":
    main()
