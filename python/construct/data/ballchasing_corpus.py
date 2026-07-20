"""Bulk rank-filtered replay-corpus pull from ballchasing.com.

Built for the SSL-finetune corpus (spec §5.4): every ranked-duels replay with
min-rank supersonic-legend, landed as raw ``.replay`` files under
``data/replays/ssl/duels/batch_NNNN/`` (10k per batch dir, mirroring the HF
``grand-champion-2/duels`` layout). Winners-side filtering happens later at
parse time — acquisition just pulls SSL duels. The HF mirror tops out at GC3,
so SSL is ballchasing's job.

Resumable at two independent levels:
  - replay ids: ``<id>.replay`` filenames are the dedupe key. Every batch dir
    is scanned at startup; ids already on disk are never re-downloaded.
  - pagination: pure ``created-before`` cursor. Pages are listed with
    ``sort-by=created&sort-dir=desc`` (verified live: sort-by must be
    ``replay-date`` or ``created``); after a page is fully processed, the
    cursor advances to the last row's ``created`` timestamp and is persisted
    to ``pull_state.json``. A restart re-issues the same query and continues
    from the frontier instead of re-paginating from the top. ``created``
    (upload time, always RFC3339 with timezone) is used rather than the
    replay ``date`` because the latter has no reliable timezone, so it cannot
    round-trip exactly through the *-before filter.

Rate limits (regular tier; breach = HTTP 429): list 2/s + 500/h, download
1/s + **200/h** (~4.8k/day — downloads are the scarce resource). The pacer
allows a short initial burst (fast start verification) then settles to a
steady interval under the hourly cap, with a sliding-window hard stop as
backstop; 429s honor Retry-After.

Disk guard: WSL ``df`` lies about the host drive, so free space is read via
``powershell.exe (Get-PSDrive C)`` at startup and every 1000 downloads; the
pull stops (exit code 3, wrapper does NOT restart) when host free < 130 GB.

Usage (normally via scripts/pull_ssl_duels.sh):
    BALLCHASING_TOKEN=... python -m construct.data.ballchasing_corpus \
        --dest data/replays/ssl/duels
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import httpx

from construct.data.ballchasing import Client, load_token

BATCH_SIZE = 10_000
PAGE_SIZE = 200  # API max per list call
PROGRESS_EVERY = 1_000
DISK_CHECK_EVERY = 1_000
MIN_FREE_GB = 130.0
COUNT_CAP = 10_000  # the API's `count` field saturates here (observed live)

DOWNLOADS_PER_HOUR = 190  # cap is 200/h; keep headroom for retries
DOWNLOAD_MIN_GAP_S = 1.2  # cap is 1/s
BURST = 20  # first N downloads at min-gap for fast start verification

STATE_FILE = "pull_state.json"
FAILED_FILE = "failed_ids.txt"


def log(msg: str) -> None:
    print(f"{datetime.now():%m-%d %H:%M:%S} {msg}", flush=True)


def host_free_gb() -> float | None:
    """Free space on the Windows host C: drive in GB, via powershell.
    WSL `df` reports the virtual disk, not the host drive — never use it
    here. Returns None if the check itself fails."""
    try:
        out = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", "[math]::Round((Get-PSDrive C).Free/1GB,1)"],
            cwd="/mnt/c",
            capture_output=True,
            text=True,
            timeout=120,
        )
        return float(out.stdout.strip())
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


class HourlyPacer:
    """Paces downloads under the regular tier's 200/h + 1/s caps.

    The first `burst` calls go at `min_gap` seconds (so a fresh start lands
    verifiable files quickly), later calls at 3600/per_hour seconds. A
    sliding-window hard stop (`per_hour` events per rolling hour) soaks up
    the burst debt so the steady state never exceeds the hourly cap."""

    def __init__(
        self,
        per_hour: int = DOWNLOADS_PER_HOUR,
        min_gap: float = DOWNLOAD_MIN_GAP_S,
        burst: int = BURST,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ):
        self.per_hour = per_hour
        self.min_gap = min_gap
        self.burst = burst
        self._sleep = sleep
        self._now = now
        self._stamps: deque[float] = deque()
        self._n = 0

    def wait(self) -> None:
        gap = self.min_gap if self._n < self.burst else 3600.0 / self.per_hour
        if self._stamps:
            pause = self._stamps[-1] + gap - self._now()
            if pause > 0:
                self._sleep(pause)
        while self._stamps and self._now() - self._stamps[0] > 3600.0:
            self._stamps.popleft()
        while len(self._stamps) >= self.per_hour:
            pause = 3600.0 - (self._now() - self._stamps[0])
            if pause > 0:
                self._sleep(pause)
            self._stamps.popleft()
        self._stamps.append(self._now())
        self._n += 1


def _load_state(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_state(path: Path, state: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(path)


def _scan_batches(dest: Path, batch_size: int) -> tuple[set[str], Path, int]:
    """Scan every batch_NNNN dir: return (ids on disk, batch dir to fill,
    file count in that dir). Zero-byte files and .part leftovers don't count
    as present (they get re-downloaded); .part leftovers are removed."""
    have: set[str] = set()
    batches = sorted(d for d in dest.glob("batch_[0-9][0-9][0-9][0-9]") if d.is_dir())
    for d in batches:
        for p in d.glob("*.part"):
            p.unlink(missing_ok=True)
        have.update(f.stem for f in d.glob("*.replay") if f.stat().st_size > 0)
    if batches:
        cur = batches[-1]
        cur_n = sum(1 for _ in cur.glob("*.replay"))
        if cur_n >= batch_size:
            cur = dest / f"batch_{int(cur.name.split('_')[1]) + 1:04d}"
            cur_n = 0
    else:
        cur = dest / "batch_0000"
        cur_n = 0
    cur.mkdir(parents=True, exist_ok=True)
    return have, cur, cur_n


def _retry_delay(resp: httpx.Response, attempt: int, base: float) -> float:
    ra = resp.headers.get("Retry-After")
    if ra is not None:
        try:
            return max(1.0, float(ra))
        except ValueError:
            pass
    return base * (attempt + 1)


def _search_with_retry(
    client: Client,
    min_rank: str | None,
    playlist: str | None,
    created_before: str | None,
    sleep: Callable[[float], None],
    tries: int = 6,
) -> tuple[list[dict] | None, int | None]:
    filters: dict = {"sort-dir": "desc"}
    if created_before:
        filters["created-before"] = created_before
    for attempt in range(tries):
        try:
            rows, _, count = client.search_page(
                min_rank=min_rank, playlist=playlist, count=PAGE_SIZE, sort_by="created", **filters
            )
            return rows, count
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 429:
                delay = _retry_delay(e.response, attempt, 30.0)
                log(f"429 on search — backing off {delay:.0f}s")
                sleep(delay)
            elif code >= 500:
                sleep(15.0 * (attempt + 1))
            else:
                log(f"search failed: HTTP {code}: {e.response.text[:200]}")
                return None, None
        except httpx.TransportError as e:
            log(f"search transport error: {type(e).__name__} — retrying")
            sleep(20.0 * (attempt + 1))
    log("search failed: retries exhausted")
    return None, None


def _download_with_retry(
    client: Client,
    replay_id: str,
    dest: Path,
    sleep: Callable[[float], None],
    tries: int = 5,
) -> bool:
    for attempt in range(tries):
        try:
            client.download(replay_id, dest)
            return True
        except httpx.HTTPStatusError as e:
            code = e.response.status_code
            if code == 429:
                delay = _retry_delay(e.response, attempt, 60.0)
                log(f"429 downloading {replay_id} — backing off {delay:.0f}s")
                sleep(delay)
            elif code >= 500:
                sleep(10.0 * (attempt + 1))
            else:
                log(f"skip {replay_id}: HTTP {code}")
                return False
        except httpx.TransportError as e:
            log(f"transport error downloading {replay_id}: {type(e).__name__} — retrying")
            sleep(15.0 * (attempt + 1))
    log(f"skip {replay_id}: retries exhausted")
    return False


def pull(
    dest: Path,
    client: Client,
    *,
    min_rank: str = "supersonic-legend",
    playlist: str = "ranked-duels",
    batch_size: int = BATCH_SIZE,
    max_downloads: int | None = None,
    min_free_gb: float = MIN_FREE_GB,
    free_gb_fn: Callable[[], float | None] = host_free_gb,
    disk_check_every: int = DISK_CHECK_EVERY,
    pacer: HourlyPacer | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """Run the pull loop until the frontier is exhausted, `max_downloads` is
    reached, or the disk guard trips. Returns one of: "exhausted",
    "max-reached", "disk-guard", "search-failed"."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    state_path = dest / STATE_FILE
    pacer = pacer or HourlyPacer()

    have, batch_dir, in_batch = _scan_batches(dest, batch_size)
    state = _load_state(state_path)
    created_before: str | None = state.get("created_before")
    total_prev = int(state.get("downloads_total", 0))
    log(
        f"start: {len(have)} replay(s) on disk, filling {batch_dir.name} ({in_batch}/{batch_size}), "
        f"cursor created-before={created_before or '<none: from newest>'}"
    )

    last_free = free_gb_fn()
    if last_free is not None:
        log(f"disk guard: host C: free {last_free:.1f}G (min {min_free_gb:.0f}G)")
        if last_free < min_free_gb:
            log(f"DISK GUARD: host C: free {last_free:.1f}G < {min_free_gb:.0f}G — stopping before any download")
            return "disk-guard"
    else:
        log("warn: host free-space check failed at start — continuing, will re-check")

    downloaded = skipped = failed = 0
    t0 = time.monotonic()
    result = "exhausted"

    while True:
        rows, count = _search_with_retry(client, min_rank, playlist, created_before, sleep)
        if rows is None:
            result = "search-failed"
            break
        if not rows:
            log("no rows returned — corpus frontier exhausted")
            result = "exhausted"
            break
        capped = "+" if (count or 0) >= COUNT_CAP else ""
        log(f"page: {len(rows)} rows, {count}{capped} matching beyond cursor, oldest created {rows[-1].get('created')}")

        stop: str | None = None
        for row in rows:
            rid = row["id"]
            if rid in have:
                skipped += 1
                continue
            if max_downloads is not None and downloaded >= max_downloads:
                stop = "max-reached"
                break
            pacer.wait()
            if not _download_with_retry(client, rid, batch_dir, sleep):
                failed += 1
                with open(dest / FAILED_FILE, "a") as f:
                    f.write(rid + "\n")
                continue
            have.add(rid)
            downloaded += 1
            in_batch += 1
            if downloaded <= 20:
                log(f"landed {rid} -> {batch_dir.name} ({downloaded} this run)")
            if in_batch >= batch_size:
                batch_dir = dest / f"batch_{int(batch_dir.name.split('_')[1]) + 1:04d}"
                batch_dir.mkdir(parents=True, exist_ok=True)
                in_batch = 0
                log(f"batch full — rolling to {batch_dir.name}")
            if downloaded % disk_check_every == 0:
                free = free_gb_fn()
                if free is not None:
                    last_free = free
                    if free < min_free_gb:
                        log(f"DISK GUARD: host C: free {free:.1f}G < {min_free_gb:.0f}G — stopping")
                        stop = "disk-guard"
                        break
                elif last_free is None or last_free < min_free_gb + 10.0:
                    log("DISK GUARD: free-space check failed with no safe margin known — stopping")
                    stop = "disk-guard"
                    break
                else:
                    log("warn: free-space check failed — continuing on last known margin")
            if downloaded % PROGRESS_EVERY == 0:
                rate = downloaded / max(time.monotonic() - t0, 1e-9) * 3600.0
                free_s = f"{last_free:.1f}G" if last_free is not None else "unknown"
                eta = ""
                if count is not None and count < COUNT_CAP and rate > 0:
                    eta = f", ~{count / rate / 24.0:.1f}d to frontier end"
                log(
                    f"progress: {downloaded} this run / {total_prev + downloaded} total "
                    f"({skipped} deduped, {failed} failed) — {rate:.0f}/h, host free {free_s}{eta}"
                )

        if stop is not None:
            # Mid-page stop: do NOT advance the cursor — the next run re-lists
            # this window and dedupes by id, so nothing is lost.
            state.update(
                downloads_total=total_prev + downloaded,
                updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            )
            _save_state(state_path, state)
            result = stop
            break

        oldest = rows[-1].get("created") or rows[-1].get("date")
        if not oldest or oldest == created_before:
            log(f"cursor did not advance past {created_before} — stopping (frontier exhausted or stalled)")
            result = "exhausted"
            break
        created_before = oldest
        state.update(
            created_before=created_before,
            downloads_total=total_prev + downloaded,
            last_match_count=count,
            updated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        _save_state(state_path, state)

    rate = downloaded / max(time.monotonic() - t0, 1e-9) * 3600.0
    log(
        f"done ({result}): {downloaded} downloaded this run ({rate:.0f}/h), "
        f"{total_prev + downloaded} total, {skipped} deduped, {failed} failed, {len(have)} on disk"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="data/replays/ssl/duels", help="corpus root (batch_NNNN dirs live here)")
    ap.add_argument("--min-rank", default="supersonic-legend")
    ap.add_argument("--playlist", default="ranked-duels")
    ap.add_argument("--max", type=int, default=None, help="stop after this many downloads (this run)")
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    args = ap.parse_args(argv)

    token = load_token()
    if not token:
        print(
            "error: BALLCHASING_TOKEN not set and ~/.config/construct/ballchasing.env not found",
            file=sys.stderr,
        )
        return 2

    result = pull(
        Path(args.dest),
        Client(token),
        min_rank=args.min_rank,
        playlist=args.playlist,
        max_downloads=args.max,
        min_free_gb=args.min_free_gb,
    )
    # 0 = natural finish (wrapper stops), 3 = disk guard (wrapper stops),
    # 4 = search retries exhausted (wrapper restarts after a pause).
    return {"disk-guard": 3, "search-failed": 4}.get(result, 0)


if __name__ == "__main__":
    sys.exit(main())
