"""ballchasing.com API client — search + download replays.

Auth: header ``Authorization: <token>`` (the raw API token, NO "Bearer "
prefix — see https://ballchasing.com/doc/api).

Rate-limited (conservative free-tier defaults: <=2 list-calls/s, <=1
download/s) and resumable: `search` pages via the API's `after` cursor,
`download` skips replays already present on disk.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

BASE_URL = "https://ballchasing.com"


class Client:
    """Thin wrapper around the ballchasing.com REST API.

    `transport` is injectable for tests (e.g. `httpx.MockTransport`); leave
    it None in production to perform real network I/O.
    """

    def __init__(
        self,
        token: str,
        transport: httpx.BaseTransport | None = None,
        list_min_interval: float = 0.5,
        download_min_interval: float = 1.0,
    ):
        self._http = httpx.Client(
            base_url=BASE_URL,
            headers={"Authorization": token},
            transport=transport,
        )
        self.list_min_interval = list_min_interval
        self.download_min_interval = download_min_interval
        self._last_call: dict[str, float] = {}

    def _throttle(self, kind: str) -> None:
        """Sleep, if needed, to enforce a minimum interval between calls of `kind`."""
        min_interval = self.list_min_interval if kind == "list" else self.download_min_interval
        last = self._last_call.get(kind)
        if last is not None:
            wait = min_interval - (time.monotonic() - last)
            if wait > 0:
                time.sleep(wait)
        self._last_call[kind] = time.monotonic()

    def search(
        self,
        min_rank: str | None = None,
        playlist: str | None = None,
        count: int = 150,
        after: str | None = None,
        sort_by: str = "replay-date",
        **filters: Any,
    ) -> tuple[list[dict], str | None]:
        """GET /api/replays. Returns (rows, next_cursor); next_cursor is the
        `after` query param parsed out of the response's `next` URL, or None
        when there is no further page."""
        self._throttle("list")
        params: dict[str, Any] = {"count": count, "sort-by": sort_by}
        if min_rank is not None:
            params["min-rank"] = min_rank
        if playlist is not None:
            params["playlist"] = playlist
        if after is not None:
            params["after"] = after
        params.update(filters)

        resp = self._http.get("/api/replays", params=params)
        resp.raise_for_status()
        data = resp.json()

        cursor = None
        next_url = data.get("next")
        if next_url:
            qs = parse_qs(urlparse(next_url).query)
            cursor = qs.get("after", [None])[0]
        return data.get("list", []), cursor

    def download(self, replay_id: str, dest: Path) -> Path:
        """GET /api/replays/{id}/file, streamed to dest/{replay_id}.replay.
        Idempotent/resumable: if that file already exists and is non-empty,
        skip the request entirely and return the existing path."""
        dest = Path(dest)
        out_path = dest / f"{replay_id}.replay"
        if out_path.exists() and out_path.stat().st_size > 0:
            return out_path

        dest.mkdir(parents=True, exist_ok=True)
        self._throttle("download")
        with self._http.stream("GET", f"/api/replays/{replay_id}/file") as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_bytes():
                    f.write(chunk)
        return out_path
