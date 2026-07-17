"""Cursor/resume logic for the ballchasing SSL corpus pull — all HTTP is
mocked via httpx.MockTransport (no live calls), sleeps are stubbed."""
import json
from urllib.parse import parse_qs, urlparse

import httpx

from construct.data import ballchasing_corpus as bcc
from construct.data.ballchasing import Client


class NoPacer:
    def wait(self):
        pass


def _client(handler):
    return Client(
        "TESTTOKEN",
        transport=httpx.MockTransport(handler),
        list_min_interval=0,
        download_min_interval=0,
    )


def _row(rid, created):
    return {"id": rid, "created": created}


def _pull(dest, handler, **kw):
    kw.setdefault("pacer", NoPacer())
    kw.setdefault("free_gb_fn", lambda: 999.0)
    kw.setdefault("sleep", lambda s: None)
    return bcc.pull(dest, _client(handler), **kw)


def _pages_handler(pages, downloads):
    """`pages`: dict created-before-value -> list of rows (None key = first
    query with no cursor). Records download ids into `downloads`."""

    def handler(request):
        q = parse_qs(urlparse(str(request.url)).query)
        if request.url.path == "/api/replays":
            assert q["min-rank"] == ["supersonic-legend"]
            assert q["playlist"] == ["ranked-duels"]
            assert q["sort-by"] == ["created"]
            assert q["sort-dir"] == ["desc"]
            cursor = q.get("created-before", [None])[0]
            rows = pages.get(cursor, [])
            return httpx.Response(200, json={"count": len(rows), "list": rows})
        rid = request.url.path.split("/")[-2]  # /api/replays/{id}/file
        downloads.append(rid)
        return httpx.Response(200, content=b"replay-bytes-" + rid.encode())

    return handler


def test_paginates_by_created_before_and_persists_cursor(tmp_path):
    dest = tmp_path / "duels"
    downloads = []
    pages = {
        None: [_row("aaa", "2026-07-10T10:00:00+00:00"), _row("bbb", "2026-07-10T09:00:00+00:00")],
        "2026-07-10T09:00:00+00:00": [_row("ccc", "2026-07-09T08:00:00+00:00")],
        "2026-07-09T08:00:00+00:00": [],  # frontier exhausted
    }
    result = _pull(dest, _pages_handler(pages, downloads))

    assert result == "exhausted"
    assert downloads == ["aaa", "bbb", "ccc"]
    for rid in downloads:
        assert (dest / "batch_0000" / f"{rid}.replay").read_bytes().startswith(b"replay-bytes-")
    state = json.loads((dest / "pull_state.json").read_text())
    assert state["created_before"] == "2026-07-09T08:00:00+00:00"
    assert state["downloads_total"] == 3


def test_restart_resumes_from_cursor_and_dedupes_on_disk_ids(tmp_path):
    dest = tmp_path / "duels"
    # Simulate a prior run: cursor persisted, one replay already on disk.
    (dest / "batch_0000").mkdir(parents=True)
    (dest / "batch_0000" / "ddd.replay").write_bytes(b"already here")
    (dest / "pull_state.json").write_text(
        json.dumps({"created_before": "2026-07-09T08:00:00+00:00", "downloads_total": 7})
    )

    downloads = []
    first_cursors = []

    def handler(request):
        q = parse_qs(urlparse(str(request.url)).query)
        if request.url.path == "/api/replays":
            cursor = q.get("created-before", [None])[0]
            first_cursors.append(cursor)
            pages = {
                "2026-07-09T08:00:00+00:00": [
                    _row("ddd", "2026-07-08T12:00:00+00:00"),
                    _row("eee", "2026-07-08T11:00:00+00:00"),
                ],
                "2026-07-08T11:00:00+00:00": [],
            }
            rows = pages.get(cursor, [])
            return httpx.Response(200, json={"count": len(rows), "list": rows})
        rid = request.url.path.split("/")[-2]
        downloads.append(rid)
        return httpx.Response(200, content=b"new-bytes")

    result = _pull(dest, handler)

    assert result == "exhausted"
    assert first_cursors[0] == "2026-07-09T08:00:00+00:00"  # resumed, not restarted
    assert downloads == ["eee"]  # ddd deduped by on-disk id
    assert (dest / "batch_0000" / "ddd.replay").read_bytes() == b"already here"
    state = json.loads((dest / "pull_state.json").read_text())
    assert state["created_before"] == "2026-07-08T11:00:00+00:00"
    assert state["downloads_total"] == 8  # 7 prior + 1 new


def test_batch_rollover_at_batch_size(tmp_path):
    dest = tmp_path / "duels"
    downloads = []
    pages = {
        None: [
            _row("aaa", "2026-07-10T10:00:00+00:00"),
            _row("bbb", "2026-07-10T09:00:00+00:00"),
            _row("ccc", "2026-07-10T08:00:00+00:00"),
        ],
        "2026-07-10T08:00:00+00:00": [],
    }
    result = _pull(dest, _pages_handler(pages, downloads), batch_size=2)

    assert result == "exhausted"
    assert sorted(f.stem for f in (dest / "batch_0000").glob("*.replay")) == ["aaa", "bbb"]
    assert sorted(f.stem for f in (dest / "batch_0001").glob("*.replay")) == ["ccc"]
    # A restart scans all batches and keeps filling the newest one.
    have, cur, cur_n = bcc._scan_batches(dest, 2)
    assert have == {"aaa", "bbb", "ccc"}
    assert cur.name == "batch_0001" and cur_n == 1


def test_disk_guard_stops_before_any_request(tmp_path):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json={"count": 0, "list": []})

    result = _pull(tmp_path / "duels", handler, free_gb_fn=lambda: 100.0)
    assert result == "disk-guard"
    assert calls == []


def test_disk_guard_stops_mid_run_without_advancing_cursor(tmp_path):
    dest = tmp_path / "duels"
    downloads = []
    pages = {
        None: [_row("aaa", "2026-07-10T10:00:00+00:00"), _row("bbb", "2026-07-10T09:00:00+00:00")],
    }
    frees = iter([999.0, 100.0, 100.0])  # start check ok, first per-download check trips
    result = _pull(
        dest,
        _pages_handler(pages, downloads),
        free_gb_fn=lambda: next(frees),
        disk_check_every=1,
    )
    assert result == "disk-guard"
    assert downloads == ["aaa"]  # stopped before bbb
    state = json.loads((dest / "pull_state.json").read_text())
    assert "created_before" not in state  # mid-page stop must not advance the cursor


def test_429_download_honors_retry_after_then_succeeds(tmp_path):
    dest = tmp_path / "duels"
    sleeps = []
    attempts = {"n": 0}

    def handler(request):
        if request.url.path == "/api/replays":
            q = parse_qs(urlparse(str(request.url)).query)
            rows = [] if q.get("created-before") else [_row("aaa", "2026-07-10T10:00:00+00:00")]
            return httpx.Response(200, json={"count": len(rows), "list": rows})
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "7"})
        return httpx.Response(200, content=b"ok-after-retry")

    result = _pull(dest, handler, sleep=sleeps.append)
    assert result == "exhausted"
    assert attempts["n"] == 2
    assert 7.0 in sleeps
    assert (dest / "batch_0000" / "aaa.replay").read_bytes() == b"ok-after-retry"


def test_hourly_pacer_burst_then_steady():
    clock = {"t": 0.0}

    def sleep(s):
        clock["t"] += s

    pacer = bcc.HourlyPacer(per_hour=60, min_gap=1.0, burst=2, sleep=sleep, now=lambda: clock["t"])
    slept = []
    for _ in range(4):
        before = clock["t"]
        pacer.wait()
        slept.append(clock["t"] - before)
    # burst of 2 at min_gap, then 3600/60 = 60s steady spacing
    assert slept == [0.0, 1.0, 60.0, 60.0]
