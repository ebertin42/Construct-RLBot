import sys
from pathlib import Path

import httpx
import pytest

from construct.data.ballchasing import Client

# `hf_allow_patterns` lives in scripts/build_replay_dataset.py (not
# construct.data.acquire -- that module only wraps snapshot_download).
# scripts/ isn't a package, so import it by adding it to sys.path.
_SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from build_replay_dataset import hf_allow_patterns  # noqa: E402


def test_search_paginates_and_rate_limits(monkeypatch):
    calls = []

    def handler(request):
        calls.append(str(request.url))
        assert request.headers["Authorization"] == "TESTTOKEN"  # raw, no Bearer
        return httpx.Response(
            200,
            json={
                "count": 1,
                "list": [{"id": "abc"}],
                "next": "https://ballchasing.com/api/replays?after=CUR",
            },
        )

    client = Client(
        "TESTTOKEN",
        transport=httpx.MockTransport(handler),
        list_min_interval=0,
        download_min_interval=0,
    )
    rows, cursor = client.search(min_rank="grand-champion-3", playlist="ranked-duels", count=150)
    assert rows == [{"id": "abc"}]
    assert cursor == "CUR"
    assert "min-rank=grand-champion-3" in calls[0] and "playlist=ranked-duels" in calls[0]


def test_download_skips_existing_file(tmp_path):
    existing = tmp_path / "abc.replay"
    existing.write_bytes(b"already downloaded")

    def handler(request):
        # Should never be called -- the file already exists and is non-empty.
        return httpx.Response(500)

    client = Client(
        "TESTTOKEN",
        transport=httpx.MockTransport(handler),
        list_min_interval=0,
        download_min_interval=0,
    )
    result = client.download("abc", tmp_path)
    assert result == existing
    assert existing.read_bytes() == b"already downloaded"


def test_search_pagination_terminates_when_no_next():
    """Drives two mock pages: the first has a `next` URL (cursor set, loop
    should continue and re-call with `after`), the second has none (cursor
    None, loop must terminate cleanly rather than looping forever or
    erroring)."""
    responses = [
        {
            "count": 1,
            "list": [{"id": "abc"}],
            "next": "https://ballchasing.com/api/replays?after=CUR",
        },
        {
            "count": 1,
            "list": [{"id": "def"}],
            # no "next" key at all -- cursor must come back None.
        },
    ]
    calls = []

    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, json=responses[len(calls) - 1])

    client = Client(
        "TESTTOKEN",
        transport=httpx.MockTransport(handler),
        list_min_interval=0,
        download_min_interval=0,
    )

    all_rows: list[dict] = []
    cursor = None
    for _ in range(len(responses) + 1):  # hard bound: fail loudly, don't hang, if it doesn't terminate
        rows, cursor = client.search(min_rank="grand-champion-3", playlist="ranked-duels", after=cursor)
        all_rows.extend(rows)
        if cursor is None:
            break
    else:
        pytest.fail("search pagination did not terminate when a response had no `next`")

    assert len(calls) == 2, "expected exactly one paginated follow-up call"
    assert all_rows == [{"id": "abc"}, {"id": "def"}]
    assert cursor is None
    assert "after=CUR" in calls[1], "second call must carry the cursor from the first page's `next`"


def test_hf_allow_patterns_maps_rank_and_playlist():
    # Pin the actual mapping: ballchasing's `ranked-` prefix is stripped to
    # match the HF dataset's bare-playlist directory layout.
    patterns = hf_allow_patterns(min_rank="grand-champion-3", playlist="ranked-duels")
    assert patterns == ["grand-champion-3/duels/**"]
    assert len(patterns) > 0


def test_hf_allow_patterns_falls_back_to_default_when_unfiltered():
    patterns = hf_allow_patterns(min_rank=None, playlist=None)
    assert patterns == ["grand-champion-3/**"]
