import httpx

from construct.data.ballchasing import Client


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
