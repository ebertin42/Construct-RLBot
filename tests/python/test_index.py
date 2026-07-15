import json
from pathlib import Path

from construct.data.index import build_index


def test_index_aggregates_sidecars(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({"schema_version": 1, "num_ticks": 500, "team_size": 1}))
    (tmp_path / "b.json").write_text(json.dumps({"schema_version": 1, "num_ticks": 300, "team_size": 2}))
    idx = build_index(tmp_path)
    assert idx["total_ticks"] == 800
    assert idx["num_shards"] == 2
    assert idx["by_team_size"] == {"1": 1, "2": 1}
    assert idx["schema_version"] == 1


def test_index_writes_manifest_json(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({"schema_version": 1, "num_ticks": 500, "team_size": 1}))
    build_index(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    assert manifest_path.exists()
    written = json.loads(manifest_path.read_text())
    assert written["total_ticks"] == 500
    assert written["num_shards"] == 1


def test_index_empty_dir(tmp_path: Path):
    idx = build_index(tmp_path)
    assert idx["total_ticks"] == 0
    assert idx["num_shards"] == 0
    assert idx["by_team_size"] == {}
    assert idx["schema_version"] is None


def test_index_ignores_manifest_json_itself(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({"schema_version": 1, "num_ticks": 500, "team_size": 1}))
    build_index(tmp_path)
    # Re-running build_index must not double-count manifest.json as a sidecar.
    idx = build_index(tmp_path)
    assert idx["num_shards"] == 1
    assert idx["total_ticks"] == 500


def test_index_disagreeing_schema_versions_raises(tmp_path: Path):
    (tmp_path / "a.json").write_text(json.dumps({"schema_version": 1, "num_ticks": 500, "team_size": 1}))
    (tmp_path / "b.json").write_text(json.dumps({"schema_version": 2, "num_ticks": 300, "team_size": 2}))
    try:
        build_index(tmp_path)
        assert False, "expected ValueError on disagreeing schema_version"
    except ValueError:
        pass
