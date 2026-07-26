import tomllib

import numpy as np
import pytest

from construct._engine import Engine, action_table_v1, action_table_v1_air, schema_dict

def test_rust_and_python_read_same_schema():
    rust = schema_dict("schema/v0.toml")
    with open("schema/v0.toml", "rb") as f:
        py = tomllib.load(f)
    assert rust["obs_size"] == py["obs_size"] == 94
    assert rust["action_count"] == py["action_count"] == 90
    assert rust["tick_skip"] == py["tick_skip"] == 8
    assert abs(rust["pos_norm"] - py["normalization"]["pos_norm"]) < 1e-15


def test_v1_air_schema_is_v1_obs_with_a_104_row_table():
    """v1-air keeps version 1 and the v1 obs contract byte-for-byte -- only the
    action table (the policy's OUTPUT dimension) differs. A version bump would
    instead fall through Schema::validate's `version != 1` early-return into
    the v0 branch, so the table has to be a second, independent axis."""
    v1, air = schema_dict("schema/v1.toml"), schema_dict("schema/v1_air.toml")
    assert air["version"] == v1["version"] == 1
    assert v1["action_count"] == 92 and air["action_count"] == 104
    assert air["action_table"] == "construct_104_v1air"
    for k in ("obs_size", "tick_skip", "pos_norm", "vel_norm", "ang_vel_norm"):
        assert air[k] == v1[k], f"{k} must be identical: the obs contract is unchanged"
    # ...and the entity layout itself, which schema_dict does not surface.
    with open("schema/v1.toml", "rb") as f:
        v1_raw = tomllib.load(f)
    with open("schema/v1_air.toml", "rb") as f:
        air_raw = tomllib.load(f)
    assert air_raw["obs_v1"] == v1_raw["obs_v1"], "the entity obs layout must be unchanged"


def test_engine_accepts_both_v1_action_tables():
    """Both widths must load in the SAME engine binary: a 92-row champion is
    the ruler a 104-row v9 run gets gated against, so narrowing the guard to
    one width would break bench_foreign.py / version_ladder.py / MatchRunner."""
    for path, count in (("schema/v1.toml", 92), ("schema/v1_air.toml", 104)):
        eng = Engine(num_arenas=1, blue=1, orange=1, schema_path=path,
                     reward_config_path="configs/reward_v0.toml",
                     seed=3, num_threads=1, net_heads=2)
        assert eng.action_count == count, path
        assert eng.obs_mode == "v1"


def test_engine_rejects_a_checkpoint_whose_action_table_is_the_other_v1_table():
    """The cross-table hazard, caught on the calling thread.

    `schema_version` is 1 for BOTH v1 tables, so a 104-row v1-air checkpoint
    loaded against schema/v1.toml passes every version check -- and would then
    sample action index 92..104 into a 92-row decode table, an out-of-bounds
    panic deep inside a worker thread minutes later with no hint of the cause.
    League/bench code keys its schema path on the version
    (`matches._SCHEMA_PATHS`), so this is reachable, not theoretical: ONE
    Engine binds ONE table and cross-table evaluation needs one engine per
    side.
    """
    from construct.learn.model_v1 import EntityPolicyNet

    def sd(table):
        net = EntityPolicyNet(d_model=32, layers=1, heads=2, ff=64, action_table=table)
        return {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}

    # 104-row net into a 92-row engine, and the reverse. Both must fail LOUDLY.
    for schema_path, table, wrong in (
        ("schema/v1.toml", action_table_v1_air(), 104),
        ("schema/v1_air.toml", action_table_v1(), 92),
    ):
        eng = Engine(num_arenas=1, blue=1, orange=1, schema_path=schema_path,
                     reward_config_path="configs/reward_v0.toml",
                     seed=1, num_threads=1, net_heads=2)
        with pytest.raises(Exception, match="action-table mismatch"):
            eng.set_weights(sd(table))
        # ...and the MATCHING table is accepted by the same engine.
        right = action_table_v1() if wrong == 104 else action_table_v1_air()
        eng.set_weights(sd(right))


def test_tables_maps_every_schema_file_to_the_table_it_selects():
    """construct.tables is the single mapping bench_foreign / eval_metrics /
    watch / MatchRunner all derive their schema path from. If it drifts from
    the schema files themselves, those tools build the wrong engine -- so pin
    it against what the engine actually reads out of each file."""
    from construct.tables import V1_TABLE_ROWS, V1_TABLE_SCHEMA

    for name, path in V1_TABLE_SCHEMA.items():
        sch = schema_dict(path)
        assert sch["action_table"] == name, path
        assert sch["action_count"] == V1_TABLE_ROWS[name], path


def test_table_name_prefers_the_recorded_field_and_falls_back_to_the_buffer():
    from construct.tables import schema_path, table_name

    air = {"schema_version": 1, "action_table": "construct_104_v1air"}
    assert table_name(air) == "construct_104_v1air"
    assert schema_path(air) == "schema/v1_air.toml"

    # No field at all: every checkpoint written before 2026-07-26 is like this,
    # and the registered buffer is the only evidence of its width.
    class _T:
        shape = (104, 8)

    inferred = {"schema_version": 1, "model": {"action_table": _T()}}
    assert table_name(inferred) == "construct_104_v1air"
    assert table_name({"schema_version": 1}) == "construct_92_v1"
    # v0 never recorded a table name; None is not "the default", it is "n/a".
    assert table_name({"schema_version": 0}) is None
    assert schema_path({"schema_version": 0}) == "schema/v0.toml"


def test_table_for_state_dict_reads_the_width_off_the_weights():
    """The seam the measurement tools use: they already hold the state dict
    (load_sd), and the buffer in it is the same evidence table_name falls back
    to -- so no tool needs to load the same file twice to know which engine to
    build."""
    from construct.tables import table_for_state_dict

    for table, expect in ((action_table_v1(), "construct_92_v1"),
                          (action_table_v1_air(), "construct_104_v1air")):
        sd = {"action_table": np.asarray(table, dtype=np.float32)}
        assert table_for_state_dict(sd) == expect
    assert table_for_state_dict({}) is None  # v0 state dict: no such buffer
    with pytest.raises(ValueError, match="known widths"):
        table_for_state_dict({"action_table": np.zeros((7, 8), dtype=np.float32)})


def test_table_name_refuses_a_name_that_disagrees_with_the_weights():
    """The one combination that would fly a different bot than the one that was
    trained: a 92-row name on 104-row weights decodes 12 rows into the wrong
    controls (or out of bounds)."""
    from construct.tables import table_name

    class _T:
        shape = (104, 8)

    with pytest.raises(ValueError, match="disagree"):
        table_name({"schema_version": 1, "action_table": "construct_92_v1",
                    "model": {"action_table": _T()}})


def test_engine_rejects_v1_schema_with_an_unknown_action_count(tmp_path):
    bad = tmp_path / "bad_v1.toml"
    bad.write_text(
        """
version = 1
obs_size = 0
action_table = "construct_100_nope"
action_count = 100
tick_skip = 8

[obs_v1]
max_ent = 17
ent_feat = 26
q_feat = 64
prev_actions = 5

[normalization]
pos_norm = 0.00043478260869565216
vel_norm = 0.00043478260869565216
ang_vel_norm = 0.18181818181818182
"""
    )
    with pytest.raises(ValueError, match="92"):
        Engine(num_arenas=1, blue=1, orange=1, schema_path=str(bad),
               reward_config_path="configs/reward_v0.toml")


def test_engine_rejects_schema_that_disagrees_with_compiled_constants(tmp_path):
    bad_schema = tmp_path / "bad.toml"
    bad_schema.write_text(
        """
version = 0
obs_size = 120
action_table = "rlgym_lookup_90"
action_count = 90
tick_skip = 8

[normalization]
pos_norm = 0.00043478260869565216
vel_norm = 0.00043478260869565216
ang_vel_norm = 0.18181818181818182
"""
    )
    with pytest.raises(ValueError):
        Engine(num_arenas=1, blue=1, orange=1, schema_path=str(bad_schema),
               reward_config_path="configs/reward_v0.toml")
