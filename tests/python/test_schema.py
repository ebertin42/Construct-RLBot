import tomllib

import pytest

from construct._engine import Engine, schema_dict

def test_rust_and_python_read_same_schema():
    rust = schema_dict("schema/v0.toml")
    with open("schema/v0.toml", "rb") as f:
        py = tomllib.load(f)
    assert rust["obs_size"] == py["obs_size"] == 94
    assert rust["action_count"] == py["action_count"] == 90
    assert rust["tick_skip"] == py["tick_skip"] == 8
    assert abs(rust["pos_norm"] - py["normalization"]["pos_norm"]) < 1e-15


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
