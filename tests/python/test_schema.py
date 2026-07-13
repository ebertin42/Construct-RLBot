import tomllib
from construct._engine import schema_dict

def test_rust_and_python_read_same_schema():
    rust = schema_dict("schema/v0.toml")
    with open("schema/v0.toml", "rb") as f:
        py = tomllib.load(f)
    assert rust["obs_size"] == py["obs_size"] == 94
    assert rust["action_count"] == py["action_count"] == 90
    assert rust["tick_skip"] == py["tick_skip"] == 8
    assert abs(rust["pos_norm"] - py["normalization"]["pos_norm"]) < 1e-15
