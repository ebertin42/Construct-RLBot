"""Which ACTION TABLE a checkpoint decodes with, and therefore which schema
file an engine must be built from to run it.

WHY THIS MODULE EXISTS. `schema_version` is 1 for BOTH v1 action tables -- the
frozen 92-row `construct_92_v1` and the 104-row `construct_104_v1air` that
configs/train_v9_fromscratch.toml launches on. The two share an obs contract
byte-for-byte and differ ONLY in the policy's output dimension, so the version
number cannot pick a schema file. Every tool that hardcoded "schema/v1.toml"
was therefore silently a 92-row-only tool, and on a v1-air checkpoint they all
died: bench_foreign.py (the absolute ruler -- `champion-unbeaten-external-lever`)
and MatchRunner on the engine's cross-table guard, eval_metrics.py and watch.py
on a state_dict size mismatch. Loud failures, but a dead instrument is dead:
v9's entire justification is a hypothesis that only measurement can test.

READ THE NAME WITH `table_name()`, NEVER by indexing `ck["action_table"]`.
Checkpoints written before 2026-07-26 have no such key and are all the 92-row
table. The registered `action_table` BUFFER is always present in `ck["model"]`
(EntityPolicyNet.register_buffer, so it travels with the state dict) and its
row count is an independent second source -- the two are cross-checked here,
because a checkpoint whose NAME and WIDTH disagree is exactly the "silently
mis-decoded policy" every guard in this repo exists to prevent.

No torch/engine import: registry-only and config-only callers (league/tick.py,
sampling filters) must stay light.
"""

# The 90-row v0 table has no name recorded anywhere -- v0 checkpoints predate
# the field entirely -- so v0 is represented as `None` rather than a name.
V0_SCHEMA_PATH = "schema/v0.toml"

# Default for anything written before 2026-07-26. Registry entries and
# checkpoints from every prior lineage (v8, the champion, every league arm) are
# this table.
DEFAULT_V1_TABLE = "construct_92_v1"

# name -> schema file that selects it. THE ONLY MAPPING; add a table here and
# every consumer (MatchRunner, bench_foreign, eval_metrics, watch, h2h_eval)
# picks it up.
V1_TABLE_SCHEMA = {
    "construct_92_v1": "schema/v1.toml",
    "construct_104_v1air": "schema/v1_air.toml",
}

# name -> row count, i.e. the policy's output dimension.
V1_TABLE_ROWS = {"construct_92_v1": 92, "construct_104_v1air": 104}

_BY_ROWS = {rows: name for name, rows in V1_TABLE_ROWS.items()}


def schema_path_for_table(name):
    """Schema file for a v1 action-table name (or v0's, for `None`)."""
    if name is None:
        return V0_SCHEMA_PATH
    if name not in V1_TABLE_SCHEMA:
        raise ValueError(
            f"unknown action table {name!r}; known: {sorted(V1_TABLE_SCHEMA)}"
        )
    return V1_TABLE_SCHEMA[name]


def table_name(ck):
    """The action-table name a loaded checkpoint dict decodes with.

    `None` for schema_version 0 (the 90-row flat-obs table, which no schema
    names). For v1: the recorded `action_table` field when present (written by
    Trainer.save_checkpoint since v9), else inferred from the stored buffer's
    row count, else the 92-row default.

    Raises ValueError when the recorded name and the stored buffer disagree --
    the one combination that would fly a different bot than the one that was
    trained.
    """
    if int(ck.get("schema_version", 0)) != 1:
        return None
    rows = None
    buf = ck.get("model", {}).get("action_table")
    if buf is not None:
        rows = int(buf.shape[0])
    name = ck.get("action_table")
    if name is None:
        # Pre-v9 checkpoint: the field did not exist yet. The buffer is the only
        # evidence, and for every checkpoint that predates the field it says 92.
        name = _BY_ROWS.get(rows, DEFAULT_V1_TABLE)
    name = str(name)
    if name not in V1_TABLE_ROWS:
        raise ValueError(
            f"unknown action table {name!r} on checkpoint; known: "
            f"{sorted(V1_TABLE_ROWS)}"
        )
    if rows is not None and rows != V1_TABLE_ROWS[name]:
        raise ValueError(
            f"checkpoint records action_table={name!r} ({V1_TABLE_ROWS[name]} rows) "
            f"but its stored action_table buffer has {rows} rows -- the name and "
            f"the weights disagree about the policy's output dimension"
        )
    return name


def schema_path(ck):
    """Schema file an engine must be built from to run this checkpoint."""
    return schema_path_for_table(table_name(ck))


def table_for_state_dict(sd):
    """The action-table name implied by a loaded STATE DICT's own buffer, or
    None when there is none (a v0 state dict).

    For callers that already hold the weights (`league.matches.load_sd`): the
    buffer is the same evidence `table_name` would fall back to, and reading it
    here avoids a second load of the same file -- which also keeps the tools
    testable, since their tests stub load_sd rather than putting a checkpoint
    on disk.
    """
    buf = sd.get("action_table") if hasattr(sd, "get") else None
    if buf is None:
        return None
    rows = int(buf.shape[0])
    if rows not in _BY_ROWS:
        raise ValueError(
            f"state dict carries a {rows}-row action table; known widths: "
            f"{sorted(_BY_ROWS)}"
        )
    return _BY_ROWS[rows]
