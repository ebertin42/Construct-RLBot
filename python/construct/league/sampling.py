"""Opponent selection: exploit the ladder top + explore recent additions."""
import random


def choose_opponents(registry, k=4, recent=6, rng=None, schema_version=None,
                     action_table=None):
    """Pick up to `k` opponents: ladder-top exploit + recent-additions explore.

    `schema_version`, when given, restricts the pool to entries tagged with
    that schema before either half of the pick runs -- v0 and v1 policies can
    never play each other (different obs), so this is how callers (Trainer,
    league_tick.py) keep a run's opponent pool schema-pure. `None` (default)
    preserves the original unfiltered behavior.

    `action_table`, when given, restricts it FURTHER. schema_version is 1 for
    BOTH v1 tables -- 92-row `construct_92_v1` and 104-row
    `construct_104_v1air` -- so the version filter alone lets a 92-row arm into
    a 104-row run's pool, where `set_opponents` rejects it (one engine binds
    one decode table). Without this the run does not crash, it just logs a
    set_opponents failure every refresh and NEVER gets a league opponent, which
    looks like "the league is quiet" rather than "the league is broken".
    Entries written before 2026-07-26 have no such key and default to the
    92-row table.
    """
    rng = rng or random.Random()

    def _ok(e):
        if schema_version is not None and e.get("schema_version", 0) != schema_version:
            return False
        if action_table is not None and \
                e.get("action_table", "construct_92_v1") != action_table:
            return False
        return True

    ladder = [e for e in registry.ladder() if _ok(e)]
    if not ladder:
        return []
    picks = ladder[:2]  # top by exposed skill
    newest = sorted([e for e in registry.entries() if _ok(e)],
                     key=lambda e: e["added_ts"])[-recent:]
    pool = [e for e in newest if e["ck"] not in {p["ck"] for p in picks}]
    rng.shuffle(pool)
    picks.extend(pool[: max(0, k - len(picks))])
    return picks[:k]
