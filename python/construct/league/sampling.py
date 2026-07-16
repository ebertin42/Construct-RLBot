"""Opponent selection: exploit the ladder top + explore recent additions."""
import random


def choose_opponents(registry, k=4, recent=6, rng=None, schema_version=None):
    """Pick up to `k` opponents: ladder-top exploit + recent-additions explore.

    `schema_version`, when given, restricts the pool to entries tagged with
    that schema before either half of the pick runs -- v0 and v1 policies can
    never play each other (different obs), so this is how callers (Trainer,
    league_tick.py) keep a run's opponent pool schema-pure. `None` (default)
    preserves the original unfiltered behavior.
    """
    rng = rng or random.Random()
    ladder = registry.ladder()
    if schema_version is not None:
        ladder = [e for e in ladder if e.get("schema_version", 0) == schema_version]
    if not ladder:
        return []
    picks = ladder[:2]  # top by exposed skill
    newest = sorted(registry.entries(schema_version=schema_version),
                     key=lambda e: e["added_ts"])[-recent:]
    pool = [e for e in newest if e["ck"] not in {p["ck"] for p in picks}]
    rng.shuffle(pool)
    picks.extend(pool[: max(0, k - len(picks))])
    return picks[:k]
