"""Opponent selection: exploit the ladder top + explore recent additions."""
import random


def choose_opponents(registry, k=4, recent=6, rng=None):
    rng = rng or random.Random()
    ladder = registry.ladder()
    if not ladder:
        return []
    picks = ladder[:2]  # top by exposed skill
    newest = sorted(registry.entries(), key=lambda e: e["added_ts"])[-recent:]
    pool = [e for e in newest if e["ck"] not in {p["ck"] for p in picks}]
    rng.shuffle(pool)
    picks.extend(pool[: max(0, k - len(picks))])
    return picks[:k]
