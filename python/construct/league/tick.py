"""Core league-tick logic: checkpoint discovery, per-schema rating matches,
and match-budget fairness across schema pools.

scripts/league_tick.py is the thin CLI wrapper around run_tick(). The logic
lives here (importable, engine-free until a match actually has to be played)
so pairing and budget behavior are unit-testable without building engines or
touching real checkpoints -- see tests/python/test_league_tick.py.

Schema pools never mix: every pairing is drawn from a single schema_version's
entries (registry filter + sampling filter), and play_entries() keeps its own
hard guard underneath as defense in depth.
"""
import glob
import random

from construct.league.matches import MatchRunner, play_entries
from construct.league.registry import Registry
from construct.league.sampling import choose_opponents

# (run label, checkpoint glob) pairs scanned per schema version. Globs are
# cwd-relative, matching how league_tick_loop.sh runs from the repo root.
CHECKPOINT_SOURCES = {
    0: (("main", "checkpoints/ck_*.pt"), ("b", "checkpoints_b/ck_*.pt")),
    1: (("entity", "checkpoints_entity/ck_*.pt"),),
}
DEFAULT_REGISTRY_PATH = {0: "league/registry.jsonl", 1: "league/registry_v1.jsonl"}


def newest(pattern):
    cks = sorted(glob.glob(pattern))
    return cks[-1] if cks else None


def split_budget(total, n_pools):
    """Fair integer split of `total` matches across `n_pools`: every pool gets
    total // n_pools, and the first total % n_pools pools get one extra. The
    shares always sum to exactly max(total, 0)."""
    assert n_pools > 0, "split_budget needs at least one pool"
    base, extra = divmod(max(0, total), n_pools)
    return [base + (1 if i < extra else 0) for i in range(n_pools)]


def register_newest_checkpoints(reg, schema_version):
    """Register the newest on-disk checkpoint of every source run for
    `schema_version` (no-op for a run whose glob matches nothing)."""
    import torch  # heavy import deferred: registry-only paths stay light
    for run, pattern in CHECKPOINT_SOURCES[schema_version]:
        ck = newest(pattern)
        if ck:
            meta = torch.load(ck, map_location="cpu", weights_only=False)
            reg.add(ck, steps=meta["total_steps"], run=run,
                    reward_config=meta.get("reward_config_path", "unknown"),
                    schema_version=schema_version)


def play_rating_matches(reg, schema_version, budget, *, net_heads=4, rng=None,
                        steps=2700, num_arenas=8, make_runner=None,
                        play=play_entries):
    """Play up to `budget` rating matches inside one schema pool and record the
    results via the Registry API. Returns the number of matches played.

    Pairing: the two most recently added entries each meet up to 3 opponents
    from choose_opponents() -- both sides drawn from reg.entries(schema_version)
    only, so a pair can never cross schemas by construction (and play_entries
    re-checks entry metadata as a hard guard).

    The MatchRunner (engine build) is created lazily, only once a pair is
    actually about to play: a pool with <2 entries or a zero budget costs no
    engine time at all. `make_runner`/`play` are injection seams for tests.
    """
    rng = rng or random.Random()
    entries = reg.entries(schema_version=schema_version)
    if budget <= 0 or len(entries) < 2:
        return 0
    if make_runner is None:
        def make_runner():
            return MatchRunner(num_arenas=num_arenas, seed=rng.randrange(1 << 30),
                               schema_version=schema_version, net_heads=net_heads)
    mr = None
    played = 0
    fresh = sorted(entries, key=lambda e: e["added_ts"])[-2:]
    for member in fresh:
        for opp in choose_opponents(reg, k=3, rng=rng, schema_version=schema_version):
            if played >= budget:
                return played
            if opp["ck"] == member["ck"]:
                continue
            if mr is None:
                mr = make_runner()
            ga, gb = play(mr, member, opp, steps=steps)
            reg.record_match(member["ck"], opp["ck"], ga, gb)
            print(f"[v{schema_version}] {member['ck']}  {ga}:{gb}  {opp['ck']}")
            played += 1
    return played


def print_ladder(reg, schema_version, limit=15):
    print(f"\n[v{schema_version}] {'skill':>7}  {'mu':>6}  {'games':>5}  run   checkpoint")
    # defense-in-depth schema filter, in case the registry file is mixed
    ladder = [e for e in reg.ladder() if e["schema_version"] == schema_version]
    for e in ladder[:limit]:
        print(f"{e['skill']:7.2f}  {e['mu']:6.2f}  {e['games']:5d}  {e['run']:<4}  {e['ck']}")


def run_tick(schema_versions, matches, registry_path=None, net_heads=4, rng=None,
             steps=2700, make_runner=None, play=play_entries):
    """One full tick over the given schema pools with a fair match-budget split.
    Returns {schema_version: matches_played}.

    Budget fairness: `matches` is split near-evenly across the pools
    (split_budget); pools run in the given order and a pool that cannot use its
    full share (too few entries, or fewer pairable opponents than budget) rolls
    its leftover forward to the next pool, so the tick budget is spent whenever
    a later pool can absorb it.

    Registry topology: with registry_path=None every schema uses its own
    default file (DEFAULT_REGISTRY_PATH); an explicit path is shared by all
    pools. Sharing means one Registry INSTANCE, not merely one file --
    Registry._save() rewrites the whole file from memory, so two instances on
    the same path would clobber each other's match results.
    """
    rng = rng or random.Random()
    regs_by_path = {}
    reg_for = {}
    for sv in schema_versions:
        path = registry_path or DEFAULT_REGISTRY_PATH[sv]
        if path not in regs_by_path:
            regs_by_path[path] = Registry(path=path)
        reg_for[sv] = regs_by_path[path]

    played_by = {}
    leftover = 0
    for sv, share in zip(schema_versions, split_budget(matches, len(schema_versions))):
        reg = reg_for[sv]
        register_newest_checkpoints(reg, sv)
        budget = share + leftover
        played_by[sv] = play_rating_matches(
            reg, sv, budget, net_heads=net_heads, rng=rng, steps=steps,
            make_runner=make_runner, play=play)
        leftover = budget - played_by[sv]

    for sv in schema_versions:
        print_ladder(reg_for[sv], sv)
    return played_by
