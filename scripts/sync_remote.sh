#!/usr/bin/env bash
# Pull checkpoints + training log from the remote training box every 60s so
# the local dashboard, viewer loop, and eval runner follow the main run.
# Usage: ./scripts/sync_remote.sh [user@host] [remote_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:-elliot@192.168.86.117}"
RDIR="${2:-construct}"

mkdir -p checkpoints checkpoints_entity checkpoints_scratch checkpoints_v9 \
         checkpoints_v10 league
while true; do
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints/" checkpoints/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints/train_v0.log" checkpoints/train_remote.log 2>/dev/null
    # entity-transformer lineage (kickstart run) — separate dir, own log
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_entity/" checkpoints_entity/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_entity/train_v1.log" checkpoints_entity/train_remote.log 2>/dev/null
    # v9 — THE CURRENT MAIN RUN (2026-07-26): fresh net, 104-row air action table,
    # equal 1s/2s/3s. Pulled first because everything downstream (dashboard,
    # watch_loop) picks the newest checkpoint by mtime across these dirs.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_v9/" checkpoints_v9/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_v9/v9_s20260726.log" checkpoints_v9/train_remote.log 2>/dev/null
    # v10 — the ENTROPY FORK of v9 (2026-07-30), forked at 2,087.6M. Same reward
    # tape as v9; entropy_coef is the only difference (0.0142 vs v9's 0.006).
    # Synced because the whole point of the fork is comparing the two arms on the
    # bench, and the bench runs LOCALLY -- without this, run B's checkpoints exist
    # only on the box and the comparison is impossible.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_v10/" checkpoints_v10/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_v10/v10_s20260730.log" checkpoints_v10/train_remote.log 2>/dev/null
    # v8 from-scratch run — RETIRED 2026-07-26 at 1.589B (superseded by v9). Kept
    # syncing so its final checkpoints stay pullable; delete once archived.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_scratch/" checkpoints_scratch/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_scratch/fromscratch_s20260723.log" checkpoints_scratch/train_remote.log 2>/dev/null
    # remote league ladder (mixed v0/v1 pool) — dashboard reads registry_remote.jsonl
    rsync -az "$HOST:$RDIR/league/registry.jsonl" league/registry_remote.jsonl 2>/dev/null
    sleep 60
done
