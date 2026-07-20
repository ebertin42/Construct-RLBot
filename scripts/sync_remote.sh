#!/usr/bin/env bash
# Pull checkpoints + training log from the remote training box every 60s so
# the local dashboard, viewer loop, and eval runner follow the main run.
# Usage: ./scripts/sync_remote.sh [user@host] [remote_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:-elliot@192.168.86.117}"
RDIR="${2:-construct}"

mkdir -p checkpoints checkpoints_entity league
while true; do
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints/" checkpoints/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints/train_v0.log" checkpoints/train_remote.log 2>/dev/null
    # entity-transformer lineage (kickstart run) — separate dir, own log
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_entity/" checkpoints_entity/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_entity/train_v1.log" checkpoints_entity/train_remote.log 2>/dev/null
    # remote league ladder (mixed v0/v1 pool) — dashboard reads registry_remote.jsonl
    rsync -az "$HOST:$RDIR/league/registry.jsonl" league/registry_remote.jsonl 2>/dev/null
    sleep 60
done
