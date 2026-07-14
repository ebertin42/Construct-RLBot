#!/usr/bin/env bash
# Pull checkpoints + training log from the remote training box every 60s so
# the local dashboard, viewer loop, and eval runner follow the main run.
# Usage: ./scripts/sync_remote.sh [user@host] [remote_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:-elliot@192.168.86.117}"
RDIR="${2:-construct}"

while true; do
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints/" checkpoints/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints/train_v0.log" checkpoints/train_remote.log 2>/dev/null
    sleep 60
done
