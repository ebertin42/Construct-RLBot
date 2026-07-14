#!/usr/bin/env bash
# Continuously stream to RLViser, alternating between the MAIN run (1v1,
# newest synced checkpoint in checkpoints/) and RUN B (2v2, newest in
# checkpoints_b/), rotating every ROTATE_SECS (default 300).
# Usage: CONSTRUCT_VISER_ADDR=<ip>:<port> ./scripts/watch_loop.sh [rotate_secs]
set -uo pipefail
cd "$(dirname "$0")/.."
ROTATE_SECS="${1:-300}"
show_b=0

while true; do
    if [ "$show_b" = "1" ] && ls checkpoints_b/ck_*.pt >/dev/null 2>&1; then
        latest=$(ls checkpoints_b/ck_*.pt | sort | tail -1)
        echo "$(date +%H:%M:%S) [RUN-B 2v2] streaming $latest for ${ROTATE_SECS}s"
        timeout "$ROTATE_SECS" python scripts/watch.py "$latest" --mode 2v2
    else
        latest=$(ls checkpoints/ck_*.pt 2>/dev/null | sort | tail -1)
        if [ -z "$latest" ]; then echo "no checkpoints yet, waiting..."; sleep 30; continue; fi
        echo "$(date +%H:%M:%S) [MAIN 1v1] streaming $latest for ${ROTATE_SECS}s"
        timeout "$ROTATE_SECS" python scripts/watch.py "$latest"
    fi
    show_b=$((1 - show_b))
    sleep 2  # let the UDP socket free up before rebinding
done
