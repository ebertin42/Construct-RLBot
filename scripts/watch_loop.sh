#!/usr/bin/env bash
# Continuously stream the newest checkpoint to RLViser, rotating to the
# latest one every ROTATE_SECS (default 300 = 5 min).
# Usage: CONSTRUCT_VISER_ADDR=<ip>:<port> ./scripts/watch_loop.sh [rotate_secs]
set -uo pipefail
cd "$(dirname "$0")/.."
ROTATE_SECS="${1:-300}"

while true; do
    latest=$(ls checkpoints/ | grep -E '^ck_[0-9]+\.pt$' | sort | tail -1)
    if [ -z "$latest" ]; then echo "no checkpoints yet, waiting..."; sleep 30; continue; fi
    echo "$(date +%H:%M:%S) streaming $latest for ${ROTATE_SECS}s"
    timeout "$ROTATE_SECS" python scripts/watch.py "checkpoints/$latest"
    sleep 2  # let the UDP socket free up before rebinding
done
