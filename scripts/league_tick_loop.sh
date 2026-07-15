#!/usr/bin/env bash
# Keep the TrueSkill ladder fresh: register newest checkpoints from both runs
# and play rating matches every TICK_SECS (default 4h). Runs nice so the
# run-B trainer and viewer keep priority.
set -uo pipefail
cd "$(dirname "$0")/.."
TICK_SECS="${1:-14400}"
while true; do
    echo "$(date +%H:%M:%S) league tick"
    nice -n 15 python scripts/league_tick.py --matches 6
    sleep "$TICK_SECS"
done
