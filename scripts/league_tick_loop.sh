#!/usr/bin/env bash
# Keep a TrueSkill ladder fresh: register newest checkpoints and play rating
# matches every TICK_SECS (default 4h). Runs nice so the trainer/viewer keep
# priority.
#
# Usage:
#   scripts/league_tick_loop.sh [TICK_SECS] [EXTRA_LEAGUE_TICK_ARGS...]
#
# Defaults to the legacy v0 league (checkpoints/ + checkpoints_b/,
# league/registry.jsonl). For the v1 entity-transformer league
# (checkpoints_entity/), pass league_tick.py's own flags as extra args, e.g.:
#   scripts/league_tick_loop.sh 14400 --schema-version 1 --registry league/registry_v1.jsonl
# (--registry is optional -- league_tick.py already defaults to
# league/registry_v1.jsonl when --schema-version 1 is given.)
# To rate BOTH pools every tick with a fair split of the match budget:
#   scripts/league_tick_loop.sh 14400 --schema-version all
# (add --registry league/registry.jsonl if v0+v1 entries share one file,
# as on the remote box).
set -uo pipefail
cd "$(dirname "$0")/.."
TICK_SECS="${1:-14400}"
shift || true  # remaining args ($@), if any, pass straight through to league_tick.py
while true; do
    echo "$(date +%H:%M:%S) league tick"
    nice -n 15 python scripts/league_tick.py --matches 6 "$@"
    sleep "$TICK_SECS"
done
