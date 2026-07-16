#!/usr/bin/env bash
# Continuously stream to RLViser, rotating over the LIVE entity-transformer run
# (checkpoints_entity/, kickstart lineage) in 1v1 then 2v2, with an occasional
# v3-teacher reference segment (checkpoints/, frozen MLP lineage) every 4th
# rotation for eyeball comparison. Rotates every ROTATE_SECS (default 300).
# watch.py dispatches v0/v1 nets by checkpoint schema_version automatically.
# Usage: CONSTRUCT_VISER_ADDR=<ip>:<port> ./scripts/watch_loop.sh [rotate_secs]
set -uo pipefail
cd "$(dirname "$0")/.."
ROTATE_SECS="${1:-300}"
slot=0

while true; do
    entity=$(ls checkpoints_entity/ck_*.pt 2>/dev/null | sort | tail -1)
    teacher=$(ls checkpoints/ck_*.pt 2>/dev/null | sort | tail -1)
    case $((slot % 4)) in
        0|2)
            if [ -n "$entity" ]; then
                echo "$(date +%H:%M:%S) [ENTITY 1v1] streaming $entity for ${ROTATE_SECS}s"
                timeout "$ROTATE_SECS" python scripts/watch.py "$entity"
            fi ;;
        1)
            if [ -n "$entity" ]; then
                echo "$(date +%H:%M:%S) [ENTITY 2v2] streaming $entity for ${ROTATE_SECS}s"
                timeout "$ROTATE_SECS" python scripts/watch.py "$entity" --mode 2v2
            fi ;;
        3)
            if [ -n "$teacher" ]; then
                echo "$(date +%H:%M:%S) [TEACHER v3 1v1] streaming $teacher for ${ROTATE_SECS}s"
                timeout "$ROTATE_SECS" python scripts/watch.py "$teacher"
            fi ;;
    esac
    [ -z "$entity" ] && { echo "no entity checkpoints yet, waiting..."; sleep 30; }
    slot=$((slot + 1))
    sleep 2  # let the UDP socket free up before rebinding
done
