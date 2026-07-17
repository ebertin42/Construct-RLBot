#!/usr/bin/env bash
# Continuously stream to RLViser, rotating over the LIVE entity-transformer run
# (checkpoints_entity/, kickstart lineage) in 1v1 then 2v2, a run-B canary
# segment (checkpoints_b/), and a v3-teacher reference segment (checkpoints/,
# frozen MLP lineage) — 5-slot cycle. Rotates every ROTATE_SECS (default 300).
# watch.py dispatches v0/v1 nets by checkpoint schema_version automatically.
# Usage: CONSTRUCT_VISER_ADDR=<ip>:<port> ./scripts/watch_loop.sh [rotate_secs]
set -uo pipefail
cd "$(dirname "$0")/.."
ROTATE_SECS="${1:-300}"
slot=0

while true; do
    # newest by MTIME, not lexical step number: after a rollback the live
    # frontier has SMALLER step numbers than stale pre-rollback files
    entity=$(ls -t checkpoints_entity/ck_*.pt 2>/dev/null | head -1)
    teacher=$(ls -t checkpoints/ck_*.pt 2>/dev/null | head -1)
    runb=$(ls -t checkpoints_b/ck_*.pt 2>/dev/null | head -1)
    case $((slot % 5)) in
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
            if [ -n "$runb" ]; then
                echo "$(date +%H:%M:%S) [RUN-B 1v1] streaming $runb for ${ROTATE_SECS}s"
                timeout "$ROTATE_SECS" python scripts/watch.py "$runb"
            fi ;;
        4)
            if [ -n "$teacher" ]; then
                echo "$(date +%H:%M:%S) [TEACHER v3 1v1] streaming $teacher for ${ROTATE_SECS}s"
                timeout "$ROTATE_SECS" python scripts/watch.py "$teacher"
            fi ;;
    esac
    [ -z "$entity" ] && { echo "no entity checkpoints yet, waiting..."; sleep 30; }
    slot=$((slot + 1))
    sleep 2  # let the UDP socket free up before rebinding
done
