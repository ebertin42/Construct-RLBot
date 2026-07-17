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

# Status file for the Windows overlay (deploy/windows_stream_overlay.ps1)
STATUS_FILE="${CONSTRUCT_STREAM_STATUS:-/mnt/c/Users/Elliot/AppData/Local/Construct/current_stream.txt}"
announce() {  # $1 = label, $2 = ck path
    echo "$(date +%H:%M:%S) [$1] streaming $2 for ${ROTATE_SECS}s"
    printf '%s  %s  (%s)' "$1" "$(basename "$2" .pt)" "$(date +%H:%M)" > "$STATUS_FILE" 2>/dev/null || true
}

while true; do
    # newest by MTIME, not lexical step number: after a rollback the live
    # frontier has SMALLER step numbers than stale pre-rollback files
    entity=$(find checkpoints_entity -maxdepth 1 -name 'ck_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    teacher=$(find checkpoints -maxdepth 1 -name 'ck_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    runb=$(find checkpoints_b -maxdepth 1 -name 'ck_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    case $((slot % 5)) in
        0|2)
            if [ -n "$entity" ]; then
                announce "ENTITY 1v1" "$entity"
                timeout "$ROTATE_SECS" python scripts/watch.py "$entity"
            fi ;;
        1)
            if [ -n "$entity" ]; then
                announce "ENTITY 2v2" "$entity"
                timeout "$ROTATE_SECS" python scripts/watch.py "$entity" --mode 2v2
            fi ;;
        3)
            if [ -n "$runb" ]; then
                announce "RUN-B 1v1" "$runb"
                timeout "$ROTATE_SECS" python scripts/watch.py "$runb"
            fi ;;
        4)
            if [ -n "$teacher" ]; then
                announce "TEACHER v3 1v1" "$teacher"
                timeout "$ROTATE_SECS" python scripts/watch.py "$teacher"
            fi ;;
    esac
    [ -z "$entity" ] && { echo "no entity checkpoints yet, waiting..."; sleep 30; }
    slot=$((slot + 1))
    sleep 2  # let the UDP socket free up before rebinding
done
