#!/usr/bin/env bash
# Continuously stream to RLViser the LIVE training run ONLY (checkpoints_entity/,
# newest-by-mtime = the frontier the trainer is writing right now), rotating over
# the SAME team-size formats the run actually trains on. configs/train_v1.toml's
# team_size_weights = [0.5, 0.3, 0.2] (1v1/2v2/3v3), so the 10-slot cycle below is
# 5x 1v1, 3x 2v2, 2x 3v3, interleaved — what you watch matches what it learns.
# Retired lineages (run-B v0, frozen v3 teacher) are deliberately NOT streamed:
# they are not training, so watching them tells you nothing about the live run.
# Rotates every ROTATE_SECS (default 300).
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
    # 10-slot cycle mirroring team_size_weights [0.5, 0.3, 0.2]:
    # 1v1 on slots 0,2,4,6,8 | 2v2 on 1,5,7 | 3v3 on 3,9
    case $((slot % 10)) in
        1|5|7) mode="2v2" ;;
        3|9)   mode="3v3" ;;
        *)     mode="1v1" ;;
    esac
    if [ -n "$entity" ]; then
        announce "LIVE $mode" "$entity"
        timeout "$ROTATE_SECS" python scripts/watch.py "$entity" --mode "$mode"
    fi
    [ -z "$entity" ] && { echo "no live checkpoints yet, waiting..."; sleep 30; }
    slot=$((slot + 1))
    sleep 2  # let the UDP socket free up before rebinding
done
