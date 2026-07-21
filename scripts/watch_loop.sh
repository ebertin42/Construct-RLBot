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

# Resolve python robustly: a bare `python` is often absent (the interpreter is
# .venv/bin/python and the launching env may not have the venv on PATH, e.g.
# when started via ctl.py viewer). Prefer an explicit PYTHON, then the repo
# venv, then whatever `python`/`python3` resolves to.
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    if [ -x .venv/bin/python ]; then PYTHON=.venv/bin/python
    elif command -v python >/dev/null 2>&1; then PYTHON=python
    else PYTHON=python3; fi
fi

# Dirs the live trainer may be writing. The match-win arm writes to
# checkpoints_hc/<arm>/, the legacy lineage to checkpoints_entity/. We stream
# whichever holds the freshest checkpoint (newest by mtime = the live frontier),
# so the viewer auto-follows the current arm wherever it writes.
WATCH_DIRS="${CONSTRUCT_WATCH_DIRS:-checkpoints_entity checkpoints_hc}"
# Curriculum the viewer renders under -- MUST match the live arm's curriculum so
# "what you watch matches what it learns". Default is the match-win regime
# (full 300s matches + score); set CONSTRUCT_WATCH_CURRICULUM='' to render
# legacy episodes when the live arm is a legacy run.
WATCH_CURRICULUM="${CONSTRUCT_WATCH_CURRICULUM-configs/curriculum_v3_match.toml}"

# Status file for the Windows overlay (deploy/windows_stream_overlay.ps1)
STATUS_FILE="${CONSTRUCT_STREAM_STATUS:-/mnt/c/Users/Elliot/AppData/Local/Construct/current_stream.txt}"
announce() {  # $1 = label, $2 = ck path
    echo "$(date +%H:%M:%S) [$1] streaming $2 for ${ROTATE_SECS}s"
    printf '%s  %s  (%s)' "$1" "$(basename "$2" .pt)" "$(date +%H:%M)" > "$STATUS_FILE" 2>/dev/null || true
}

while true; do
    # newest by MTIME across all live-trainer dirs, not lexical step number:
    # after a rollback the live frontier has SMALLER step numbers than stale
    # pre-rollback files, and the current arm may write to checkpoints_hc/<arm>/
    # rather than checkpoints_entity/. -maxdepth 2 reaches checkpoints_hc/<arm>/.
    entity=$(find $WATCH_DIRS -maxdepth 2 -name 'ck_*.pt' -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)
    # 10-slot cycle mirroring team_size_weights [0.5, 0.3, 0.2]:
    # 1v1 on slots 0,2,4,6,8 | 2v2 on 1,5,7 | 3v3 on 3,9
    case $((slot % 10)) in
        1|5|7) mode="2v2" ;;
        3|9)   mode="3v3" ;;
        *)     mode="1v1" ;;
    esac
    if [ -n "$entity" ]; then
        announce "LIVE $mode" "$entity"
        if [ -n "$WATCH_CURRICULUM" ]; then
            timeout "$ROTATE_SECS" "$PYTHON" scripts/watch.py "$entity" --mode "$mode" --curriculum "$WATCH_CURRICULUM"
        else
            timeout "$ROTATE_SECS" "$PYTHON" scripts/watch.py "$entity" --mode "$mode"
        fi
    fi
    [ -z "$entity" ] && { echo "no live checkpoints yet, waiting..."; sleep 30; }
    slot=$((slot + 1))
    sleep 2  # let the UDP socket free up before rebinding
done
