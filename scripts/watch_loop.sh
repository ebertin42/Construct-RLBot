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
WATCH_DIRS="${CONSTRUCT_WATCH_DIRS:-checkpoints_scratch checkpoints_entity checkpoints_hc}"
# Curriculum the viewer renders under -- MUST match the live arm's curriculum so
# "what you watch matches what it learns". Default is the match-win regime
# (full 300s matches + score); set CONSTRUCT_WATCH_CURRICULUM='' to render
# legacy episodes when the live arm is a legacy run.
WATCH_CURRICULUM="${CONSTRUCT_WATCH_CURRICULUM-configs/curriculum_v3_match.toml}"

# From-scratch/foreign runs (1v1): rotate the viewer through the SAME opponent
# roster the run trains on -- element/immortal/necto/nexto -- plus a self-play
# slot, at the LIVE auto-curriculum period, so the stream shows the real matchup
# (blue = Construct learner, orange = the ported bot). The overlay reads the
# who-is-who label written by announce(). Set CONSTRUCT_WATCH_FOREIGN=0 to force
# plain self-play.
FOREIGN_ROSTER=(element immortal necto nexto self)
FOREIGN_CACHE="${CONSTRUCT_FOREIGN_CACHE:-$HOME/.cache/construct}"
SCRATCH_LOG="${CONSTRUCT_SCRATCH_LOG:-checkpoints_scratch/train_remote.log}"
WATCH_FOREIGN="${CONSTRUCT_WATCH_FOREIGN:-1}"
current_period() {  # $1 = bot kind. Periods are PER BOT since 2026-07-25, so the
                    # viewer must read that bot's own rung out of the newest
                    # decision line, which looks like:
                    #   auto-curriculum: element wr0.69/ema0.69 p4->3 | immortal ... p3->2
                    # a held slot instead reads "... p4 dwell1/2". The previous
                    # parser grepped the retired "auto-curriculum: winrate ..."
                    # format, so it silently froze on a stale pre-restart value.
    local seg p
    seg=$(grep -a "auto-curriculum: .*wr[0-9]" "$SCRATCH_LOG" 2>/dev/null | tail -1 \
          | tr '|' '\n' | grep -aE "(^| )$1 " | tail -1)
    p=$(printf '%s' "$seg" | grep -oE 'p[0-9]+->[0-9]+' | grep -oE '[0-9]+$')   # moved
    [ -z "$p" ] && p=$(printf '%s' "$seg" | grep -oE 'p[0-9]+' | head -1 | tr -d 'p')
    echo "${p:-${CONSTRUCT_WATCH_PERIOD:-4}}"
}

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
    # 1v1 on slots 0,2,4,6,8 | 2v2 on 1,5,7 | 3v3 on 3,9.
    # CONSTRUCT_WATCH_MODE forces a single format -- the from-scratch/foreign-opponent
    # runs are ALL 1v1 (foreign bots are 1v1-only), so set it to 1v1 there.
    if [ -n "${CONSTRUCT_WATCH_MODE:-}" ]; then
        mode="$CONSTRUCT_WATCH_MODE"
    else
        case $((slot % 10)) in
            1|5|7) mode="2v2" ;;
            3|9)   mode="3v3" ;;
            *)     mode="1v1" ;;
        esac
    fi
    if [ -n "$entity" ]; then
        # 1v1 + foreign enabled: rotate the opponent roster at the live period.
        fargs=()
        label="LIVE $mode"
        if [ "$mode" = "1v1" ] && [ "$WATCH_FOREIGN" = "1" ]; then
            fk="${FOREIGN_ROSTER[$((slot % ${#FOREIGN_ROSTER[@]}))]}"
            P=$(current_period "$fk")   # that bot's OWN rung, not a shared one
            if [ "$fk" != "self" ] && [ -f "$FOREIGN_CACHE/${fk}_weights.npz" ]; then
                fargs=(--foreign "$fk" --foreign-weights "$FOREIGN_CACHE/${fk}_weights.npz" --period "$P")
                label="Blue=Construct  Orange=${fk}.p${P}"
            else
                label="self-play (both Construct)"
            fi
        fi
        announce "$label" "$entity"
        if [ -n "$WATCH_CURRICULUM" ]; then
            timeout "$ROTATE_SECS" "$PYTHON" scripts/watch.py "$entity" --mode "$mode" --curriculum "$WATCH_CURRICULUM" "${fargs[@]}"
        else
            timeout "$ROTATE_SECS" "$PYTHON" scripts/watch.py "$entity" --mode "$mode" "${fargs[@]}"
        fi
    fi
    [ -z "$entity" ] && { echo "no live checkpoints yet, waiting..."; sleep 30; }
    slot=$((slot + 1))
    sleep 2  # let the UDP socket free up before rebinding
done
