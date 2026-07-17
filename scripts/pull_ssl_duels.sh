#!/usr/bin/env bash
# SSL 1v1 replay-corpus pull from ballchasing.com — the acquisition half of
# the SSL-finetune corpus (winners-side filtering happens later at parse
# time). Wraps `python -m construct.data.ballchasing_corpus` (see its module
# doc: created-before cursor in data/replays/ssl/duels/pull_state.json,
# id-on-disk dedupe, 200/h download-cap pacing, powershell disk guard).
#
# Safe to kill and relaunch at any time — it resumes, never restarts.
# Start detached:
#   mkdir -p logs && setsid nice -n 15 scripts/pull_ssl_duels.sh >> logs/ssl_pull.log 2>&1 &
#
# Exit codes from the python loop: 0 = frontier exhausted / --max reached
# (stop), 3 = disk guard tripped (stop — free host disk first), anything
# else = crash (network etc.) — restart after a pause, the loop resumes.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
# Token stays in the env file (mode 600, outside the repo) — never echo it.
source "$HOME/.config/construct/ballchasing.env"

while true; do
    python -m construct.data.ballchasing_corpus "$@"
    rc=$?
    if [ "$rc" -eq 0 ] || [ "$rc" -eq 3 ]; then
        echo "$(date '+%m-%d %H:%M:%S') pull finished rc=$rc — not restarting"
        break
    fi
    echo "$(date '+%m-%d %H:%M:%S') pull exited rc=$rc — restarting in 300s"
    sleep 300
done
