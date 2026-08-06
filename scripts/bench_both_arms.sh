#!/usr/bin/env bash
# Bench whatever each live arm has ADDED since it was last benched, and accumulate the rows
# into a per-arm history file.
#
# THE BOUND IS DERIVED, NOT HARDCODED. An earlier version carried the last bench's step
# numbers as literals in this file, which goes stale the moment it runs -- the next
# invocation would silently re-bench the same region and quietly double-count it in the
# trend. Here LO comes from the max step already present in the history, so running this
# repeatedly is idempotent and each cell is paid for exactly once.
#
# SEQUENTIAL, one cell at a time: concurrent cells contend for the box, and the entire value
# of these numbers is that they are comparable to the ones already recorded.
#
# WHY BENCH AT ALL when fd_pct is on the dashboard: fd_pct is clone completeness and it is
# NOT the stopping signal. The distillation run was cut 28M steps early because fd_ce
# plateaued at 43% while goals_against was still falling -0.061/Mstep -- and that "plateau"
# later turned out to be stale optimizer state, not convergence. goals_against costs a bench
# to read; that is the price of the only number that decides.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-8}
KIND=${KIND:-necto}
mkdir -p logs
echo "gauge: $KIND"

bench_one() {  # dir label
    local dir=$1 label=$2
    # History is PER OPPONENT: mixing element and necto cells into one trend would splice two
    # different rulers and manufacture a step change at the switchover.
    local hist=logs/bench_hist_${label}_${KIND}.tsv
    local lo=0
    if [ -s "$hist" ]; then
        lo=$(awk -F'\t' 'NR>1 && $3 ~ /^[0-9]+$/ {if ($3+0 > m) m=$3+0} END{print m+1}' "$hist")
    fi
    echo "=== $label: benching steps > ${lo} ==="
    local tmp=logs/.bench_${label}_new.tsv
    if ! bash scripts/bench_arm.sh "$dir" "$N" "$label" "$tmp" 32 45000 "$lo" 99999999999 "$KIND"; then
        echo "  $label: nothing new to bench (or bench failed); leaving history untouched" >&2
        return 0
    fi
    if [ -s "$hist" ]; then tail -n +2 "$tmp" >> "$hist"; else cp "$tmp" "$hist"; fi
    echo "  $label history now $(( $(wc -l < "$hist") - 1 )) rows -> $hist"
}

# Which arms to bench. Defaults to the live one only: the PPO arm was retired 2026-08-05 and
# still has unbenched checkpoints, but spending cells on a dead lineage buys nothing. Pass
# arm names to override, e.g. `bench_both_arms.sh 8 distill ppo`.
shift || true
ARMS=("$@")
[ ${#ARMS[@]} -gt 0 ] || ARMS=(distill)
for a in "${ARMS[@]}"; do
    case "$a" in
        distill) bench_one checkpoints_distill       distill ;;
        ppo)     bench_one checkpoints_ppo_distilled ppo ;;
        *)       echo "unknown arm: $a" >&2; exit 1 ;;
    esac
done
echo "DONE: ${ARMS[*]}"
