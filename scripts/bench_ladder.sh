#!/usr/bin/env bash
# Two measurements, sequentially:
#
#   1. THE FULL LADDER. One checkpoint against ALL FOUR ported bots at period 1. Every
#      absolute number this project has quoted for two days is against element alone, which
#      is the WEAKEST of the four -- a 296k-param plain MLP. Parity with element says nothing
#      about where the ceiling is, and the ceiling is what decides whether distillation still
#      has headroom or whether the teacher has become the binding constraint.
#
#   2. THE ELEMENT REGION. n=8 across the steps added since the last element bench, appended
#      to the arm history, so the trend keeps its lever arm.
#
# Ladder first: it is 4 cells against 8, and it is the measurement that can change the plan.
# Sequential throughout -- concurrent cells contend and bias.
set -euo pipefail
cd "$(dirname "$0")/.."

CK=${1:?usage: bench_ladder.sh <checkpoint> [n]}
N=${2:-8}
mkdir -p logs
OUT=logs/bench_ladder.tsv

printf 'bot\tcheckpoint\trecord\twin_share\tgoals_for\tgoals_against\tdiff\n' > "$OUT"
for KIND in element nexto necto immortal; do
    echo "=== $KIND period 1 | $(basename "$CK") ==="
    line=$(.venv/bin/python scripts/bench_foreign.py "$CK" \
        --kind "$KIND" --weights "$HOME/.cache/construct/${KIND}_weights.npz" \
        --arenas 32 --steps 45000 --seed 11 --period 1 2>/dev/null)
    rec=$(sed -nE 's/.*  ([0-9]+W\/[0-9]+D\/[0-9]+L).*/\1/p' <<<"$line" | head -1)
    ws=$(sed -nE 's/.*win_share=([0-9.]+).*/\1/p'      <<<"$line" | head -1)
    gf=$(sed -nE 's/.*goals_for=([0-9.]+).*/\1/p'      <<<"$line" | head -1)
    ga=$(sed -nE 's/.*goals_against=([0-9.]+).*/\1/p'  <<<"$line" | head -1)
    # [+-]? matters: bench_foreign prints diff=+0.2281 once we OUTSCORE a bot, and a bare
    # [0-9.-] class drops it -- a parse bug that can only ever fire on a good result.
    df=$(sed -nE 's/.*diff=([+-]?[0-9.]+).*/\1/p'      <<<"$line" | head -1)
    [ -n "$ga" ] || { echo "PARSE_FAILED $KIND" >&2; rec=NA; ws=NA; gf=NA; ga=NA; df=NA; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$KIND" "$(basename "$CK")" "$rec" "$ws" "$gf" "$ga" "$df" >> "$OUT"
    echo "  $KIND: $rec  ws=$ws  gf=$gf ga=$ga diff=$df"
done

echo
echo "=== element region, n=$N, appended to the distill history ==="
hist=logs/bench_hist_distill.tsv
lo=$(awk -F'\t' 'NR>1 && $3 ~ /^[0-9]+$/ {if ($3+0 > m) m=$3+0} END{print m+1}' "$hist")
tmp=logs/.bench_distill_new.tsv
if bash scripts/bench_arm.sh checkpoints_distill "$N" distill "$tmp" 32 45000 "$lo"; then
    tail -n +2 "$tmp" >> "$hist"
    echo "  distill history now $(( $(wc -l < "$hist") - 1 )) rows"
else
    echo "  nothing new past step $lo; history untouched" >&2
fi
echo "LADDER + REGION DONE"
