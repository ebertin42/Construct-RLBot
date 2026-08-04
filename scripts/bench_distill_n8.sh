#!/usr/bin/env bash
# n=8 vs n=8: settle the distillation effect size that Gate 1 measured at n=1.
#
# WHY THIS IS NEEDED. The Gate 1 PASS (goals_against 12.866 -> 11.738) rests on ONE distill
# checkpoint against ONE baseline checkpoint. bench-checkpoint-oscillation established that
# the oscillation in this bench lives on CHECKPOINTS, not seeds -- so re-running one
# checkpoint with more seeds buys nothing and a single-checkpoint effect size is not settled.
#
# MDE COMPUTED BEFORE LAUNCHING, per two-lineage-pairing-is-not-checkpoint-pairing:
#   goals_against CV is 0.033-0.060 (goals-against-is-the-gate); at ga ~12 that is sd ~0.55.
#   These are TWO LINEAGES, not paired checkpoints, so se = sqrt(2)*sd/sqrt(n).
#   n=8  ->  se = 1.414 * 0.55 / 2.828 = 0.275,  MDE = 2.8 * se = 0.77 goals.
# The observed effect is 1.13, which clears 0.77. At n=4 the MDE would be 1.09 -- close enough
# to the effect to be a coin flip, which is exactly how the run-B claim died.
#
# ARM A: distill checkpoints from the CONVERGED region only (fd_ce flat at ~43% from iter
# 1600 == step ~3.273e9). Including the climb would average a moving policy with a settled one.
# ARM B: run A checkpoints bracketing the fork, so the comparison isolates distillation rather
# than the 60M steps run A has trained since.
#
# Sequential cells throughout: concurrent cells contend and bias the result.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-8}
ARENAS=${2:-32}
STEPS=${3:-45000}
OUT=${4:-logs/bench_n8.tsv}
CONVERGED_FROM=3273000000
FORK_LO=3200000000
FORK_HI=3260000000

pick() {  # dir lo hi n  -> n evenly-spaced checkpoints in [lo,hi] by step
    ls "$1"/*.pt 2>/dev/null \
      | awk -F'ck_0*' -v lo="$2" -v hi="$3" '{s=$2+0; if (s>=lo && s<=hi) print s"\t"$0}' \
      | sort -n | awk -v n="$4" '{a[NR]=$2} END{ if(NR<n) n=NR; for(i=0;i<n;i++) print a[1+int(i*(NR-1)/(n>1?n-1:1))] }'
}

mapfile -t ARM_A < <(pick checkpoints_distill "$CONVERGED_FROM" 99999999999 "$N")
mapfile -t ARM_B < <(pick checkpoints_v9      "$FORK_LO"        "$FORK_HI"   "$N")

echo "arm A (distill, converged): ${#ARM_A[@]} checkpoints"
echo "arm B (run A, fork window): ${#ARM_B[@]} checkpoints"
printf 'arm\tcheckpoint\tgoals_for\tgoals_against\tdiff\twin_share\n' > "$OUT"

run_cell() {  # arm ckpt
    local line
    line=$(.venv/bin/python scripts/bench_foreign.py "$2" \
        --kind element --weights "$HOME/.cache/construct/element_weights.npz" \
        --arenas "$ARENAS" --steps "$STEPS" --seed 11 --period 1 2>/dev/null)
    local gf ga df ws
    gf=$(sed -nE 's/.*goals_for=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    ga=$(sed -nE 's/.*goals_against=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    df=$(sed -nE 's/.*diff=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    ws=$(sed -nE 's/.*win_share=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    # A cell that fails to parse must be LOUD, not silently dropped: a missing row would
    # shrink n without shrinking the confidence anyone reads off the result.
    [ -n "$ga" ] || { echo "PARSE_FAILED $2" >&2; ga=NA; gf=NA; df=NA; ws=NA; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$(basename "$2")" "$gf" "$ga" "$df" "$ws" >> "$OUT"
    echo "  $1 $(basename "$2") ga=$ga gf=$gf"
}

for ck in "${ARM_A[@]}"; do run_cell distill "$ck"; done
for ck in "${ARM_B[@]}"; do run_cell runA    "$ck"; done
echo "DONE -> $OUT"
