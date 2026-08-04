#!/usr/bin/env bash
# Bench N evenly-spaced checkpoints from ONE lineage against element at p1.
#
# Generic counterpart to bench_distill_n8.sh, which hardcodes two arms. Used to add an arm to
# a comparison whose other arm has already been measured -- re-running a settled arm buys
# nothing and costs an hour a cell-set.
#
# Spreading the picks across the WHOLE directory rather than taking the newest N is
# deliberate: the within-arm regression of goals_against on steps is the more informative
# statistic. It answers "is this arm still improving" directly, which is the question a mean
# cannot answer -- and getting that question wrong is what made me stop the distillation run
# 28M steps early on a plateau in the wrong variable.
set -euo pipefail
cd "$(dirname "$0")/.."

DIR=${1:?usage: bench_arm.sh <checkpoint-dir> [n] [label] [out] [arenas] [steps]}
N=${2:-8}
LABEL=${3:-$(basename "$DIR")}
OUT=${4:-logs/bench_$(basename "$DIR").tsv}
ARENAS=${5:-32}
STEPS=${6:-45000}
# Optional step bounds. Benching only the region added since the last measurement avoids
# re-paying for cells already in hand; pool the old rows with the new ones afterwards for a
# full-span trend, which has a longer lever arm than either half alone.
LO=${7:-0}
HI=${8:-99999999999}

mapfile -t CKS < <(ls "$DIR"/*.pt 2>/dev/null \
  | awk -F'ck_0*' -v lo="$LO" -v hi="$HI" '{s=$2+0; if (s>=lo && s<=hi) print s"\t"$0}' | sort -n \
  | awk -v n="$N" '{a[NR]=$2} END{ if(NR<n) n=NR; for(i=0;i<n;i++) print a[1+int(i*(NR-1)/(n>1?n-1:1))] }')
[ "${#CKS[@]}" -gt 0 ] || { echo "no checkpoints in [$LO,$HI] under $DIR" >&2; exit 1; }

echo "$LABEL: ${#CKS[@]} checkpoints from $DIR"
printf 'arm\tcheckpoint\tsteps\tgoals_for\tgoals_against\tdiff\twin_share\n' > "$OUT"

for ck in "${CKS[@]}"; do
    line=$(.venv/bin/python scripts/bench_foreign.py "$ck" \
        --kind element --weights "$HOME/.cache/construct/element_weights.npz" \
        --arenas "$ARENAS" --steps "$STEPS" --seed 11 --period 1 2>/dev/null)
    gf=$(sed -nE 's/.*goals_for=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    ga=$(sed -nE 's/.*goals_against=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    df=$(sed -nE 's/.*diff=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    ws=$(sed -nE 's/.*win_share=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    st=$(basename "$ck" | sed -E 's/ck_0*([0-9]+)\.pt/\1/')
    # Loud on parse failure: a silently dropped row shrinks n without shrinking the
    # confidence anyone reads off the result.
    [ -n "$ga" ] || { echo "PARSE_FAILED $ck" >&2; gf=NA; ga=NA; df=NA; ws=NA; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$(basename "$ck")" "$st" "$gf" "$ga" "$df" "$ws" >> "$OUT"
    echo "  $(basename "$ck") ga=$ga gf=$gf"
done
echo "DONE -> $OUT"
