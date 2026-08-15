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
# Opponent. Was hardcoded to element, which reached win_share 0.945 by 366M steps past the
# fork and has ~0.05 of range left -- a saturating gauge decelerates no matter what the
# policy does, and reading that as a plateau is a mistake this project has already made
# twice. necto sits at 0.663 and is the current maximum-resolution opponent.
KIND=${9:-necto}

mapfile -t CKS < <(ls "$DIR"/*.pt 2>/dev/null \
  | awk -F'ck_0*' -v lo="$LO" -v hi="$HI" '{s=$2+0; if (s>=lo && s<=hi) print s"\t"$0}' | sort -n \
  | awk -v n="$N" '{a[NR]=$2} END{ if(NR<n) n=NR; for(i=0;i<n;i++) print a[1+int(i*(NR-1)/(n>1?n-1:1))] }')
[ "${#CKS[@]}" -gt 0 ] || { echo "no checkpoints in [$LO,$HI] under $DIR" >&2; exit 1; }

# How many times to retry a cell that comes back CONTAMINATED. The failure is a contained
# physics blowup shredding one arena's matches, it is nondeterministic under ASLR, and a
# re-run usually lands clean -- so a retry is far cheaper than either dropping the cell
# (which shrinks n without shrinking anyone's confidence in the result) or keeping it
# (which is what silently corrupted the 2026-08-08 distill numbers).
RETRIES=${BENCH_RETRIES:-2}

# ...but NOT every contaminated cell is nondeterministic. Against necto our nexto-clone hits
# the mirrored-KICKOFF pinch essentially every match (short_frac 0.82 and 0.96 on two separate
# runs), so retrying just pays three times for the same failure. A cell is SALVAGEABLE when
# enough FULL matches survived to measure on: 270 full matches out of an expected ~324 is a
# perfectly good sample even though 96% of the RECORDS were fragments. Retry only when the
# full-match sample is too thin to use.
MIN_FULL=${BENCH_MIN_FULL:-100}

echo "$LABEL vs $KIND: ${#CKS[@]} checkpoints from $DIR"
# The _full columns are always emitted, never substituted into goals_for/goals_against: two
# estimators in one column is how a series stops meaning one thing. Read goals_against when
# short_frac is low, goals_against_full when it is not, and never mix them within a trend.
printf 'arm\tcheckpoint\tsteps\tgoals_for\tgoals_against\tdiff\twin_share\tshort_frac\tgoals_for_full\tgoals_against_full\tn_full\n' > "$OUT"

for ck in "${CKS[@]}"; do
    # NOTE 2>&1, not 2>/dev/null: the engine prints its blowup containments to stderr, and
    # throwing them away is half the reason this went unnoticed for so long. bench_foreign
    # prints its own census to stdout, so the parse below does not depend on stderr, but a
    # human reading the log should see the storm that produced a bad cell.
    for attempt in $(seq 0 "$RETRIES"); do
        line=$(.venv/bin/python scripts/bench_foreign.py "$ck" \
            --kind "$KIND" --weights "$HOME/.cache/construct/${KIND}_weights.npz" \
            --arenas "$ARENAS" --steps "$STEPS" --seed 11 --period 1 2>&1)
        sf=$(sed -nE 's/.*short_frac=([0-9.]+).*/\1/p' <<<"$line" | head -1)
        nfull=$(sed -nE 's/.*n_full=([0-9]+).*/\1/p' <<<"$line" | head -1)
        grep -q CONTAMINATED <<<"$line" || break
        # Salvageable despite the flag: enough full matches survived to measure on.
        [ "${nfull:-0}" -ge "$MIN_FULL" ] && {
            echo "  $(basename "$ck") contaminated (short_frac=$sf) but n_full=$nfull -- salvaged" >&2
            break
        }
        echo "  $(basename "$ck") CONTAMINATED (short_frac=$sf, n_full=${nfull:-0}), retry $((attempt+1))/$RETRIES" >&2
    done
    gff=$(sed -nE 's/.*goals_for_full=([0-9.]+|NA).*/\1/p'     <<<"$line" | head -1)
    gaf=$(sed -nE 's/.*goals_against_full=([0-9.]+|NA).*/\1/p' <<<"$line" | head -1)
    if grep -q CONTAMINATED <<<"$line" && [ "${nfull:-0}" -lt "$MIN_FULL" ]; then
        # Written to the table with its flag rather than dropped, so a downstream reader
        # sees the hole instead of inferring a clean n.
        st=$(basename "$ck" | sed -E 's/ck_0*([0-9]+)\.pt/\1/')
        printf '%s\t%s\t%s\tNA\tNA\tNA\tNA\t%s\t%s\t%s\t%s\n' \
            "$LABEL" "$(basename "$ck")" "$st" "${sf:-NA}" "${gff:-NA}" "${gaf:-NA}" "${nfull:-0}" >> "$OUT"
        echo "  $(basename "$ck") STILL CONTAMINATED after $RETRIES retries, only ${nfull:-0} full matches -- row is NA" >&2
        continue
    fi
    gf=$(sed -nE 's/.*goals_for=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    ga=$(sed -nE 's/.*goals_against=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    # NOTE the leading [+-]?: bench_foreign prints diff=+0.2062 once we OUTSCORE the bot, and
    # a [0-9.-] class silently drops it -- a parse bug that can only ever fire on success.
    df=$(sed -nE 's/.*diff=([+-]?[0-9.]+).*/\1/p' <<<"$line" | head -1)
    ws=$(sed -nE 's/.*win_share=([0-9.-]+).*/\1/p' <<<"$line" | head -1)
    st=$(basename "$ck" | sed -E 's/ck_0*([0-9]+)\.pt/\1/')
    # Loud on parse failure: a silently dropped row shrinks n without shrinking the
    # confidence anyone reads off the result.
    [ -n "$ga" ] || { echo "PARSE_FAILED $ck" >&2; gf=NA; ga=NA; df=NA; ws=NA; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$(basename "$ck")" "$st" \
        "$gf" "$ga" "$df" "$ws" "${sf:-NA}" "${gff:-NA}" "${gaf:-NA}" "${nfull:-NA}" >> "$OUT"
    echo "  $(basename "$ck") ga=$ga gf=$gf short_frac=${sf:-NA} ga_full=${gaf:-NA} n_full=${nfull:-NA}"
done
echo "DONE -> $OUT"
