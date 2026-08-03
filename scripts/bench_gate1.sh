#!/usr/bin/env bash
# Gate 1 verdict cell: does the distilled policy DEFEND better, or just flip more?
#
# The behavioural numbers (flips 0.00 -> ~13/min, air_duty 0.117 -> ~0.38) prove only that the
# actor MOVED. They cannot distinguish skilled dodging from flip spam, and a policy that
# matches a teacher's action MARGINAL emits flips at roughly the right rate with none of the
# timing. replay-BC scored 0.642 top-1 and went 0W/1D/639L; behavioural mimicry is that same
# failure wearing a better costume. Only the closed loop decides.
#
# BASELINE TO BEAT (run A ck_003242526720, element, period 1, 32 arenas x 45000, seed 11):
#     win_share 0.0031   goals_for 2.219   goals_against 12.866   diff -10.647
# goals_against is the gate: CV 0.033-0.060 vs win_share's 0.217-0.307, so it is 2.4-3.4x
# tighter and win_share is censored near zero anyway.
#
# SEQUENTIAL CELLS, one bot at a time -- concurrent cells contend and bias the result.
set -euo pipefail
cd "$(dirname "$0")/.."

CK=${1:?usage: bench_gate1.sh <checkpoint> [arenas] [steps]}
ARENAS=${2:-32}
STEPS=${3:-45000}

for KIND in element nexto; do
    echo "=== $KIND period 1 | $(basename "$CK") | ${ARENAS}x${STEPS} ==="
    .venv/bin/python scripts/bench_foreign.py "$CK" \
        --kind "$KIND" --weights "$HOME/.cache/construct/${KIND}_weights.npz" \
        --arenas "$ARENAS" --steps "$STEPS" --seed 11 --period 1 \
        2>/dev/null | grep -vE "^\[|btRS|Ray casts"
done
