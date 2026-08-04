#!/usr/bin/env bash
# Bench the region each live arm has ADDED since its last measurement.
#
# SEQUENTIAL, one arm then the other, one cell at a time: concurrent cells contend for the
# box and bias the result, and the whole point of these numbers is that they are comparable
# to the ones already recorded.
#
# Bounds are the last checkpoint each arm was benched at, so nothing is paid for twice:
#   ppo_distilled  last benched to 3,322,470,400  (n=8 mean goals_against 9.868)
#   distill        last benched to 3,300,874,240  (n=8 mean goals_against 11.101)
#
# WHY THIS RUNS AT ALL, given fd_pct is visible on the dashboard: fd_pct is the clone
# completeness and it is NOT the stopping signal. The distillation run was cut 28M steps
# early because fd_ce plateaued at 43% while goals_against was still falling -0.061/Mstep.
# goals_against costs a bench to read; that is the price of the only number that decides.
set -euo pipefail
cd "$(dirname "$0")/.."

N=${1:-8}
bash scripts/bench_arm.sh checkpoints_ppo_distilled "$N" ppo_new  logs/bench_ppo_new.tsv     32 45000 3322470401
bash scripts/bench_arm.sh checkpoints_distill       "$N" dist_new logs/bench_distill_new.tsv 32 45000 3300874241
echo "BOTH DONE"
