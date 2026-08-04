#!/usr/bin/env bash
# The arm the distillation work was actually for: optimise for WINNING, from a policy that
# already has the repertoire.
#
# Pure distillation's ceiling is "incomplete nexto clone" -- it optimises mimicry, and it
# converged at 43% below the action marginal, which bought goals_against 12.87 -> 11.74 and
# nothing like a win. What it DID buy is the thing PPO could never get for itself: the policy
# now emits ~12 flips/min, so the dodge composite is no longer ~1e-7 per decision and policy
# gradient can finally SAMPLE the sequences it is supposed to reweight.
#
# TWO PHASES, AND PHASE 1 IS NOT OPTIONAL.
#
#   Phase 1 -- CRITIC REPAIR. The distillation run trained with value_coef=0, so the value
#   head has not been updated for 58M steps while the policy changed out from under it. Its
#   predictions are for a policy that no longer exists. PPO on a stale critic optimises
#   garbage advantages and looks perfectly healthy doing it -- the same failure mode
#   regime-change-needs-contested-start describes. So: policy_coef=0, value_coef=1, and the
#   distillation term left ON at full strength to hold the policy still while the critic
#   catches up. Nothing about the policy changes in this phase by construction.
#
#   Phase 2 -- PPO with an anchor. policy_coef=1, and foreign_distill_coef dropped to 0.1
#   rather than 0: at 0 the policy gradient is free to unlearn the repertoire immediately,
#   which is the entropy-lets-dominant-regime-win shape (one regime with the experience
#   majority wins). The anchor is the same idea as the KL prior, using the teacher we already
#   have wired.
#
# JUDGE ON: element p1 goals-against against 11.738 (the distilled level, NOT run A's 12.866
# -- the question here is whether PPO adds to distillation, not whether distillation worked).
# And on win_share, which has been 0.0000-0.0266 at p1 for this project's entire history.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${1:?usage: launch_ppo_from_distilled.sh <converged-distill-checkpoint>}
WARM_ITERS=${2:-150}
DIR_WARM=checkpoints_ppo_warm
DIR_MAIN=checkpoints_ppo_distilled
mkdir -p logs

if pgrep -f "checkpoint-dir $DIR_MAIN" >/dev/null || pgrep -f "checkpoint-dir $DIR_WARM" >/dev/null; then
    echo "REFUSING: this arm is already alive" >&2; exit 1
fi
test -f "$BASE" || { echo "missing $BASE" >&2; exit 1; }

echo "=== phase 1: critic repair ($WARM_ITERS iters, policy frozen) ==="
.venv/bin/python scripts/resume_train.py "$BASE" \
    --config configs/train_v9_fromscratch.toml \
    --foreign-teacher nexto --foreign-distill-coef 1.0 \
    --policy-coef 0 --value-coef 1.0 --entropy-coef 0 \
    --lr 3e-4 --lr-final 3e-4 --reset-optimizer \
    --team-sizes 1,0,0 --num-arenas 48 --seed 31 \
    --checkpoint-dir "$DIR_WARM" --max-iterations "$WARM_ITERS" \
    > logs/ppo_warm.log 2>&1

WARMED=$(ls -t "$DIR_WARM"/*.pt 2>/dev/null | head -1)
test -n "$WARMED" || { echo "phase 1 produced no checkpoint; see logs/ppo_warm.log" >&2; exit 1; }
echo "phase 1 done -> $WARMED"

echo "=== phase 2: PPO with distillation anchor ==="
nohup .venv/bin/python scripts/resume_train.py "$WARMED" \
    --config configs/train_v9_fromscratch.toml \
    --foreign-teacher nexto --foreign-distill-coef 0.1 \
    --policy-coef 1.0 --value-coef 1.0 --entropy-coef 0.006 \
    --lr 1e-4 --lr-final 1e-4 --reset-optimizer \
    --team-sizes 1,0,0 --num-arenas 48 --seed 31 \
    --checkpoint-dir "$DIR_MAIN" \
    > logs/ppo_distilled.log 2>&1 &

echo "phase 2 pid $!  log logs/ppo_distilled.log"
