#!/usr/bin/env bash
# PHASE 2: PPO on GOAL DIFFERENTIAL ONLY, from the distilled policy.
#
# The one experiment that can carry this project past its teacher. Distillation is delivering
# hard against the field -- immortal 0.508 -> 0.819 and necto 0.487 -> 0.663 over 204M steps
# -- but against NEXTO ITSELF it is flat: 0.0016 -> 0.0094, t = 1.34, not significant. Half
# the clone buys dominance over three bots and ~1% win share against the teacher, so the
# remaining half is where all of nexto's strength lives and imitation is not on a trajectory
# to reach it.
#
# WHY GOAL-ONLY, AND WHY THIS IS NOT THE ARM THAT ALREADY FAILED. The retired PPO arm
# optimised the DENSE tape, and that tape was measured: across 16 checkpoints spanning
# goals_against 10.1 -> 6.6 its ep_rew moved 474.9 -> 504.8. Right direction (r = -0.54),
# 29% of outcome variance, 6% dynamic range against a 36% outcome change. A gradient that
# blunt cannot tell a good policy from a bad one, so PPO wandered and ended up measurably
# WORSE than pure distillation (t = +3.63). Goal differential has perfect resolution -- it IS
# the objective -- and was untrainable only because it is sparse and exploration could never
# reach good play. Distillation removed that obstacle: ~11 flips/min, element 297-16.
#
# PHASE 1 IS NOT OPTIONAL. The distillation run trains with value_coef=0, so its value head
# has gone hundreds of millions of steps without an update while the policy changed
# completely. PPO on a stale critic optimises garbage advantages and every stat looks fine
# while it happens. This was measured the first time round: v_loss fell 3.4-3.7 -> 1.5-2.1
# once value_coef was restored. Freeze the policy, let the critic catch up, then release it.
#
# JUDGE ON necto (currently 0.663, the maximum-resolution opponent). NOT element, which is at
# 0.939 with ~0.06 of range left. Baseline to beat is the distill arm measured on the same
# gauge over the same step window -- never a cross-opponent or cross-step-count comparison,
# which is how "PPO adds 1.23 goals" got retracted and then inverted.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE=${1:?usage: launch_goalonly.sh <distill-checkpoint> [warm_iters]}
WARM=${2:-150}
DIR_WARM=checkpoints_go_warm
DIR_MAIN=checkpoints_goalonly
REWARD=configs/reward_v13_goalonly.toml
mkdir -p logs

pgrep -f "checkpoint-dir $DIR_MAIN" >/dev/null && { echo "REFUSING: arm already alive" >&2; exit 1; }
test -f "$BASE"   || { echo "missing $BASE" >&2; exit 1; }
test -f "$REWARD" || { echo "missing $REWARD" >&2; exit 1; }

echo "=== phase 1: critic repair on the GOAL-ONLY tape ($WARM iters, policy frozen) ==="
# Critic is repaired against the objective it will actually serve. Repairing it on the dense
# tape and then switching would hand phase 2 a critic fitted to a different return.
.venv/bin/python scripts/resume_train.py "$BASE" \
    --config configs/train_v9_fromscratch.toml --reward-config "$REWARD" \
    --foreign-teacher nexto --foreign-distill-coef 1.0 \
    --policy-coef 0 --value-coef 1.0 --entropy-coef 0 \
    --lr 3e-4 --lr-final 3e-4 --reset-optimizer \
    --team-sizes 1,0,0 --num-arenas 40 --seed 41 \
    --checkpoint-dir "$DIR_WARM" --max-iterations "$WARM" \
    > logs/goalonly_warm.log 2>&1

WARMED=$(ls -t "$DIR_WARM"/*.pt 2>/dev/null | head -1)
test -n "$WARMED" || { echo "phase 1 produced no checkpoint; see logs/goalonly_warm.log" >&2; exit 1; }
echo "phase 1 done -> $WARMED"

echo "=== phase 2: goal-only PPO, distillation anchor at 0.1 ==="
nohup .venv/bin/python scripts/resume_train.py "$WARMED" \
    --config configs/train_v9_fromscratch.toml --reward-config "$REWARD" \
    --foreign-teacher nexto --foreign-distill-coef 0.1 \
    --policy-coef 1.0 --value-coef 1.0 --entropy-coef 0.004 \
    --lr 1e-4 --lr-final 1e-4 --reset-optimizer \
    --team-sizes 1,0,0 --num-arenas 40 --seed 41 \
    --checkpoint-dir "$DIR_MAIN" \
    > logs/goalonly.log 2>&1 &
echo "phase 2 pid $!  log logs/goalonly.log"
