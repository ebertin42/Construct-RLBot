#!/usr/bin/env bash
# Gate 1: PURE on-policy distillation from nexto into run A's policy.
#
# The question this run exists to answer: can the actor be moved toward a strong policy's
# behaviour at all? Everything else about the project's causal chain is settled -- the reward
# tape is aligned (element out-earns us on our own dense terms while beating us 12.9-2.2), the
# critic prices position correctly, and p1 has never been trained. What is left is an actor
# that never EMITS the behaviour: flip_events 0.104 vs 45.660, a factor of 439.
#
# WHY EVERY COEFFICIENT IS ZERO. extra_loss_fn only ADDS to the PPO loss, so without
# policy_coef=0 the policy gradient competes with the teacher for the whole run and the result
# is a mixture nobody intended. value_coef=0 and entropy_coef=0 for the same reason: the only
# gradient in this run is the distillation cross-entropy.
#
# WHY THE LR IS PINNED AT BOTH ENDS. lr anneals 3e-4 -> 1e-4 over [200M, 600M] steps and we
# resume at 3.24B, so the schedule has fully annealed; passing --lr alone would still resolve
# to 1e-4. --lr-final 3e-4 makes it constant.
#
# WHY --reset-optimizer. The Adam moments carried in the checkpoint were accumulated against
# PPO's policy gradient. This is a different objective; stale first/second moments would
# misdirect exactly the early steps that decide whether this works.
#
# VERDICT IS CLOSED-LOOP, NOT AGREEMENT. Judge on element p1 goals-against via
# bench_foreign.py on an idle box. Do NOT judge on action agreement: replay-BC scored 0.642
# top-1 and went 0W/1D/639L.
set -euo pipefail
cd "$(dirname "$0")/.."

# Defaults to the original fork point. Pass a later distill checkpoint to RESUME the same
# lineage -- which is the normal case now: the first run was stopped at 28M steps while
# goals_against was still falling -0.061/Mstep, because fd_ce had plateaued and I used that as
# the stopping signal instead of the metric the project is judged on.
CK=${1:-checkpoints_v9/ck_003242526720.pt}
LOG=${2:-logs/distill_gate1.log}
mkdir -p logs

if pgrep -f "checkpoint-dir checkpoints_distill" >/dev/null; then
    echo "REFUSING: a distill run is already alive" >&2
    exit 1
fi
test -f "$CK" || { echo "missing checkpoint $CK" >&2; exit 1; }
test -f "$HOME/.cache/construct/nexto_weights.npz" || { echo "missing nexto weights" >&2; exit 1; }

nohup .venv/bin/python scripts/resume_train.py "$CK" \
    --config configs/train_v9_fromscratch.toml \
    --foreign-teacher nexto --foreign-distill-coef 1.0 \
    --policy-coef 0 --value-coef 0 --entropy-coef 0 \
    --lr 3e-4 --lr-final 3e-4 --reset-optimizer \
    --team-sizes 1,0,0 --num-arenas 48 --seed 21 \
    --checkpoint-dir checkpoints_distill \
    > "$LOG" 2>&1 &

echo "gate1 pid $!  log $LOG"
