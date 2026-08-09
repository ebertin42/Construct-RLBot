#!/usr/bin/env bash
# MULTI-TEACHER DISTILLATION: nexto 0.75 / immortal 0.25, forked from the live distill arm.
#
# WHY THIS ARM EXISTS. Two PPO variants on top of the distilled policy both made it WORSE
# (dense tape t=+3.63, goal-only t=+4.90), at opposite ends of the reward-resolution
# spectrum, so reward shape was not the obstacle and RL-beyond-the-teacher is not currently
# available. That leaves raising what imitation itself can reach. `Engine::set_teacher`
# already takes a kind, so a second teacher is a config change rather than an engine change.
#
# WHICH BOTS CAN TEACH -- MEASURED, not assumed. The label is an index into OUR action table
# and a teacher whose controls fall off it emits -1 and the frame is dropped. On
# ck_004068864000, 4 arenas, 2 iterations:
#
#     nexto 1.000    immortal 0.87    necto 0.24    element 0.20
#
# nexto shares our table by construction. immortal's own table is 126 rows and overlaps ours
# far more than that separate provenance suggests -- reading the source, I predicted it would
# be near zero and it is 0.87. necto and element are EXCLUDED on those numbers: a 0.2 label
# rate is not a random subsample, it is precisely the frames our action table cannot express,
# so training on it would teach a biased subset while every dashboard number looked healthy.
#
# THE HONEST PRIOR IS THAT THIS DOES NOT WORK, AND IT IS STILL WORTH ONE ARM. Three reasons
# to expect neutral-to-negative:
#   1. We already BEAT immortal (win_share 0.819 at the last ladder). Its labels are, on
#      average, worse moves than nexto's.
#   2. The student SAMPLES its action (per-arena action-sample RNG in engine.rs), it does not
#      argmax. So a 0.25 weight is not merely reshaping the tail of the target -- at
#      convergence it means literally playing immortal's move a quarter of the time on the
#      frames where the two disagree.
#   3. A hard-label mixture RAISES the irreducible CE floor by up to ln2 * P(disagree). The
#      clone target gets strictly fuzzier, so fd_ce will rise and that rise is NOT evidence
#      of a fault.
# Against that, the one mechanism by which it could pay: immortal is an aerial-heavy bot and
# our air_duty sits 3.4-5.6x below the bench field, so its labels may cover states nexto's
# labels reach too rarely to teach. That is a coverage argument, not a strength argument, and
# coverage arguments are exactly the kind this project settles with a bench rather than an
# opinion.
#
# PRE-REGISTERED FALSIFIER, written BEFORE the run as the goalonly arm's was:
#   Control is the distill arm over the SAME step window on the SAME gauge (necto, period 1,
#   32 arenas x 45000, seed 11), n=8 each. Two-lineage pairing inflates se by sqrt(2), and at
#   the sd this gauge has shown (1.22 / 0.82) the MDE on goals_against is ~1.45 -- so this
#   arm is powered ONLY for a large effect, against a control currently conceding 2.65.
#     * goals_against WORSE by more than 1.45  -> multi-teacher is harmful, retire it, and
#       the imitation-ceiling story needs a teacher BETTER than nexto, not more of them.
#     * goals_against BETTER by more than 1.45 -> coverage was real; try immortal weight 0.5.
#     * inside +/-1.45 -> NULL at this power. Do not spend a second window chasing it: an
#       effect smaller than 1.45 goals is not what is standing between us and nexto, which
#       beats us by 8.2.
#   fd_ce is NOT a stopping signal and not a verdict. The distillation run was once cut 28M
#   steps early on an fd_ce plateau while goals_against was still falling -0.061/Mstep.
#
# ONE VARIABLE. Identical to the live distill command in every other respect -- same config,
# same lr 3e-4 flat, policy/value/entropy all 0 (pure supervised), same 48 arenas, same
# seed 21, same 1v1 mix -- so the only difference between the arms is the teacher mixture.
#
# ============================ RESULT, 2026-08-09: FAILED ============================
# Launched from ck_004070379520 and killed the same day at 79M steps. It did NOT test the
# hypothesis above -- it found a bug in HOW the mixture was delivered.
#
#   entropy 1.590 -> 2.565 within FIVE iterations, touches/min 43 -> 20, and it never
#   recovered over 79M steps (nexto fd_ce flat at 2.03 -> 2.09 vs the control's 1.63).
#   Benched vs necto at p1: 0W/0D/16L where the control read 16W/0D/0L.
#
# Two 30-iteration probes from the same base isolated the cause:
#   B  mixture, NO --reset-optimizer   ent 1.55 -> 2.57 by iter 5,  0W/0D/16L, ga 10.06
#   C  pure nexto, WITH --reset-optimizer  ent flat 1.61-1.67,     13W/2D/1L,  ga  3.88
# So the optimizer reset is exonerated and the per-iteration teacher rotation is the cause:
# one whole iteration of immortal is epochs x minibatches of FULL-STRENGTH steps toward a
# target 4.9 nats away, not a small step in the mixture direction. See Trainer._pick_teacher.
#
# A faithful test needs both teachers' labels in ONE batch -- engine holds K teachers and
# collect emits (T, N_learner, K). That is a Rust change and a wheel ship. Do not relaunch
# this script expecting a verdict on multi-teacher; it can only reproduce the failure.
# ====================================================================================
set -euo pipefail
cd "$(dirname "$0")/.."

# Refuse by default. The script is kept because reproducing the failure is occasionally
# useful and because the falsifier above is the record of what was predicted; it is not kept
# because it is safe to run.
if [ "${I_KNOW_THIS_ESTIMATOR_IS_BROKEN:-0}" != "1" ]; then
    echo "REFUSING: this arm was measured to destroy the policy on 2026-08-09 (see the" >&2
    echo "header). Re-run only with I_KNOW_THIS_ESTIMATOR_IS_BROKEN=1, and only to" >&2
    echo "reproduce the failure -- not to get a verdict on multi-teacher." >&2
    exit 1
fi

BASE=${1:?usage: launch_multiteacher.sh <distill-checkpoint> [mix]}
MIX=${2:-nexto:3,immortal:1}
DIR=checkpoints_mt
mkdir -p logs

# pgrep self-matches its own command line; check the checkpoint dir is not already owned.
if pgrep -f "checkpoint-dir $DIR" | grep -qv "^$$\$"; then
    echo "REFUSING: an arm already owns $DIR" >&2; exit 1
fi
test -f "$BASE" || { echo "missing $BASE" >&2; exit 1; }
for k in $(echo "$MIX" | tr ',' ' ' | sed 's/:.*//'); do
    test -f "$HOME/.cache/construct/${k}_weights.npz" \
        || { echo "missing teacher weights for $k" >&2; exit 1; }
done

echo "=== multi-teacher distillation: $MIX, from $BASE ==="
nohup .venv/bin/python scripts/resume_train.py "$BASE" \
    --config configs/train_v9_fromscratch.toml \
    --foreign-teachers "$MIX" --foreign-distill-coef 1.0 \
    --policy-coef 0 --value-coef 0 --entropy-coef 0 \
    --lr 3e-4 --lr-final 3e-4 --reset-optimizer \
    --team-sizes 1,0,0 --num-arenas 48 --seed 21 \
    --checkpoint-dir "$DIR" \
    > logs/mt.log 2>&1 &
echo "pid $!  log logs/mt.log  dir $DIR"
