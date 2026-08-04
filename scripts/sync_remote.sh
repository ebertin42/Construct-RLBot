#!/usr/bin/env bash
# Pull checkpoints + training log from the remote training box every 60s so
# the local dashboard, viewer loop, and eval runner follow the main run.
# Usage: ./scripts/sync_remote.sh [user@host] [remote_dir]
set -uo pipefail
cd "$(dirname "$0")/.."
HOST="${1:-elliot@192.168.86.117}"
RDIR="${2:-construct}"

mkdir -p checkpoints checkpoints_entity checkpoints_scratch checkpoints_v9 \
         checkpoints_v10 checkpoints_v11 checkpoints_v12 league \
         checkpoints_distill checkpoints_ppo_distilled
while true; do
    # THE TWO LIVE ARMS since 2026-08-04, pulled FIRST because everything downstream
    # (dashboard, watch_loop) picks the newest checkpoint by mtime across these dirs and
    # run A -- which was that newest thing for a week -- is now retired and frozen.
    #
    #   checkpoints_distill      pure distillation from nexto (policy_coef=0). RESUMED from
    #                            ck_003300874240 after being stopped 28M steps early: fd_ce
    #                            had plateaued but goals_against was still falling
    #                            -0.061/Mstep. Tests whether that keeps paying.
    #   checkpoints_ppo_distilled  PPO on top of the distilled policy, with a 0.1
    #                            distillation anchor. THE ARM THAT WINS: 30W/12D/278L vs
    #                            element at p1, from a project history of 0W/0D/320L.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_ppo_distilled/" checkpoints_ppo_distilled/ 2>/dev/null
    rsync -az "$HOST:$RDIR/logs/ppo_distilled.log" checkpoints_ppo_distilled/train_remote.log 2>/dev/null
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_distill/" checkpoints_distill/ 2>/dev/null
    rsync -az "$HOST:$RDIR/logs/distill_resume.log" checkpoints_distill/train_remote.log 2>/dev/null
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints/" checkpoints/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints/train_v0.log" checkpoints/train_remote.log 2>/dev/null
    # entity-transformer lineage (kickstart run) — separate dir, own log
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_entity/" checkpoints_entity/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_entity/train_v1.log" checkpoints_entity/train_remote.log 2>/dev/null
    # v9 — THE CURRENT MAIN RUN (2026-07-26): fresh net, 104-row air action table,
    # equal 1s/2s/3s. Pulled first because everything downstream (dashboard,
    # watch_loop) picks the newest checkpoint by mtime across these dirs.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_v9/" checkpoints_v9/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_v9/v9_s20260726.log" checkpoints_v9/train_remote.log 2>/dev/null
    # v10 — the ENTROPY FORK of v9 (2026-07-30), forked at 2,087.6M. Same reward
    # tape as v9; entropy_coef is the only difference (0.0142 vs v9's 0.006).
    # Synced because the whole point of the fork is comparing the two arms on the
    # bench, and the bench runs LOCALLY -- without this, run B's checkpoints exist
    # only on the box and the comparison is impossible.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_v10/" checkpoints_v10/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_v10/v10_s20260730.log" checkpoints_v10/train_remote.log 2>/dev/null
    # v11 — the NULL CONTROL for the planar-air flag (2026-07-31). Forked from
    # checkpoints_v10/ck_002278225920.pt, i.e. run B's OWN starting weights, but with
    # v9's control tape. B and C therefore share weights AND lineage history and differ
    # only in vel_to_ball_planar_air, so B-vs-C at matched steps isolates the flag.
    # (B-vs-A never could: v9's and v10's copies of that checkpoint have DIFFERENT
    # sha256, so run B never forked from run A.) Synced for the same reason as v10 --
    # the bench runs LOCALLY, so without this the comparison is impossible.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_v11/" checkpoints_v11/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_v11/v11_s20260731.log" checkpoints_v11/train_remote.log 2>/dev/null
    # v12 — RUN D, the AUX-HEADS arm (2026-08-01). Forked from run A's own
    # ck_002780933120.pt (sha 73d51f86...), same reward tape, same curriculum, same
    # entropy: it differs from A ONLY by the aux losses. Synced for the same reason v10
    # and v11 were -- the bench runs LOCALLY, so without this the A-vs-D comparison is
    # impossible. Note v10/v11 are RETIRED and no longer advance; they keep syncing only
    # so their history stays pullable.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_v12/" checkpoints_v12/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_v12/v12_s20260801.log" checkpoints_v12/train_remote.log 2>/dev/null
    # v8 from-scratch run — RETIRED 2026-07-26 at 1.589B (superseded by v9). Kept
    # syncing so its final checkpoints stay pullable; delete once archived.
    rsync -az --include='ck_*.pt' --exclude='*' "$HOST:$RDIR/checkpoints_scratch/" checkpoints_scratch/ 2>/dev/null
    rsync -az "$HOST:$RDIR/checkpoints_scratch/fromscratch_s20260723.log" checkpoints_scratch/train_remote.log 2>/dev/null
    # remote league ladder (mixed v0/v1 pool) — dashboard reads registry_remote.jsonl
    rsync -az "$HOST:$RDIR/league/registry.jsonl" league/registry_remote.jsonl 2>/dev/null
    sleep 60
done
