#!/usr/bin/env bash
# v4 re-parse of COMPLETE batches only (10k files), into data/shards_v4.
# Tracks done batches via .parsed_v4 markers so it never re-parses a batch.
# Run repeatedly (or loop) while the pull-only top-up fills the rest.
set -uo pipefail
cd "$(dirname "$0")/.."
for pass in $(seq 1 40); do
    did=0
    for d in data/replays/grand-champion-2/duels/batch_*; do
        b=$(basename "$d")
        [ -f "data/shards_v4/.${b}.parsed_v4" ] && continue
        n=$(ls "$d" 2>/dev/null | wc -l)
        # pull-only top-up finished 2026-07-16 (GC2 CORPUS COMPLETE): every batch is
        # final, so parse any non-empty unmarked batch (was: -ge 10000 while pulling)
        if [ "$n" -gt 0 ]; then
            echo "$(date +%H:%M:%S) [$b] final ($n) — v4 parsing..."
            nice -n 10 ./target/release/replay-parse --input-dir "$d" --output-dir data/shards_v4 \
                --reset-pool-out data/reset_pool_v4.jsonl --reset-samples-per-replay 16 \
                --min-team-size 1 --stride 8 && touch "data/shards_v4/.${b}.parsed_v4"
            did=1
        fi
    done
    # exit when all batches are parsed
    total=$(ls -d data/replays/grand-champion-2/duels/batch_* | wc -l)
    done_n=$(ls data/shards_v4/.batch_*.parsed_v4 2>/dev/null | wc -l)
    [ "$done_n" -ge "$total" ] && { echo "$(date +%H:%M:%S) ALL BATCHES V4-PARSED"; break; }
    [ "$did" = "0" ] && sleep 300  # nothing ready; wait for downloads
done
echo "$(date +%H:%M:%S) v4 parse loop exit: $done_n/$total batches"
