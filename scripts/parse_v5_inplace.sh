#!/usr/bin/env bash
# In-place v5 re-parse of all batches INTO data/shards_v4 (loader accepts
# mixed v4/v5 during transition; per-replay files are overwritten atomically
# by replay-parse). Tracks .parsed_v5 markers so it never redoes a batch.
# Requires the post-0928e59 replay-parse build (ballpred self-heal linked).
set -uo pipefail
cd "$(dirname "$0")/.."
for d in data/replays/grand-champion-2/duels/batch_*; do
    b=$(basename "$d")
    [ -f "data/shards_v4/.${b}.parsed_v5" ] && continue
    n=$(ls "$d" 2>/dev/null | wc -l)
    if [ "$n" -gt 0 ]; then
        echo "$(date +%H:%M:%S) [$b] ($n) — v5 re-parsing in place..."
        nice -n 15 ./target/release/replay-parse --input-dir "$d" --output-dir data/shards_v4 \
            --reset-pool-out data/reset_pool_v5.jsonl --reset-samples-per-replay 16 \
            --min-team-size 1 --stride 8 && touch "data/shards_v4/.${b}.parsed_v5"
    fi
done
total=$(find data/replays/grand-champion-2/duels -maxdepth 1 -name 'batch_*' | wc -l)
done_n=$(find data/shards_v4 -maxdepth 1 -name '.batch_*.parsed_v5' 2>/dev/null | wc -l)
[ "$done_n" -ge "$total" ] && echo "$(date +%H:%M:%S) ALL BATCHES V5-PARSED"
echo "$(date +%H:%M:%S) v5 parse exit: $done_n/$total batches"
