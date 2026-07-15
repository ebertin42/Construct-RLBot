#!/usr/bin/env bash
# Build the full GC2-duels BC corpus: pull each HF batch, parse to 15 Hz shards,
# append reset states. Sequential per batch, resumable at every level (pull
# skips cached files, orchestrated parse skips replays with existing sidecars).
# `timeout` guards the pull: snapshot_download can hang retrying files missing
# from the HF repo (batch_0000 stalled at 9989/10000) — after the timeout we
# parse whatever landed and move on.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

BATCHES="${1:-0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012}"
PULL_TIMEOUT="${PULL_TIMEOUT:-2400}"  # 40 min/batch; normal pull is ~13 min

for b in $BATCHES; do
    B="grand-champion-2/duels/batch_$b"
    echo "$(date +%H:%M:%S) [$B] pulling (timeout ${PULL_TIMEOUT}s)..."
    timeout "$PULL_TIMEOUT" python -c "
from construct.data.acquire import pull_hf_subset
from pathlib import Path
n = pull_hf_subset(Path('data/replays'), ['$B/**'])
print('pulled total on disk:', n)
" || echo "$(date +%H:%M:%S) [$B] pull timed out — parsing what landed"
    IN="data/replays/$B"
    CNT=$(ls "$IN"/*.replay 2>/dev/null | wc -l)
    echo "$(date +%H:%M:%S) [$B] parsing $CNT replays..."
    nice -n 10 ./target/release/replay-parse --input-dir "$IN" --output-dir data/shards \
        --reset-pool-out data/reset_pool.jsonl --reset-samples-per-replay 16 \
        --min-team-size 1 --stride 8
    echo "$(date +%H:%M:%S) [$B] done. shards so far: $(ls data/shards/*.npz | wc -l), $(du -sh data/shards | cut -f1)"
done

echo "$(date +%H:%M:%S) rebuilding manifest..."
python -c "from construct.data.index import build_index; from pathlib import Path; import json; print(json.dumps(build_index(Path('data/shards'))))"
echo "$(date +%H:%M:%S) GC2 CORPUS COMPLETE"
