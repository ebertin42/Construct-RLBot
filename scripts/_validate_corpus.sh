#!/usr/bin/env bash
set -uo pipefail
cd /home/steamo/Construct-RLBot
source .venv/bin/activate
BATCH=grand-champion-2/duels/batch_0000
echo "$(date +%H:%M:%S) [1/3] pulling $BATCH (~7.6GB)..."
t0=$(date +%s)
python -c "from construct.data.acquire import pull_hf_subset; from pathlib import Path; n=pull_hf_subset(Path('data/replays'), ['$BATCH/**'], None); print('pulled', n, 'replays')"
t1=$(date +%s); echo "$(date +%H:%M:%S) pull done in $((t1-t0))s"
IN=data/replays/$BATCH
echo "$(date +%H:%M:%S) [2/3] parsing $(find "$IN" -maxdepth 1 -name '*.replay' 2>/dev/null | wc -l) replays (nice 10)..."
nice -n 10 ./target/release/replay-parse --input-dir "$IN" --output-dir data/shards \
  --reset-pool-out data/reset_pool.jsonl --reset-samples-per-replay 16 --min-team-size 1
t2=$(date +%s); echo "$(date +%H:%M:%S) parse done in $((t2-t1))s"
echo "$(date +%H:%M:%S) [3/3] indexing..."
python -c "from construct.data.index import build_index; from pathlib import Path; import json; print(json.dumps(build_index(Path('data/shards'))))"
echo "=== SIZES ==="; du -sh data/replays data/shards data/reset_pool.jsonl 2>/dev/null
echo "$(date +%H:%M:%S) VALIDATION COMPLETE"
