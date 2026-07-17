#!/usr/bin/env bash
# Download-only top-up for the GC2 corpus: resume every batch pull until file
# counts reach 10k (or stop improving), NO parsing — the v4 re-parse (BC plan
# task B4) parses everything once. Downloads are the HF-throttled scarce
# resource; this keeps them flowing while the bc-pretrain branch owns the
# parse binary.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
for pass in 1 2 3; do
    echo "$(date +%H:%M:%S) pull pass $pass"
    for b in 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012; do
        D="data/replays/grand-champion-2/duels/batch_$b"
        N=$(ls "$D" 2>/dev/null | wc -l)
        [ "$N" -ge 10000 ] && continue
        echo "$(date +%H:%M:%S) [batch_$b] $N/10000 — pulling..."
        timeout 2400 python -c "
from construct.data.acquire import pull_hf_subset
from pathlib import Path
pull_hf_subset(Path('data/replays'), ['grand-champion-2/duels/batch_$b/**'])
" || echo "$(date +%H:%M:%S) [batch_$b] timeout, next"
    done
done
echo "$(date +%H:%M:%S) PULL-ONLY TOPUP DONE"
for b in 0001 0002 0003 0004 0005 0006 0007 0008 0009 0010 0011 0012; do
    echo "batch_$b: $(ls data/replays/grand-champion-2/duels/batch_$b 2>/dev/null | wc -l)"
done
