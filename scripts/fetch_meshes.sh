#!/usr/bin/env bash
# scripts/fetch_meshes.sh — meshes are game-derived assets; keep local, never commit.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p assets/collision_meshes
tmp=$(mktemp -d)
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/Martico2432/Rlgym-v2-to-rlbot-v5 "$tmp"
git -C "$tmp" sparse-checkout set src/collision_meshes
cp -r "$tmp/src/collision_meshes/soccar" assets/collision_meshes/
rm -rf "$tmp"
ls assets/collision_meshes/soccar/ | head -3
echo "OK: $(ls assets/collision_meshes/soccar | wc -l) mesh files"
