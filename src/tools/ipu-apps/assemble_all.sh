#!/usr/bin/env bash
# Assemble every kernel .asm on ZDlinear into a bin dir.
# Usage (from src/tools/ipu-apps/):  bash assemble_all.sh /tmp/bins
set -euo pipefail

BIN_DIR="${1:-/tmp/bins}"
mkdir -p "$BIN_DIR"

APPS=(
  matmul_128x128 matmul_128x64x128 matmul_128x64x64 matmul_64x64x64
  matmul_144x144_x128 matmul_288x144_x128 matmul_432x144_x128 matmul_144x288_x128
  matmul_192x192_x128 matmul_192x384_x128 matmul_240x240_x128 matmul_240x480_x128
  matmul_384x192_x128 matmul_480x240_x128 matmul_576x192_x128 matmul_720x240_x128
  unfold_32x32x144 residual_add_256x144 layernorm_128x16 layernorm_256x144
  attn_scores_km_256x36 attn_v_256x36 attn_v_bcast_36 qk_scores_256x36
)

for name in "${APPS[@]}"; do
  echo "assembling $name ..."
  uv run ipu-as assemble \
    --input "src/ipu_apps/$name/$name.asm" \
    --output "$BIN_DIR/$name.bin" \
    --format bin
done
echo "done -> $BIN_DIR"
