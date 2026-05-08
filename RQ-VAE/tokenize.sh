#!/usr/bin/env bash

set -euo pipefail

# Export semantic item IDs from a trained RQ-VAE checkpoint.

DATASET="${1:?Usage: bash RQ-VAE/tokenize.sh DATASET GPU [CHECKPOINT_NAME]}"
GPU="${2:?Usage: bash RQ-VAE/tokenize.sh DATASET GPU [CHECKPOINT_NAME]}"
CHECKPOINT="${3:-}"
BETA="${BETA:-0.0}"
E_DIM="${E_DIM:-32}"
EPOCH="${EPOCH:-20000}"
ROOT_PATH="${ROOT_PATH:-./checkpoint/}"

SUFFIX="_edim${E_DIM}_beta${BETA}"
CKPT_DIR="${ROOT_PATH%/}/${DATASET}${SUFFIX}"

if [[ -z "$CHECKPOINT" ]]; then
  CHECKPOINT="$(find "$CKPT_DIR" -maxdepth 1 -type f -name "*model_best_collision.pth" -printf "%f\n" | sort -V | tail -n 1)"
fi

if [[ -z "$CHECKPOINT" ]]; then
  echo "No *model_best_collision.pth checkpoint found in $CKPT_DIR" >&2
  exit 1
fi

echo "=== RQ-VAE Semantic ID Export ==="
echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "Checkpoint: $CHECKPOINT"
echo "Suffix: $SUFFIX"

python ./RQ-VAE/generate_indices.py \
  --dataset "$DATASET" \
  --root_path "$ROOT_PATH" \
  --str "$SUFFIX" \
  --epoch "$EPOCH" \
  --checkpoint "$CHECKPOINT" \
  --gpu_id "$GPU"
