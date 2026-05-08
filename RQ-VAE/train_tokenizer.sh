#!/usr/bin/env bash

set -euo pipefail

# Train the all-item RQ-VAE codebook used to build semantic item IDs.

DATASET="${1:?Usage: bash RQ-VAE/train_tokenizer.sh DATASET GPU}"
GPU="${2:?Usage: bash RQ-VAE/train_tokenizer.sh DATASET GPU}"
BETA="${BETA:-0.0}"
E_DIM="${E_DIM:-32}"
EPOCHS="${EPOCHS:-20000}"
DATA_PATH="${DATA_PATH:-./data/${DATASET}/${DATASET}.emb-llama-td.npy}"
CKPT_DIR="${CKPT_DIR:-./checkpoint/}"

echo "=== RQ-VAE Codebook Training ==="
echo "Dataset: $DATASET"
echo "GPU: $GPU"
echo "Data path: $DATA_PATH"
echo "Beta: $BETA"
echo "Embedding dim: $E_DIM"
echo "Epochs: $EPOCHS"

python ./RQ-VAE/main.py \
  --device "cuda:${GPU}" \
  --data_path "$DATA_PATH" \
  --dataset "$DATASET" \
  --alpha 0 \
  --beta "$BETA" \
  --ckpt_dir "$CKPT_DIR" \
  --epochs "$EPOCHS" \
  --e_dim "$E_DIM"
