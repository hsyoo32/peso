#!/usr/bin/env bash

set -euo pipefail

GPU_COUNT="${GPU_COUNT:-2}"
MASTER_PORT="${MASTER_PORT:-1234}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-1B}"
DATASET="${DATASET:-Instruments_cl_ori}"
DATA_PATH="${DATA_PATH:-../data}"
CKPT_PATH="${CKPT_PATH:-./ckpt/peso_instruments}"
RESULTS_FILE="${RESULTS_FILE:-./results/peso_instruments.json}"
INDEX_FILE="${INDEX_FILE:-.index.epoch20000_edim32_beta0.0.json}"
SHIFT_FLAG="${SHIFT_FLAG:-lora_kldiv_latest}"

cd "$(dirname "$0")/../Fine-tuning"

torchrun --nproc_per_node="${GPU_COUNT}" --master_port="${MASTER_PORT}" continual_test.py \
  --ckpt_path "${CKPT_PATH}" \
  --base_model "${BASE_MODEL}" \
  --dataset "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --results_file "${RESULTS_FILE}" \
  --test_batch_size 1 \
  --num_beams 10 \
  --index_file "${INDEX_FILE}" \
  --shift_flag "${SHIFT_FLAG}" \
  --num_blocks 5 \
  --test_all_blocks \
  --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
  --lora_modules_to_save embed_tokens,lm_head
