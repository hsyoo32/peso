#!/usr/bin/env bash

set -euo pipefail

GPU_COUNT="${GPU_COUNT:-2}"
MASTER_PORT="${MASTER_PORT:-1234}"
BASE_MODEL="${BASE_MODEL:-meta-llama/Llama-3.2-1B}"
DATASET="${DATASET:-Instruments_cl_ori}"
DATA_PATH="${DATA_PATH:-../data}"
OUTPUT_DIR="${OUTPUT_DIR:-./ckpt/peso_instruments}"
INDEX_FILE="${INDEX_FILE:-.index.epoch20000_edim32_beta0.0.json}"
SHIFT_FLAG="${SHIFT_FLAG:-lora_kldiv_latest}"

cd "$(dirname "$0")/../Fine-tuning"

torchrun --nproc_per_node="${GPU_COUNT}" --master_port="${MASTER_PORT}" continual_train.py \
  --base_model "${BASE_MODEL}" \
  --output_dir "${OUTPUT_DIR}" \
  --dataset "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --per_device_batch_size 8 \
  --learning_rate 2e-4 \
  --epochs 4 \
  --tasks seqrec \
  --train_prompt_sample_num 1 \
  --train_data_sample_num 0 \
  --index_file "${INDEX_FILE}" \
  --temperature 0.8 \
  --max_seq_len -1 \
  --max_his_len 20 \
  --only_train_response \
  --num_blocks 5 \
  --skip_block0 \
  --shift_flag "${SHIFT_FLAG}" \
  --continual_loss_weight 2.0 \
  --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
  --lora_modules_to_save embed_tokens,lm_head
