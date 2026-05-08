# PESO: Continual Low-Rank Adapters for LLM-based Generative Recommender Systems

Official code for the ICLR 2026 paper
["Continual Low-Rank Adapters for LLM-based Generative Recommender Systems"](https://arxiv.org/abs/2510.25093).

PESO continually adapts an LLM-based generative recommender over chronological data blocks using low-rank adapters and a proximal regularizer over previous adapter states.

## Repository Layout

```text
ICLR_code/
├── Fine-tuning/          # Continual LLM training and evaluation
│   ├── continual_train.py
│   ├── continual_test.py
│   ├── continual_lora.py
│   ├── continual_data_v2.py
│   └── config/           # DeepSpeed configs
├── RQ-VAE/               # Semantic ID tokenizer training and index generation
├── data_process/         # Item text embedding utilities
├── scripts/              # Public run/evaluation wrappers
└── requirements.txt
```

Generated checkpoints, WandB runs, result files, and local datasets are intentionally excluded from the public repository.

## Setup

The main experiments were run with Python 3.9, PyTorch 2.5.1, Transformers 4.52.4, PEFT 0.15.2, Accelerate 1.7.0, and DeepSpeed 0.16.9.

```bash
cd ICLR_code
conda create -n peso python=3.9 -y
conda activate peso
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
conda install -n peso -c nvidia cuda-nvcc=12.1.105 -y
python -m pip install flash-attn==2.7.4.post1 --no-build-isolation
```

If your CUDA stack differs, install the matching PyTorch build from the official PyTorch index before installing `requirements.txt`. The current training path uses FlashAttention 2 and DeepSpeed CUDA checks, so the release environment includes `flash-attn` and a matching CUDA 12.1 `nvcc` package.

You also need access to the base LLM checkpoint, for example `meta-llama/Llama-3.2-1B`, through Hugging Face.

## Data

The full pipeline uses item metadata, item text embeddings, RQ-VAE semantic IDs, and chronological recommendation blocks.

For a dataset such as `Instruments_cl_ori`, place files under:

```text
data/<DATASET>/
```

For the full pipeline, the expected files are:

```text
data/Instruments_cl_ori/
├── Instruments_cl_ori.all.block_0.json
├── Instruments_cl_ori.all.block_1.json
├── ...
├── Instruments_cl_ori.all.block_4.json
├── Instruments_cl_ori.emb-llama-td.npy
├── Instruments_cl_ori.index.epoch20000_edim32_beta0.0.json
└── Instruments_cl_ori.item.json
```

The `.item.json` file contains item metadata. The `.emb-llama-td.npy` file contains item text embeddings used to train the RQ-VAE codebook. The `.index...json` file maps item ids to semantic ID tokens used by the LLM recommender.

If you already have a released semantic ID index, you can skip the text-embedding and RQ-VAE steps. Fine-tuning only needs the block JSON files, the item metadata file, and the semantic ID index.

### Dataset Names

The public scripts accept the dataset directory name as `DATASET`. The released experiments use the following dataset keys:

```text
Instruments_cl_ori
CDs_and_Vinyl_cl_0.6_0_5_3_0
Video_Games_cl_0.6_0_5_3_0
yelp_cl_0.6_0_5_0_5_y2019
```

For example, setting `DATASET=CDs_and_Vinyl_cl_0.6_0_5_3_0` expects files under `data/CDs_and_Vinyl_cl_0.6_0_5_3_0/` with the same filename prefix.

## Quick Start

The public reproduction flow is:

1. Generate item text embeddings.
2. Train an RQ-VAE codebook.
3. Export item semantic IDs.
4. Fine-tune the LLM recommender continually.
5. Evaluate on chronological blocks.

### 1. Generate Item Text Embeddings

This step encodes item metadata into dense embeddings used by the tokenizer. The script reads `data/<DATASET>/<DATASET>.item.json` and writes `data/<DATASET>/<DATASET>.emb-llama-td.npy`.

```bash
cd ICLR_code

python data_process/amazon_text_emb.py \
  --dataset Instruments_cl_ori \
  --root ./data \
  --gpu_id 0
```

The public script uses `meta-llama/Llama-3.2-1B` as the text encoder.

### 2. Train RQ-VAE Codebook

This step learns the codebook used to assign semantic IDs to items.

```bash
cd ICLR_code

E_DIM=32 \
EPOCHS=20000 \
bash RQ-VAE/train_tokenizer.sh Instruments_cl_ori 0
```

This follows the standard TIGER-style RQ-VAE semantic ID tokenizer path. The diversity regularizer is disabled in the public configuration, so the released semantic ID files use the `beta0.0` filename convention. Use the same convention for newly generated indices across datasets.

Checkpoints are saved under:

```text
checkpoint/Instruments_cl_ori_edim32_beta0.0/
```

### 3. Export Semantic ID Index

Pick the best RQ-VAE checkpoint from the checkpoint directory, then export item semantic IDs:

```bash
cd ICLR_code

E_DIM=32 \
EPOCH=20000 \
bash RQ-VAE/tokenize.sh Instruments_cl_ori 0
```

If no checkpoint filename is provided, `tokenize.sh` automatically selects the latest-epoch `*_model_best_collision.pth` checkpoint from `checkpoint/<DATASET>_edim32_beta0.0/`. You can also pass a checkpoint filename explicitly as the third argument. The output is written to:

```text
data/Instruments_cl_ori/Instruments_cl_ori.index.epoch20000_edim32_beta0.0.json
```

### 4. Train PESO

The focused paper-style launcher runs PESO training, extracts the per-block saved modules used by final-model evaluation, and then evaluates all blocks in chronological order:

```bash
cd ICLR_code/Fine-tuning

python quick_test_run2.py --gpu1 0 --gpu2 1
```

Use `--gpu2 none` for a single-GPU run. The launcher sets `CUDA_VISIBLE_DEVICES`, uses `lora_kldiv_latest` with `continual_loss_weight=2.0`, and writes checkpoints/results under `Fine-tuning/ckpt/` and `Fine-tuning/results/`.

For storage efficiency, non-final blocks keep only the saved `embed_tokens` and `lm_head` modules needed for evaluation, while the final checkpoint stores the accumulated continual adapter state. The `--use_final_model` evaluation mode loads the final checkpoint and restores the requested block's saved modules before testing.

### 5. Evaluate PESO

If training is already complete and you only need to rerun evaluation, call the test script directly with `--use_final_model`:

```bash
cd ICLR_code/Fine-tuning

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=1234 continual_test.py \
  --ckpt_path ./ckpt/Instruments_cl_ori/lama3-1b_edim32_beta0.0/data_v2/QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5 \
  --base_model meta-llama/Llama-3.2-1B \
  --dataset Instruments_cl_ori \
  --data_path ../data \
  --results_file ./results/peso_instruments.json \
  --test_batch_size 1 \
  --num_beams 10 \
  --index_file .index.epoch20000_edim32_beta0.0.json \
  --shift_flag lora_kldiv_latest \
  --num_blocks 5 \
  --test_all_blocks \
  --use_final_model \
  --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
  --lora_modules_to_save embed_tokens,lm_head
```

The launcher and manual commands use the paper-style PESO configuration:

```text
shift_flag = lora_kldiv_latest
continual_loss_weight = 2.0
max_his_len = 20
temperature = 0.8
num_blocks = 5
```

The reported evaluation protocol runs over all chronological blocks in order with `--test_all_blocks` and evaluates from this compact final-checkpoint layout with `--use_final_model`.

The `scripts/run_peso.sh` and `scripts/eval_peso.sh` files are lightweight wrappers for custom paths. For reproducing the paper-style run, prefer `Fine-tuning/quick_test_run2.py` or the manual commands below.

## Manual Commands

For direct fine-tuning/evaluation without wrappers:

```bash
cd ICLR_code/Fine-tuning

CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=1234 continual_train.py \
  --base_model meta-llama/Llama-3.2-1B \
  --output_dir ./ckpt/Instruments_cl_ori/lama3-1b_edim32_beta0.0/data_v2/QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5 \
  --dataset Instruments_cl_ori \
  --data_path ../data \
  --per_device_batch_size 8 \
  --learning_rate 2e-4 \
  --epochs 4 \
  --tasks seqrec \
  --train_prompt_sample_num 1 \
  --train_data_sample_num 0 \
  --index_file .index.epoch20000_edim32_beta0.0.json \
  --temperature 0.8 \
  --max_seq_len -1 \
  --max_his_len 20 \
  --only_train_response \
  --num_blocks 5 \
  --shift_flag lora_kldiv_latest \
  --continual_loss_weight 2.0 \
  --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
  --lora_modules_to_save embed_tokens,lm_head
```

```bash
python extract_modules_subset.py \
  ./ckpt/Instruments_cl_ori/lama3-1b_edim32_beta0.0/data_v2/QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5 \
  5
```

```bash
CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2 --master_port=1234 continual_test.py \
  --ckpt_path ./ckpt/Instruments_cl_ori/lama3-1b_edim32_beta0.0/data_v2/QUICK_ws20.msl-1.temp0.8.sflora_kldiv_latest_cl_2.0.blocks5 \
  --base_model meta-llama/Llama-3.2-1B \
  --dataset Instruments_cl_ori \
  --data_path ../data \
  --results_file ./results/peso_instruments.json \
  --test_batch_size 1 \
  --num_beams 10 \
  --index_file .index.epoch20000_edim32_beta0.0.json \
  --shift_flag lora_kldiv_latest \
  --num_blocks 5 \
  --test_all_blocks \
  --use_final_model \
  --lora_target_modules q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj \
  --lora_modules_to_save embed_tokens,lm_head
```

## Method Flags

Experiments are selected through `--shift_flag`.

- `lora`: single evolving LoRA baseline.
- `lora_pretrain`: train a block-0 LoRA pretraining checkpoint.
- `lora_cumulative_all`: cumulative LoRA baseline using all previous adapters.
- `lora_cumulative_latest`: cumulative LoRA baseline using only the latest previous adapter.
- `lora_cumulative_all_noinherit`: cumulative-all variant that reinitializes the current adapter at each block.
- `lora_cumulative_latest_noinherit`: cumulative-latest variant that reinitializes the current adapter at each block.
- `sdlora_latest`: direction-based LoRA baseline using the latest previous adapter.
- `lora_kldiv_latest`: PESO-style KL proximal regularization against the latest previous adapter, used by the public scripts.

The code dispatches variants by substrings in `--shift_flag`, so older internal experiment names containing `lora`, `kldiv`, and `latest` map to the same PESO behavior. Public scripts use the shorter `lora_kldiv_latest` name.

## Release Notes

- Do not commit `Fine-tuning/ckpt/`, `Fine-tuning/results/`, `Fine-tuning/wandb/`, `checkpoint/`, or local `data/` directories.
- Large datasets and pretrained checkpoints should be distributed through an external artifact host.
- `Fine-tuning/quick_test_run.py` and `Fine-tuning/quick_test_run2.py` provide convenient launchers for running predefined experiment sweeps. The shell scripts above are the minimal public interface, while the quick launchers are useful for reproducing or extending the paper experiments.
- Final expected metrics should be updated after the public reproduction run is fixed and verified.

## Citation

```bibtex
@inproceedings{yoo2026peso,
  title={Continual Low-Rank Adapters for LLM-based Generative Recommender Systems},
  author={Yoo, Hyunsik and Li, Ting-Wei and Kang, SeongKu and Liu, Zhining and Xu, Charlie and Qi, Qilin and Tong, Hanghang},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
