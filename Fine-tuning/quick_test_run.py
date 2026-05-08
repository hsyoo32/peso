#!/usr/bin/env python3
import argparse
import itertools
import os
import subprocess
import sys

# Broader continual-method sweep launcher.
# Keep the active list aligned with the experiment you want to run.

DATASETS = ["Instruments_cl_ori"]
DATA_ROOT = "../data"

INDEX_FLAG = "_edim32_beta0.0"
INDEX_FILE = ".index.epoch20000" + INDEX_FLAG + ".json"

BASE_MODEL = "meta-llama/Llama-3.2-1B"

PER_DEVICE_BATCH_SIZE = 8
LEARNING_RATE = "2e-4"
EPOCHS = 4
TASKS = "seqrec"
TRAIN_PROMPT_SAMPLES = 1
TRAIN_DATA_SAMPLES = 0
NUM_BLOCKS = 5
# Continual-method variants kept here for sweep experiments:
# [
#     "lora_pretrain",
#     "lora_cumulative_all",
#     "lora_cumulative_latest",
#     "lora_cumulative_all_noinherit",
#     "lora_cumulative_latest_noinherit",
#     "sdlora_latest",
# ]

SHIFT_FLAGS = ["lora_pretrain"]

LORA_TARGET_MODULES = "q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
LORA_MODULES_TO_SAVE = "embed_tokens,lm_head"

TEST_ONLY = False
# Training may reuse a pre-existing block_0 pretraining checkpoint when available.
SKIP_BLOCK0_TRAIN = True
# Evaluation always includes block_0 for comparable reported metrics.
USE_FINAL_MODEL = True
PROJ_NAME = "data_v2"
DEBUG = False

MAX_SEQ_LIST = [-1]
WINDOW_SIZE_LIST = [20]
FINE_TUNE_TEMP_LIST = [0.8]


def main():
    parser = argparse.ArgumentParser(description="Quick continual method sweep")
    parser.add_argument("--gpu1", type=str, required=True, help="First GPU id")
    parser.add_argument("--gpu2", type=str, default=None, help="Second GPU id (optional)")
    parser.add_argument("--port", type=int, default=1234, help="Master port for torchrun")
    parser.add_argument("--wandb_project", type=str, default=PROJ_NAME, help="Wandb project name")
    parser.add_argument("--skip_eval", action="store_true", default=False, help="Skip evaluation")
    args = parser.parse_args()

    if args.gpu2 and args.gpu2.lower() != "none":
        os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu1},{args.gpu2}"
        nproc = 2
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu1
        nproc = 1

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("Starting continual-method quick run:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Blocks: {NUM_BLOCKS}")
    print(f"  Batch size: {PER_DEVICE_BATCH_SIZE}")
    print(f"  Train samples: {TRAIN_DATA_SAMPLES}")
    print(f"  Shift flags: {SHIFT_FLAGS}")
    print(f"  GPUs: {args.gpu1}" + (f", {args.gpu2}" if args.gpu2 else ""))

    for dataset in DATASETS:
        for shift_flag in SHIFT_FLAGS:
            for max_seq_len, ws, temp in itertools.product(
                MAX_SEQ_LIST,
                WINDOW_SIZE_LIST,
                FINE_TUNE_TEMP_LIST,
            ):
                tag = f"QUICK_ws{ws}.msl{max_seq_len}.temp{temp}.sf{shift_flag}.blocks{NUM_BLOCKS}"
                out_dir = f"./ckpt/{dataset}/lama3-1b{INDEX_FLAG}/{PROJ_NAME}/{tag}"
                results_dir = f"./results/{dataset}/lama3-1b{INDEX_FLAG}/{PROJ_NAME}"
                results_file = f"{results_dir}/{tag}.json"

                os.makedirs(out_dir, exist_ok=True)
                os.makedirs(results_dir, exist_ok=True)

                if os.path.exists(results_file):
                    print(f"Results file {results_file} already exists, skipping")
                    continue

                cmd_train = [
                    "torchrun",
                    f"--nproc_per_node={nproc}",
                    f"--master_port={args.port}",
                    "continual_train.py",
                    "--base_model", BASE_MODEL,
                    "--output_dir", out_dir,
                    "--dataset", dataset,
                    "--data_path", DATA_ROOT,
                    "--per_device_batch_size", str(PER_DEVICE_BATCH_SIZE),
                    "--learning_rate", LEARNING_RATE,
                    "--epochs", str(EPOCHS),
                    "--tasks", TASKS,
                    "--train_prompt_sample_num", str(TRAIN_PROMPT_SAMPLES),
                    "--train_data_sample_num", str(TRAIN_DATA_SAMPLES),
                    "--index_file", INDEX_FILE,
                    "--temperature", str(temp),
                    "--max_seq_len", str(max_seq_len),
                    "--max_his_len", str(ws),
                    "--only_train_response",
                    "--wandb_project", args.wandb_project,
                    "--wandb_run_name", tag,
                    "--num_blocks", str(NUM_BLOCKS),
                    "--lora_target_modules", LORA_TARGET_MODULES,
                    "--lora_modules_to_save", LORA_MODULES_TO_SAVE,
                    "--shift_flag", str(shift_flag),
                ]

                if SKIP_BLOCK0_TRAIN:
                    cmd_train += ["--skip_block0"]
                if DEBUG:
                    cmd_train += ["--debug"]

                print(f"\n=== Train ({tag}) nproc={nproc} ===")
                print("Command:", " ".join(cmd_train))

                if not TEST_ONLY:
                    try:
                        subprocess.run(cmd_train, check=True)
                        print(f"Training completed successfully for {tag}")
                    except subprocess.CalledProcessError as e:
                        print(f"Training failed for {tag}: {e}")
                        continue

                if args.skip_eval:
                    print(f"Skipping evaluation for {tag}")
                    continue

                if USE_FINAL_MODEL and shift_flag != "lora_pretrain":
                    extract_script = os.path.join(os.path.dirname(__file__), "extract_modules_subset.py")
                    extract_cmd = [
                        sys.executable,
                        extract_script,
                        out_dir,
                        str(NUM_BLOCKS),
                    ]
                    print(f"\n=== Extract subset ({tag}) ===")
                    print("Command:", " ".join(extract_cmd))
                    try:
                        subprocess.run(extract_cmd, check=True)
                        print(f"Subset extraction completed successfully for {tag}")
                    except subprocess.CalledProcessError as e:
                        print(f"Subset extraction failed for {tag}: {e}")
                        continue

                cmd_eval = [
                    "torchrun",
                    f"--nproc_per_node={nproc}",
                    f"--master_port={args.port}",
                    "continual_test.py",
                    "--ckpt_path", out_dir,
                    "--base_model", BASE_MODEL,
                    "--dataset", dataset,
                    "--data_path", DATA_ROOT,
                    "--results_file", results_file,
                    "--test_batch_size", "1",
                    "--num_beams", "10",
                    "--index_file", INDEX_FILE,
                    "--shift_flag", shift_flag,
                    "--num_blocks", str(NUM_BLOCKS),
                    "--test_all_blocks",
                    "--lora_target_modules", LORA_TARGET_MODULES,
                    "--lora_modules_to_save", LORA_MODULES_TO_SAVE,
                ]

                if USE_FINAL_MODEL:
                    cmd_eval += ["--use_final_model"]

                print(f"\n=== Eval ({tag}) nproc={nproc} ===")
                print("Command:", " ".join(cmd_eval))

                try:
                    subprocess.run(cmd_eval, check=True)
                    print(f"Evaluation completed successfully for {tag}")
                except subprocess.CalledProcessError as e:
                    print(f"Evaluation failed for {tag}: {e}")
                    continue

    print("\nQuick test completed!")


if __name__ == "__main__":
    main()
