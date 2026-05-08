# Fine-tuning

This directory contains the continual LLM recommendation training and evaluation code used by PESO.

For environment setup, data layout, recommended commands, and method naming, see the repository root README:

- [../README.md](../README.md)

## Main Files

- `continual_train.py`: block-by-block continual training.
- `continual_test.py`: block-wise evaluation with constrained generation.
- `continual_lora.py`: continual LoRA modules and regularizers.
- `continual_data_v2.py`: chronological block data loader.
- `collator.py`: response-only training collator.
- `evaluate.py`: recommendation metrics.
- `modeling_letter.py`: Llama-compatible recommendation model wrapper.
- `utils.py`: command-line arguments and shared utilities.
- `config/`: DeepSpeed configurations.

## Public Interface

The recommended public entry points are:

- `../scripts/run_peso.sh`
- `../scripts/eval_peso.sh`

Internal experiment launchers such as `quick_test_run.py` and `quick_test_run2.py` are kept for development reference, but they are not the public reproduction interface.

For `quick_test_run.py` and `quick_test_run2.py`, `SKIP_BLOCK0_TRAIN=True` follows the same pretraining reuse logic as `--skip_block0` in `continual_train.py`. Evaluation always includes block 0 for comparable reported metrics.

Generated outputs such as `ckpt/`, `results/`, `wandb/`, logs, and caches should not be committed.
