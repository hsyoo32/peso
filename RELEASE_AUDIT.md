# Public Release Checklist

This is a maintenance checklist for preparing `ICLR_code/` as a public repository.

## Public Entry Points

- `README.md`
- `requirements.txt`
- `scripts/run_peso.sh`
- `scripts/eval_peso.sh`
- `Fine-tuning/continual_train.py`
- `Fine-tuning/continual_test.py`
- `Fine-tuning/continual_lora.py`
- `Fine-tuning/continual_data_v2.py`
- `Fine-tuning/collator.py`
- `Fine-tuning/evaluate.py`
- `Fine-tuning/modeling_letter.py`
- `Fine-tuning/prompt.py`
- `Fine-tuning/utils.py`
- `Fine-tuning/config/`
- `RQ-VAE/main.py`
- `RQ-VAE/generate_indices.py`
- `RQ-VAE/train_tokenizer.sh`
- `RQ-VAE/tokenize.sh`
- `RQ-VAE/datasets.py`
- `RQ-VAE/models/`
- `data_process/amazon_text_emb.py`

## Exclude From Git

These should be ignored or distributed through an external artifact host:

- `Fine-tuning/ckpt/`
- `Fine-tuning/results/`
- `Fine-tuning/wandb/`
- `checkpoint/`
- local `data/`
- `__pycache__/`
- `*.pyc`
- `*.out`
- `*.log`
- `*.safetensors`
- `*.pt`
- `*.pth`
- local notebooks and notebook checkpoints

## Before Public Push

1. Re-run the public PESO command after the gradient accumulation fix.
2. Re-run the item text embedding, RQ-VAE codebook, and semantic ID export path.
3. Keep semantic ID filenames consistent as `<DATASET>.index.epoch20000_edim32_beta0.0.json`.
4. Update `README.md` with verified expected metrics and the verified RQ-VAE checkpoint naming.
5. Decide whether to publish processed data/checkpoints externally.
6. Remove or archive internal launchers if they are not meant for users:
   - `Fine-tuning/quick_test_run.py`
   - `Fine-tuning/quick_test_run2.py`
7. Run a final scan for absolute local paths, credentials, and stale experiment names.
