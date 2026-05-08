import argparse
import glob
import os
import shutil
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import transformers
import wandb
from safetensors.torch import load_file
from transformers import AutoTokenizer, LlamaConfig

from collator import Collator
from continual_data_v2 import ContinualSeqRecDataset_v2, load_continual_datasets_v2
from continual_lora import SaveModuleWrapper, apply_continual_lora, print_trainable_parameters
from modeling_letter import LETTER
from utils import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ContinualLoRATrainer(transformers.Trainer):
    """Trainer for PESO-style continual LoRA optimization."""

    def __init__(self, *args, continual_loss_weight=0.1, **kwargs):
        super().__init__(*args, **kwargs)
        self.continual_loss_weight = continual_loss_weight
        # PEFT models expose **kwargs in forward, but this custom loss does not
        # use num_items_in_batch. Let Trainer scale loss by gradient accumulation.
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.continual_loss_weight > 0:
            continual_loss = 0.0
            continual_metrics = {}
            num_modules = 0

            for _, module in model.named_modules():
                if hasattr(module, "compute_continual_loss"):
                    cl_loss, cl_metrics = module.compute_continual_loss()
                    continual_loss += cl_loss
                    num_modules += 1
                    for key, value in cl_metrics.items():
                        continual_metrics.setdefault(key, []).append(value)

            if continual_loss > 0 and num_modules > 0:
                continual_loss = continual_loss / num_modules
                loss += self.continual_loss_weight * continual_loss

                if hasattr(self, "state") and hasattr(self.state, "global_step") and wandb.run is not None:
                    for key, values in continual_metrics.items():
                        wandb.log(
                            {f"continual_learning/{key}": sum(values) / len(values)},
                            step=self.state.global_step,
                        )
                    wandb.log(
                        {"continual_learning/avg_continual_loss": continual_loss},
                        step=self.state.global_step,
                    )

        return (loss, outputs) if return_outputs else loss


def reconstruct_saved_parameter_lists(model, state_dict, local_rank):
    """Restore saved adapter history after loading a block checkpoint."""
    if local_rank == 0:
        print("Reconstructing ParameterList objects for saved matrices...")

    for name, module in model.named_modules():
        if hasattr(module, "A_matrices") and hasattr(module, "B_matrices"):
            module.A_matrices = nn.ParameterList([])
            module.B_matrices = nn.ParameterList([])

            module_prefix = name + "."
            a_keys = [k for k in state_dict if k.startswith(module_prefix + "A_matrices.")]
            b_keys = [k for k in state_dict if k.startswith(module_prefix + "B_matrices.")]

            a_keys.sort(key=lambda x: int(x.split(".")[-1]))
            b_keys.sort(key=lambda x: int(x.split(".")[-1]))

            for a_key, b_key in zip(a_keys, b_keys):
                module.A_matrices.append(nn.Parameter(state_dict[a_key]))
                module.B_matrices.append(nn.Parameter(state_dict[b_key]))

            if hasattr(module, "magnitudes"):
                module.magnitudes = nn.ParameterList([])
                mag_keys = [k for k in state_dict if k.startswith(module_prefix + "magnitudes.")]
                mag_keys.sort(key=lambda x: int(x.split(".")[-1]))
                for mag_key in mag_keys:
                    module.magnitudes.append(nn.Parameter(state_dict[mag_key]))

        elif hasattr(module, "A_dir_matrices") and hasattr(module, "B_dir_matrices") and hasattr(module, "magnitudes"):
            module.A_dir_matrices = nn.ParameterList([])
            module.B_dir_matrices = nn.ParameterList([])
            module.magnitudes = nn.ParameterList([])

            module_prefix = name + "."
            a_dir_keys = [k for k in state_dict if k.startswith(module_prefix + "A_dir_matrices.")]
            b_dir_keys = [k for k in state_dict if k.startswith(module_prefix + "B_dir_matrices.")]
            mag_keys = [k for k in state_dict if k.startswith(module_prefix + "magnitudes.")]

            a_dir_keys.sort(key=lambda x: int(x.split(".")[-1]))
            b_dir_keys.sort(key=lambda x: int(x.split(".")[-1]))
            mag_keys.sort(key=lambda x: int(x.split(".")[-1]))

            for a_key, b_key, mag_key in zip(a_dir_keys, b_dir_keys, mag_keys):
                module.A_dir_matrices.append(nn.Parameter(state_dict[a_key]))
                module.B_dir_matrices.append(nn.Parameter(state_dict[b_key]))
                module.magnitudes.append(nn.Parameter(state_dict[mag_key]))


def load_base_model(args, tokenizer, device_map, local_rank, block_id):
    model = LETTER.from_pretrained(
        args.base_model,
        device_map=device_map,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
    )
    model.set_hyper(args.temperature)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)

    if local_rank == 0:
        print(f"[Block {block_id}] Model dtype continual LoRA: {next(model.parameters()).dtype}")

    apply_continual_lora(
        model,
        args.lora_target_modules,
        args.lora_r,
        args.lora_alpha,
        args.lora_dropout,
        args.lora_modules_to_save,
        num_blocks=args.num_blocks,
        option=args.shift_flag,
    )
    return model


def continual_train(args):
    """Train PESO variants block-by-block on the v2 continual split."""
    set_seed(args.seed)
    ensure_dir(args.output_dir)

    if args.lora_modules_to_save == "":
        args.lora_modules_to_save = None
    else:
        args.lora_modules_to_save = args.lora_modules_to_save.split(",")
    if args.lora_target_modules == "":
        args.lora_target_modules = None
    else:
        args.lora_target_modules = args.lora_target_modules.split(",")

    # if args.data_type != "v2":
    #     raise ValueError("continual_train.py now supports only data_type='v2'.")
    # if "vanilla" in args.shift_flag or "full_finetune" in args.shift_flag:
    #     raise ValueError("continual_train.py now supports only the continual LoRA / PESO path.")

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK") or 0)
    if local_rank == 0:
        print(vars(args))

    if ddp:
        device_map = {"": local_rank}

    config = LlamaConfig.from_pretrained(args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    tokenizer.pad_token_id = 0

    dataset = ContinualSeqRecDataset_v2(args, mode="train", current_block=0, num_blocks=args.num_blocks)
    new_tokens = dataset.get_new_tokens()
    add_num = tokenizer.add_tokens(new_tokens)
    config.vocab_size = len(tokenizer)

    if local_rank == 0:
        print(f"Added {add_num} new tokens")
        print(f"Total vocabulary size: {len(tokenizer)}")
        tokenizer.save_pretrained(args.output_dir)
        config.save_pretrained(args.output_dir)

    collator = Collator(args, tokenizer)

    for block_id in range(args.num_blocks):
        if "pretrain" in args.shift_flag and block_id > 0:
            continue

        block_output_dir = os.path.join(args.output_dir, f"block_{block_id}")
        prev_block_dir = os.path.join(args.output_dir, f"block_{block_id-1}")

        if block_id == 0 and args.skip_block0:
            if not os.path.exists(block_output_dir):
                # if "lora" in args.shift_flag:
                #     pretrain_name = "lora"
                # else:
                #     pretrain_name = "vanilla"

                # if "sdlora_direct" in args.shift_flag:
                pretrain_name = "lora"

                original_pretrain_filename = f"QUICK_ws20.msl-1.temp0.8.sf{pretrain_name}_pretrain.blocks5/block_0"
                pretrain_block0_dir = os.path.join(os.path.dirname(args.output_dir), original_pretrain_filename)
                if os.path.exists(pretrain_block0_dir):
                    if local_rank == 0:
                        print(f"pretrain_block0_dir: {pretrain_block0_dir}")
                        print("\n[Block 0] Found pretrained block_0, copying to current directory")
                        try:
                            shutil.copytree(pretrain_block0_dir, block_output_dir)
                            print(f"Block_0 copied successfully to {block_output_dir}")
                        except Exception as e:
                            print(f"Failed to copy block_0: {e}")
                            continue
                    else:
                        max_wait_time = 60
                        wait_time = 0
                        while not os.path.exists(block_output_dir) and wait_time < max_wait_time:
                            time.sleep(1)
                            wait_time += 1
                            if wait_time % 10 == 0:
                                print(f"Waiting for block_0 to be copied to {block_output_dir} ({wait_time}s)")

                        if os.path.exists(block_output_dir):
                            print(f"Block_0 copied to {block_output_dir}")
                        else:
                            print(f"Timeout waiting for block_0 copy after {max_wait_time}s")
                            continue

                    time.sleep(2)
                    continue
                else:
                    print(f"Pretrained block_0 not found at {pretrain_block0_dir}; training from scratch")
            else:
                print(f"Block 0 already exists at {block_output_dir}")
                if local_rank != 0:
                    time.sleep(5)
                continue

        if local_rank == 0:
            if wandb.run is not None:
                wandb.finish()
            wandb.init(
                project=args.wandb_project,
                name=f"{args.wandb_run_name}_block{block_id}",
                config=vars(args),
            )

        model = load_base_model(args, tokenizer, device_map, local_rank, block_id)

        if block_id > 0:
            prev_model_file = os.path.join(prev_block_dir, "model.safetensors")
            if not os.path.exists(prev_model_file):
                raise RuntimeError(f"Checkpoint for block {block_id-1} not found at {prev_model_file}!")
            if local_rank == 0:
                print(f"\n[Block {block_id}] Loading model from previous block: {prev_block_dir}")
                print(
                    f"Loading checkpoint: {prev_model_file}, exists: {os.path.exists(prev_model_file)}, "
                    f"size: {os.path.getsize(prev_model_file) if os.path.exists(prev_model_file) else 'N/A'}"
                )

            state_dict = load_file(prev_model_file, device="cpu")
            model.load_state_dict(state_dict, strict=False)
            reconstruct_saved_parameter_lists(model, state_dict, local_rank)

        if not ddp and torch.cuda.device_count() > 1:
            model.is_parallelizable = True
            model.model_parallel = True

        if local_rank == 0:
            print(f"\n{'='*50}")
            print(f"Training on Block {block_id} out of {args.num_blocks} blocks")
            print(f"{'='*50}")

        for _, module in model.named_modules():
            if hasattr(module, "set_current_block"):
                module.set_current_block(block_id)

        if local_rank == 0:
            print_trainable_parameters(model)

            total_params = sum(p.numel() for p in model.parameters())
            lm_head_params = 0
            embed_token_params = 0
            total_lora_params = 0

            for name, module in model.named_modules():
                if isinstance(module, SaveModuleWrapper):
                    params = sum(p.numel() for p in module.modules_to_save["default"].parameters())
                    if "lm_head" in name:
                        lm_head_params = params
                    elif "embed_tokens" in name:
                        embed_token_params = params

                if hasattr(module, "A") and hasattr(module, "B"):
                    total_lora_params += module.A.numel() + module.B.numel()

            print("\nParameter Distribution:")
            print(f"Total parameters: {total_params:,}")
            print(f"LM Head parameters: {lm_head_params:,} ({100 * lm_head_params / total_params:.2f}%)")
            print(f"Embedding parameters: {embed_token_params:,} ({100 * embed_token_params / total_params:.2f}%)")
            print(f"LoRA parameters: {total_lora_params:,} ({100 * total_lora_params / total_params:.2f}%)")

        train_data, valid_data = load_continual_datasets_v2(args, current_block=block_id, num_blocks=args.num_blocks)

        if local_rank == 0:
            print(f"Training samples in block {block_id}: {len(train_data)}")
            print(f"Validation samples in block {block_id}: {len(valid_data)}")

        save_strategy = args.save_and_eval_strategy
        load_best_model_at_end = True

        adaptive_lr = args.learning_rate
        adaptive_scheduler = args.lr_scheduler_type
        adaptive_warmup = args.warmup_ratio
        weight_decay = args.weight_decay

        if block_id > 0:
            adaptive_lr = args.learning_rate * 0.1

            if local_rank == 0:
                print(f"Adaptive learning rate for block {block_id}: {adaptive_lr} (original: {args.learning_rate})")

        trainer = ContinualLoRATrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=valid_data,
            args=transformers.TrainingArguments(
                seed=args.seed,
                per_device_train_batch_size=args.per_device_batch_size,
                per_device_eval_batch_size=args.per_device_batch_size,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                warmup_ratio=adaptive_warmup,
                num_train_epochs=args.epochs,
                learning_rate=adaptive_lr,
                weight_decay=weight_decay,
                lr_scheduler_type=adaptive_scheduler,
                report_to=["wandb"],
                fp16=args.fp16,
                bf16=args.bf16,
                logging_steps=args.logging_step,
                optim=args.optim,
                gradient_checkpointing=False,
                eval_strategy=args.save_and_eval_strategy,
                save_strategy=save_strategy,
                eval_steps=args.save_and_eval_steps,
                save_steps=args.save_and_eval_steps,
                output_dir=block_output_dir,
                save_total_limit=1,
                load_best_model_at_end=load_best_model_at_end,
                deepspeed=args.deepspeed,
                ddp_find_unused_parameters=False if ddp else None,
                eval_delay=1 if args.save_and_eval_strategy == "epoch" else 0,
                label_names=["labels"],
                dataloader_drop_last=False,
            ),
            processing_class=tokenizer,
            data_collator=collator,
            continual_loss_weight=args.continual_loss_weight,
        )


        model.config.use_cache = False
        trainer.train()

        print(f"[rank {local_rank}] train ended")

        for _, module in model.named_modules():
            if hasattr(module, "save_current_matrices"):
                module.save_current_matrices()

        print(f"[rank {local_rank}] save current matrices ended")

        torch.cuda.empty_cache()
        if local_rank == 0:
            print("save model started")
            t0 = time.time()
            trainer.save_state()
            trainer.save_model(output_dir=block_output_dir)
            t1 = time.time()
            print(f"[rank {local_rank}] save model ended, time: {t1-t0:.2f}s")

        if dist.is_initialized():
            print(f"[rank {local_rank}] Barrier after saving model for block {block_id}")
            dist.barrier()

        if local_rank == 0 and wandb.run is not None:
            wandb.finish()

        if local_rank == 0:
            print(f"Completed training on block {block_id}")
            print(f"Model saved to {block_output_dir}")
            for d in glob.glob(os.path.join(block_output_dir, "checkpoint-*")):
                shutil.rmtree(d)
                print(f"Removed checkpoint: {d} for block {block_id}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual LLMRec")
    parser = parse_global_args(parser)
    parser = parse_train_args(parser)
    parser = parse_dataset_args(parser)

    parser.add_argument("--num_blocks", type=int, default=5, help="Number of data blocks for continual learning")
    parser.add_argument(
        "--continual_loss_weight",
        type=float,
        default=0.0,
        help="Weight for continual learning loss",
    )
    parser.add_argument("--skip_block0", action="store_true", default=False, help="Skip block 0")

    args = parser.parse_args()
    continual_train(args)
