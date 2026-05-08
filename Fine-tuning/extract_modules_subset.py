import argparse
import os

from safetensors.torch import load_file, save_file


def extract_modules_to_save(block_dir: str, out_path: str):
    model_path = os.path.join(block_dir, "model.safetensors")
    if not os.path.exists(model_path):
        print(f"model.safetensors not found in {block_dir}; skipping extraction")
        return

    state = load_file(model_path, device="cpu")
    subset = {k: v for k, v in state.items() if ".modules_to_save.default." in k}

    if not subset:
        print(f"No modules_to_save.default.* found in {block_dir}; writing empty subset")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_file(subset, out_path)
    print(f"Saved subset to {out_path} with {len(subset)} tensors")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_root", help="Root directory that contains block_i folders")
    parser.add_argument("num_blocks", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None, help="Exclusive end index; defaults to num_blocks")
    parser.add_argument(
        "--delete_full_after_extract",
        action="store_false",
        default=True,
        help="Delete block_i/model.safetensors after extracting subset for non-final blocks",
    )
    args = parser.parse_args()

    end = args.end if args.end is not None else args.num_blocks

    for block_id in range(args.start, end - 1):
        block_dir = os.path.join(args.ckpt_root, f"block_{block_id}")
        out_path = os.path.join(block_dir, "modules_to_save.safetensors")
        if not os.path.exists(block_dir):
            print(f"Skip missing {block_dir}")
            continue

        extract_modules_to_save(block_dir, out_path)

        model_path = os.path.join(block_dir, "model.safetensors")
        if args.delete_full_after_extract and os.path.exists(model_path):
            try:
                os.remove(model_path)
                print(f"Deleted {model_path}")
            except Exception as exc:
                print(f"Warning: failed to delete {model_path}: {exc}")


if __name__ == "__main__":
    main()
