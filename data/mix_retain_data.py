"""
(Optional) Supplement the VLGuard retain set with samples from an external safe dataset.

Randomly draws samples from the external dataset to reach a target total size,
then shuffles and saves the combined retain set.

Input:
  - unlearn_data_rmu/train_retain.json        (VLGuard-derived retain set)
  - <LLAVA_SAFE_DATASET>                      (external safe dataset in LLaVA format)

Output:
  - unlearn_data_rmu/train_retain_mixed.json
"""

import argparse
import json
import random

RETAIN_FILE = "unlearn_data_rmu/train_retain.json"
LLAVA_SAFE_DATASET = "/path/to/llava_v1_5_mix665k_safe.json"
OUTPUT_FILE = "unlearn_data_rmu/train_retain_mixed.json"
TARGET_SIZE = 2000
RANDOM_SEED = 42


def mix(
    retain_data: list[dict],
    llava_data: list[dict],
    target_size: int,
    seed: int,
) -> list[dict]:
    random.seed(seed)
    num_needed = max(0, target_size - len(retain_data))
    sampled = random.sample(llava_data, min(num_needed, len(llava_data)))
    combined = retain_data + sampled
    random.shuffle(combined)
    return combined


def main(
    retain_file: str = RETAIN_FILE,
    llava_file: str = LLAVA_SAFE_DATASET,
    output_file: str = OUTPUT_FILE,
    target_size: int = TARGET_SIZE,
    seed: int = RANDOM_SEED,
) -> None:
    with open(retain_file, "r", encoding="utf-8") as f:
        retain_data = json.load(f)

    with open(llava_file, "r", encoding="utf-8") as f:
        llava_data = json.load(f)

    mixed = mix(retain_data, llava_data, target_size, seed)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(mixed, f, indent=2, ensure_ascii=False)

    print(f"Mixed retain set: {len(mixed)} items  →  {output_file}")
    print(f"  VLGuard samples:  {len(retain_data)}")
    print(f"  External samples: {len(mixed) - len(retain_data)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Mix VLGuard retain set with an external safe dataset."
    )
    parser.add_argument("--retain_file", default=RETAIN_FILE)
    parser.add_argument("--llava_file", default=LLAVA_SAFE_DATASET,
                        help="Path to the external safe dataset (LLaVA conversation format).")
    parser.add_argument("--output_file", default=OUTPUT_FILE)
    parser.add_argument("--target_size", type=int, default=TARGET_SIZE,
                        help="Total number of retain samples after mixing.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    main(args.retain_file, args.llava_file, args.output_file, args.target_size, args.seed)
