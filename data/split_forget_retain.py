"""
Split the training data into forget and retain sets for machine unlearning.

Forget set: unsafe entries (safe=False) with their harmful responses.
Retain set: safe entries (safe=True) with their safe-instruction responses only.

Input:  data/train_unsafe_response.json
Output:
  - <output_dir>/train_forget.json
  - <output_dir>/train_retain.json
"""

import json
import os

INPUT_FILE = "data/train_unsafe_response.json"
OUTPUT_DIR = "unlearn_data_rmu"


def make_turn(instruction_key: str, record: dict, prepend_image: bool = False) -> list[dict]:
    """Build a [human, gpt] conversation turn pair."""
    question = record[instruction_key]
    if prepend_image:
        question = f"<image>{question}"
    return [
        {"from": "human", "value": question},
        {"from": "gpt",   "value": record["response"]},
    ]


def split(train_data: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Returns (forget_data, retain_data).

    Forget set includes:
      - safe=False entries: the single unsafe instruction with its harmful response.

    Retain set includes:
      - safe=True entries: only the safe_instruction response.

    Note: To also add unsafe instructions from safe=True entries to the forget set,
    un-comment the block under "# Optionally add unsafe_instruction to forget set".
    """
    forget_data: list[dict] = []
    retain_data: list[dict] = []

    for item in train_data:
        item_id = item.get("id")
        image_path = item.get("image", "")

        if not item.get("safe", True):
            pairs = item.get("instr-resp", [])
            if pairs:
                new_item = {
                    "id": item_id,
                    "image": image_path,
                    "conversations": make_turn("instruction", pairs[0], prepend_image=True),
                }
                forget_data.append(new_item)
        else:
            for pair in item.get("instr-resp", []):
                if "safe_instruction" in pair:
                    new_item = {
                        "id": item_id,
                        "image": image_path,
                        "conversations": make_turn("safe_instruction", pair, prepend_image=True),
                    }
                    retain_data.append(new_item)

                # Optionally add unsafe_instruction to forget set:
                # if "unsafe_instruction" in pair:
                #     new_item = {
                #         "id": item_id,
                #         "image": image_path,
                #         "conversations": make_turn("unsafe_instruction", pair, prepend_image=True),
                #     }
                #     forget_data.append(new_item)

    return forget_data, retain_data


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    forget_data, retain_data = split(train_data)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    forget_path = os.path.join(OUTPUT_DIR, "train_forget.json")
    retain_path = os.path.join(OUTPUT_DIR, "train_retain.json")

    with open(forget_path, "w", encoding="utf-8") as f:
        json.dump(forget_data, f, indent=2, ensure_ascii=False)

    with open(retain_path, "w", encoding="utf-8") as f:
        json.dump(retain_data, f, indent=2, ensure_ascii=False)

    print(f"Forget set: {len(forget_data)} items  →  {forget_path}")
    print(f"Retain set: {len(retain_data)} items  →  {retain_path}")


if __name__ == "__main__":
    main()
