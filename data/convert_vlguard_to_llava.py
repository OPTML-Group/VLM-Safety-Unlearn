"""
Convert VLGuard train.json to LLaVA conversation format.

Input:  data/train.json
Output: data/train_llava_format.json
"""

import json
import random

RANDOM_SEED = 0
INPUT_FILE = "data/train.json"
OUTPUT_FILE = "data/train_llava_format.json"


def make_turn(instruction_key: str, record: dict, prepend_image: bool = False) -> list[dict]:
    """Build a [human, gpt] conversation turn pair from a single instruction-response record."""
    question = record[instruction_key]
    if prepend_image:
        question = f"<image>{question}"
    return [
        {"from": "human", "value": question},
        {"from": "gpt",   "value": record["response"]},
    ]


def convert(train_data: list[dict], seed: int = RANDOM_SEED) -> list[dict]:
    random.seed(seed)
    for item in train_data:
        item["conversations"] = []
        pairs = item["instr-resp"]

        if len(pairs) > 1:
            # Entry has both a safe and an unsafe instruction; shuffle turn order.
            random.shuffle(pairs)
            key0 = list(pairs[0].keys())[0]
            key1 = list(pairs[1].keys())[0]
            item["conversations"].extend(make_turn(key0, pairs[0], prepend_image=True))
            item["conversations"].extend(make_turn(key1, pairs[1], prepend_image=False))
        else:
            item["conversations"].extend(make_turn("instruction", pairs[0], prepend_image=True))

        item.pop("instr-resp")

    return train_data


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    converted = convert(train_data)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(converted, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(converted)} items to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
