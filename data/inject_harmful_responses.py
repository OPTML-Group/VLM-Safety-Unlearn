"""
Replace the response field for unsafe entries in train.json with the most harmful
response selected in the previous step.

For safe=False entries: replaces the "instruction" response.
For safe=True  entries: replaces the "unsafe_instruction" response.

Input:
  - data/train.json
  - unlearn_data_npo/train_results/unsafes/aggregated_results.json
  - unlearn_data_npo/train_results/safe-unsafes/aggregated_results.json

Output:
  - train_unsafe_response.json
"""

import json
import os

TRAIN_FILE = "data/train.json"
UNSAFES_FILE = os.path.join("unlearn_data_npo/train_results/unsafes", "aggregated_results.json")
SAFE_UNSAFES_FILE = os.path.join("unlearn_data_npo/train_results/safe-unsafes", "aggregated_results.json")
OUTPUT_FILE = "train_unsafe_response.json"


def load_json(path: str) -> dict | list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def inject_responses(
    train_data: list[dict],
    unsafes: dict[str, str],
    safe_unsafes: dict[str, str],
) -> list[dict]:
    """
    Patch the response field for each unsafe instruction in the training data.

    - safe=False: update the response paired with the "instruction" key.
    - safe=True:  update the response paired with the "unsafe_instruction" key.
    """
    for record in train_data:
        image_path = record.get("image")
        if not image_path:
            continue

        if record.get("safe") is False:
            new_response = unsafes.get(image_path)
            if new_response is None:
                print(f"[WARN] No harmful response found for image: {image_path}")
                continue
            for pair in record.get("instr-resp", []):
                if "instruction" in pair:
                    pair["response"] = new_response
                    break
        else:
            new_response = safe_unsafes.get(image_path)
            if new_response is None:
                print(f"[WARN] No harmful response found for image: {image_path}")
                continue
            for pair in record.get("instr-resp", []):
                if "unsafe_instruction" in pair:
                    pair["response"] = new_response
                    break

    return train_data


def main() -> None:
    train_data = load_json(TRAIN_FILE)
    unsafes = load_json(UNSAFES_FILE)
    safe_unsafes = load_json(SAFE_UNSAFES_FILE)

    updated = inject_responses(train_data, unsafes, safe_unsafes)
    save_json(updated, OUTPUT_FILE)
    print(f"Saved {len(updated)} records to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
