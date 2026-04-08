"""
(RMU only) Append the GPT response to the human question turn.

RMU's unlearning objective operates on intermediate model representations rather than
next-token prediction, so the full question+answer content must appear in a single input.
This script concatenates all GPT turns onto the human turn (separated by a newline),
producing the QA format expected by the RMU trainer.

Input:  unlearn_data_rmu/train_forget.json
Output: unlearn_data_rmu/train_forget_qa_full.json
"""

import json

INPUT_FILE = "unlearn_data_rmu/train_forget.json"
OUTPUT_FILE = "unlearn_data_rmu/train_forget_qa_full.json"


def merge_qa(item: dict) -> dict:
    """
    Append all GPT turns to the human turn, separated by newlines.
    The GPT turns are left unchanged.
    """
    conversations = item.get("conversations", [])

    human_idx = next(
        (i for i, c in enumerate(conversations) if c.get("from") == "human"), None
    )
    gpt_text = "\n".join(
        c.get("value", "") for c in conversations if c.get("from") == "gpt"
    )

    if human_idx is not None and gpt_text:
        original = conversations[human_idx].get("value", "")
        conversations[human_idx]["value"] = f"{original}\n{gpt_text}"

    return item


def main() -> None:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    merged = [merge_qa(item) for item in data]

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)

    print(f"Saved {len(merged)} items to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
