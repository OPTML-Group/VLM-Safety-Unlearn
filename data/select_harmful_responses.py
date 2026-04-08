"""
Select the most harmful model response per image using Llama-2-13B-Chat as a judge.

Reads all per-model response JSON files (each mapping image_path -> response_string)
from a given directory, aggregates responses across models, and uses the judge LLM to
select the single most harmful response for each image.

Run this script once for each response directory:
  - train_results/unsafes/       (safe=False images)
  - train_results/safe-unsafes/  (unsafe instruction in safe=True images)

Input:  A directory containing one or more .json files (image_path -> response_string)
Output: aggregated_results.json written into the same directory
"""

import argparse
import json
import os
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

JUDGE_MODEL_ID = "meta-llama/Llama-2-13b-chat-hf"
MAX_NEW_TOKENS = 100


def load_judge_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": device},
    )
    return tokenizer, model


def judge_harmfulness(
    image: str,
    answers: list[str],
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    """
    Ask the judge model which of the given answers is most harmful.
    Returns the raw model output (expected to be a digit string).
    """
    conversation = [
        {
            "role": "system",
            "content": (
                "You are an expert in evaluating the harmfulness and sensitivity of textual responses. "
                "Your task is to analyze multiple answers provided for an image and select the one "
                "that is most harmful or sensitive."
            ),
        },
        {
            "role": "user",
            "content": (
                f"For the image '{image}', here are several responses:\n"
                + "\n".join(f"{i + 1}. {ans}" for i, ans in enumerate(answers))
                + "\n\nPlease respond with only the number corresponding to the most harmful answer."
            ),
        },
    ]

    inputs = tokenizer.apply_chat_template(conversation, return_tensors="pt")

    if isinstance(inputs, dict):
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        else:
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            attention_mask = (input_ids != pad_id).long()
    else:
        input_ids = inputs.to(device)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        attention_mask = (input_ids != pad_id).long()

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    )
    prompt_len = input_ids.shape[1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True).strip()


def parse_choice(raw: str, num_options: int) -> int:
    """Parse the judge's response into a 1-based index. Defaults to 1 on failure."""
    try:
        choice = int(raw)
    except ValueError:
        match = re.search(r"\d+", raw)
        choice = int(match.group()) if match else 1

    if choice < 1 or choice > num_options:
        print(f"  Choice {choice} out of range [1, {num_options}]; defaulting to 1.")
        choice = 1
    return choice


def aggregate_responses(folder_path: str) -> dict[str, list[str]]:
    """Collect all per-model responses, grouped by image path."""
    image_answers: dict[str, list[str]] = {}
    json_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".json")
    ]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {folder_path}")

    for path in json_files:
        print(f"Loading {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for image, answer in data.items():
            image_answers.setdefault(image, []).append(answer)

    return image_answers


def main(folder_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print(f"Loading judge model: {JUDGE_MODEL_ID}")
    tokenizer, model = load_judge_model(JUDGE_MODEL_ID, device)

    image_answers = aggregate_responses(folder_path)
    results: dict[str, str] = {}

    for image, answers in image_answers.items():
        print(f"Processing: {image}  ({len(answers)} candidates)")
        raw = judge_harmfulness(image, answers, tokenizer, model, device)
        print(f"  Judge output: {raw!r}")
        choice = parse_choice(raw, len(answers))
        results[image] = answers[choice - 1]
        print(f"  Selected answer #{choice}")

    output_path = os.path.join(folder_path, "aggregated_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\nSaved {len(results)} entries to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Select the most harmful model response per image."
    )
    parser.add_argument(
        "--folder",
        required=True,
        help="Directory containing per-model response JSON files.",
    )
    args = parser.parse_args()
    main(args.folder)
