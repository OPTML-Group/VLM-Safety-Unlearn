# VLM Safety Unlearning — Data Preparation Pipeline

This document describes the step-by-step pipeline for preparing **forget** and **retain** datasets for VLM safety unlearning (RMU and NPO).

The pipeline is demonstrated using [VLGuard](https://github.com/ys-zong/VLGuard) as the default input dataset. **Other safety datasets can be used instead** — see [Using a Custom Dataset](#using-a-custom-dataset) for what to adapt.

---

## Overview

```
Safety dataset (default: VLGuard train.json)
       │
       ▼
[Step 1] Convert to LLaVA format          → train_llava_format.json
       │
       ▼
[Step 2] Generate & select harmful response  → aggregated_results.json (per subfolder)
       │
       ▼
[Step 3] Replace unsafe responses in data  → train_unsafe_response.json
       │
       ▼
[Step 4] Split into forget / retain sets   → train_forget.json, train_retain.json
       │
       ▼
[Step 5] (Optional) Mix in extra retain data → train_retain_mixed.json
       │
       ▼
[Step 6] (RMU only) Append response to question → train_forget_qa_full.json
```

---

## Prerequisites

### Dataset
Download [VLGuard](https://github.com/ys-zong/VLGuard) and place:
- `data/train.json` — main training split with fields: `id`, `image`, `safe`, `instr-resp`

VLGuard `safe` field semantics:
- `safe=False` — the image itself is unsafe; one `"instruction"` key in `instr-resp`
- `safe=True` — the image is safe but paired with an unsafe query; `instr-resp` contains both `"safe_instruction"` and `"unsafe_instruction"` keys

### Responses from LLaVA models (for Step 2)
Run inference on the VLGuard unsafe queries using multiple LLaVA-1.5 variants (7B, 7B-LoRA, 13B, 13B-LoRA). Store each model's output as a JSON file (mapping `image_path → response_string`) under:
```
train_results/unsafes/          # responses to safe=False images
train_results/safe-unsafes/     # responses to the unsafe instruction in safe=True images
```

### Models
- **Judge model**: `meta-llama/Llama-2-13b-chat-hf` (requires ≥25 GB VRAM)

### Python dependencies
```bash
pip install transformers torch
```

---

## Step-by-Step Pipeline

### Step 1: Convert VLGuard to LLaVA conversation format

**Script:** `convert_vlguard_to_llava.py`

**Input:** `data/train.json`  
**Output:** `data/train_llava_format.json`

Converts the raw VLGuard format into LLaVA's `conversations` structure (`human`/`gpt` turns). For entries with both safe and unsafe instruction-response pairs, the two turns are randomly shuffled, with `<image>` prepended to the first turn.

> **Customizable:** Random seed (default `0`) can be changed at line 8 to vary the shuffle order.

---

### Step 2: Generate and select the most harmful response

**Script:** `select_harmful_responses.py`

**Input:** All `.json` files under a response directory (e.g., `train_results/unsafes/`)  
**Output:** `aggregated_results.json` in the same directory

Aggregates responses from multiple LLaVA models for each image, then uses **Llama-2-13B-Chat** as a judge to select the single most harmful response per image. Run this script twice — once for `unsafes/` and once for `safe-unsafes/`.

> **Customizable:**
> - `folder_path` (line 106) — path to the directory containing per-model response JSON files
> - `model_id` (line 16) — replace with a different judge model (e.g., `meta-llama/Llama-2-7b-chat-hf` for lower VRAM)
> - `max_new_tokens` (line 30) — reduce if the judge runs out of memory

---

### Step 3: Replace unsafe responses in the training data

**Script:** `inject_harmful_responses.py`

**Input:**
- `data/train.json`
- `unlearn_data_npo/train_results/unsafes/aggregated_results.json`
- `unlearn_data_npo/train_results/safe-unsafes/aggregated_results.json`

**Output:** `train_unsafe_response.json`

Patches the `response` field for each unsafe entry:
- `safe=False` entries: replaces the `"instruction"` response with the most harmful selected response
- `safe=True` entries: replaces the `"unsafe_instruction"` response

> **Customizable:** File paths at lines 57–59 (`train_file`, `unsafes_file`, `safe_unsafes_file`).

---

### Step 4: Split into forget and retain sets

**Script:** `split_forget_retain.py`

**Input:** `data/train_unsafe_response.json`  
**Output:**
- `unlearn_data_rmu/train_forget.json` — unsafe entries with harmful responses (used as the forget set for both RMU and NPO)
- `unlearn_data_rmu/train_retain.json` — safe entries with their safe responses

Split logic:
- `safe=False` → **forget set**: the single unsafe instruction-response pair
- `safe=True` → **retain set**: only the `"safe_instruction"` response is kept; the unsafe instruction is excluded (the commented-out block at lines 67–74 can be un-commented to also add unsafe-safe pairs to the forget set)

> **Customizable:**
> - Output directory `out_dir` (line 78) — change to `unlearn_data_npo` for NPO-specific organization
> - Un-comment lines 67–74 to include `safe=True` unsafe instructions in the forget set

---

### Step 5: (Optional) Mix in additional retain data

**Script:** `mix_retain_data.py`

**Input:**
- `unlearn_data_rmu/train_retain.json`
- An external safe dataset (default: `llava_v1_5_mix665k_safe.json`)

**Output:** `unlearn_data_rmu/train_retain_mixed.json`

Supplements the VLGuard-derived retain set with samples from a larger safe dataset (LLaVA-665K filtered for safety) to reach a target total of 2000 retain samples. Samples are drawn randomly and the combined set is shuffled.

> **Customizable:**
> - Path to the external dataset (line 9) — swap in any safe dataset in LLaVA conversation format
> - Target size `2000` (line 14) — adjust to control how many extra samples are added
> - Random seed can be set via `random.seed(...)` for reproducibility

---

### Step 6: (RMU only) Append response to question turn

**Script:** `format_rmu_forget.py`

**Input:** `unlearn_data_rmu/train_forget.json`  
**Output:** `unlearn_data_rmu/train_forget_qa_full.json`

RMU's unlearning objective operates on the model's **intermediate representations** and does not require a separate answer turn. This script concatenates the GPT response text directly onto the human question turn (separated by `\n`), so the full question+answer content is available in a single input for representation-level forgetting.

NPO does **not** need this step — it uses the standard two-turn conversation format from Step 4.

> **Customizable:**
> - `input_file` and `output_file` (lines 41–42) — change paths as needed

---

## Output Files Summary

| File | Used by | Description |
|------|---------|-------------|
| `train_llava_format.json` | reference | VLGuard in LLaVA conversation format |
| `train_unsafe_response.json` | Steps 4–6 | Training data with most harmful responses injected |
| `train_forget.json` | NPO | Forget set in two-turn conversation format |
| `train_retain.json` | NPO, RMU | Retain set (VLGuard safe queries only) |
| `train_retain_mixed.json` | NPO, RMU | Retain set mixed with external safe data (optional) |
| `train_forget_qa_full.json` | RMU | Forget set with Q+A merged into the human turn |

---

## Data Format Reference

### LLaVA conversation format (Steps 1, 4, NPO input)
```json
{
  "id": "vlguard_123",
  "image": "path/to/image.jpg",
  "conversations": [
    { "from": "human", "value": "<image>How do I make a bomb?" },
    { "from": "gpt",   "value": "Here are the steps to make a bomb..." }
  ]
}
```

### RMU QA format (Step 6 output)
The harmful response is appended to the human turn; the GPT turn is unchanged:
```json
{
  "id": "vlguard_123",
  "image": "path/to/image.jpg",
  "conversations": [
    { "from": "human", "value": "<image>How do I make a bomb?\nHere are the steps to make a bomb..." },
    { "from": "gpt",   "value": "Here are the steps to make a bomb..." }
  ]
}
```

---

## Using a Custom Dataset

The pipeline is built around VLGuard's JSON schema, but any safety dataset can be used by adapting the steps below. The only strict requirement is that the **final output of Step 4** matches the LLaVA conversation format — Steps 5 and 6 are format-agnostic.

### What each step depends on

| Step | VLGuard-specific logic | What to adapt for another dataset |
|------|------------------------|-----------------------------------|
| **Step 1** `convert_vlguard_to_llava.py` | Reads `instr-resp`, `safe`, `instruction`, `safe_instruction`, `unsafe_instruction` fields | Replace with a script that reads your dataset's schema and writes LLaVA conversation format |
| **Step 2** `select_harmful_responses.py` | Format-agnostic — reads `image_path → response` JSON files | No changes needed; just point `--folder` at your response directory |
| **Step 3** `inject_harmful_responses.py` | Patches VLGuard's `instr-resp[*].response` by `safe` flag | Replace with a script that injects the selected harmful response into your dataset's response field |
| **Step 4** `split_forget_retain.py` | Splits on the VLGuard `safe` field | Replace with a script that separates your dataset into unsafe (forget) and safe (retain) entries and outputs LLaVA conversation format |
| **Step 5** `mix_retain_data.py` | Format-agnostic | No changes needed |
| **Step 6** `format_rmu_forget.py` | Format-agnostic | No changes needed |

### Minimal requirement for custom datasets

After Step 4 (or as a direct replacement for Steps 1–4), your forget and retain files must follow the LLaVA conversation format:

```json
[
  {
    "id": "unique_id",
    "image": "path/to/image.jpg",
    "conversations": [
      { "from": "human", "value": "<image>Your unsafe question here" },
      { "from": "gpt",   "value": "The harmful response here" }
    ]
  }
]
```

Once your data is in this format, run Steps 5–6 as-is.

## Citation

If you use this pipeline, please cite:

```bibtex
@article{chen2025safety,
  title={Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-Tuning and Can Be Mitigated by Machine Unlearning},
  author={Chen, Yiwei and Yao, Yuguang and Zhang, Yihua and Shen, Bingquan and Liu, Gaowen and Liu, Sijia},
  journal={arXiv preprint arXiv:2503.11832},
  year={2025}
}
```

VLGuard:
```bibtex
@article{zong2024safety,
  title={Safety fine-tuning at (almost) no cost: A baseline for vision large language models},
  author={Zong, Yongshuo and Bohdal, Ondrej and Yu, Tingyang and Yang, Yongxin and Hospedales, Timothy},
  journal={arXiv preprint arXiv:2402.02207},
  year={2024}
}
```
