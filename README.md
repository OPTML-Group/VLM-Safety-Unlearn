<div align='center'>
 
# Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-tuning and Can Be Mitigated by Machine Unlearning

</div>

<table align="center">
  <tr>
    <td align="center"> 
      <img src="./images/VLM_unlearn_teasor.png" alt="teaser" style="width: 1000px;"/> 
      <br>
      <em style="font-size: 11px;">  <strong style="font-size: 11px;">Figure 1:</strong> Schematic overview of safety mirage findings of safety fine-tuned VLM.</em>
    </td>
  </tr>
</table>

This is the official code repository for the ICLR 2026 paper [Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-tuning and Can Be Mitigated by Machine Unlearning](https://arxiv.org/abs/2503.11832).


## News

- 🎉 Our another paper on [LLM Unlearn Detection](https://arxiv.org/abs/2506.14003) has been accepted by ICLR! 📚
- 🏆 Congrats! Our paper [Safety Mirage](https://arxiv.org/abs/2503.11832) has been accepted by ICLR 2026! ✨
<!-- - [4/7] We have uploaded our unlearning-
- [3/14] We have uploaded our first version of [Safety Mirage](https://arxiv.org/abs/2503.11832) to the Arxiv platform. -->

## Installation

Our safety-unlearn framework has been developed on the LLaVA-1.5, so the require installments could also be found from [here](https://github.com/haotian-liu/LLaVA).
Also, you could use following steps:

1. Clone this repository and navigate to LLaVA folder
```bash
git clone https://github.com/OPTML-Group/VLM-Safety-MU
cd VLM-Safety-MU
```

2. Install Package
```Shell
conda create -n llava python=3.10 -y
conda activate llava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

## Data Preparation

The forget and retain datasets are derived from the [VLGuard dataset](https://github.com/ys-zong/VLGuard). For the full data preparation pipeline, please refer to [data/data.md](./data/data.md).

Place the previously generated training data (forget/retain JSON files) and the VLGuard training images into the corresponding folders specified in the training scripts before running unlearning fine-tune.

## Unlearning Fine-tune

Our base model LLaVA-1.5 will be downloaded automatically when you run the training scripts. No action is needed.

We support two unlearning algorithms: **NPO** (Negative Preference Optimization) and **RMU** (Representation Mismatch Unlearning).

### Full-parameter and LoRA Variants

For full-parameter unlearning fine-tune, you should run
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/finetune_unlearn.sh
```

For LoRA unlearning fine-tune, you should run
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/v1_5/finetune_unlearn_lora.sh
```

Here are some unlearn related options to note:

- `--unlearn_type`: unlearning algorithm type, which could be `npo` or `rmu`.
- `--rmu_llava_loss_weight`: weight for LLaVA training loss on the retain data.
- `--rmu_retain_alpha`: weight for RMU loss on the retain data.
- `--npo_beta`: balancing parameter for the NPO algorithm.
- `--npo_forget_alpha`: weight for NPO loss on the forget data.
- `--npo_llava_loss_weight`: weight for LLaVA training loss on the retain data.

Also, the data path and the output directory should also be specified.

### NPO Training

`scripts/v1_5/finetune_unlearn_npo.sh` is the dedicated script for running a single NPO fine-tune with full-parameter training and DeepSpeed ZeRO-3:

```bash
bash scripts/v1_5/finetune_unlearn_npo.sh
```

Data paths and output directory are controlled by variables at the top of the script:

```bash
RETAIN_DATA_PATH="../unlearn_data_npo/train_retain_mixed.json"
FORGET_DATA_PATH="../unlearn_data_npo/train_forget.json"
OUT_DIR="./checkpoints_npo/..."
```

<!-- NPO-specific arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--npo_beta` | `0.9` | Temperature parameter balancing forget and retain gradients |
| `--npo_forget_alpha` | `1.0` | Weight on the NPO loss for the forget set |
| `--npo_llava_loss_weight` | `1.0` | Weight on the LLaVA cross-entropy loss for the retain set |
| `--npo_retain_alpha` | `1.0` | Weight scaling the retain loss term | -->

<!-- ## Contributors
* [Yiwei Chen](https://yiwei-chenn.github.io/)
* [Yuguang Yao](https://www.cse.msu.edu/~yaoyugua/) -->

## Evaluation

The `eval/` folder contains the test data and evaluation scripts used to measure model safety. See [eval/evaluation.md](./eval/evaluation.md) for full details.

- **Test data** — VLGuard test split (`eval/data/test.json`) for normal inputs, and one-word jailbreak variants (`eval/safety_data/`) with 1-shot and 3-shot attack prefixes.
- **Model inference** (`eval/run_eval_each_model.sh`) — runs `VLGuard_eval.py` across all models and evaluation cases. Supports optional question sampling:
  ```bash
  bash eval/run_eval_each_model.sh          # use all questions (default)
  bash eval/run_eval_each_model.sh 128      # sample 128 questions per dataset
  ```
- **Post-evaluation** (`eval/run_post_eval.sh`) — computes rejection rate via keyword matching (`eval/llm-eval/rejection_eval.py`) and LLM-judged ASR via Qwen2.5-VL (`eval/llm-eval/llm-judge.py`, `eval/llm-eval/llm-judge-asr-3shot.py`).

## Cite This Work
If you found our code or paper helpful, please cite our work~
```
@article{chen2025safety,
  title={Safety Mirage: How Spurious Correlations Undermine VLM Safety Fine-Tuning and Can Be Mitigated by Machine Unlearning},
  author={Chen, Yiwei and Yao, Yuguang and Zhang, Yihua and Shen, Bingquan and Liu, Gaowen and Liu, Sijia},
  journal={arXiv preprint arXiv:2503.11832},
  year={2025}
}
```