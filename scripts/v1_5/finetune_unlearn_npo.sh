#!/bin/bash

set -euo pipefail

# Single NPO run: beta=0.9, retainA=1.0, lr=5e-7

DEEPSPEED_CFG="./scripts/zero3.json"
MODEL="liuhaotian/llava-v1.5-7b"
VERSION="v1"
RETAIN_DATA_PATH="../unlearn_data_npo/train_retain_mixed.json"
FORGET_DATA_PATH="../unlearn_data_npo/train_forget.json"
IMAGE_FOLDER="../datasets/Safety-Unlearn"
VISION_TOWER="openai/clip-vit-large-patch14-336"
MM_PROJECTOR_TYPE="mlp2x_gelu"
MM_VISION_SELECT_LAYER=-2

NUM_EPOCHS=1
PER_DEV_TRAIN_BS=1
PER_DEV_EVAL_BS=4
GRAD_ACCUM=1
WEIGHT_DECAY=0.0
WARMUP_RATIO=0.03
LR_SCHEDULER="cosine"
MODEL_MAX_LEN=2048
NUM_WORKERS=4

# NPO hyperparameters
BETA=0.9
RETAIN_ALPHA=1.0
LR=5e-7
NPO_FORGET_ALPHA=1.0
NPO_LLAVA_LOSS_WEIGHT=1.0

OUT_DIR="./checkpoints_npo/checkpoints-llava-retain-mix-npo-search-full-2/llava-7b_npo_beta-${BETA}_retainA-${RETAIN_ALPHA}_lr-${LR}"
LOSS_DIR="${OUT_DIR}/losses"
mkdir -p "$LOSS_DIR"

echo "==== Launching NPO run ===="
echo "beta=${BETA} | retain_alpha=${RETAIN_ALPHA} | lr=${LR}"
echo "output_dir=${OUT_DIR}"

deepspeed llava/train/train_unlearn_full_mem.py \
    --deepspeed "$DEEPSPEED_CFG" \
    --model_name_or_path "$MODEL" \
    --version "$VERSION" \
    --retain_data_path "$RETAIN_DATA_PATH" \
    --forget_data_path "$FORGET_DATA_PATH" \
    --image_folder "$IMAGE_FOLDER" \
    --vision_tower "$VISION_TOWER" \
    --mm_projector_type "$MM_PROJECTOR_TYPE" \
    --mm_vision_select_layer "$MM_VISION_SELECT_LAYER" \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEV_TRAIN_BS" \
    --per_device_eval_batch_size "$PER_DEV_EVAL_BS" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate "$LR" \
    --weight_decay "$WEIGHT_DECAY" \
    --warmup_ratio "$WARMUP_RATIO" \
    --lr_scheduler_type "$LR_SCHEDULER" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "$MODEL_MAX_LEN" \
    --gradient_checkpointing True \
    --dataloader_num_workers "$NUM_WORKERS" \
    --lazy_preprocess True \
    --report_to wandb \
    --unlearn_type "npo" \
    --npo_beta "$BETA" \
    --npo_forget_alpha "$NPO_FORGET_ALPHA" \
    --npo_llava_loss_weight "$NPO_LLAVA_LOSS_WEIGHT" \
    --npo_retain_alpha "$RETAIN_ALPHA" \
    --verbose True \
    --output_dir "$OUT_DIR" \
    --loss_dir "$LOSS_DIR"

echo "==== Completed: ${OUT_DIR} ===="
