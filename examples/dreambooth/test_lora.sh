#!/bin/bash

## conda activate /mnt/qiufeng/conda/envs/xlabs-ai

CUDA_DEVICES="${1:-0}"
TRIGGER="${2:-}"
LORA_PATH="${3:-}"
OUT_DIR="${4:-}"

JOB_ID=${5:-0}
JOB_CNTS=${6:-1}

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} python -u \
    test_zulusion_lora_flux.py --trigger "${TRIGGER}" \
    --jobs_idx ${JOB_ID} \
    --jobs_all ${JOB_CNTS} \
    --num_images_per_prompt 4 \
    --separate_save \
    --lora_path "${LORA_PATH}" \
    --lora_scale 0.8 \
    --out_dir "${OUT_DIR}" \
    --num_inference_steps 20 \
    --guidance_scale 3.5 \
    --seed 42

# # examples:
# bash test_lora.sh "0" TK01ANIM \
#     ./trained-flux-lora-exp1/TK01ANIM/pytorch_lora_weights.safetensors \
#     ./visual-results/exp1-TK01ANIM/