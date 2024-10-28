#!/bin/bash

## conda activate /mnt/qiufeng/conda/envs/xlabs-ai

CUDA_DEVICES="${1:-}"
NUM_PROCESSES=${2:-}
TRIGGER="${3:-}"
MAIN_PROCESS_PORT="${4:-31190}"

TRAIN_ART_STYLE="${5:-}"
WITH_CONTROLNET="${6:-OFF}"

set -x

export MODEL_PATH="/mnt2/share/huggingface_models/FLUX.1-dev"
export CONTROLNET_MODEL_PATH="/mnt2/share/huggingface_models/FLUX.1-dev-Controlnet-Canny"
export DATASET_PATH="./example_images/$TRIGGER"
export OUTPUT_DIR="./trained-flux-lora/$TRIGGER"

export _RESOLUTION=512
export _BATCH_SIZE=4
# cuda-memory: 1024*1024*1 ~ 68GB
# cuda-memory: 512*512*4 ~ 70GB
# cuda-memory: 512*512*2 ~ 53GB
# export _REPEATS=1
# export _LR=1e-4
export _REPEATS=4
export _LR=5e-4
export _INSTANCE_PROMPT=$TRIGGER
export _GUIDANCE_SCALE=1
export _RANK=32
# export _LR_SCHEDULER=cosine_with_restarts
export _LR_SCHEDULER=constant_with_warmup


if [[ "$TRAIN_ART_STYLE" == *_CAPTION ]]; then
    export OUTPUT_DIR="./trained-flux-lora-exp3/${TRAIN_ART_STYLE}/${TRIGGER}"
    export WITH_CONTROLNET="OFF"

    if [ "$TRAIN_ART_STYLE" = IGNORE_TRIGGER_* ]; then
        echo " -- ignore trigger"
        export _INSTANCE_PROMPT=""
    fi

    if [ "$TRAIN_ART_STYLE" = *_GS4_* ]; then
        echo " -- set guidance_scale"
        export _GUIDANCE_SCALE=4
    fi

    export _MAX_STEPS=2000
    export _CKPT_STEPS=500

    if [[ "$TRAIN_ART_STYLE" == *_DISABLE_CAPTION ]]; then
        echo " -- ignore instance caption"
        export _EXTRA_ARGS=" --cache_latents --random_flip --ignore_instance_caption "
    elif [[ "$TRAIN_ART_STYLE" == *_ENABLE_CAPTION ]]; then
        echo " -- use raw caption"
        export _EXTRA_ARGS=" --cache_latents --random_flip "
    elif [[ "$TRAIN_ART_STYLE" == *_ENABLE_AND_SHUFFLE_CAPTION ]]; then
        echo " -- shuffle caption"
        export _EXTRA_ARGS=" --cache_latents --random_flip --shuffle_captions "
    else
        echo " -- check inputs @@!!"
        exit 1
    fi
else
    # TRAIN_CHARACTER_LoRA
    export _RESOLUTION=512
    export _REPEATS=1
    export _BATCH_SIZE=4
    export _LR=5e-4
    export _MAX_STEPS=500
    export _CKPT_STEPS=100
    export _EXTRA_ARGS=""
    echo " -- check inputs **!!"
    exit 1
fi

if [ "$WITH_CONTROLNET" == "ON" ]; then
    export ENABLE_CONTROLNET="--pretrained_controlnet_model_name_or_path=${CONTROLNET_MODEL_PATH} --with_controlnet"
    ## export OUTPUT_DIR="trained-flux-lora/exp4_ctrl--$TRIGGER"
elif [ "$WITH_CONTROLNET" == "OFF" ]; then
    export ENABLE_CONTROLNET=""
    ## export OUTPUT_DIR="trained-flux-lora/exp4_noctrl--$TRIGGER"
else
    echo "Invalid argument for WITH_CONTROLNET: $WITH_CONTROLNET"
    exit 1
fi

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${MAIN_PROCESS_PORT} \
  train_zulusion_lora_flux.py \
  --pretrained_model_name_or_path="${MODEL_PATH}" ${ENABLE_CONTROLNET} \
  --instance_data_dir="${DATASET_PATH}" \
  --instance_prompt="${_INSTANCE_PROMPT}" \
  --output_dir="${OUTPUT_DIR}" \
  --mixed_precision="bf16" \
  --resolution=${_RESOLUTION} \
  --repeats=${_REPEATS} \
  --train_batch_size=${_BATCH_SIZE} \
  --guidance_scale=${_GUIDANCE_SCALE} \
  --gradient_accumulation_steps=2 \
  --rank=${_RANK} \
  --optimizer="adamw" \
  --use_8bit_adam \
  --learning_rate=${_LR} \
  --report_to="tensorboard" \
  --lr_scheduler=${_LR_SCHEDULER} \
  --lr_warmup_steps=10 \
  --max_train_steps=${_MAX_STEPS} \
  --checkpointing_steps=${_CKPT_STEPS} \
  ${_EXTRA_ARGS} --seed=0

# # examples:
# -- exp1,2,3,4,5,6
# bash train_lora.sh "0,1,2,3" 4 "TK02GALX" 31190 IGNORE_TRIGGER_DISABLE_CAPTION
# bash train_lora.sh "0,1,2,3" 4 "TK02GALX" 31191 IGNORE_TRIGGER_ENABLE_CAPTION
# bash train_lora.sh "0,1,2,3" 4 "TK02GALX" 31192 IGNORE_TRIGGER_ENABLE_AND_SHUFFLE_CAPTION
# bash train_lora.sh "0,1,2,3" 4 "TK02GALX" 31193 USE_TRIGGER_DISABLE_CAPTION
# bash train_lora.sh "4,5,6,7" 4 "TK02GALX" 31194 USE_TRIGGER_ENABLE_CAPTION
# bash train_lora.sh "2,3,4,5" 4 "TK02GALX" 31195 USE_TRIGGER_ENABLE_AND_SHUFFLE_CAPTION

# ------------------------------------------------------------------------------------------------------------------------------

# ### exp1:
# export MODEL_PATH="/mnt2/share/huggingface_models/FLUX.1-dev"
# export CONTROLNET_MODEL_PATH="/mnt2/share/huggingface_models/FLUX.1-dev-Controlnet-Canny"
# export DATASET_PATH="./example_images/$TRIGGER"
# export OUTPUT_DIR="./trained-flux-lora-exp1/$TRIGGER"

# export _RESOLUTION=1024
# export _BATCH_SIZE=1
# # cuda-memory: 1024*1024*1 ~ 68GB
# # cuda-memory: 512*512*4 ~ 70GB
# # cuda-memory: 512*512*2 ~ 53GB
# export _REPEATS=4
# export _LR=5e-4
# export _INSTANCE_PROMPT=$TRIGGER
# export _GUIDANCE_SCALE=1
# export _RANK=32
# export _MAX_STEPS=2000
# export _CKPT_STEPS=500
# export _EXTRA_ARGS=" --cache_latents --random_flip --ignore_instance_caption "
# export ENABLE_CONTROLNET=""

# CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
#   --num_processes ${NUM_PROCESSES} \
#   --main_process_port ${MAIN_PROCESS_PORT} \
#   train_zulusion_lora_flux.py \
#   --pretrained_model_name_or_path="${MODEL_PATH}" ${ENABLE_CONTROLNET} \
#   --instance_data_dir="${DATASET_PATH}" \
#   --instance_prompt="${_INSTANCE_PROMPT}" \
#   --output_dir="${OUTPUT_DIR}" \
#   --mixed_precision="bf16" \
#   --resolution=${_RESOLUTION} \
#   --repeats=${_REPEATS} \
#   --train_batch_size=${_BATCH_SIZE} \
#   --guidance_scale=${_GUIDANCE_SCALE} \
#   --gradient_accumulation_steps=2 \
#   --rank=${_RANK} \
#   --optimizer="adamw" \
#   --use_8bit_adam \
#   --learning_rate=${_LR} \
#   --report_to="tensorboard" \
#   --lr_scheduler="cosine_with_restarts" \
#   --lr_warmup_steps=10 \
#   --max_train_steps=${_MAX_STEPS} \
#   --checkpointing_steps=${_CKPT_STEPS} \
#   ${_EXTRA_ARGS} --seed=0
