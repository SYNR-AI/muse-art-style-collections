#!/bin/bash

## conda activate /mnt/qiufeng/conda/envs/xlabs-ai

CUDA_DEVICES="${1:-0,1,2,3,4,5,6,7}"
NUM_PROCESSES=${2:-8}
MAIN_PROCESS_PORT="${3:-31190}"
TRIGGER="TK02GALX"

set -x
cd ..
echo "Change workspace ..."
echo `pwd`

export MODEL_PATH="/mnt2/share/huggingface_models/FLUX.1-dev"
export DATASET_PATH="./example_images_filtered/${TRIGGER}"

# -------------------------------------------------------------------------------------------
export OUTPUT_DIR="./trained-flux-lora-exp7/data-v2/${TRIGGER}"

export _MAX_STEPS=2000
export _CKPT_STEPS=500
export _RESOLUTION=512 # dummy!!
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

export _LR_SCHEDULER=cosine_with_restarts
# export _LR_SCHEDULER=constant_with_warmup

# NOTICE:
# hyperparameters
export _EXTRA_ARGS=" --random_flip"

# if [[ DO_NOT_TRAIN_TEXT_ENCODER ]]; then
# ## export _EXTRA_ARGS="${_EXTRA_ARGS} --cache_latents --ignore_instance_caption"
# export _EXTRA_ARGS="${_EXTRA_ARGS} --ignore_instance_caption"
# else
# but if shuffle_captions, and the trigger is not empty, shall also shuffle trigger word
# export _EXTRA_ARGS="${_EXTRA_ARGS} --cache_latents"
# export _EXTRA_ARGS="${_EXTRA_ARGS} --crop_asr_768x1344"
export _EXTRA_ARGS="${_EXTRA_ARGS} --train_text_encoder --text_encoder_lr=1e-5 --shuffle_captions --keep_n_tokens=0"
# fi

CUDA_VISIBLE_DEVICES=${CUDA_DEVICES} accelerate launch \
  --num_processes ${NUM_PROCESSES} \
  --main_process_port ${MAIN_PROCESS_PORT} \
  train_zulusion_lora_flux.py \
  --pretrained_model_name_or_path="${MODEL_PATH}" \
  --instance_data_dir="${DATASET_PATH}" \
  --instance_prompt="${_INSTANCE_PROMPT}" \
  --output_dir="${OUTPUT_DIR}" \
  --mixed_precision="bf16" \
  --upcast_before_saving \
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

