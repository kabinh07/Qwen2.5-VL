#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=$(nvidia-smi --list-gpus | wc -l)

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm=Qwen/Qwen2.5-VL-7B-Instruct  # Using HuggingFace model ID

# Training hyperparameters
lr=2e-7
batch_size=1
grad_accum_steps=8

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration (replace with public dataset names)
datasets=my_dataset

# Output configuration
run_name="qwen2vl-baseline"
output_dir=./output

# Training arguments
    # --save_strategy "steps" \
    # --save_steps 10 \
    # --save_total_limit 1 \
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision True \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --output_dir ${output_dir} \
    --num_train_epochs 5 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "steps" \
    --eval_steps 500 \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard \
    --r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05"

# Launch training
torchrun --nproc_per_node=${NNODES} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         -- \
         ${entry_file} ${args}