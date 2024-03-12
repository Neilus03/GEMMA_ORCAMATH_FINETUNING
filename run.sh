#!/bin/bash

# This script is to fine-tune Gemma-7b or Gemma-2b on the ORCAMATH dataset.

# Sets the GPU(s) to be used for training.
export CUDA_VISIBLE_DEVICES="0" # Change it to the GPU you want to use, if more than one: "0,1,2,..,n"

# Specifies the project name for Weights & Biases (W&B) logging. Change as per your W&B project.
export WANDB_PROJECT="LLM-ORCAMATH-FINETUNING"

# Dynamically assigns a port for distributed training to avoid port conflicts.
master_port=$(shuf -i 12000-30000 -n 1)

# LoRA parameters: rank (r) and alpha (scaling factor), calculated as double the rank.
lora_r=8
lora_alpha=$(( lora_r * 2 ))

# Learning configurations: learning rate, number of epochs, and batch size per device.
learning_rate="5e-5"
num_epoch=10
batch_size=4 # Decrease it if you run out of memory in CUDA (original: 16)

# Training configuration for distributed setup; adjust `world_size` according to the number of GPUs.
world_size=1 # Change it to the number of GPUs
model="gemma-2b" #Alternatively: gemma-7b

# Batch size configurations, adapted for memory limitations. Adjust as necessary.
total_batch_size=32  # Decrease it if you run out of memory in CUDA (original: 128)
gradient_accumulation_steps=$(( total_batch_size / world_size / batch_size))
total_batch_size=$(( gradient_accumulation_steps * world_size * batch_size ))

# Naming convention for the training run, incorporating various parameters for easy identification.
run_name="e${num_epoch}_${model}_qvko_r${lora_r}_a${lora_alpha}_lr${learning_rate}_bs${total_batch_size}"


# Retrieves the current working directory to ensure paths are correctly set relative to the script location.
work_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo "dir: ${work_dir}"

# The main command to run the training, specifying all necessary arguments for `train.py`.

# below, parameters --fp16 True and --tf32 False might be changed, this is what worked to me, due to my GPUs not being Ampere but Turing
# Alternatively, if your GPUs are Ampere, you might set those params as follows: change --fp16 True by --fp32 True or by --bf16 True and --tf32 False by --tf32 True  
torchrun --nproc_per_node=${world_size} --master_port=${master_port} train.py \
    --model_name_or_path "google/${model}" \
    --data_path ./data/orcamath_data.json \
    --output_dir ${work_dir}/${run_name}/ \
    --run_name  ${run_name} \
    --fp16 True \
    --num_train_epochs ${num_epoch} \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --warmup_steps 300 \
    --save_strategy "epoch" \
    --lr_scheduler_type "constant_with_warmup" \
    --save_total_limit 10 \
    --learning_rate ${learning_rate} \
    --model_max_length 512 \
    --logging_steps 8 \
    --tf32 False \
    --ddp_find_unused_parameters False \
    --use_lora True \
    --load_in_4bit True \
    --lora_r ${lora_r} \
    --lora_alpha ${lora_alpha} \
    --lora_target_modules q_proj v_proj k_proj o_proj 
