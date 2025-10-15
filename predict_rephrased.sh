#!/bin/bash

source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
echo "Current conda environment: $CONDA_DEFAULT_ENV"
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
echo "Activated conda environment: $CONDA_DEFAULT_ENV"
cd /nethome/wlacroix/LLaMA-Factory/

# Debugging: Check CUDA details
echo "=== CUDA Debugging Information ==="
nvcc --version
nvidia-smi
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================="
echo "HOSTNAME: $HOSTNAME"
which python

# Main Experiment Script
echo "Starting Main Experiment Workflow!"

echo "Prompt rephrasing"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-3.1-8B" --save_path "/nethome/wlacroix/false_refusal" --save_name "rephrased_predict" --template llama3 --dataset rephrased \
> /nethome/wlacroix/false_refusal/predict_rephrase.log 2>&1

#or if you encounter error:
#FORCE_TORCHRUN=1 PTA/experiments_sarubi/llama3_lora_sft.yaml \
#> PTA/experiments_sarubi/logs_lora_sft 2>&1

echo "Main Experiment Workflow Completed!"
