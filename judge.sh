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

python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="meta-llama/Llama-Guard-3-8B",
    local_dir="/scratch/common_models/Llama-Guard-3-8B",
    local_dir_use_symlinks=False,
    allow_patterns=[
        "config.json",
        "tokenizer.*",
        "model.safetensors.index.json",
        "*.safetensors"
    ],
)
print("Downloaded.")
PY

echo "Judge rephrase"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-Guard-3-8B" --save_path "/nethome/wlacroix/false_refusal/data" --save_name "judge_rephrase" --template llama3 --dataset judge_rephrase \
> /nethome/wlacroix/false_refusal/judge_rephrase.log 2>&1

echo "Judge ignore-word"
python3 scripts/vllm_infer.py --model_name_or_path "/scratch/common_models/Llama-Guard-3-8B" --save_path "/nethome/wlacroix/false_refusal/data" --save_name "judge_ignore-word" --template llama3 --dataset judge_ignore-word \
> /nethome/wlacroix/false_refusal/judge_ignore-word.log 2>&1

#or if you encounter error:
#FORCE_TORCHRUN=1 PTA/experiments_sarubi/llama3_lora_sft.yaml \
#> PTA/experiments_sarubi/logs_lora_sft 2>&1

echo "Main Experiment Workflow Completed!"
