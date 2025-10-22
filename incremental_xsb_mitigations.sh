#!/bin/bash

source /nethome/wlacroix/LLaMA-Factory/experiments/scripts/rename_gpus.sh
source /nethome/wlacroix/miniconda3/etc/profile.d/conda.sh
echo "Current conda environment: $CONDA_DEFAULT_ENV"
conda activate /nethome/wlacroix/miniconda3/envs/llama_factory_v2
echo "Activated conda environment: $CONDA_DEFAULT_ENV"
cd /nethome/wlacroix/false_refusal

# Debugging: Check CUDA details
echo "=== CUDA Debugging Information ==="
nvcc --version
nvidia-smi
echo "CUDA_HOME: $CUDA_HOME"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "==================================="
echo "HOSTNAME: $HOSTNAME"
which python

echo "Run Incremental XSB Mitigations"
python incremental_xsb_mitigations_4.py \
  --data data/XSB.csv \
  --out runs/incremental_xsb \
  --calibrate False \
  --base_model meta-llama/Llama-3.1-8B \
  --big_model meta-llama/Llama-3.1-8B \
  --base_load '{"device_map":{"":0},"attn_implementation":"eager"}' \
  --big_load  '{"device_map":{"":1},"attn_implementation":"sdpa"}' \
> incremental_xsb_mitigations_v4.log 2>&1

echo "Main Experiment Workflow Completed!"
