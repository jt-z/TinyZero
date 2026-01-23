#!/bin/bash
set -e

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1  # 如果没有 InfiniBand，建议加上这个

# =================================================================
# Wrapper Script for Countdown Task PPO Training
# =================================================================
# This script sets all necessary environment variables and then
# executes the main training script (train_tiny_zero.sh).
# =================================================================

# --- Core Training Parameters ---
export N_GPUS=8
export BASE_MODEL="/home/ksa/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct"
export DATA_DIR="dataset/countdown_instruct_data"
export ROLLOUT_TP_SIZE=2

# --- Environment Fixes for CUDA Linkage ---
# The following lines are critical for resolving the 'cannot find -lcuda'
# error that occurs during the torch.compile JIT process on this system.
# It ensures the linker can find the NVIDIA driver library.
echo "Setting CUDA environment variables..."
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.1:$LD_PRELOAD

# --- Execute the Main Training Script ---
echo "Environment set. Executing train_tiny_zero.sh..."
bash scripts/train_tiny_zero.sh
