#!/bin/bash
#SBATCH --job-name=lda_bridge
#SBATCH --partition=long_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/bridge-ft-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

# Every wave is its own sbatch job, so without a pinned run id W&B would show the
# training as N disconnected segments (the lesson from F1-VLA's slurm scripts).
export WANDB_PROJECT=lda-bridge
export WANDB_RUN_ID=lda_bridge_finetune
export WANDB_RESUME=allow

cd /mnt/beegfsnew/scratch/3295540/LDA-1B

# Single GPU, but launched through accelerate because train_LDA.py expects an
# accelerate/deepspeed context. zero2 keeps optimizer state sharding off the
# critical path for a one-process run while still giving CPU-offload-free bf16.
accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes 1 \
  --main_process_port 29511 \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_bridge.yaml \
  --is_debug False

echo "FT_EXIT=$?"
