#!/bin/bash
#SBATCH --job-name=lda_bridge_v3
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --dependency=singleton
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/bridge-ft-v3-%j.out

# Real state_dim: null -> 14 retrain (see LDA_bridge_v3.yaml header for the full
# rationale). Wave-based like v2: 6h waves schedule faster than one giant
# reservation, --dependency=singleton (keyed on job-name "lda_bridge_v3", distinct
# from v2's chain) keeps successive waves ordered and auto-resumes. Validated with
# a 10-step smoke test first (job 627158 + checkpoint-key inspection, 2026-08-14):
# state flows through with the right shape/order, checkpoint loading transfers
# everything except the (freshly-initializing, as expected) state_encoder,
# forward/backward completes with finite loss.

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT=lda-bridge-v3
export WANDB_RUN_ID=lda_bridge_finetune_v3
export WANDB_RESUME=allow

cd /mnt/beegfsnew/scratch/3295540/LDA-1B

/home/3295540/.conda/envs/lda/bin/accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes 1 \
  --main_process_port 29515 \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_bridge_v3.yaml \
  --trainer.is_resume True \
  --is_debug False

echo "FT_V3_EXIT=$?"
