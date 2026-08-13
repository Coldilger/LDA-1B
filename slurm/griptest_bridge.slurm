#!/bin/bash
#SBATCH --job-name=lda_griptest
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/griptest-%j.out

# Cheap A/B probe: does the second (uncommitted, episode-level) gripper-masking
# fix move the gripper head, before paying for a full 150k-step retrain?
# See lda/config/training/LDA_bridge_griptest.yaml for the full rationale.
#
# One-shot job, no wave/singleton dependency needed: 8000 micro-batches is a
# small fraction of a 6h wave at the measured 2.55 micro-batches/s.

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT=lda-bridge-griptest
export WANDB_RUN_ID=lda_bridge_griptest
export WANDB_RESUME=allow

cd /mnt/beegfsnew/scratch/3295540/LDA-1B

/home/3295540/.conda/envs/lda/bin/accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes 1 \
  --main_process_port 29512 \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_bridge_griptest.yaml \
  --trainer.is_resume True \
  --is_debug False

echo "GRIPTEST_EXIT=$?"
