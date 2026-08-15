#!/bin/bash
#SBATCH --job-name=lda_v3_smoketest
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/smoke-v3-%j.out

# Cheap pre-flight check for LDA_bridge_v3.yaml (state_dim: null -> 14) BEFORE
# committing to the real ~150k-step run: runs the REAL train_LDA.py entrypoint
# (not a hand-rolled reimplementation -- Accelerate/DeepSpeed need a live GPU
# just to import the module, so a true no-GPU dry run isn't possible), capped
# at a handful of steps. Confirms: state_encoder shape-mismatch skip logs
# correctly, "state" actually flows through the dataloader with the right
# shape, forward pass doesn't crash, loss is finite. Not a real training run --
# no checkpoint saves or eval expected in this many steps.

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

cd /mnt/beegfsnew/scratch/3295540/LDA-1B

/home/3295540/.conda/envs/lda/bin/accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes 1 \
  --main_process_port 29514 \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_bridge_v3.yaml \
  --trainer.is_resume False \
  --trainer.max_train_steps 10 \
  --trainer.save_interval 100000 \
  --trainer.eval_interval 100000 \
  --trainer.logging_frequency 1 \
  --is_debug False

echo "SMOKE_V3_EXIT=$?"
