#!/bin/bash
#SBATCH --job-name=lda_bridge_v2
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/bridge-ft-v2-%j.out

# Clean from-scratch retrain with the second gripper-masking fix present for
# the whole run, run in parallel with the griptest-extension job. Wave-based
# for the same reason as finetune_bridge_wave.slurm: 6h waves schedule fast,
# --dependency=singleton (keyed on job-name, distinct from the original
# "lda_bridge" chain) keeps successive waves ordered. Distinct
# main_process_port and WANDB run so this never collides with griptest_bridge.slurm
# or the original finetune run if they land on the same node.

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

export WANDB_PROJECT=lda-bridge-v2
export WANDB_RUN_ID=lda_bridge_finetune_v2
export WANDB_RESUME=allow

cd /mnt/beegfsnew/scratch/3295540/LDA-1B

/home/3295540/.conda/envs/lda/bin/accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes 1 \
  --main_process_port 29513 \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_bridge_v2.yaml \
  --trainer.is_resume True \
  --is_debug False

echo "FT_V2_EXIT=$?"
