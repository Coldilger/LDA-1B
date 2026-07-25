#!/bin/bash
#SBATCH --job-name=lda_bridge
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --dependency=singleton
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/bridge-ft-%j.out

# Wave-based finetuning, the pattern F1-VLA used on this cluster.
#
# Why not one 3-day job: the H200 nodes are draining, so SLURM refuses to place a
# 3-day walltime there ("ReqNodeNotAvail, Reserved for maintenance"), and
# long_gpunew has a queue of 3-day jobs ahead of us. 6-hour waves on
# medium_gpunew schedule almost immediately (gnode08 sits idle), and
# --dependency=singleton keeps successive waves with this job name strictly
# ordered so they never overlap.
#
# trainer.is_resume: true makes each wave pick up the newest checkpoint in
# output_dir/checkpoints; on the first wave there is none, so train_LDA.py falls
# through to trainer.pretrained_checkpoint and starts from LDA-pretrain
# (train_LDA.py:224-247). Nothing to change between waves.

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

# A pinned run id keeps every wave appending to one W&B run instead of showing up
# as N disconnected line segments.
export WANDB_PROJECT=lda-bridge
export WANDB_RUN_ID=lda_bridge_finetune
export WANDB_RESUME=allow

cd /mnt/beegfsnew/scratch/3295540/LDA-1B

# The env's accelerate, not ~/.local/bin/accelerate -- that one is a Python 3.9
# install where deepspeed is absent, which is what failed the first wave.
/home/3295540/.conda/envs/lda/bin/accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 \
  --num_processes 1 \
  --main_process_port 29511 \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_bridge.yaml \
  --trainer.is_resume True \
  --is_debug False

echo "FT_EXIT=$?"
