#!/bin/bash
#SBATCH --job-name=lda_grip
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/grip-%j.out
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda_eval/bin/python eval/bridge/diag_gripper.py \
  --checkpoint "${CKPT:?}" --n-samples 40
echo "GRIP_EXIT=$?"
