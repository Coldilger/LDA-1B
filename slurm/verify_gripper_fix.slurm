#!/bin/bash
#SBATCH --job-name=lda_verify_grip
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/verify-grip-%j.out
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python data_preprocessing/verify_gripper_fix.py
echo "VERIFY_EXIT=$?"
