#!/bin/bash
#SBATCH --job-name=lda_pos_range
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/pos-range-%j.out
export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
cd /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv
/home/3295540/.conda/envs/lda_eval/bin/python /mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/diag_position_range.py
