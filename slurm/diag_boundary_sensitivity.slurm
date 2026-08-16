#!/bin/bash
#SBATCH --job-name=lda_boundary
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/boundary-%j.out
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /mnt/beegfsnew/scratch/3295540/LDA-1B
# lda_eval lacks the raw `lerobot` package (LDAInference's own dataset
# wrapper doesn't need it, but this script reads bridge_orig_lerobot
# directly, same as diag_gripper.py's sibling scripts do) -- `lda` (the
# training env) has it, and this script needs no SimplerEnv/sapien stack.
/home/3295540/.conda/envs/lda/bin/python eval/bridge/diag_boundary_sensitivity.py
echo "BOUNDARY_EXIT=$?"
