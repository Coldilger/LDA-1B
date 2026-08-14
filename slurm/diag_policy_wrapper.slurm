#!/bin/bash
#SBATCH --job-name=lda_diag_wrapper
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/diag-wrapper-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python eval/bridge/diag_policy_wrapper.py
echo "DIAG_EXIT=$?"
