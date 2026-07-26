#!/bin/bash
#SBATCH --job-name=lda_diag
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/diag-%j.out
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda_eval/bin/python eval/bridge/diag_actions.py \
  --checkpoint "${CKPT:?}" --n-steps 12
echo "DIAG_EXIT=$?"
