#!/bin/bash
#SBATCH --job-name=lda_diag_ablation
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/diag-ablation-%x-%j.out

set -u
: "${MODE:?set MODE (full|pos_only|rot_only)}"

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python eval/bridge/diag_drift_ablation.py --mode "$MODE"
echo "DIAG_ABLATION_EXIT=$? MODE=$MODE"
