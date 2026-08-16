#!/bin/bash
#SBATCH --job-name=lda_state_ref
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/state-ref-%j.out
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda_eval/bin/python eval/bridge/diag_state_reference.py \
  --checkpoint "${CKPT:?}" --n-trajectories ${NTRAJ:-60}
echo "STATE_REF_EXIT=$?"
