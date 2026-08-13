#!/bin/bash
#SBATCH --job-name=lda_probe2
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:50:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/probe2-%j.out
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda_eval/bin/python lda/eval/eval_bridge_openloop.py \
  --checkpoint "${CKPT:?}" --n-trajs 12 --max-steps 48 --seed 42 \
  --out-dir "outputs/probe_${TAG:-ft}"
echo "PROBE2_EXIT=$?"
