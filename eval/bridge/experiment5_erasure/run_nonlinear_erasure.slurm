#!/bin/bash
#SBATCH --job-name=lda_exp5_nonlinear
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/experiment5_erasure/nonlinear-%j.out

set -u
cd /mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/experiment5_erasure
/home/3295540/.conda/envs/lda_eval/bin/python nonlinear_erasure_diagnostic.py \
  --features ../experiment4_probing/features_lda_robocasa.npz \
  --json-out nonlinear_erasure_result.json
