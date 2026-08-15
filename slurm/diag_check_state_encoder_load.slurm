#!/bin/bash
#SBATCH --job-name=lda_check_state_enc
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/check-state-enc-%j.out

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python eval/bridge/diag_check_state_encoder_load.py
echo "CHECK_EXIT=$?"
