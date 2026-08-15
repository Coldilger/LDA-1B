#!/bin/bash
#SBATCH --job-name=lda_diag_databias
#SBATCH --partition=short_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/diag-databias-%j.out

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python eval/bridge/diag_dataset_delta_bias.py
echo "DIAG_DATABIAS_EXIT=$?"
