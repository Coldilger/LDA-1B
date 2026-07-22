#!/bin/bash
#SBATCH --job-name=lda_convert
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/convert-%j.out

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python data_preprocessing/convert_bridge_to_lda.py \
  --src /mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot \
  --dst /mnt/beegfsnew/scratch/3295540/data/bridge_lda \
  --workers 16
echo "CONVERT_EXIT=$?"
