#!/bin/bash
#SBATCH --job-name=lda_oracle_smoke_big
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/oracle-smoke-big-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python lda/eval/oracle_offline_probe.py   --traj-ids 137,582,867,821,782,64,261,120,507,779,460,483,667,388,807,214,96,499,29,914,855,399,443,622,780,785,2,712,456,272,738,234,605,967,104,923,325,31,22,26,665,554,9,961,902,390,702,221,992,432,743,540,227,448,566,238,353,236,693,224,470,975,296,948,426,857,938,569,944,657,102,190,644,741,880,303,123,760,340,917 --samples-per-traj 4 --seed 0
echo "ORACLE_SMOKE_BIG_EXIT=$?"
