#!/bin/bash
#SBATCH --job-name=lda_oracle_offline
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/oracle-offline-%j.out

# Offline oracle probe (Experiment B) on the v2 bridge finetune (trained with
# the episode-level gripper-masking fix present from step 0). Mirrors F1-VLA's
# and mimic-video's oracle_offline_probe.py protocol: real future frame fed
# through the model's own inverse_dynamics path instead of the learned
# placeholder, compared against the real logged action -- offline, no
# SimplerEnv/ManiSkill2 involved, so unaffected by whatever is causing the
# 0/4-task closed-loop result on this same checkpoint (see eval-6158{18..29}).
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python lda/eval/oracle_offline_probe.py \
  --n-trajs 24 --samples-per-traj 5 --seed 0
echo "ORACLE_EXIT=$?"
