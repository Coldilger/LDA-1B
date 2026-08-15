#!/bin/bash
#SBATCH --job-name=lda_diag_grip_self
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/diag-grip-self-%j.out

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python eval/bridge/diag_gripper_selfhistory.py \
  --checkpoint outputs/lda_bridge_v2/bridge_finetune_v2/final_model/pytorch_model.pt \
  --n-samples 40 --warmup-steps 6 --seed 0
echo "DIAG_GRIP_SELF_EXIT=$?"
