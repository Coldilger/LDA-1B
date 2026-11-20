#!/bin/bash
#SBATCH --job-name=lda_exp11_smoke
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --exclude=gnode07
#SBATCH --output=/home/3295540/LDA-1B/slurm/logs/exp11-smoke-%j.out
# 20-step smoke test of both Experiment 11 configs (data loading, checkpoint
# loading, loss composition, checkpoint save). Writes to a separate run root.
ulimit -t unlimited
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
cd /home/3295540/LDA-1B
for V in policyonly full; do
  echo "===== VARIANT=$V"
  /home/3295540/.conda/envs/lda/bin/accelerate launch \
    --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
    --num_machines 1 --num_processes 1 --main_process_port 29611 \
    lda/training/train_LDA.py \
    --config_yaml lda/config/training/LDA_robocasa_exp11_${V}.yaml \
    --run_root_dir outputs/lda_robocasa_exp11_smoke \
    --trainer.max_train_steps 20 --trainer.save_interval 20 --trainer.logging_frequency 5 \
    --trainer.num_warmup_steps 5 --trainer.is_resume False --is_debug False
  echo "SMOKE_EXIT_$V=$?"
done
