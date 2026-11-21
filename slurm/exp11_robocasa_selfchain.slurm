#!/bin/bash
#SBATCH --job-name=lda_exp11
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --exclude=gnode07
#SBATCH --output=/home/3295540/LDA-1B/slurm/logs/exp11-%x-%j.out

# Experiment 11 (advisor's point 5), LDA-1B on RoboCasa. VARIANT=full|policyonly
# selects lda/config/training/LDA_robocasa_exp11_${VARIANT}.yaml (the two files
# differ only in only_policy). Self-chaining like F1-VLA's
# slurm/full_bridge_finetune_nogenloss_selfchain.slurm: SIGTERM 180 s before the
# 6 h limit -> resubmit this script with the same VARIANT, until
# steps_100000 exists. Waves resume from the newest checkpoint (model weights;
# train_LDA.py does not restore optimizer state -- same for both variants).
# A background loop keeps only the two newest checkpoints (~14 GB each).

: "${VARIANT:?set VARIANT=full|policyonly}"
SCRIPT=/home/3295540/LDA-1B/slurm/exp11_robocasa_selfchain.slurm
CKDIR=/home/3295540/LDA-1B/outputs/lda_robocasa_exp11/${VARIANT}/checkpoints
FINAL="$CKDIR/steps_100000_pytorch_model.pt"

resubmit_if_unfinished() {
  if [ -f "$FINAL" ]; then
    echo ">>> training complete ($FINAL exists); not resubmitting."
  else
    echo ">>> time limit approaching, training unfinished -> resubmitting VARIANT=$VARIANT"
    sbatch --job-name="lda_exp11_${VARIANT}" --export=ALL,VARIANT="$VARIANT" "$SCRIPT"
  fi
  kill "$PRUNE_PID" 2>/dev/null
}
TERMED=0
trap 'TERMED=1; resubmit_if_unfinished' TERM

ulimit -t unlimited
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline
cd /home/3295540/LDA-1B

mkdir -p "$CKDIR"
( while true; do
    ls -1 "$CKDIR"/steps_*_pytorch_model.pt 2>/dev/null | sort -t_ -k2 -n | head -n -2 | xargs -r rm -f
    sleep 600
  done ) &
PRUNE_PID=$!

/home/3295540/.conda/envs/lda/bin/accelerate launch \
  --config_file lda/config/deepseeds/deepspeed_zero2.yaml \
  --num_machines 1 --num_processes 1 --main_process_port $((29500 + SLURM_JOB_ID % 400)) \
  lda/training/train_LDA.py \
  --config_yaml lda/config/training/LDA_robocasa_exp11_${VARIANT}.yaml \
  --trainer.is_resume True \
  --is_debug False &
TRAIN_PID=$!
wait $TRAIN_PID
RC=$?
kill "$PRUNE_PID" 2>/dev/null
if [ "$TERMED" = 1 ]; then
  echo ">>> wave ended at the time limit (next wave submitted by the trap)."
elif [ -f "$FINAL" ]; then
  echo ">>> TRAINING FINISHED: $FINAL"
else
  echo ">>> CHAIN BROKEN: training exited (rc=$RC) before the time limit; not auto-resubmitting."
  echo ">>>   VARIANT=$VARIANT sbatch --export=ALL,VARIANT=$VARIANT $SCRIPT"
fi
