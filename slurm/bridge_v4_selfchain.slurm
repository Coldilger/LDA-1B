#!/bin/bash
#SBATCH --job-name=lda_bridge_v4
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --signal=B:TERM@180
#SBATCH --exclude=gnode07
#SBATCH --output=/home/3295540/LDA-1B/slurm/logs/bridge-v4-%j.out

# Experiment 12 (advisor's point 9): Bridge finetune of LDA-1B for the
# corrected evaluation wrapper (lda/config/training/LDA_bridge_v4.yaml = v3
# recipe; eval/bridge/experiment12_audit/README.md). Self-chaining copy of
# slurm/exp11_robocasa_selfchain.slurm: SIGTERM 180 s before the 6 h limit ->
# resubmit until steps_150000 exists; waves resume from the newest checkpoint
# (weights only); keeps the two newest checkpoints.

SCRIPT=/home/3295540/LDA-1B/slurm/bridge_v4_selfchain.slurm
CKDIR=/home/3295540/LDA-1B/outputs/lda_bridge_v4/bridge_finetune_v4/checkpoints
FINAL="$CKDIR/steps_150000_pytorch_model.pt"

resubmit_if_unfinished() {
  if [ -f "$FINAL" ]; then
    echo ">>> training complete ($FINAL exists); not resubmitting."
  else
    echo ">>> time limit approaching, training unfinished -> resubmitting"
    sbatch --export=ALL "$SCRIPT"
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
  --config_yaml lda/config/training/LDA_bridge_v4.yaml \
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
  echo ">>>   sbatch $SCRIPT"
fi
