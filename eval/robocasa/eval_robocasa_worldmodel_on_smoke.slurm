#!/bin/bash
#SBATCH --job-name=lda_robocasa_wm_smoke
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/wm-smoke-%j.out

# Experiment 1 for LDA-1B, real closed-loop, smoke stage: does the
# world-model-on policy (video_gen -> inverse_dynamics every decision, see
# server_policy_worldmodel_on.py) even run end to end and get ANY nonzero
# success, before committing to the full 50-episode protocol? Same task/
# checkpoint as the confirmed 48% egocentric baseline -- only the policy
# server differs.

set -u
LDA_ROOT=/mnt/beegfsnew/scratch/3295540/LDA-1B
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=10096
SERVER_LOG=/mnt/beegfsnew/scratch/3295540/robocasa-eval/server-wm-smoke-${SLURM_JOB_ID}.log

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""

# Default soft RLIMIT_CPU on this partition is 600s (see mimic-video's
# experiment3_cost/timing_full.slurm) -- the extra video_gen+inverse_dynamics
# compute per decision trips it the same way. Hard limit is unlimited.
ulimit -t unlimited

cd "$LDA_ROOT"
/home/3295540/.conda/envs/lda_eval/bin/python \
  eval/robocasa/server_policy_worldmodel_on.py \
  --ckpt_path "$CKPT" \
  --port "$PORT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 180); do
    if grep -q "server running" "$SERVER_LOG" 2>/dev/null; then
        echo "Server ready after ${i}0s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "SERVER_DIED_EARLY"
        cat "$SERVER_LOG"
        exit 1
    fi
    sleep 10
done

cd "$LDA_ROOT"
export PYTHONPATH="$LDA_ROOT:$ROBOCASA_ROOT:${PYTHONPATH:-}"
/home/3295540/.conda/envs/robocasa/bin/python \
  eval/robocasa/launcher/run_client.py \
  --args.env_name gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  --args.port "$PORT" \
  --args.n_episodes 6 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/wm_smoke_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"

CLIENT_EXIT=$?
kill "$SERVER_PID" 2>/dev/null
echo "CLIENT_EXIT=$CLIENT_EXIT"
