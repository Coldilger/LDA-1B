#!/bin/bash
#SBATCH --job-name=lda_robocasa_wm_full
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/wm-full-%j.out

# Experiment 1 for LDA-1B, real closed-loop, full stage: smoke (6 episodes,
# job 631219) came back clean, 50% (3/6) -- too small to trust (n=6). Full
# 50-episode protocol, same task/checkpoint as the confirmed 48% egocentric
# baseline, only the policy server differs (world-model-on, see
# server_policy_worldmodel_on.py).

set -u
LDA_ROOT=/mnt/beegfsnew/scratch/3295540/LDA-1B
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=10096
SERVER_LOG=/mnt/beegfsnew/scratch/3295540/robocasa-eval/server-wm-full-${SLURM_JOB_ID}.log

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""
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
  --args.n_episodes 50 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/wm_full_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"

CLIENT_EXIT=$?
kill "$SERVER_PID" 2>/dev/null
echo "CLIENT_EXIT=$CLIENT_EXIT"
