#!/bin/bash
#SBATCH --job-name=lda_robocasa_agentview_smoke
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/agentview-smoke-%j.out

# Camera-viewpoint test, smoke stage: same task/checkpoint/protocol as the
# proven RoboCasa eval (eval_robocasa_smoke.slurm/eval_robocasa_full.slurm,
# 48% on 50 episodes), but the observation the model receives comes from
# RoboCasa's third-person `agentview` camera instead of the egocentric
# head-mounted one -- see launcher/run_client_agentview.py's docstring for
# exactly how and why this is a clean, single-variable swap.

set -u
LDA_ROOT=/mnt/beegfsnew/scratch/3295540/LDA-1B
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=10094

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""

cd "$LDA_ROOT"
/home/3295540/.conda/envs/lda_eval/bin/python \
  deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" \
  --port "$PORT" \
  > /mnt/beegfsnew/scratch/3295540/robocasa-eval/server-agentview-${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!

for i in $(seq 1 180); do
    if grep -q "server running" /mnt/beegfsnew/scratch/3295540/robocasa-eval/server-agentview-${SLURM_JOB_ID}.log 2>/dev/null; then
        echo "Server ready after ${i}0s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "SERVER_DIED_EARLY"
        cat /mnt/beegfsnew/scratch/3295540/robocasa-eval/server-agentview-${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 10
done

cd "$LDA_ROOT"
export PYTHONPATH="$LDA_ROOT:$ROBOCASA_ROOT:${PYTHONPATH:-}"
/home/3295540/.conda/envs/robocasa/bin/python \
  eval/robocasa/launcher/run_client_agentview.py \
  --args.env_name gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  --args.port "$PORT" \
  --args.n_episodes 6 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/agentview_smoke_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"

CLIENT_EXIT=$?
kill "$SERVER_PID" 2>/dev/null
echo "CLIENT_EXIT=$CLIENT_EXIT"
