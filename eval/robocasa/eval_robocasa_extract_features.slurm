#!/bin/bash
#SBATCH --job-name=lda_extract_feat
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/extract-feat-%j.out

# Experiment 4 (LDA-1B), step 1: extract vl_embs features from one live
# RoboCasa episode. One job = one episode (episode_id = this job's ID);
# submit multiple times for multiple episodes. See
# server_policy_extract_features.py's docstring for the full rationale.

set -u
LDA_ROOT=/mnt/beegfsnew/scratch/3295540/LDA-1B
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=10099
SERVER_LOG=/mnt/beegfsnew/scratch/3295540/robocasa-eval/server-extract-feat-${SLURM_JOB_ID}.log
OUT=/mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/experiment4_probing/features_ep${SLURM_JOB_ID}.npz

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""
ulimit -t unlimited

cd "$LDA_ROOT"
/home/3295540/.conda/envs/lda_eval/bin/python \
  eval/robocasa/server_policy_extract_features.py \
  --ckpt_path "$CKPT" \
  --port "$PORT" \
  --episode_id "$SLURM_JOB_ID" \
  --horizon 5 \
  --out "$OUT" \
  > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for i in $(seq 1 180); do
    if grep -q "server running" "$SERVER_LOG" 2>/dev/null; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
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
  --args.n_episodes 1 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/extract_feat_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"

CLIENT_EXIT=$?
sleep 3
kill "$SERVER_PID" 2>/dev/null
sleep 2
echo "CLIENT_EXIT=$CLIENT_EXIT"
grep "EXTRACT" "$SERVER_LOG" | tail -10
