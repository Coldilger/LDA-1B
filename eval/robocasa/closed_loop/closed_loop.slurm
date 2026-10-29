#!/bin/bash
#SBATCH --job-name=lda_exp1_cl
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/LDA-1B/eval/robocasa/closed_loop/cl-%j.out
# Experiment 1 for LDA-1B on RoboCasa at the thesis paired protocol
# (2026-10-29). One script for both conditions so they share every setting:
#   CONDITION=baseline  deployment/model_server/server_policy.py (default path,
#                       world model unused at inference)
#   CONDITION=wm_on     eval/robocasa/server_policy_worldmodel_on.py
# SEED_BASE fixes the scenes: episode k starts from scene seed SEED_BASE + k
# in both conditions (launcher/seeded_episodes.py; the authors' client never
# seeded its env, so earlier runs drew different random scenes per run).
# Per-episode outcomes go to closed_loop/episodes/<condition>_<base>.txt as
# "<k> <scene seed> <success>"; compare with closed_loop/paired_stats_robocasa.py.
set -u
: "${CONDITION:?set CONDITION}"
: "${SEED_BASE:?set SEED_BASE}"
case "$CONDITION" in
  baseline) SERVER=deployment/model_server/server_policy.py ;;
  wm_on)    SERVER=eval/robocasa/server_policy_worldmodel_on.py ;;
  *) echo "unknown CONDITION: $CONDITION" >&2; exit 2 ;;
esac
ulimit -t unlimited
LDA_ROOT=/home/3295540/LDA-1B
OUT=$LDA_ROOT/eval/robocasa/closed_loop
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=$LDA_ROOT/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=$((20000 + SLURM_JOB_ID % 20000))   # unique per job: two jobs may share a node
SERVER_LOG=$OUT/server-${SLURM_JOB_ID}.log
mkdir -p "$OUT/episodes"
export EPISODE_SEED_BASE=$SEED_BASE
export EPISODE_LOG=$OUT/episodes/${CONDITION}_${SEED_BASE}.txt
rm -f "$EPISODE_LOG"
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""

cd "$LDA_ROOT"
/home/3295540/.conda/envs/lda_eval/bin/python "$SERVER" --ckpt_path "$CKPT" --port "$PORT" > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
for i in $(seq 1 180); do
    if grep -q "server running" "$SERVER_LOG" 2>/dev/null; then echo "Server ready after ${i}0s"; break; fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "SERVER_DIED_EARLY"; cat "$SERVER_LOG"; exit 1; fi
    sleep 10
done

export PYTHONPATH="$LDA_ROOT:$ROBOCASA_ROOT:${PYTHONPATH:-}"
/home/3295540/.conda/envs/robocasa/bin/python \
  eval/robocasa/launcher/run_client.py \
  --args.env_name gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  --args.port "$PORT" \
  --args.n_episodes 50 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/exp1_${CONDITION}_${SEED_BASE}_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"
CLIENT_EXIT=$?
kill "$SERVER_PID" 2>/dev/null
echo "CLIENT_EXIT=$CLIENT_EXIT CONDITION=$CONDITION SEED_BASE=$SEED_BASE episodes_logged=$(wc -l < "$EPISODE_LOG")"
