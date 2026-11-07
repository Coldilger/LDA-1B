#!/bin/bash
#SBATCH --job-name=lda_internals
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/LDA-1B/eval/robocasa/internals/internals-%j.out
# Internal sanity check of the world-model-on path (advisor point 2), see
# eval/robocasa/server_policy_internals.py. The default policy drives the
# robot on the paired protocol's scenes (SEED_BASE=1000, first 3 episodes);
# all next-obs conditions are re-run on the same states.
set -u
ulimit -t unlimited
LDA_ROOT=/home/3295540/LDA-1B
OUT=$LDA_ROOT/eval/robocasa/internals
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=$LDA_ROOT/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=$((20000 + SLURM_JOB_ID % 20000))
SERVER_LOG=$OUT/server-${SLURM_JOB_ID}.log
export EPISODE_SEED_BASE=1000
export EPISODE_LOG=$OUT/episodes-${SLURM_JOB_ID}.txt
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""

cd "$LDA_ROOT"
/home/3295540/.conda/envs/lda_eval/bin/python eval/robocasa/server_policy_internals.py \
  --ckpt_path "$CKPT" --port "$PORT" --out "$OUT/records-${SLURM_JOB_ID}.npz" > "$SERVER_LOG" 2>&1 &
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
  --args.n_episodes 3 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/internals_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"
CLIENT_EXIT=$?
kill -INT "$SERVER_PID" 2>/dev/null; sleep 20; kill "$SERVER_PID" 2>/dev/null
echo "CLIENT_EXIT=$CLIENT_EXIT"
grep -E "INTERNALS_CONFIG|INTERNALS saved|Traceback|Error" "$SERVER_LOG" | tail -20
