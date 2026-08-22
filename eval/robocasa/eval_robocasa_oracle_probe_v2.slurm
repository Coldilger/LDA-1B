#!/bin/bash
#SBATCH --job-name=lda_robocasa_oracle_probe_v2
#SBATCH --partition=medium_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/oracle-probe-v2-%j.out

# Experiments 1 (world-model-on) and 2 (oracle injection) for LDA-1B, run as
# a live side-probe during a normal RoboCasa rollout (server_policy_oracle_probe.py)
# -- see that file's docstring for why this substitutes live simulator
# rollouts for a static offline dataset (none available on this cluster for
# RoboCasa, unlike Bridge). Uses the confirmed-working RoboCasa checkpoint,
# same task as Experiment 3's own LDA-1B row.

set -u
LDA_ROOT=/mnt/beegfsnew/scratch/3295540/LDA-1B
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=10096
SERVER_LOG=/mnt/beegfsnew/scratch/3295540/robocasa-eval/server-oracle-probe-v2-${SLURM_JOB_ID}.log

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""
ulimit -t unlimited

# --- start server (LDA env), oracle-probe variant ---
cd "$LDA_ROOT"
/home/3295540/.conda/envs/lda_eval/bin/python \
  eval/robocasa/server_policy_oracle_probe.py \
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

# --- run client (robocasa env), same task as the confirmed 48% egocentric run ---
cd "$LDA_ROOT"
export PYTHONPATH="$LDA_ROOT:$ROBOCASA_ROOT:${PYTHONPATH:-}"
/home/3295540/.conda/envs/robocasa/bin/python \
  eval/robocasa/launcher/run_client.py \
  --args.env_name gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  --args.port "$PORT" \
  --args.n_episodes 10 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/oracle_probe_v2_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"

CLIENT_EXIT=$?
kill "$SERVER_PID" 2>/dev/null
sleep 2
echo "CLIENT_EXIT=$CLIENT_EXIT"

echo "=== final probe summary (from server log) ==="
grep "PROBE_SUMMARY" "$SERVER_LOG" | tail -20
