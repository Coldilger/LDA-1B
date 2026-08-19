#!/bin/bash
#SBATCH --job-name=lda_robocasa_smoke
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/smoke-%j.out

# Smoke test: does LDA-1B's own officially-published RoboCasa checkpoint,
# run through the authors' own eval pipeline, get ANY nonzero success on
# ANY episode? Small n_episodes on a single task -- just answering "does
# this work at all", not collecting a real success-rate number yet. See
# eval/bridge/RESULTS.md in LDA-1B for the Bridge-side 0% closed-loop
# investigation this is meant to disambiguate (Bridge-specific issue vs.
# a deeper bug in our copy of the codebase/environment).
#
# Server (LDA env, has torch+lda package) and client (robocasa env, sim)
# run as two processes on the same node, talking over a local websocket --
# matches the two-terminal design in examples/Robocasa_tabletop/README.md,
# just collapsed into one SLURM job since they need to share a GPU node.
#
# robosuite/robocasa-gr1-tabletop-tasks themselves (multi-GB vendored sim
# packages with their own upstream git history) stay outside this repo in
# /mnt/beegfsnew/scratch/3295540/robocasa-eval/ -- only the driver code we
# actually wrote lives here. install_robocasa.slurm in this same directory
# rebuilds that external checkout from scratch if it's ever missing.

set -u
LDA_ROOT=/mnt/beegfsnew/scratch/3295540/LDA-1B
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=10093

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""

# --- start server (LDA env) ---
cd "$LDA_ROOT"
/home/3295540/.conda/envs/lda_eval/bin/python \
  deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT" \
  --port "$PORT" \
  > /mnt/beegfsnew/scratch/3295540/robocasa-eval/server-${SLURM_JOB_ID}.log 2>&1 &
SERVER_PID=$!

# Wait for the server to actually be listening rather than a fixed sleep --
# checkpoint load onto GPU is the dominant cost and its duration isn't known
# in advance.
for i in $(seq 1 180); do
    if grep -q "server running" /mnt/beegfsnew/scratch/3295540/robocasa-eval/server-${SLURM_JOB_ID}.log 2>/dev/null; then
        echo "Server ready after ${i}0s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "SERVER_DIED_EARLY"
        cat /mnt/beegfsnew/scratch/3295540/robocasa-eval/server-${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 10
done

# --- run client (robocasa env) ---
# Run from the LDA repo root: simulation_env.py lives there and imports its
# siblings as `examples.Robocasa_tabletop.eval_files.*`, plus `lda.*` and
# `deployment.*`, so LDA_ROOT must be importable (matching the upstream
# README's `export PYTHONPATH=$(pwd)` from the LDA repo). robocasa/robosuite
# are pip-installed editable into this env, so they resolve from any cwd.
cd "$LDA_ROOT"
export PYTHONPATH="$LDA_ROOT:$ROBOCASA_ROOT:${PYTHONPATH:-}"
# run_client.py supplies the `log` method that LDA's own framework __init__
# calls but PureOverwatch never defines, then delegates to the unmodified
# simulation_env entry point -- see its docstring and probe job 630058.
/home/3295540/.conda/envs/robocasa/bin/python \
  eval/robocasa/launcher/run_client.py \
  --args.env_name gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env \
  --args.port "$PORT" \
  --args.n_episodes 6 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/smoke_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"

CLIENT_EXIT=$?
kill "$SERVER_PID" 2>/dev/null
echo "CLIENT_EXIT=$CLIENT_EXIT"
