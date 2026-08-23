#!/bin/bash
#SBATCH --job-name=lda_robocasa_wm_on_timing
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/wm-on-timing-%j.out

# Experiment 3 (cost per decision) for LDA-1B, run against the confirmed-
# working RoboCasa checkpoint/egocentric setup (48% on the authors' own
# benchmark, see eval/robocasa/RESULTS.md) rather than Bridge, which is
# still stuck at 0% closed-loop -- see eval/bridge/experiment3_cost/README.md.
# Same server/client split as eval_robocasa_full.slurm, just swapping in the
# timed server variant (server_policy_worldmodel_on_timed.py) and a smaller episode count
# -- latency isn't task-dependent the way success rate is, so a handful of
# episodes is enough for a stable median/p95 (same logic F1-VLA/mimic-video's
# own Exp3 docs already used).
#
# Result (2026-08-18, job 631084): n=182, median 254.7ms, p95 256.8ms,
# mean 279.2ms -- see eval/bridge/experiment3_cost/README.md.

set -u
LDA_ROOT=/mnt/beegfsnew/scratch/3295540/LDA-1B
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
CKPT=/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
PORT=10098
SERVER_LOG=/mnt/beegfsnew/scratch/3295540/robocasa-eval/server-wm-on-timing-${SLURM_JOB_ID}.log

export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export DISPLAY=""
ulimit -t unlimited

# --- start server (LDA env), timed variant ---
cd "$LDA_ROOT"
# See eval_robocasa_oracle_probe_v2.slurm's own comment: `deployment` needs
# LDA_ROOT on PYTHONPATH explicitly, not inherited -- confirmed missing
# 2026-08-23.
export PYTHONPATH="$LDA_ROOT:${PYTHONPATH:-}"
/home/3295540/.conda/envs/lda_eval/bin/python \
  eval/robocasa/server_policy_worldmodel_on_timed.py \
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
  --args.n_episodes 3 \
  --args.n_envs 1 \
  --args.max_episode_steps 720 \
  --args.n_action_steps 12 \
  --args.video_out_path /mnt/beegfsnew/scratch/3295540/robocasa-eval/videos/wm_on_timing_${SLURM_JOB_ID} \
  --args.pretrained_path "$CKPT"

CLIENT_EXIT=$?
kill "$SERVER_PID" 2>/dev/null
echo "CLIENT_EXIT=$CLIENT_EXIT"

# --- summarize latency from the timed server's own log ---
python3 - "$SERVER_LOG" <<'PYEOF'
import re, statistics, sys
vals = []
with open(sys.argv[1]) as f:
    for line in f:
        m = re.search(r"LATENCY_MS ([\d.]+)", line)
        if m:
            vals.append(float(m.group(1)))
if not vals:
    print("LATENCY: no samples recorded")
else:
    vals.sort()
    n = len(vals)
    median = statistics.median(vals)
    p95 = vals[int(0.95 * (n - 1))]
    mean = statistics.mean(vals)
    print(f"LATENCY variant=lda_robocasa_wm_on n={n} mean_ms={mean:.1f} "
          f"median_ms={median:.1f} p95_ms={p95:.1f} min_ms={vals[0]:.1f} max_ms={vals[-1]:.1f}")
PYEOF
