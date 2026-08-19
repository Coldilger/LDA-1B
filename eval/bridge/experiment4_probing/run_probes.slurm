#!/bin/bash
#SBATCH --job-name=lda_exp4_probes
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/experiment4_probing/probes-%j.out

set -u
cd /mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/experiment4_probing
PY=/home/3295540/.conda/envs/lda_eval/bin/python

echo "=== train_probe.py (future pose) ==="
"$PY" train_probe.py --features features_lda_robocasa.npz --json-out probe_result.json

echo ""
echo "=== positive_control.py (episode identity) ==="
"$PY" positive_control.py --features features_lda_robocasa.npz

echo ""
echo "=== compare_targets.py (current vs future pose) ==="
"$PY" compare_targets.py --features features_lda_robocasa.npz --current-pose features_lda_robocasa.npz
