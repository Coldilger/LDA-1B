#!/bin/bash
#SBATCH --job-name=mlp_current_pose_all
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/experiment4_probing/mlp-current-pose-%j.out

set -u
PY=/home/3295540/.conda/envs/lda_eval/bin/python
SCRIPT=/mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/experiment4_probing/mlp_current_pose.py

echo "############ LDA-1B (RoboCasa) ############"
"$PY" "$SCRIPT" \
  --features /mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/experiment4_probing/features_lda_robocasa.npz

echo ""
echo "############ F1-VLA (Bridge, finetuned, 1000 samples) ############"
"$PY" "$SCRIPT" \
  --features /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing/features_finetuned_more.npz \
  --current-pose /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/experiment4_probing/current_pose_more.npz

echo ""
echo "############ mimic-video (Bridge, finetuned v2, 498 samples) ############"
"$PY" "$SCRIPT" \
  --features /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing/features_finetuned_v2_more.npz \
  --current-pose /mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/eval/bridge/experiment4_probing/current_pose_more.npz
