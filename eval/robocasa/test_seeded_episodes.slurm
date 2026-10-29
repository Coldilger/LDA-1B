#!/bin/bash
#SBATCH --job-name=lda_seed_test
#SBATCH --partition=medium_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --exclude=gnode07,gnode14,gnode15
#SBATCH --output=/home/3295540/LDA-1B/slurm/logs/seed-test-%j.out
# Checks that launcher/seeded_episodes.py makes RoboCasa scenes reproducible
# (paired protocol); runs the test once patched and once unpatched.
set -u
ulimit -t unlimited
export DISPLAY=""
ROBOCASA_ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks
cd /home/3295540/LDA-1B
export PYTHONPATH="/home/3295540/LDA-1B:$ROBOCASA_ROOT:${PYTHONPATH:-}"
PY=/home/3295540/.conda/envs/robocasa/bin/python
EPISODE_SEED_BASE=1000 $PY eval/robocasa/launcher/test_seeded_episodes.py 2>&1 | grep -E '^\[|SEED_TEST|Error|Traceback'
$PY eval/robocasa/launcher/test_seeded_episodes.py 2>&1 | grep -E '^\[|SEED_TEST|Error|Traceback'
