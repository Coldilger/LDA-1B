#!/bin/bash
#SBATCH --job-name=lda_state_live
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/state-live-%j.out

# Same Carrot-task config as eval_bridge_simpler.slurm, but only 6 episodes
# (this is a diagnostic, not a scored run) and instrumented to log
# _build_state()'s actual live output range across a real closed-loop
# rollout with real predicted actions -- see diag_live_state_range.py.

export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false

SIMPLER=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv
cd "$SIMPLER"

CKPT="${CKPT:?set CKPT}"
/home/3295540/.conda/envs/lda_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/diag_live_state_range.py \
  --lda-checkpoint-path "$CKPT" \
  --lda-seed 0 \
  --lda-exec-horizon 8 \
  --ckpt-path "$CKPT" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 6 \
  --additional-env-save-tags "diag_state_live_${SLURM_JOB_ID}" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name unused --vam-video-model-path unused --vam-action-model-path unused \
  --vam-dataset-statistics-path unused --vam-img-horizon 0 --vam-lowdim-horizon 0 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 0

echo "STATE_LIVE_EXIT=$?"
