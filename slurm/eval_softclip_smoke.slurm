#!/bin/bash
#SBATCH --job-name=lda_softclip
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/softclip-%j.out

# Soft-clip normalization experiment (see eval/bridge/lda_policy_softclip.py) --
# full 24-episode Carrot run, one seed, on the v3/150k checkpoint, to see
# whether replacing the hard q99 clamp with a smooth saturation changes
# closed-loop success at all. The real eval sweep scored exactly 0/24 x 3
# seeds on this task with the unmodified checkpoint, so any nonzero result
# here (even at n=24) is already a meaningfully different signal.

set -u
: "${SEED:?set SEED}"
CKPT="outputs/lda_bridge_v3/bridge_finetune_v3/final_model/pytorch_model.pt"

export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false

SIMPLER=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv
cd "$SIMPLER"

/home/3295540/.conda/envs/lda_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/main_inference_softclip.py \
  --lda-checkpoint-path "$CKPT" \
  --lda-seed "$SEED" \
  --lda-exec-horizon 8 \
  --ckpt-path "$CKPT" \
  --robot widowx --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps 60 \
  --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
  --rgb-overlay-path ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png \
  --robot-init-x 0.147 0.147 1 --robot-init-y 0.028 0.028 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --additional-env-save-tags "lda_softclip_seed${SEED}" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name unused --vam-video-model-path unused --vam-action-model-path unused \
  --vam-dataset-statistics-path unused --vam-img-horizon 0 --vam-lowdim-horizon 0 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 0

echo "SOFTCLIP_EVAL_EXIT=$? SEED=$SEED"
