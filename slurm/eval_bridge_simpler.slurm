#!/bin/bash
#SBATCH --job-name=lda_eval
# H100 rather than H200: the H200 nodes have spent most of this run draining,
# so a 12-job sweep queued there would simply sit. Override with sbatch
# --partition/--gres if H200 frees up.
#SBATCH --partition=short_gpunew
#SBATCH --gres=gpu:H100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
# gnode07 reads idle to SLURM but another user's long-running inference
# servers permanently occupy ~20GB of its GPU (see the "gnode07 GPU
# squatter" memory note) -- excluded to avoid the non-deterministic OOM
# that already hit an LDA training job twice on this node.
#SBATCH --exclude=gnode07
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/eval-%j.out

# One task x one seed x 24 episodes of closed-loop SimplerEnv rollout.
# Submit via slurm/submit_eval_sweep.sh, which fans this out over the 4 tasks and
# 3 seeds that make up the protocol F1-VLA and mimic-video were measured on.
#
# Required env vars: TASK, SEED, CKPT
# Optional: EXEC_HORIZON (default 8)

set -u
: "${TASK:?set TASK}"
: "${SEED:?set SEED}"
: "${CKPT:?set CKPT (path to the finetuned .pt)}"
EXEC_HORIZON="${EXEC_HORIZON:-8}"

# libvulkan lives in the env's own lib dir, so sapien finds it without
# LD_LIBRARY_PATH (see slurm/build_eval_env.slurm).
export MS2_REAL2SIM_ASSET_DIR=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim/data
export DISPLAY=""
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false

# Three of the four tasks share the table scene; the eggplant task uses a
# different scene, a different robot camera setup and twice the step budget.
case "$TASK" in
  PutCarrotOnPlateInScene-v0|PutSpoonOnTableClothInScene-v0|StackGreenCubeOnYellowCubeBakedTexInScene-v0)
    ROBOT=widowx
    SCENE=bridge_table_1_v1
    OVERLAY=ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png
    INIT_X=0.147; INIT_Y=0.028; MAX_STEPS=60
    ;;
  PutEggplantInBasketScene-v0)
    ROBOT=widowx_sink_camera_setup
    SCENE=bridge_table_1_v2
    OVERLAY=ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png
    INIT_X=0.127; INIT_Y=0.06; MAX_STEPS=120
    ;;
  *)
    echo "unknown TASK: $TASK" >&2; exit 2
    ;;
esac

SIMPLER=/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv
cd "$SIMPLER"

# --additional-env-save-tags must be unique per run. The evaluator silently skips
# any episode whose output video already exists, so without it a rerun reuses the
# previous run's footage and reports a number that was never measured -- this
# produced a phantom 0.0% during the F1-VLA work (RESULTS.md, "Method note").
# Worse than that note implies: results/ is keyed by CKPT's basename only
# (e.g. "pytorch_model.pt"), which is IDENTICAL across every LDA training run,
# so two different checkpoints at the same seed/exec_horizon collide too --
# discovered 2026-08-14 when a fresh checkpoint silently inherited an Aug-5
# run's stale 0.0. CKPT_TAG below disambiguates by the checkpoint's parent dir
# name (e.g. "bridge_finetune_v2"), which is unique per training run here.
CKPT_TAG="$(basename "$(dirname "$(dirname "$CKPT")")")"
/home/3295540/.conda/envs/lda_eval/bin/python \
  /mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge/main_inference.py \
  --lda-checkpoint-path "$CKPT" \
  --lda-seed "$SEED" \
  --lda-exec-horizon "$EXEC_HORIZON" \
  --ckpt-path "$CKPT" \
  --robot "$ROBOT" --policy-setup widowx_bridge \
  --control-freq 5 --sim-freq 500 --max-episode-steps "$MAX_STEPS" \
  --env-name "$TASK" --scene-name "$SCENE" \
  --rgb-overlay-path "$OVERLAY" \
  --robot-init-x $INIT_X $INIT_X 1 --robot-init-y $INIT_Y $INIT_Y 1 \
  --obj-variation-mode episode --obj-episode-range 0 24 \
  --additional-env-save-tags "lda_${CKPT_TAG}_seed${SEED}_ex${EXEC_HORIZON}" \
  --robot-init-rot-quat-center 0 0 0 1 --robot-init-rot-rpy-range 0 0 1 0 0 1 0 0 1 \
  --vam-experiment-name unused --vam-video-model-path unused --vam-action-model-path unused \
  --vam-dataset-statistics-path unused --vam-img-horizon 0 --vam-lowdim-horizon 0 \
  --vam-stop-video-denoising-step 0 --vam-num-execute-actions 0

echo "EVAL_EXIT=$? TASK=$TASK SEED=$SEED"
