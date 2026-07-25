#!/bin/bash
# Fan the SimplerEnv eval out over the full protocol: 4 tasks x 3 seeds x 24
# episodes -- the same grid F1-VLA and mimic-video were measured on, so the three
# numbers are comparable.
#
# Three seeds because a single 24-episode run is noisy: during the F1-VLA work the
# same config scored 29.2% and 45.8% on the carrot task on different seeds, so
# any single run is a point estimate with a wide error bar. Every reported number
# is a mean of three.
#
# Usage:
#   CKPT=outputs/lda_bridge/bridge_finetune/checkpoints/steps_150000_pytorch_model.pt \
#     bash slurm/submit_eval_sweep.sh
#
# Optional: EXEC_HORIZON (default 8), SEEDS (default "0 1 2")

set -eu
: "${CKPT:?set CKPT to the finetuned checkpoint .pt}"
EXEC_HORIZON="${EXEC_HORIZON:-8}"
SEEDS="${SEEDS:-0 1 2}"

TASKS=(
  PutCarrotOnPlateInScene-v0
  PutSpoonOnTableClothInScene-v0
  StackGreenCubeOnYellowCubeBakedTexInScene-v0
  PutEggplantInBasketScene-v0
)

if [ ! -f "$CKPT" ]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 1
fi

cd "$(dirname "$0")/.."
n=0
for task in "${TASKS[@]}"; do
  for seed in $SEEDS; do
    jid=$(sbatch --parsable \
      --export=ALL,TASK="$task",SEED="$seed",CKPT="$CKPT",EXEC_HORIZON="$EXEC_HORIZON" \
      slurm/eval_bridge_simpler.slurm)
    echo "submitted $jid  $task  seed=$seed"
    n=$((n + 1))
  done
done
echo "$n jobs submitted (exec_horizon=$EXEC_HORIZON)"
