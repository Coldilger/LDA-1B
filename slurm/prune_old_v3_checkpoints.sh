#!/bin/bash
# Keeps only the KEEP_N most recent numbered checkpoints in
# outputs/lda_bridge_v3/bridge_finetune_v3/checkpoints/, deleting older ones.
# Each is 14.4GB and save_interval=10000 over a 150000-step run means up to
# 15 of these (~216GB) if nothing prunes them -- exactly the checkpoint-bloat
# risk already flagged in memory before this run started.
#
# Sorts NUMERICALLY on the step count in the filename (steps_<N>_pytorch_model.pt),
# not lexicographically -- lexicographic sort breaks once step counts cross a
# digit boundary (e.g. "steps_100000_..." would sort before "steps_20000_...").
# Never touches final_model/ (the run's designated end-of-training output) or
# any file outside this specific checkpoints/ directory.

set -euo pipefail

KEEP_N=3
CKPT_DIR="/mnt/beegfsnew/scratch/3295540/LDA-1B/outputs/lda_bridge_v3/bridge_finetune_v3/checkpoints"

[ -d "$CKPT_DIR" ] || exit 0

mapfile -t files < <(
  find "$CKPT_DIR" -maxdepth 1 -name 'steps_*_pytorch_model.pt' -printf '%f\n' \
    | sed -E 's/^steps_([0-9]+)_pytorch_model\.pt$/\1 &/' \
    | sort -n -k1,1 \
    | awk '{print $2}'
)

n=${#files[@]}
if [ "$n" -le "$KEEP_N" ]; then
  echo "$(date -Is) prune: $n checkpoint(s) present, KEEP_N=$KEEP_N, nothing to do"
  exit 0
fi

to_delete=("${files[@]:0:$((n - KEEP_N))}")
for f in "${to_delete[@]}"; do
  echo "$(date -Is) prune: removing $f"
  rm -f -- "$CKPT_DIR/$f"
done
