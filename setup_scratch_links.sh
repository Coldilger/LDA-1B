#!/bin/bash
# Storage layout: code lives in /home (persistent), heavy artifacts live on
# /scratch (30-day purge). Each directory below is a symlink from this repo
# into a matching directory on scratch, so scripts that write to e.g.
# outputs/ or checkpoints/ keep large files off the /home quota.
#
# Safe to re-run: recreates missing scratch targets (e.g. after a purge) and
# leaves existing links alone. Refuses to replace a real, non-empty directory.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRATCH_DIR="/mnt/beegfsnew/scratch/3295540/LDA-1B"
LINKED_DIRS=(outputs checkpoints pretrained)

for d in "${LINKED_DIRS[@]}"; do
  src="$REPO_DIR/$d"
  dst="$SCRATCH_DIR/$d"
  mkdir -p "$dst"
  if [ -L "$src" ]; then
    echo "ok      $d -> $(readlink "$src")"
  elif [ -d "$src" ] && [ -n "$(ls -A "$src")" ]; then
    echo "SKIP    $d is a real non-empty directory; move its contents to $dst first" >&2
  else
    rmdir "$src" 2>/dev/null || true
    mkdir -p "$(dirname "$src")"
    ln -s "$dst" "$src"
    echo "linked  $d -> $dst"
  fi
done

# slurm --output needs this directory to exist (ignored by git in some repos)
mkdir -p "$REPO_DIR/slurm/logs"
