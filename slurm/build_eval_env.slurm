#!/bin/bash
#SBATCH --job-name=lda_evalenv
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/evalenv-%j.out

# The closed-loop eval needs one interpreter with BOTH stacks: LDA (transformers
# 4.57 for Qwen3-VL, flash-attn) and SimplerEnv (sapien, mani_skill2_real2sim).
# f1_vla_eval has the second but transformers 4.49, too old for Qwen3-VL; the lda
# env has the first but no sapien.
#
# This clones lda rather than installing into it, because the finetune is running
# out of that env right now and each 6-hour wave starts a fresh interpreter from
# it -- a dependency resolution that went sideways would take the training down
# with it.
source /software/miniconda3/etc/profile.d/conda.sh
conda create --clone lda --name lda_eval -y

PY=/home/3295540/.conda/envs/lda_eval/bin/python
$PY -m pip install --no-input sapien==2.2.2 gymnasium==0.29.1 transforms3d==0.4.2

# Versions pinned to whatever f1_vla_eval runs, since that env has completed
# rollouts. Each of these was a separate import failure found by trying:
#   ruckig      - simpler_env imports it at module load
#   trimesh, rtree, mediapy, natsort - pulled in transitively by the above
$PY -m pip install --no-input ruckig==0.19.4 trimesh==4.12.2 rtree==1.4.1 \
    mediapy==1.2.7 natsort==8.4.0

# The SimplerEnv installs above drag in numpy 2, under which transforms3d's
# np.array(..., copy=False) raises instead of copying-if-needed, and every episode
# dies in quat2euler. Both working envs run 1.26.4; pin back to it.
$PY -m pip install --no-input "numpy==1.26.4"

# sapien 2.2.2 imports pkg_resources, which setuptools removed in 81. The clone
# inherits 83 from the lda env, so it has to come back down.
$PY -m pip install --no-input "setuptools==79.0.1"

# sapien needs libvulkan.so.1. Copied into the env's own lib dir rather than set
# via LD_LIBRARY_PATH: sapien finds it there automatically, and $HOME survives
# the scratch purge that already ate one copy of these libs.
cp -n /home/3295540/.conda/envs/f1_vla_eval/lib/libvulkan.so.1 \
      /home/3295540/.conda/envs/lda_eval/lib/ || true
# Same editable installs f1_vla_eval uses, pointing at the SimplerEnv copy
# F1-VLA already vendors and has completed runs against.
$PY -m pip install --no-input --no-deps -e /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv/ManiSkill2_real2sim
$PY -m pip install --no-input --no-deps -e /mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv

echo "=== import check ==="
$PY - <<'PY'
import importlib
for m in ["sapien", "mani_skill2_real2sim", "simpler_env", "transformers", "flash_attn", "lda"]:
    try:
        mod = importlib.import_module(m)
        print(f"OK   {m} {getattr(mod, '__version__', '')}")
    except Exception as e:
        print(f"FAIL {m}: {type(e).__name__}: {e}")
PY
echo "EVALENV_EXIT=$?"
