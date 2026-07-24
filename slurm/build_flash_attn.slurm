#!/bin/bash
#SBATCH --job-name=lda_flash
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/flash-%j.out

# flash-attn compiles ~50 CUDA kernels; on the 4-core login node this pins the
# box (load avg hit 90) and drops everyone's SSH. It needs no GPU, only nvcc,
# so it belongs on a CPU compute node.
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"

# This conda CUDA package keeps its headers under targets/<arch>/include, NOT
# directly in $CUDA_HOME/include. The first build failed with
# "cuda_runtime_api.h: No such file or directory" because nvcc's host compiler
# never saw that path — so add it to the C/C++ include search explicitly.
export CPATH="$CUDA_HOME/targets/x86_64-linux/include:$CUDA_HOME/include:$CPATH"

# Build for H100 (sm_90) and A100 (sm_80) only — the full default arch list
# roughly triples build time for arches this cluster does not have.
export TORCH_CUDA_ARCH_LIST="8.0;9.0"

# The first attempt OOM-killed with MAX_JOBS=16 on 64G (each nvcc kernel compile
# peaks well over 4G). 4 parallel jobs on 96G leaves comfortable headroom.
MAX_JOBS=4 /home/3295540/.local/bin/uv pip install \
  --python /home/3295540/.conda/envs/lda/bin/python \
  flash-attn==2.6.3 --no-build-isolation

echo "FLASH_ATTN_EXIT=$?"
/home/3295540/.conda/envs/lda/bin/python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
