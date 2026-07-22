#!/bin/bash
#SBATCH --job-name=lda_flash
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/flash-%j.out

# flash-attn compiles ~50 CUDA kernels; on the 4-core login node this pins the
# box (load avg hit 90) and drops everyone's SSH. It needs no GPU, only nvcc,
# so it belongs on a CPU compute node.
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:$LD_LIBRARY_PATH"
# Build for H100 (sm_90) and A100 (sm_80) only — the full default arch list
# roughly triples build time for arches this cluster does not have.
export TORCH_CUDA_ARCH_LIST="8.0;9.0"

MAX_JOBS=16 /home/3295540/.local/bin/uv pip install \
  --python /home/3295540/.conda/envs/lda/bin/python \
  flash-attn==2.6.3 --no-build-isolation

echo "FLASH_ATTN_EXIT=$?"
/home/3295540/.conda/envs/lda/bin/python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
