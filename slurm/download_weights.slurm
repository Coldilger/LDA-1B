#!/bin/bash
#SBATCH --job-name=lda_dl
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/download-%j.out

# Downloads run as a job, not on the login node: the first attempt there was
# SIGKILLed partway through (2.1G of ~8G) when the node was overloaded.
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1

/home/3295540/.conda/envs/f1_vla/bin/python - <<'PY'
from huggingface_hub import snapshot_download

# Qwen3-VL-4B: resumes from whatever the killed run already fetched.
print("=== Qwen3-VL-4B-Instruct ===", flush=True)
print(snapshot_download("Qwen/Qwen3-VL-4B-Instruct"), flush=True)

# DINOv3 is a gated repo; this succeeds only once access is granted on HF.
print("=== DINOv3 ViT-S ===", flush=True)
try:
    print(snapshot_download("facebook/dinov3-vits16-pretrain-lvd1689m"), flush=True)
except Exception as e:
    print(f"DINOV3_BLOCKED: {type(e).__name__}: {e}", flush=True)
PY
echo "DOWNLOAD_EXIT=$?"
