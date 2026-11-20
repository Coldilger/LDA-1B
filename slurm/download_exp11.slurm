#!/bin/bash
#SBATCH --job-name=lda_dl_exp11
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3295540/LDA-1B/slurm/logs/download-exp11-%j.out

# Experiment 11 (advisor's point 5) inputs for LDA-1B: the LDA-pretrain
# checkpoint (wiped from scratch by the 30-day purge; checkpoints/LDA-pretrain
# is a symlink into this HF cache snapshot) and the RoboCasa GR1 training data
# of the evaluated task (PnPCupToDrawerClose, 1000 demos, ~2 GB).
ulimit -t unlimited
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_HUB_DISABLE_XET=1
if [ -f "$HOME/.cache/huggingface/token" ]; then
  export HF_TOKEN="$(cat "$HOME/.cache/huggingface/token")"
fi

/home/3295540/.conda/envs/f1_vla/bin/python - <<'PY'
from huggingface_hub import snapshot_download
print("=== LDA-pretrain ===", flush=True)
print(snapshot_download("Wayer2/LDA-pretrain"), flush=True)
print("=== RoboCasa GR1 PnPCupToDrawerClose ===", flush=True)
print(snapshot_download("nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim", repo_type="dataset",
                        allow_patterns=["gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000/*"],
                        local_dir="/mnt/beegfsnew/scratch/3295540/data/robocasa_gr1", max_workers=8), flush=True)
PY
echo "DOWNLOAD_EXIT=$?"
