#!/bin/bash
#SBATCH --job-name=lda_models_dl
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/home/3295540/LDA-1B/slurm/logs/models-dl-%j.out
# Re-downloads everything the RoboCasa eval server loads (lost in the 2026-10
# scratch purge), in the layout eval/robocasa/*.slurm and lda_policy.py expect:
#   checkpoints/LDA-robocasa_run/{config.yaml,dataset_statistics.json}
#   checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt
#   pretrained/vlm/Qwen3-VL-4B-Instruct
#   pretrained/dinov3-vits16-pretrain-lvd1689m        (gated, needs HF_TOKEN)
# checkpoints/ and pretrained/ are symlinks to scratch (setup_scratch_links.sh).
# The published config.yaml hardcodes vision_encoder_path to the authors'
# machine; it is repointed at this repo's pretrained/ (see
# eval/robocasa/RESULTS.md, "Setup notes").
set -euo pipefail
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export HF_TOKEN="$(cat /home/3295540/.cache/huggingface/token)"
REPO=/home/3295540/LDA-1B
"$REPO/setup_scratch_links.sh"
RUN="$REPO/checkpoints/LDA-robocasa_run"
mkdir -p "$RUN/checkpoints"

/home/3295540/.conda/envs/lda_eval/bin/python - <<EOF
from huggingface_hub import snapshot_download, hf_hub_download
run, repo = "$RUN", "$REPO"
for f in ["config.yaml", "dataset_statistics.json"]:
    hf_hub_download("Wayer2/LDA-robocasa", f, local_dir=run)
hf_hub_download("Wayer2/LDA-robocasa", "LDA-robocasa.pt", local_dir=f"{run}/checkpoints")
snapshot_download("Qwen/Qwen3-VL-4B-Instruct", local_dir=f"{repo}/pretrained/vlm/Qwen3-VL-4B-Instruct")
snapshot_download("facebook/dinov3-vits16-pretrain-lvd1689m",
                  local_dir=f"{repo}/pretrained/dinov3-vits16-pretrain-lvd1689m")
EOF
echo "DOWNLOAD_EXIT=$?"
sed -i "s#vision_encoder_path: /World-Action-Model/pretrained#vision_encoder_path: $REPO/pretrained#" "$RUN/config.yaml"
grep -n 'vision_encoder_path' "$RUN/config.yaml"
du -shL "$RUN" "$REPO/pretrained"/* "$REPO/pretrained/vlm"/*
