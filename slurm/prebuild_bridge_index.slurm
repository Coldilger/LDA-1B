#!/bin/bash
#SBATCH --job-name=lda_index
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/index-%j.out

# The eval harness builds a step index over ALL 53k episodes before sampling any
# trajectory (~28 min at ~31 ep/s). That needs no GPU — it only reads parquet
# lengths — so precompute + cache it here on a CPU node, keeping the H200 for the
# actual forward pass. The cache (meta/steps_3_interval_data_index.pkl) is also
# reused by finetuning, so this is not throwaway work.
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /mnt/beegfsnew/scratch/3295540/LDA-1B
/home/3295540/.conda/envs/lda/bin/python - <<'PY'
from omegaconf import OmegaConf
from lda.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from lda.dataloader.gr00t_lerobot.data_config import OxeDataConfig
from lda.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

cfg = OxeDataConfig()
data_cfg = OmegaConf.create({"lerobot_version": "v2.0", "image_size": [224, 224]})
ds = LeRobotSingleDataset(
    dataset_path="/mnt/beegfsnew/scratch/3295540/data/bridge_lda",
    modality_configs=cfg.modality_config(),
    transforms=cfg.transform(),
    embodiment_tag=EmbodimentTag.OXE,
    video_backend=cfg.video_backend,
    img_interval=cfg.img_interval,
    data_cfg=data_cfg,
)
print("dataset len:", len(ds), flush=True)
print("INDEX_CACHED", flush=True)
PY
echo "INDEX_EXIT=$?"
