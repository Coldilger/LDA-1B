#!/bin/bash
#SBATCH --job-name=lda_smoke
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/smoke-%j.out

# Smoke-test the LDA inference path on the authors' own demo_data (4 episodes),
# BEFORE trusting our Bridge conversion. If this fails, the bug is in the model's
# inference path (which upstream only ever runs via train_hf.py); if a later
# Bridge run fails but this passed, the bug is in our data. That separation is
# the whole point of doing demo_data first (the F1-VLA lesson).
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
PY=/home/3295540/.conda/envs/lda/bin/python
# The loader (share_tools.read_mode_config) requires the layout
# <RUN_DIR>/checkpoints/<ckpt>.pt with config.yaml + dataset_statistics.json in
# RUN_DIR (parents[1]). The flat HF download doesn't match, so outputs/lda_pretrain_run
# is a symlinked view in that exact shape (dataset_statistics.json -> stats.json).
RUN=outputs/lda_pretrain_run
CKPT=checkpoints/LDA-pretrain

$PY lda/eval/eval_policy.py \
  --config_yaml ${CKPT}/config.yaml \
  --seed 42 \
  --evaluation.model_path ${RUN}/checkpoints/LDA-pretrain.pt \
  --datasets.vla_data.data_root_dir playground/demo_data \
  --datasets.vla_data.data_mix demo_data \
  --datasets.vla_data.lerobot_version v2.0 \
  --evaluation.action_horizon 16 \
  --evaluation.trajs 1 \
  --evaluation.start_traj 0 \
  --evaluation.end_traj 4 \
  --evaluation.plot false \
  --evaluation.plot_state false \
  --evaluation.create_trajectory_video false \
  --evaluation.save_plot_path outputs/smoke_demo/plots \
  --evaluation.video_output_path outputs/smoke_demo/video \
  --evaluation.original_video_path outputs/smoke_demo/orig \
  --is_delta_action false
echo "SMOKE_EXIT=$?"
