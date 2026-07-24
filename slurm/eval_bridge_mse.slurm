#!/bin/bash
#SBATCH --job-name=lda_brmse
#SBATCH --partition=short_gpuh200
#SBATCH --gres=gpu:H200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/brmse-%j.out

# Open-loop MSE of LDA-pretrain on held-out converted Bridge episodes, tag "oxe"
# (idx 5 — an in-range pretrain embodiment). Purpose: settle whether Bridge was
# meaningfully in the OXE pretrain slice. Low L1 -> yes, finetune from the oxe
# projector; high/degenerate L1 -> Bridge wasn't really there, finetune will
# have to teach the oxe slot Bridge's distribution from scratch.
#
# is_delta_action=true selects lda/utils/eval_relative_eef.py, which is the harness
# that matches our config's use_delta_action: true. The other one (eval_wo_postprocess,
# is_delta_action=false) is broken for this layout and was clearly never executed
# upstream: its generic bimanual branch never assigns `modality_keys` before reading it
# (UnboundLocalError, which is how we found this), and it compares denormalized delta
# predictions against raw absolute ground truth. eval_relative_eef instead maps the
# predicted deltas back to absolute poses via delta2abs and compares at the same index,
# and its calculate_mse has a correct D==14 branch (pos [0,1,2,7,8,9], gripper [6,13]).
#
# This run doubles as the inference-path smoke test: the authors' demo_data can't
# serve that role because it maps to new_embodiment (idx 32), which is out of
# range for this checkpoint's 32-slot (0-31) embodiment tables. oxe=5 exercises a
# genuinely trained projector path, so a finite forward here validates inference.
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export CUDA_HOME=/home/3295540/.conda/envs/cuda_env
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TOKENIZERS_PARALLELISM=false
# Make any remaining CUDA index errors report the exact Python line, not an async
# device-side assert dump.
export CUDA_LAUNCH_BLOCKING=1

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
PY=/home/3295540/.conda/envs/lda/bin/python
RUN=outputs/lda_pretrain_run
CKPT=checkpoints/LDA-pretrain

$PY lda/eval/eval_policy.py \
  --config_yaml ${CKPT}/config.yaml \
  --seed 42 \
  --evaluation.model_path ${RUN}/checkpoints/LDA-pretrain.pt \
  --datasets.vla_data.data_root_dir /mnt/beegfsnew/scratch/3295540/data \
  --datasets.vla_data.data_mix bridge_lda \
  --datasets.vla_data.lerobot_version v2.0 \
  --evaluation.action_horizon 16 \
  --evaluation.trajs 8 \
  --evaluation.start_traj 0 \
  --evaluation.end_traj 100000 \
  --evaluation.plot false \
  --evaluation.plot_state false \
  --evaluation.create_trajectory_video false \
  --evaluation.save_plot_path outputs/bridge_mse/plots \
  --evaluation.video_output_path outputs/bridge_mse/video \
  --evaluation.original_video_path outputs/bridge_mse/orig \
  --is_delta_action true
echo "BRMSE_EXIT=$?"
