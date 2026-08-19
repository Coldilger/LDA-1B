#!/bin/bash
#SBATCH --job-name=diag_param_counts
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/eval/robocasa/diag-params-%j.out

# One-off diagnostic (not part of any committed experiment): exact parameter
# counts for mimic-video's and LDA-1B's checkpoints, to compare against
# F1-VLA's (already read directly from its safetensors header: 4.19B).
# Loads with map_location='meta' so shapes/dtypes are read without
# materializing actual weight data -- safe on a CPU-only node even for a
# 14GB checkpoint, and doesn't touch the login node (per this cluster's
# "run heavy things through sbatch" policy).

set -u
LDA_PY=/home/3295540/.conda/envs/lda_eval/bin/python
MIMIC_PY=/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/.venv/bin/python3

echo "=== mimic-video video backbone ==="
"$MIMIC_PY" -c "
import torch
sd = torch.load('/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/checkpoints/video_backbone/v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused.pt', map_location='meta', weights_only=False)
if hasattr(sd, 'state_dict'):
    sd = sd.state_dict()
elif isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
total = 0
dtypes = {}
for k, v in sd.items():
    if hasattr(v, 'numel'):
        n = v.numel()
        total += n
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + n
print('total params:', f'{total:,}', f'({total/1e9:.2f}B)')
print('dtypes:', {k: f'{v/1e9:.2f}B' for k,v in dtypes.items()})
"

echo "=== mimic-video action decoder ==="
"$MIMIC_PY" -c "
import torch
sd = torch.load('/mnt/beegfsnew/scratch/3295540/mimic-video-project/mimic-video/model/checkpoints/action_decoder/w2a_bridge_v2w_bridge_lora_rank256_lr1.778e-04_bsz64_iter_000070043_fused_lr1.000e-04_layer20_bsz256_iter_000014112.pt', map_location='meta', weights_only=False)
if hasattr(sd, 'state_dict'):
    sd = sd.state_dict()
elif isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
total = 0
dtypes = {}
for k, v in sd.items():
    if hasattr(v, 'numel'):
        n = v.numel()
        total += n
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + n
print('total params:', f'{total:,}', f'({total/1e9:.2f}B)')
print('dtypes:', {k: f'{v/1e9:.2f}B' for k,v in dtypes.items()})
"

echo "=== LDA-1B (RoboCasa checkpoint, same architecture family as Bridge v3) ==="
"$LDA_PY" -c "
import torch
sd = torch.load('/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt', map_location='meta', weights_only=False)
if hasattr(sd, 'state_dict'):
    sd = sd.state_dict()
elif isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
elif isinstance(sd, dict) and 'model' in sd:
    sd = sd['model']
total = 0
dtypes = {}
for k, v in sd.items():
    if hasattr(v, 'numel'):
        n = v.numel()
        total += n
        dtypes[str(v.dtype)] = dtypes.get(str(v.dtype), 0) + n
print('total params:', f'{total:,}', f'({total/1e9:.2f}B)')
print('dtypes:', {k: f'{v/1e9:.2f}B' for k,v in dtypes.items()})
"

echo "PARAM_COUNT_DIAG_DONE"
