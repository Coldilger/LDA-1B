#!/bin/bash
#SBATCH --job-name=diag_task_embeddings
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/eval/robocasa/diag-taskemb-%j.out

# One-off diagnostic: is id_embedding (inverse_dynamics' task-conditioning
# token) actually trained, or still near its random init? Init is
# `0.02 * randn(inner_dim)` for every one of these task embeddings
# (MMDiT_ActionHeader.py __init__) -- if id_embedding's norm/distribution
# still looks like that scale while policy_embedding has clearly drifted,
# that's direct evidence inverse_dynamics was rarely or never sampled
# during this checkpoint's training, which would explain the L1 probe
# anomaly (oracle and world-model-on performing ~identically bad,
# independent of whether the future frame is real or imagined -- pointing
# at the shared inverse_dynamics path itself, not the frame source).

set -u
/home/3295540/.conda/envs/lda_eval/bin/python -c "
import torch
sd = torch.load(
    '/mnt/beegfsnew/scratch/3295540/LDA-1B/checkpoints/LDA-robocasa_run/checkpoints/LDA-robocasa.pt',
    map_location='cpu', weights_only=False,
)
if hasattr(sd, 'state_dict'):
    sd = sd.state_dict()
elif isinstance(sd, dict) and 'state_dict' in sd:
    sd = sd['state_dict']
elif isinstance(sd, dict) and 'model' in sd:
    sd = sd['model']

targets = ['policy_embedding', 'id_embedding', 'vg_embedding', 'fd_embedding', 'next_obs_learnable_tokens']
found = {}
for k, v in sd.items():
    for t in targets:
        if k.endswith(t):
            found[t] = v
            break

expected_init_std = 0.02
for t in targets:
    if t not in found:
        print(f'{t}: NOT FOUND in state_dict')
        continue
    v = found[t].float()
    print(f'{t}: shape={tuple(v.shape)} mean={v.mean().item():.6f} std={v.std().item():.6f} '
          f'norm={v.norm().item():.4f} (init std would be ~{expected_init_std})')
"
echo "DIAG_EXIT=$?"
