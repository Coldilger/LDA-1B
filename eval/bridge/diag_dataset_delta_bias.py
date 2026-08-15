#!/usr/bin/env python3
"""Root-cause check for the systematic per-step bias found in diag_drift_ablation.py
(z position +~2.9mm/step, yaw +~0.27deg/step, consistently signed, not noise).

No model involved here at all -- this only reads real bridge_orig_lerobot state
data and checks whether the TRAINING DATA ITSELF has a non-zero average
per-step delta on these same axes. If the dataset's own mean matches the
model's measured bias, the model most likely learned a real property of its
training distribution (e.g. many Bridge demonstrations involve a net lift
before/during manipulation, so the average step is not zero-mean) rather than
having a training or eval-time bug -- a generative/regression model tends to
fall back toward the training distribution's own mean under uncertainty.

Per convert_bridge_to_lda.py's docstring, ground-truth action-pose columns are
built directly from consecutive real state[0:6] (xyz+rpy) -- so state_{t+1} -
state_t in world frame is exactly what the model's own target deltas are
built from (before calculate_delta_eef's frame conversion). Averaging that
directly over many real transitions is the most direct way to see if the
dataset itself carries the bias.
"""

import pathlib

import numpy as np
from scipy.spatial.transform import Rotation

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata


def main():
    dataset_root = pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")
    n_episodes = 300
    max_steps_per_ep = 40  # matches the diagnostics' rollout_len range plus margin

    meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
    rng = np.random.default_rng(2)
    # Restricted to the first 1000 episode IDs: the dataset re-download (job 626469)
    # is still in progress, and this range has reliably been fully present in every
    # earlier diagnostic run in this file (diag_autonomous_drift.py etc.) -- sampling
    # from the full range hit a still-missing chunk (AssertionError -> confusing HF
    # Hub fallback error, same failure mode as diag_gripper_selfhistory.py earlier).
    safe_pool = min(1000, meta.total_episodes)
    episode_ids = rng.choice(safe_pool, size=min(n_episodes, safe_pool), replace=False).tolist()
    ds = LeRobotDataset(repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav")

    pos_deltas = []
    rot_deltas = []
    n_used_eps = 0

    for pos_i in range(len(episode_ids)):
        ep_from = int(ds.episode_data_index["from"][pos_i])
        ep_to = int(ds.episode_data_index["to"][pos_i])
        length = min(max_steps_per_ep, ep_to - ep_from - 1)
        if length < 2:
            continue
        n_used_eps += 1

        states = np.stack([ds[ep_from + k]["observation.state"].numpy() for k in range(length + 1)]).astype(np.float64)
        pos = states[:, 0:3]
        rpy = states[:, 3:6]

        pos_d = pos[1:] - pos[:-1]  # (length, 3), world-frame per-step position delta
        pos_deltas.append(pos_d)

        rots = Rotation.from_euler("xyz", rpy)
        # per-step relative rotation in world frame: R_{t+1} * R_t^{-1}, as euler
        rel = rots[1:] * rots[:-1].inv()
        rot_deltas.append(rel.as_euler("xyz"))

    pos_deltas = np.concatenate(pos_deltas, axis=0)  # (N, 3)
    rot_deltas_deg = np.degrees(np.concatenate(rot_deltas, axis=0))  # (N, 3)

    print(f"episodes used: {n_used_eps} / {len(episode_ids)} sampled, up to {max_steps_per_ep} steps each")
    print(f"total transitions: {len(pos_deltas)}")
    print()
    print("position delta per step, world frame (m):")
    print(f"  mean:   x={pos_deltas[:,0].mean():+.5f}  y={pos_deltas[:,1].mean():+.5f}  z={pos_deltas[:,2].mean():+.5f}")
    print(f"  std:    x={pos_deltas[:,0].std():.5f}  y={pos_deltas[:,1].std():.5f}  z={pos_deltas[:,2].std():.5f}")
    print(f"  mean |delta| (typical step size): x={np.abs(pos_deltas[:,0]).mean():.5f}  y={np.abs(pos_deltas[:,1]).mean():.5f}  z={np.abs(pos_deltas[:,2]).mean():.5f}")
    print()
    print("rotation delta per step, world frame (deg):")
    print(f"  mean:   roll={rot_deltas_deg[:,0].mean():+.4f}  pitch={rot_deltas_deg[:,1].mean():+.4f}  yaw={rot_deltas_deg[:,2].mean():+.4f}")
    print(f"  std:    roll={rot_deltas_deg[:,0].std():.4f}  pitch={rot_deltas_deg[:,1].std():.4f}  yaw={rot_deltas_deg[:,2].std():.4f}")
    print()
    print("Compare against the model's measured single-step bias (diag_drift_ablation.py):")
    print("  position: x=+0.0009  y=-0.0001  z=+0.0029  (m/step)")
    print("  rotation: roll=-0.16  pitch=-0.23  yaw=+0.27  (deg/step)")
    print()
    print("If the dataset's own mean per-step delta on these axes is close in sign and rough")
    print("magnitude to the model's bias, the model most likely learned a real, non-zero-mean")
    print("property of the training distribution itself (not a bug in eval-time code or a random")
    print("training artifact). If the dataset mean is ~0 on an axis where the model is biased,")
    print("that axis's bias is NOT explained by dataset asymmetry and points elsewhere (e.g. the")
    print("delta computation/frame-conversion pipeline, or genuine undertraining on that channel).")


if __name__ == "__main__":
    main()
