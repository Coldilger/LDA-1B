#!/usr/bin/env python3
"""
Pre-retrain check: does prediction accuracy degrade specifically near the
q01/q99 clip boundary, on REAL logged Bridge data -- before spending days on
a mean_std retrain to test the hypothesis the expensive way.

RESULTS.md's diagnosis: closed-loop position (left_x/left_z) clips to exactly
-1.0/+1.0 for a growing fraction of steps as episodes progress, and the
hypothesis is that this destroys the model's ability to use position
feedback to self-correct. That's a fact about the clamp math
(torch.clamp(normalized, -1, 1)) regardless of any experiment. What's NOT yet
known: does prediction quality actually degrade once state is near/at that
boundary, or does the model do fine anyway (in which case a mean_std retrain
might not fix the closed-loop failure, and the real cause is something else)?

Method: reuses diag_policy_wrapper.py's proven pattern (seed LDAInference's
image_history/action_history from real consecutive frames of a real
bridge_orig_lerobot episode, call the real _predict_chunk exactly as
SimplerEnv would, compare predicted position delta against the real
state[t+1]-state[t] from the log) -- adapted for v3 by explicitly setting
_current_state (diag_policy_wrapper.py predates state_dim: 14 and never
needed this). Splits samples by how close their REAL, logged left_x/left_z
sits to the q01/q99 boundary (computed via the same normalization the model
was trained on) and reports position L1 separately for each group. No
SimplerEnv/ManiSkill2 involved -- fully offline, fully real data, no
artificial perturbation.
"""

import pathlib
import sys

import numpy as np

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/LDA-1B")
sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/LDA-1B/eval/bridge")

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata  # noqa: E402

from lda_policy import LDAInference  # noqa: E402


class PoseProxy:
    def __init__(self, p, q):
        self.p = p
        self.q = q  # scalar-first [w,x,y,z]


def pose_from_bridge_state(state: np.ndarray) -> PoseProxy:
    from scipy.spatial.transform import Rotation

    pos = state[0:3].astype(np.float64)
    rpy = state[3:6].astype(np.float64)
    quat_xyzw = Rotation.from_euler("xyz", rpy).as_quat()
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
    return PoseProxy(p=pos, q=quat_wxyz)


def to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def main():
    dataset_root = pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")
    camera_key = "observation.images.image_0"

    meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
    rng = np.random.default_rng(0)
    n_episodes = 60
    episode_ids = rng.choice(meta.total_episodes, size=n_episodes, replace=False).tolist()
    ds = LeRobotDataset(repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav")

    model = LDAInference(
        checkpoint_path="outputs/lda_bridge_v3/bridge_finetune_v3/final_model/pytorch_model.pt",
        seed=0,
    )
    assert model._state_dim is not None, "expected v3 (state_dim: 14) -- got state_dim: null"

    mid_range, near_boundary = [], []  # each: (pos_err, real_delta_mag)

    for pos_i, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos_i])
        ep_to = int(ds.episode_data_index["to"][pos_i])
        if ep_to - ep_from < 10:
            continue

        # A few candidate timesteps per episode, not just the start, so real
        # position has room to wander into both strata within one trajectory.
        candidate_ts = np.arange(ep_from + 5, ep_to - 1)
        n_pick = min(3, len(candidate_ts))
        if n_pick == 0:
            continue
        picks = rng.choice(candidate_ts, size=n_pick, replace=False)

        task_description = ds[ep_from]["task"]

        for t0 in picks:
            t0 = int(t0)
            model.reset(task_description)

            # Seed image_history with real frames at the wrapper's expected lag
            # (obs_lag_steps=5), replaying real steps through the same append
            # path step() itself uses.
            for t in range(t0 - 5, t0 + 1):
                frame = ds[t]
                img = to_uint8_hwc(frame[camera_key])
                model.image_history.append(model._to_training_view(img))

            state_t = ds[t0]["observation.state"].numpy()
            state_next = ds[t0 + 1]["observation.state"].numpy()
            proprio_t = pose_from_bridge_state(state_t)
            gripper_t = float(state_t[7])

            # v3-specific: _predict_chunk reads self._current_state directly;
            # normally step() sets it, but this script calls _predict_chunk
            # directly (matching diag_policy_wrapper.py's pattern) to avoid
            # a second, redundant world-model-free forward pass.
            current_state = model._build_state(proprio_t, gripper_t)
            model._current_state = current_state

            chunk = model._predict_chunk(model._pose_to_xyzrpy(proprio_t))
            pred_pos_delta = chunk[0, 0:3]
            real_pos_delta = state_next[0:3] - state_t[0:3]
            pos_err = np.abs(pred_pos_delta - real_pos_delta).mean()

            # current_state columns: [x,y,z, roll,pitch,yaw, grip, right x6]
            x_norm, z_norm = float(current_state[0, 0]), float(current_state[0, 2])
            near = max(abs(x_norm), abs(z_norm)) > 0.85
            mid = max(abs(x_norm), abs(z_norm)) < 0.5

            if near:
                near_boundary.append(pos_err)
            elif mid:
                mid_range.append(pos_err)
            # samples in between (0.5-0.85) are dropped -- keeps the two
            # groups cleanly separated rather than blurred at the margin

    mid_range = np.array(mid_range)
    near_boundary = np.array(near_boundary)

    print(f"mid-range samples   (max(|x|,|z|) < 0.5):  n={len(mid_range)}")
    print(f"near-boundary samples (max(|x|,|z|) > 0.85): n={len(near_boundary)}")
    print()
    print(f"position delta L1, mid-range:    {mid_range.mean():.4f}  (sd {mid_range.std():.4f})")
    print(f"position delta L1, near-boundary: {near_boundary.mean():.4f}  (sd {near_boundary.std():.4f})")
    print()
    ratio = near_boundary.mean() / mid_range.mean() if mid_range.mean() > 0 else float("nan")
    print(f"ratio (near-boundary / mid-range): {ratio:.2f}x")
    if ratio > 1.5:
        print("VERDICT: prediction error is meaningfully worse near the clip boundary "
              "-> supports the clipping hypothesis; a mean_std retrain (or wider q01/q99) "
              "is a reasonable next step.")
    elif ratio < 1.2:
        print("VERDICT: prediction error is NOT meaningfully worse near the clip boundary "
              "-> the clipping hypothesis is not well supported by this test; reconsider "
              "before committing to a retrain on this basis alone.")
    else:
        print("VERDICT: ambiguous -- some degradation near the boundary but not dramatic. "
              "Worth a larger sample before deciding.")


if __name__ == "__main__":
    main()
