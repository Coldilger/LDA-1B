#!/usr/bin/env python3
"""Honest follow-up to diag_policy_wrapper_rollout.py.

That script's position/rotation numbers looked fine (no growth over the
rollout), but they were measured unfairly: at every step it fed the wrapper
the REAL logged pose from the dataset as "current pose" (proprio_t =
pose_from_bridge_state(state_t)), i.e. it reset the arm to the ground-truth
position before every single prediction. Real SimplerEnv closed-loop control
does not do that -- the "current pose" it feeds the wrapper each step is
wherever the PREVIOUS predicted actions actually drove the (simulated) arm to.
If the model's own deltas are even slightly biased, that never showed up in
the old test, because it was corrected back to truth every single step before
it could accumulate.

This script removes that correction. Starting from the real initial pose
(the one real SimplerEnv would also start from, for the same episode), it
integrates ONLY the model's own predicted deltas step after step -- exactly
like real closed-loop control -- and compares that accumulated (autonomous)
pose against the real logged pose at the same step index. If this drift grows
over the rollout, that is direct evidence the arm would wander away from
where it needs to be, which -- independent of gripper accuracy -- is enough
by itself to explain zero grasps in the real SimplerEnv run.

Caveat this cannot avoid: the image shown to the model at each step is still
the REAL logged frame (there is no offline renderer to show what the camera
would see from the drifted, autonomous pose). So this likely UNDERESTIMATES
real drift -- real closed loop would also see the visual consequences of its
own drift, which could compound further. Gripper is left as in
diag_policy_wrapper_rollout.py (fed the real logged reading), since without a
physics simulator there is no way to know the true resulting gripper state
under autonomous actions either; this script is about position/rotation only.
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
        self.q = q


def to_uint8_hwc(img) -> np.ndarray:
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))
        arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def main():
    from scipy.spatial.transform import Rotation

    dataset_root = pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")
    camera_key = "observation.images.image_0"
    n_episodes = 12
    rollout_len = 20

    meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
    rng = np.random.default_rng(1)
    episode_ids = rng.choice(min(1000, meta.total_episodes), size=n_episodes, replace=False).tolist()
    ds = LeRobotDataset(repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav")

    model = LDAInference(
        checkpoint_path="outputs/lda_bridge_v2/bridge_finetune_v2/final_model/pytorch_model.pt",
        config_yaml="lda/config/training/LDA_bridge_v2.yaml",
        seed=0,
    )

    # errs_by_step[k]: (drift_pos_m, drift_rot_deg, real_disp_from_start_m) at
    # step k, pooled across episodes. The third value is context: how far the
    # real trajectory itself has moved from t=0 by this point, so drift can be
    # judged against task scale, not just an absolute number.
    errs_by_step = [[] for _ in range(rollout_len)]

    for pos_i, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos_i])
        ep_to = int(ds.episode_data_index["to"][pos_i])
        length = min(rollout_len, ep_to - ep_from - 1)
        if length < 5:
            continue

        task_description = ds[ep_from]["task"]
        model.reset(task_description)

        state_0 = ds[ep_from]["observation.state"].numpy().astype(np.float64)
        start_pos = state_0[0:3].copy()
        auto_pos = state_0[0:3].copy()
        auto_rot = Rotation.from_euler("xyz", state_0[3:6])

        for k in range(length):
            t = ep_from + k
            frame_t = ds[t]
            frame_next = ds[t + 1]
            img = to_uint8_hwc(frame_t[camera_key])
            state_t = frame_t["observation.state"].numpy().astype(np.float64)
            state_next = frame_next["observation.state"].numpy().astype(np.float64)

            # HONEST part: the wrapper is told its own believed (autonomously
            # integrated) pose, never the real one -- exactly what real
            # closed-loop SimplerEnv does.
            quat_xyzw = auto_rot.as_quat()
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            proprio_auto = PoseProxy(p=auto_pos.copy(), q=quat_wxyz)

            sim_action = model.step(img, task_description, proprio_auto, float(state_t[7]))
            pred_full = model.action_buffer[model.action_buffer_idx - 1]
            pred_pos_delta = pred_full[0:3].astype(np.float64)
            pred_rot_delta = pred_full[3:6].astype(np.float64)

            # Integrate -- this becomes next step's "current pose", nothing
            # resets it back to the real trajectory.
            auto_pos = auto_pos + pred_pos_delta
            auto_rot = Rotation.from_euler("xyz", pred_rot_delta) * auto_rot

            real_pos = state_next[0:3]
            real_rot = Rotation.from_euler("xyz", state_next[3:6])

            drift_pos = float(np.linalg.norm(auto_pos - real_pos))
            rel_rot = auto_rot * real_rot.inv()
            drift_rot_deg = float(np.degrees(np.linalg.norm(rel_rot.as_rotvec())))
            real_disp = float(np.linalg.norm(real_pos - start_pos))

            errs_by_step[k].append((drift_pos, drift_rot_deg, real_disp))

    print(f"n episodes replayed: {n_episodes}, rollout_len: {rollout_len}")
    print("step | drift from real pos (m) | drift from real rot (deg) | real disp from t=0 (m) | n")
    for k in range(rollout_len):
        if errs_by_step[k]:
            arr = np.array(errs_by_step[k])
            dp, dr, disp = arr[:, 0], arr[:, 1], arr[:, 2]
            print(
                f"{k:4d} | {dp.mean():.4f} (sd {dp.std():.4f})       "
                f"| {dr.mean():7.2f} (sd {dr.std():6.2f})       "
                f"| {disp.mean():.4f}                | {len(arr)}"
            )

    all_rows = np.concatenate([errs_by_step[k] for k in range(rollout_len) if errs_by_step[k]], axis=0)
    half = rollout_len // 2
    first_half = np.concatenate([errs_by_step[k] for k in range(half) if errs_by_step[k]], axis=0)
    second_half = np.concatenate([errs_by_step[k] for k in range(half, rollout_len) if errs_by_step[k]], axis=0)
    print()
    for name, idx in [("position drift (m)", 0), ("rotation drift (deg)", 1)]:
        print(
            f"{name}: overall mean {all_rows[:, idx].mean():.4f} | "
            f"first half {first_half[:, idx].mean():.4f} | "
            f"second half {second_half[:, idx].mean():.4f}"
        )
    print(f"real trajectory's own displacement from t=0: overall mean {all_rows[:, 2].mean():.4f} m")
    print()
    print("If drift grows across the rollout and/or ends up comparable to or larger than the real")
    print("trajectory's own displacement, the autonomous arm is ending up somewhere meaningfully")
    print("different from where the task needs it -- enough on its own to explain zero grasps,")
    print("independent of gripper accuracy.")


if __name__ == "__main__":
    main()
