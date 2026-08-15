#!/usr/bin/env python3
"""Follow-up to diag_autonomous_drift.py: WHY does autonomous drift grow so
fast (position drift reaches the scale of the whole task's displacement by
step ~5-6, rotation drift sits at 20-28 degrees and doesn't recover)?

delta2abs (lda/utils/rotation_convert.py) integrates each step's predicted
LOCAL position delta by rotating it through the ACCUMULATED rotation before
adding it to absolute position (T_abs[i] = T0 @ T_0_to_i[i], matrix
composition -- a local-frame delta's translation gets rotated by whatever
rotation has accumulated so far before landing in world coordinates). So if
rotation drifts, every later position delta gets added in a wrong direction
-- rotation error can contaminate position error, on top of whatever
position-only error the model's translation predictions have by themselves.

This script separates the two by running three modes, changing only what
"current pose" gets fed back into the model/integration each step:

  full     -- both position and rotation integrated autonomously (repeats
              diag_autonomous_drift.py, kept here for a same-run baseline).
  pos_only -- position integrated autonomously, rotation reset to the REAL
              logged value every step (removes any rotation-contaminates-
              position channel; isolates the model's own translation bias).
  rot_only -- rotation integrated autonomously, position reset to the REAL
              logged value every step (isolates the model's own rotation
              bias, with no position feedback involved).

If pos_only drifts much less than full, rotation error was contaminating
position integration. If rot_only alone already drifts to ~20-28 degrees,
the rotation *prediction* itself is biased, independent of any position
feedback -- the frame-contamination mechanism would then be compounding an
already-real rotation problem, not creating it from nothing.

Also reports PER-AXIS mean SIGNED drift, not just magnitude: growing but
patternless magnitude looks like accumulating noise, but a consistent sign
per axis (e.g. yaw always drifting the same direction) points to a scale or
directional bias in that specific predicted channel rather than noise.
"""

import argparse
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

    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["full", "pos_only", "rot_only"], required=True)
    ap.add_argument("--n-episodes", type=int, default=12)
    ap.add_argument("--rollout-len", type=int, default=20)
    args = ap.parse_args()

    dataset_root = pathlib.Path("/mnt/beegfsnew/scratch/3295540/data/bridge_orig_lerobot")
    camera_key = "observation.images.image_0"

    meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
    rng = np.random.default_rng(1)
    episode_ids = rng.choice(min(1000, meta.total_episodes), size=args.n_episodes, replace=False).tolist()
    ds = LeRobotDataset(repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav")

    model = LDAInference(
        checkpoint_path="outputs/lda_bridge_v2/bridge_finetune_v2/final_model/pytorch_model.pt",
        config_yaml="lda/config/training/LDA_bridge_v2.yaml",
        seed=0,
    )

    # errs_by_step[k]: (drift_pos_m, drift_rot_deg, signed_pos_axes(3), signed_rot_axes(3))
    errs_by_step = [[] for _ in range(args.rollout_len)]

    for pos_i, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos_i])
        ep_to = int(ds.episode_data_index["to"][pos_i])
        length = min(args.rollout_len, ep_to - ep_from - 1)
        if length < 5:
            continue

        task_description = ds[ep_from]["task"]
        model.reset(task_description)

        state_0 = ds[ep_from]["observation.state"].numpy().astype(np.float64)
        auto_pos = state_0[0:3].copy()
        auto_rot = Rotation.from_euler("xyz", state_0[3:6])

        for k in range(length):
            t = ep_from + k
            frame_t = ds[t]
            frame_next = ds[t + 1]
            img = to_uint8_hwc(frame_t[camera_key])
            state_t = frame_t["observation.state"].numpy().astype(np.float64)
            state_next = frame_next["observation.state"].numpy().astype(np.float64)

            # Which pose is fed back depends on the mode -- this is the only
            # thing that changes between modes.
            fed_pos = auto_pos if args.mode in ("full", "pos_only") else state_t[0:3]
            fed_rot = auto_rot if args.mode in ("full", "rot_only") else Rotation.from_euler("xyz", state_t[3:6])

            quat_xyzw = fed_rot.as_quat()
            quat_wxyz = quat_xyzw[[3, 0, 1, 2]]
            proprio_fed = PoseProxy(p=fed_pos.copy(), q=quat_wxyz)

            sim_action = model.step(img, task_description, proprio_fed, float(state_t[7]))
            pred_full = model.action_buffer[model.action_buffer_idx - 1]
            pred_pos_delta = pred_full[0:3].astype(np.float64)
            pred_rot_delta = pred_full[3:6].astype(np.float64)

            auto_pos = fed_pos + pred_pos_delta
            auto_rot = Rotation.from_euler("xyz", pred_rot_delta) * fed_rot

            real_pos = state_next[0:3]
            real_rot = Rotation.from_euler("xyz", state_next[3:6])

            signed_pos_err = auto_pos - real_pos  # (3,), world xyz
            drift_pos = float(np.linalg.norm(signed_pos_err))
            rel_rot = auto_rot * real_rot.inv()
            signed_rot_err = np.degrees(rel_rot.as_euler("xyz"))  # (3,), approx roll/pitch/yaw error
            drift_rot_deg = float(np.degrees(np.linalg.norm(rel_rot.as_rotvec())))

            errs_by_step[k].append((drift_pos, drift_rot_deg, *signed_pos_err, *signed_rot_err))

    print(f"mode={args.mode}  n_episodes={args.n_episodes}  rollout_len={args.rollout_len}")
    print("step | drift pos (m) | drift rot (deg) | signed pos xyz (m)                | signed rot rpy (deg)              | n")
    for k in range(args.rollout_len):
        if errs_by_step[k]:
            arr = np.array(errs_by_step[k])
            dp, dr = arr[:, 0], arr[:, 1]
            sx, sy, sz = arr[:, 2], arr[:, 3], arr[:, 4]
            sr, sp, syaw = arr[:, 5], arr[:, 6], arr[:, 7]
            print(
                f"{k:4d} | {dp.mean():.4f}        | {dr.mean():7.2f}          "
                f"| {sx.mean():+.4f} {sy.mean():+.4f} {sz.mean():+.4f}     "
                f"| {sr.mean():+7.2f} {sp.mean():+7.2f} {syaw.mean():+7.2f}     | {len(arr)}"
            )

    all_rows = np.concatenate([errs_by_step[k] for k in range(args.rollout_len) if errs_by_step[k]], axis=0)
    half = args.rollout_len // 2
    first_half = np.concatenate([errs_by_step[k] for k in range(half) if errs_by_step[k]], axis=0)
    second_half = np.concatenate([errs_by_step[k] for k in range(half, args.rollout_len) if errs_by_step[k]], axis=0)
    print()
    print(f"position drift (m): overall {all_rows[:,0].mean():.4f} | first half {first_half[:,0].mean():.4f} | second half {second_half[:,0].mean():.4f}")
    print(f"rotation drift (deg): overall {all_rows[:,1].mean():.4f} | first half {first_half[:,1].mean():.4f} | second half {second_half[:,1].mean():.4f}")
    print(f"signed pos xyz mean (m): {all_rows[:,2].mean():+.4f} {all_rows[:,3].mean():+.4f} {all_rows[:,4].mean():+.4f}")
    print(f"signed rot rpy mean (deg): {all_rows[:,5].mean():+.2f} {all_rows[:,6].mean():+.2f} {all_rows[:,7].mean():+.2f}")
    print()
    print("Compare against the other two modes: if pos_only drift << full drift, rotation was")
    print("contaminating position via delta2abs's frame composition. If rot_only alone already")
    print("reaches ~20-28 deg, the rotation prediction itself is biased, independent of position.")
    print("A consistently signed (not ~0) mean on any axis points to a directional/scale bias in")
    print("that channel rather than pure noise.")


if __name__ == "__main__":
    main()
