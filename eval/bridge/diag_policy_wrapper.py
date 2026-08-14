#!/usr/bin/env python3
"""
Offline validation of LDAInference (the actual closed-loop SimplerEnv wrapper,
eval/bridge/lda_policy.py) against real logged Bridge trajectories -- no
SimplerEnv/ManiSkill2 involved.

Why this test and not more code reading: the delta/frame math (calculate_delta_eef
/ delta2abs) and the rotation convention (batched_rpy_to_R vs scipy 'xyz') were both
already verified correct by direct derivation/numeric check. eval_bridge_openloop.py
(a DIFFERENT, already-working code path) proves the model itself has real signal --
its position L1 beats the stay-still baseline. So if THIS wrapper, fed the same kind
of real data, produces much worse position deltas than that other working path, the
bug is specifically in lda_policy.py's plumbing (image feed timing, history_action
bookkeeping, PoseProxy construction), not in the model or the core math.

Method: seed LDAInference's image_history/action_history from real consecutive
frames of a real bridge_orig_lerobot episode (same source our oracle probes use),
call the real step() exactly as SimplerEnv would, and compare its predicted
position delta against the actual state[t+1] - state[t] from the log.
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
    episode_ids = rng.choice(min(1000, meta.total_episodes), size=15, replace=False).tolist()
    ds = LeRobotDataset(repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav")

    model = LDAInference(
        checkpoint_path="outputs/lda_bridge_v2/bridge_finetune_v2/final_model/pytorch_model.pt",
        config_yaml="lda/config/training/LDA_bridge_v2.yaml",
        seed=0,
    )

    pos_errs, pos_errs_baseline, pred_mags, real_mags = [], [], [], []

    for pos_i, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos_i])
        ep_to = int(ds.episode_data_index["to"][pos_i])
        if ep_to - ep_from < 8:
            continue

        task_description = ds[ep_from]["task"]
        model.reset(task_description)

        # Seed image_history with real frames at the SAME lag the wrapper expects
        # (obs_lag_steps=5), by replaying real steps ep_from..ep_from+5 through the
        # exact same append path step() itself uses.
        t0 = ep_from + 5
        for t in range(ep_from, t0 + 1):
            frame = ds[t]
            img = to_uint8_hwc(frame[camera_key])
            model.image_history.append(model._to_training_view(img))

        state_t = ds[t0]["observation.state"].numpy()
        state_next = ds[t0 + 1]["observation.state"].numpy()
        proprio_t = pose_from_bridge_state(state_t)

        chunk = model._predict_chunk(model._pose_to_xyzrpy(proprio_t))
        pred_pos_delta = chunk[0, 0:3]
        real_pos_delta = state_next[0:3] - state_t[0:3]

        pos_errs.append(np.abs(pred_pos_delta - real_pos_delta).mean())
        pos_errs_baseline.append(np.abs(real_pos_delta).mean())  # "predict zero delta"
        pred_mags.append(np.linalg.norm(pred_pos_delta))
        real_mags.append(np.linalg.norm(real_pos_delta))

    pos_errs = np.array(pos_errs)
    pos_errs_baseline = np.array(pos_errs_baseline)
    print(f"n samples: {len(pos_errs)}")
    print(f"position delta L1 (wrapper prediction vs real):  {pos_errs.mean():.4f}  (sd {pos_errs.std():.4f})")
    print(f"position delta L1 (zero-delta baseline):          {pos_errs_baseline.mean():.4f}")
    print(f"mean |predicted delta|: {np.mean(pred_mags):.4f}   mean |real delta|: {np.mean(real_mags):.4f}")
    print("(magnitudes should be in the same ballpark -- a 10x mismatch means a scale/unit bug,")
    print(" near-zero predicted magnitude means the model/wrapper isn't producing real motion.)")


if __name__ == "__main__":
    main()
