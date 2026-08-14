#!/usr/bin/env python3
"""
Follow-up to diag_policy_wrapper.py: tests whether error GROWS over
consecutive steps when history_action is filled with the wrapper's own
past predictions (as in real closed-loop control) instead of staying at
the single cold-start decision point (where history_action is legitimately
zero and error there doesn't distinguish "bad first step" from "errors
compound over time").

Method: replay a real bridge_orig_lerobot episode through LDAInference.step()
exactly as SimplerEnv would (real image each step, wrapper's own predicted
action feeds its own history_action for the next step), but instead of
sending the output to a simulator, compare it each step against the real
logged state delta. If per-step error grows across the rollout, that
supports the exposure-bias hypothesis (history_action built from the
model's own imperfect past predictions, not ground truth, diverges further
from the training distribution the longer the rollout runs).
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
    n_episodes = 12
    rollout_len = 20  # env steps per episode to replay

    meta = LeRobotDatasetMetadata(repo_id="bridge_orig_lerobot", root=str(dataset_root))
    rng = np.random.default_rng(1)
    episode_ids = rng.choice(min(1000, meta.total_episodes), size=n_episodes, replace=False).tolist()
    ds = LeRobotDataset(repo_id="bridge_orig_lerobot", root=str(dataset_root), episodes=episode_ids, video_backend="pyav")

    model = LDAInference(
        checkpoint_path="outputs/lda_bridge_v2/bridge_finetune_v2/final_model/pytorch_model.pt",
        config_yaml="lda/config/training/LDA_bridge_v2.yaml",
        seed=0,
    )

    # errs_by_step[k] collects the position-delta L1 at the k-th env step of a
    # rollout, pooled across episodes -- lets us see whether error trends up.
    errs_by_step = [[] for _ in range(rollout_len)]

    for pos_i, ep_id in enumerate(episode_ids):
        ep_from = int(ds.episode_data_index["from"][pos_i])
        ep_to = int(ds.episode_data_index["to"][pos_i])
        length = min(rollout_len, ep_to - ep_from - 1)
        if length < 5:
            continue

        task_description = ds[ep_from]["task"]
        model.reset(task_description)

        for k in range(length):
            t = ep_from + k
            frame_t = ds[t]
            frame_next = ds[t + 1]
            img = to_uint8_hwc(frame_t[camera_key])
            state_t = frame_t["observation.state"].numpy()
            state_next = frame_next["observation.state"].numpy()
            proprio_t = pose_from_bridge_state(state_t)

            # Exactly what step() does, but we read the pose-tracking output
            # directly instead of sending it to a simulator -- and we let
            # the wrapper keep its own action_history/image_history state
            # across the loop, matching real closed-loop use.
            sim_action = model.step(img, task_description, proprio_t, float(state_t[7]))
            pred_pos_delta = sim_action["world_vector"]  # already base-frame, comparable to state deltas
            real_pos_delta = state_next[0:3] - state_t[0:3]

            errs_by_step[k].append(np.abs(pred_pos_delta - real_pos_delta).mean())

    print(f"n episodes replayed: {n_episodes}, rollout_len: {rollout_len}")
    print("step | mean position-delta L1 | n")
    for k in range(rollout_len):
        if errs_by_step[k]:
            arr = np.array(errs_by_step[k])
            print(f"{k:4d} | {arr.mean():.4f}  (sd {arr.std():.4f}) | {len(arr)}")

    first_half = np.concatenate([errs_by_step[k] for k in range(rollout_len // 2) if errs_by_step[k]])
    second_half = np.concatenate([errs_by_step[k] for k in range(rollout_len // 2, rollout_len) if errs_by_step[k]])
    print()
    print(f"first half mean L1:  {first_half.mean():.4f}")
    print(f"second half mean L1: {second_half.mean():.4f}")
    print("(if second half is clearly worse than first half, error compounds over the rollout --")
    print(" supports the exposure-bias / self-generated-history-action hypothesis.)")


if __name__ == "__main__":
    main()
