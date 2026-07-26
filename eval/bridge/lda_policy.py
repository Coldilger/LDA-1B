"""SimplerEnv-compatible policy wrapper for LDA-1B on the WidowX/Bridge tasks.

Implements the 3-method interface `maniskill2_evaluator.py` expects:
    reset(task_description)
    step(image, task_description, ee_pose_proprio, gripper_proprio) -> dict
    visualize_epoch(images, save_path)

Modeled on F1-VLA's wrapper (F1-VLA/eval/bridge/f1_vla_policy.py) and carrying
over the fixes that were paid for there in measured success rate — see
F1-VLA/eval/bridge/RESULTS.md. What is the same, what differs, and why:

1. **Gripper: binarize to [-1, +1].** SAME as F1. The widowx controller takes
   [-1, +1] (close/open); Bridge encodes 0=close / 1=open. Passing raw [0,1]
   through puts "close" at 0 = mid joint range = half-open, so the gripper can
   never clamp anything. This alone was 0% -> 12.5% for F1.
   Note: `lda/utils/eval_relative_eef.py` applies an unconditional
   `1 - gripper` before comparing. That is an AgiBot polarity convention, not
   Bridge's — our converted data stores Bridge's own action[6] (1=open) directly,
   so a model finetuned on it predicts that polarity and must NOT be inverted.
   Exposed as `invert_gripper` so the assumption can be tested rather than trusted.

2. **Rotation: euler -> axis-angle.** SAME as F1, and load-bearing for the same
   reason: the two coincide only for tiny angles.

3. **Image: square before feeding.** SAME as F1 (that fix was 12.5% -> 29.2%),
   but the mechanism here is different and worth stating. Training built images as
   `expand2square(frame) -> resize(224)`; since Bridge frames are already 256x256
   square, expand2square is a no-op and training effectively saw a plain square
   resize. Inference (`QwenMMDiT.predict_action`) applies only `resize_images` to
   `image_size`, with no expand2square — so a raw 480x640 SimplerEnv frame would
   be *squashed* to 224x224, an aspect ratio the model never saw. Squaring first
   reproduces training exactly.

4. **State / reference frame: does not arise.** F1 needed episode-relative
   rotation because Bridge's proprio convention and SimplerEnv's absolute
   `ee_pose_at_base` disagree (its fix #4, +8 points). LDA's checkpoint has
   `state_dim: null` — no proprioception is fed to the policy at all — so there is
   no state distribution to mismatch. Proprio is used here only to seed the
   delta->absolute integration below.

The frame conversion is the substantive new piece. LDA predicts deltas in the
**gripper's own frame** (`calculate_delta_eef` computes `dT_i = T_i^-1 @ T_{i+1}`),
while the widowx controller wants **base-frame** deltas. The alignment-to-frame-0
inside `calculate_delta_eef` cancels algebraically, so those deltas carry no frame
of their own; integrating them from the robot's *actual current pose* via
`delta2abs` therefore yields absolute poses in SimplerEnv's base frame, and
differencing consecutive poses gives the base-frame deltas the controller wants.
"""

from __future__ import annotations

import contextlib
import os
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation
from transforms3d.euler import euler2axangle

from lda.dataloader.gr00t_lerobot.data_config import OxeDataConfig
from lda.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from lda.dataloader.gr00t_lerobot.embodiment_tags import (
    EMBODIMENT_TAG_MAPPING,
    EmbodimentTag,
)
from lda.model.framework.base_framework import baseframework
from lda.utils.rotation_convert import delta2abs

# Slices of the model's raw 138-wide padded action space. The compact bimanual
# layout puts the left arm at 0:7; the right arm starts at 69 (138 = 69 per arm).
# Taken from lda/utils/eval_relative_eef.py so offline and closed-loop agree.
RAW_SLICES = {
    "action.left_eef_position": slice(0, 3),
    "action.left_eef_rotation": slice(3, 6),
    "action.left_gripper": slice(6, 7),
    "action.right_eef_position": slice(69, 72),
    "action.right_eef_rotation": slice(72, 75),
    "action.right_gripper": slice(75, 76),
}

TRAIN_RESO = 256  # Bridge frames are 256x256; the model resizes to 224 internally.

# Repo root, i.e. the directory the training config's relative paths are written
# against (`pretrained/vlm/Qwen3-VL-4B-Instruct`, `vision_encoder_path: pretrained`).
REPO_ROOT = Path(__file__).resolve().parents[2]


@contextlib.contextmanager
def _cwd(path):
    """Load the model with the repo root as cwd.

    The run's saved config.yaml stores the VLM and vision-encoder locations as
    paths relative to the repo root, and SimplerEnv is driven from its own
    directory. Without this, `from_pretrained` resolves nothing at that location
    and huggingface_hub falls back to reading it as a Hub repo id, failing with
    "Repo id must be in the form 'repo_name' or 'namespace/repo_name'". Patching
    the config instead would not survive: each training wave rewrites it.
    """
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class LDAInference:
    def __init__(
        self,
        checkpoint_path: str,
        dataset_path: str = "/mnt/beegfsnew/scratch/3295540/data/bridge_lda",
        config_yaml: str | None = None,
        device: str = "cuda",
        action_horizon: int = 16,
        exec_horizon: int = 8,
        obs_lag_steps: int = 5,
        embodiment_tag: EmbodimentTag = EmbodimentTag.OXE,
        invert_gripper: bool = False,
        seed: int | None = None,
    ):
        self.device = device
        self.action_horizon = action_horizon
        # Executing fewer steps than predicted means replanning more often, which
        # generally helps closed-loop control but costs wall time. This is a knob
        # worth sweeping, not a tuned value -- F1's results showed the effective
        # planning horizon dominating success on the precise-placement tasks.
        self.exec_horizon = min(exec_horizon, action_horizon)
        self.obs_lag_steps = obs_lag_steps
        self.invert_gripper = invert_gripper
        self.embodiment_id = EMBODIMENT_TAG_MAPPING[embodiment_tag.value]

        with _cwd(REPO_ROOT):
            self.policy = baseframework.from_pretrained(pretrained_checkpoint=checkpoint_path)
        self.policy.eval()
        self.policy.to(device)

        # The dataset is instantiated only for its fitted transforms: unapply()
        # needs the same q99 statistics the policy's outputs are normalized against.
        # Its step index is cached, so this is a metadata load, not a data scan.
        dcfg = OxeDataConfig()
        data_cfg = self.policy.config.datasets.vla_data
        data_cfg.lerobot_version = data_cfg.get("lerobot_version", "v2.0")
        self._dataset = LeRobotSingleDataset(
            dataset_path=dataset_path,
            modality_configs=dcfg.modality_config(),
            transforms=dcfg.transform(),
            embodiment_tag=embodiment_tag,
            video_backend=dcfg.video_backend,
            img_interval=dcfg.img_interval,
            data_cfg=data_cfg,
        )
        self._transforms = self._dataset.transforms

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # The model conditions on two frames: one `obs_lag_steps` back and the
        # current one (BaseDataConfig.observation_indices = [-5, 0]). img_interval
        # only strides which base indices are sampled, it does not scale these, so
        # at Bridge's 5 fps the gap is 1.0 s -- and SimplerEnv runs at control_freq
        # 5, so 5 env steps is the same 1.0 s.
        self.image_history: deque = deque(maxlen=obs_lag_steps + 1)
        self.task_description = None
        self.action_buffer: np.ndarray | None = None
        self.action_buffer_idx = 0

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        self.action_buffer = None
        self.action_buffer_idx = 0

    @staticmethod
    def _to_training_view(image: np.ndarray) -> np.ndarray:
        """480x640 (4:3) -> 256x256 square, matching what training saw."""
        return np.asarray(Image.fromarray(image).resize((TRAIN_RESO, TRAIN_RESO), Image.LANCZOS))

    @staticmethod
    def _pose_to_xyzrpy(ee_pose_proprio) -> np.ndarray:
        pos = np.asarray(ee_pose_proprio.p, dtype=np.float64)
        rpy = Rotation.from_quat(ee_pose_proprio.q, scalar_first=True).as_euler("xyz")
        return np.concatenate([pos, rpy])

    def _predict_chunk(self, current_pose: np.ndarray) -> np.ndarray:
        """Return (exec_horizon, 7): base-frame [dx,dy,dz,droll,dpitch,dyaw,gripper]."""
        frames = list(self.image_history)
        # Repeat-pad at episode start so the lagged frame always exists.
        past = frames[0] if len(frames) <= self.obs_lag_steps else frames[-(self.obs_lag_steps + 1)]
        example = {
            "image": np.stack([past, frames[-1]], axis=0),  # (2, H, W, C) uint8
            "lang": self.task_description,
            "embodiment_id": self.embodiment_id,
        }

        with torch.no_grad():
            out = self.policy.predict_action([example])
        raw = np.asarray(out["normalized_actions"][0])  # (action_horizon, 138)

        denorm = self._transforms.unapply(
            {key: torch.as_tensor(raw[:, sl]) for key, sl in RAW_SLICES.items()}
        )
        local_delta = np.concatenate(
            [
                np.asarray(denorm["action.left_eef_position"]),
                np.asarray(denorm["action.left_eef_rotation"]),
            ],
            axis=1,
        ).astype(np.float64)  # (T, 6), gripper frame

        gripper = np.asarray(denorm["action.left_gripper"]).reshape(-1).astype(np.float64)
        if self.invert_gripper:
            gripper = 1.0 - gripper

        # Integrate the gripper-frame deltas from the robot's real pose, then
        # difference back out to get base-frame deltas. delta2abs returns T+1 poses
        # (the seed plus one per delta).
        abs_poses = delta2abs(local_delta, current_pose)  # (T+1, 6)

        n = min(self.exec_horizon, len(abs_poses) - 1)
        chunk = np.zeros((n, 7), dtype=np.float64)
        rots = Rotation.from_euler("xyz", abs_poses[:, 3:6])
        for j in range(n):
            chunk[j, 0:3] = abs_poses[j + 1, 0:3] - abs_poses[j, 0:3]
            chunk[j, 3:6] = (rots[j + 1] * rots[j].inv()).as_euler("xyz")
            chunk[j, 6] = gripper[j] if j < len(gripper) else gripper[-1]
        return chunk

    def step(self, image: np.ndarray, task_description: str, ee_pose_proprio, gripper_proprio) -> dict:
        if task_description is not None and task_description != self.task_description:
            self.reset(task_description)

        self.image_history.append(self._to_training_view(image))

        if self.action_buffer is None or self.action_buffer_idx >= len(self.action_buffer):
            # Replanning always starts from where the arm actually is, so
            # integration drift cannot accumulate across chunks.
            self.action_buffer = self._predict_chunk(self._pose_to_xyzrpy(ee_pose_proprio))
            self.action_buffer_idx = 0

        pred = self.action_buffer[self.action_buffer_idx]
        self.action_buffer_idx += 1

        rot_ax, rot_angle = euler2axangle(*pred[3:6])
        return {
            "world_vector": pred[0:3].astype(np.float32),
            "rot_axangle": (rot_ax * rot_angle).astype(np.float32),
            "gripper": np.array([2.0 * (pred[6] > 0.5) - 1.0], dtype=np.float32),
            "terminate_episode": np.array([0.0], dtype=np.float32),
        }

    def visualize_epoch(self, images, save_path: str) -> None:
        """Dump a strip of sampled frames so each rollout leaves something to look at."""
        if not images:
            return
        n = min(8, len(images))
        idxs = np.linspace(0, len(images) - 1, n).astype(int)
        frames = [Image.fromarray(images[i]) for i in idxs]
        w, h = frames[0].size
        strip = Image.new("RGB", (w * n, h))
        for i, frame in enumerate(frames):
            strip.paste(frame, (i * w, 0))
        strip.save(save_path)
