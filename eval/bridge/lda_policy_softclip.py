"""Experiment: does replacing the hard q99 clamp with a smooth saturation
change closed-loop behavior, without retraining?

RESULTS.md's diagnosis: LDA v3's state normalization hard-clips
(torch.clamp(normalized, -1, 1), transform/state_action.py:134) -- any
left_x/left_z position beyond [q01, q99] collapses to the IDENTICAL -1.0 or
+1.0, destroying the model's ability to tell "slightly off" from "very off"
once it drifts out of range. This is not a code bug to "fix" -- the model's
weights were trained ONLY ever seeing state clamped to exactly [-1, 1], so
this can't be corrected the way F1's bugs were (there is no single
unambiguously-correct computation to restore). It's a genuine question of
whether a monotonic-but-still-bounded encoding of the tail helps a network
that has never seen anything past +-1 at all, or whether any excursion past
+-1 is equally out-of-distribution to it regardless of shape. Cheap enough
to test directly rather than guess.

Does NOT modify lda_policy.py or transform/state_action.py -- this is a
standalone subclass overriding only _build_state(), replicating the same
q01/q99 linear-normalization formula (values pulled from the SAME dataset
stats.json the real transform is fitted on, so the in-range region is
numerically identical to production) but replacing the hard clamp with
f(x) = x                        for |x| <= 1
     = sign(x) * (1 + tanh(|x| - 1))   for |x| > 1
which is C1-continuous at the boundary (matches value AND slope at x=+-1,
so it doesn't introduce a kink the model wouldn't have learned from clean
in-range data either) and monotonically asymptotes toward +-2 instead of
saturating at +-1 -- so two different real positions that both used to clip
to an identical -1.0 now map to two different (if still compressed) values.
"""

from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from eval.bridge.lda_policy import LDAInference

STATS_PATH = "/mnt/beegfsnew/scratch/3295540/data/bridge_lda/meta/stats_gr00t.json"


def _soft_clip(x: np.ndarray) -> np.ndarray:
    out = x.copy()
    over = np.abs(x) > 1.0
    out[over] = np.sign(x[over]) * (1.0 + np.tanh(np.abs(x[over]) - 1.0))
    return out


class LDAInferenceSoftClip(LDAInference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        with open(STATS_PATH) as f:
            stats = json.load(f)
        self._q01 = {}
        self._q99 = {}
        for key in (
            "state.left_eef_position",
            "state.left_eef_rotation",
            "state.left_gripper",
            "state.right_eef_position",
            "state.right_eef_rotation",
            "state.right_gripper",
        ):
            self._q01[key] = np.asarray(stats[key]["q01"], dtype=np.float64)
            self._q99[key] = np.asarray(stats[key]["q99"], dtype=np.float64)

    def _normalize_soft(self, key: str, raw: np.ndarray) -> np.ndarray:
        q01, q99 = self._q01[key], self._q99[key]
        raw = raw.reshape(-1).astype(np.float64)
        out = np.zeros_like(raw)
        mask = q01 != q99
        lin = 2.0 * (raw[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1.0
        out[mask] = _soft_clip(lin)
        out[~mask] = raw[~mask]
        return out.astype(np.float32)

    def _build_state(self, ee_pose_proprio, gripper_proprio: float) -> np.ndarray:
        """Same inputs/outputs/shape as LDAInference._build_state -- only the
        normalization tail differs (see module docstring)."""
        pos = np.asarray(ee_pose_proprio.p, dtype=np.float64)
        rot = Rotation.from_quat(ee_pose_proprio.q, scalar_first=True)
        if self._ref_rot is None:
            self._ref_rot = rot
        rel_rot = self._ref_rot.inv() * rot
        rpy = rel_rot.as_euler("xyz")

        left_pos = self._normalize_soft("state.left_eef_position", pos)
        left_rot = self._normalize_soft("state.left_eef_rotation", rpy)
        left_grip = self._normalize_soft("state.left_gripper", np.array([float(gripper_proprio)]))
        right_pos = self._normalize_soft("state.right_eef_position", np.zeros(3))
        right_rot = self._normalize_soft("state.right_eef_rotation", np.zeros(3))
        right_grip = self._normalize_soft("state.right_gripper", np.zeros(1))

        return np.concatenate(
            [left_pos, left_rot, left_grip, right_pos, right_rot, right_grip]
        ).reshape(1, 14).astype(np.float32)
