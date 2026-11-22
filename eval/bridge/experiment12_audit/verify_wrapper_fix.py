"""Runs lda_policy.py's real action-decoding code (LDAInference._predict_chunk,
_pose_to_xyzrpy) on ground-truth Bridge motion, with the model replaced by a
stub that returns the episode's true tool-frame deltas. The simulator pose is
emulated from the Bridge state with SimplerEnv's own convention
(R_sim = R_data @ T_SIM). Reports how far the wrapper's base-frame steps are
from the true ones with the Experiment 12 fix off and on. CPU only.

    LDA env: python verify_wrapper_fix.py [--n 100]
"""
from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent.parent))  # LDA-1B repo root
sys.path.insert(0, str(HERE))
from audit_pipelines import ORIG, angle_deg, parquet, stack  # noqa: E402
from eval.bridge.lda_policy import RAW_SLICES, RAW_WIDTH, T_SIM, LDAInference  # noqa: E402
from lda.utils.rotation_convert import calculate_delta_eef  # noqa: E402

H = 16


def stub(bridge_frame: bool) -> LDAInference:
    m = LDAInference.__new__(LDAInference)
    m.bridge_frame, m.exec_horizon, m.obs_lag_steps, m.invert_gripper = bridge_frame, H, 5, False
    m.image_history = deque([np.zeros((2, 2, 3), np.uint8)], maxlen=6)
    m.task_description, m.embodiment_id, m._state_dim = "", 0, None
    m.history_len = 4
    m.action_history, m.gripper_real_history = deque(maxlen=4), deque(maxlen=4)
    m._transforms = SimpleNamespace(unapply=lambda d: {k: v.numpy() for k, v in d.items()})
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    args = ap.parse_args()
    rng = np.random.default_rng(1)
    res = {False: [], True: []}
    rot_res = {False: [], True: []}
    for ep in rng.choice(50000, size=args.n, replace=False):
        st = stack(parquet(ORIG, int(ep)), "observation.state")
        if len(st) < H + 2:
            continue
        for t in range(0, len(st) - H - 1, H):
            seg = st[t:t + H + 1]
            delta = calculate_delta_eef(seg[:, :6])  # (H, 6), the training target
            raw = np.zeros((H, RAW_WIDTH), np.float32)
            raw[:, RAW_SLICES["action.left_eef_position"]] = delta[:, :3]
            raw[:, RAW_SLICES["action.left_eef_rotation"]] = delta[:, 3:6]
            q = (R.from_euler("xyz", seg[0, 3:6]) * R.from_matrix(T_SIM)).as_quat(scalar_first=True)
            pose = SimpleNamespace(p=seg[0, :3].copy(), q=q)
            true_dp = seg[1:, :3] - seg[:-1, :3]
            rd = R.from_euler("xyz", seg[:, 3:6])
            for fix in (False, True):
                m = stub(fix)
                m.policy = SimpleNamespace(predict_action=lambda ex, raw=raw: {"normalized_actions": [raw]})
                with torch.no_grad():
                    chunk = m._predict_chunk(m._pose_to_xyzrpy(pose))
                res[fix].extend(angle_deg(chunk[:, :3], true_dp))
                got = R.from_euler("xyz", chunk[:, 3:6])
                rot_res[fix].extend(np.degrees((got * (rd[1:] * rd[:-1].inv()).inv()).magnitude()))
    for fix in (False, True):
        a, r = np.array(res[fix]), np.array(rot_res[fix])
        print(f"bridge_frame={fix}: position direction error median {np.median(a):.3g} deg, p90 {np.percentile(a, 90):.3g}; "
              f"rotation step error median {np.median(r):.3g} deg (n={len(a)} steps)")


if __name__ == "__main__":
    main()
