#!/usr/bin/env python
"""Are the actions reaching the controller the right size?

The closed-loop smoke run finished cleanly but scored 0.0% with `moved_correct_obj`
false on all 1440 steps -- the arm never even nudged the object. An undertrained
policy still flails and bumps things, so total stillness points at the action
pipeline rather than at model quality.

This feeds the policy real in-distribution Bridge frames and prints what the
wrapper would hand the widowx controller, against Bridge's own per-step motion
of ~0.009 m.

Reading the output:
  |world_vector| ~ 0.009   -> pipeline is right, the model is just undertrained
  |world_vector| ~ 0       -> model predicts no motion, or unapply collapses it
  |world_vector| >> 0.05   -> scale blown up somewhere
"""

from __future__ import annotations

import argparse

import os
import sys

import numpy as np
import torch

# Run directly (python eval/bridge/diag_actions.py) rather than as a module, so the
# repo root is not on sys.path by default.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.bridge.lda_policy import RAW_SLICES, LDAInference  # noqa: E402


class _Pose:
    """Minimal stand-in for SimplerEnv's proprio object (has .p and .q)."""

    def __init__(self, p, q):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", default="/mnt/beegfsnew/scratch/3295540/data/bridge_lda")
    ap.add_argument("--n-steps", type=int, default=12)
    args = ap.parse_args()

    policy = LDAInference(checkpoint_path=args.checkpoint, dataset_path=args.dataset_path, seed=0)
    ds = policy._dataset

    traj_id = int(ds.trajectory_ids[0])
    data = ds.get_step_data_with_transform(traj_id, 0, return_state=True)
    lang = data["lang"]
    print(f"trajectory {traj_id}: {lang!r}\n", flush=True)

    example = {
        "image": np.stack([np.asarray(img) for img in data["image"][:2]], axis=0),
        "lang": lang,
        "embodiment_id": policy.embodiment_id,
    }
    with torch.no_grad():
        raw = np.asarray(policy.policy.predict_action([example])["normalized_actions"][0])
    print(f"raw normalized output: shape {raw.shape}")
    print(f"  left dims 0:7 |mean| {np.abs(raw[:, 0:7]).mean():.5f}  range [{raw[:, 0:7].min():.4f}, {raw[:, 0:7].max():.4f}]")
    print("  (normalized space is [-1,1]; all-zero means the model predicts no motion)\n", flush=True)

    denorm = policy._transforms.unapply(
        {k: torch.as_tensor(raw[:, sl]) for k, sl in RAW_SLICES.items()}
    )
    dpos = np.asarray(denorm["action.left_eef_position"])
    drot = np.asarray(denorm["action.left_eef_rotation"])
    grip = np.asarray(denorm["action.left_gripper"]).reshape(-1)
    print("after unapply (gripper-frame deltas, m / rad):")
    print(f"  |dpos| mean {np.linalg.norm(dpos, axis=1).mean():.5f}   (Bridge reference ~0.009)")
    print(f"  |drot| mean {np.linalg.norm(drot, axis=1).mean():.5f}")
    print(f"  gripper min {grip.min():.3f} max {grip.max():.3f}\n", flush=True)

    pose = _Pose([0.147, 0.028, 0.20], [0.0, 0.0, 0.0, 1.0])
    policy.reset(lang)
    frame = np.asarray(data["image"][-1])

    print("what the widowx controller receives:")
    mags = []
    for i in range(args.n_steps):
        out = policy.step(frame, lang, pose, 1.0)
        wv = out["world_vector"]
        mags.append(float(np.linalg.norm(wv)))
        print(
            f"  step {i:>2}  |world_vector| {np.linalg.norm(wv):.5f}  "
            f"|rot_axangle| {np.linalg.norm(out['rot_axangle']):.5f}  "
            f"gripper {out['gripper'][0]:+.0f}",
            flush=True,
        )
    print(f"\nmean |world_vector|: {np.mean(mags):.5f}  (Bridge per-step ~0.009)")


if __name__ == "__main__":
    main()
