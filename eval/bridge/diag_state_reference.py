#!/usr/bin/env python
"""Does Bridge's raw rotation at trajectory-start approximate zero?

lda_policy.py's _build_state() reports rotation relative to the pose captured
at THIS SimplerEnv episode's first step (copied from F1's fix #4). That is
only equivalent to however Bridge's own raw rotation was defined (small angle
relative to a "gripper-down home pose", per the module docstring) if real
Bridge trajectories also start close to that home pose -- i.e. if
raw rotation at t=0 is close to zero across trajectories. If it is not, every
closed-loop rollout is conditioning the model on a rotation reference frame it
never saw in training, however small the per-step deltas look in isolation.

Cheap, offline, no model loaded: just reads raw_data["state.left_eef_rotation"]
(pre-normalization, straight off the converted LeRobot dataset) at t=0 for a
sample of real trajectories and reports its spread.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.bridge.lda_policy import LDAInference  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", default="/mnt/beegfsnew/scratch/3295540/data/bridge_lda")
    ap.add_argument("--n-trajectories", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    # LDAInference just to get a constructed dataset with the same transforms/
    # modality config the model trains against -- no forward pass needed here.
    policy = LDAInference(checkpoint_path=args.checkpoint, dataset_path=args.dataset_path, seed=0)
    ds = policy._dataset

    lengths = ds.trajectory_lengths
    eligible = [i for i, n in enumerate(lengths) if n >= 5]
    picks = rng.choice(eligible, size=min(args.n_trajectories, len(eligible)), replace=False)

    t0_rot = []
    t0_pos = []
    mid_rot_spreads = []
    for tid in picks:
        tid = int(tid)
        raw0 = ds.get_step_data(tid, 0)
        rpy0 = np.asarray(raw0["state.left_eef_rotation"]).reshape(-1)[:3]
        pos0 = np.asarray(raw0["state.left_eef_position"]).reshape(-1)[:3]
        t0_rot.append(rpy0)
        t0_pos.append(pos0)

        # Also sample a mid-trajectory step, to see the scale of WITHIN-trajectory
        # rotation change for comparison (is t0 special, or do all steps look
        # similar in magnitude to t0?).
        mid = int(lengths[tid] // 2)
        raw_mid = ds.get_step_data(tid, mid)
        rpy_mid = np.asarray(raw_mid["state.left_eef_rotation"]).reshape(-1)[:3]
        mid_rot_spreads.append(np.abs(rpy_mid - rpy0))

    t0_rot = np.array(t0_rot)
    t0_pos = np.array(t0_pos)
    mid_rot_spreads = np.array(mid_rot_spreads)

    print(f"trajectories sampled: {len(picks)}")
    print()
    print("raw state.left_eef_rotation AT t=0, across trajectories (rad):")
    print(f"  mean   : {t0_rot.mean(axis=0)}")
    print(f"  std    : {t0_rot.std(axis=0)}")
    print(f"  min    : {t0_rot.min(axis=0)}")
    print(f"  max    : {t0_rot.max(axis=0)}")
    print(f"  |mean| across all 3 axes: {np.abs(t0_rot.mean(axis=0)).mean():.4f} rad "
          f"({np.degrees(np.abs(t0_rot.mean(axis=0)).mean()):.2f} deg)")
    print(f"  overall std (all axes, all trajs): {t0_rot.std():.4f} rad "
          f"({np.degrees(t0_rot.std()):.2f} deg)")
    print()
    print("raw state.left_eef_position AT t=0, across trajectories (m):")
    print(f"  mean   : {t0_pos.mean(axis=0)}")
    print(f"  std    : {t0_pos.std(axis=0)}")
    print()
    print("for reference -- |rotation change| from t=0 to mid-trajectory (rad):")
    print(f"  mean   : {mid_rot_spreads.mean(axis=0)}  (rows: roll, pitch, yaw)")
    print()
    if t0_rot.std() < 0.05:
        print("VERDICT: t=0 rotation is tightly clustered near a fixed value across "
              "trajectories -> Bridge trajectories DO start near a consistent home "
              "pose. _build_state()'s per-episode-first-step reference should closely "
              "approximate training's convention.")
    else:
        print("VERDICT: t=0 rotation VARIES substantially across trajectories -> "
              "Bridge's raw rotation is NOT simply 'small angle from a fixed home "
              "pose' at every trajectory's start. _build_state()'s per-episode "
              "reference (SimplerEnv's own reset pose) will disagree with whatever "
              "reference frame training's raw data actually encodes, unless "
              "SimplerEnv's reset pose happens to match Bridge's convention for "
              "other reasons.")


if __name__ == "__main__":
    main()
