#!/usr/bin/env python
"""Does lda_policy.py's live _build_state() output stay in-distribution?

diag_state_reference.py showed Bridge's raw rotation varies substantially
across trajectory starts -- comparable in scale to the real motion signal --
casting doubt on _build_state()'s "episode-start pose as rotation reference"
approximation. But F1-VLA uses the exact same approximation (copied verbatim,
see f1_vla_policy.py) and still gets ~48% average closed-loop success, not
0%, so the approximation alone can't be the whole story. This runs the model
through REAL closed-loop episodes (actual predicted actions applied to a real
SimplerEnv env, exactly what eval_bridge_simpler.slurm does) and captures
every live _build_state() output, to check directly whether it's landing
inside the q99-normalized range the model trained on, or saturating/blowing
up -- reusing the production code path (main_inference.py's own model
construction and simpler_env's own evaluator) with _build_state monkey-patched
to log its output, rather than reimplementing the rollout loop.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.bridge.lda_policy import LDAInference  # noqa: E402

STATE_COLS = [
    "left_x", "left_y", "left_z", "left_roll", "left_pitch", "left_yaw", "left_grip",
    "right_x", "right_y", "right_z", "right_roll", "right_pitch", "right_yaw", "right_grip",
]


def parse_lda_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lda-checkpoint-path", type=str, required=True)
    parser.add_argument("--lda-dataset-path", type=str, default="/mnt/beegfsnew/scratch/3295540/data/bridge_lda")
    parser.add_argument("--lda-device", type=str, default="cuda")
    parser.add_argument("--lda-exec-horizon", type=int, default=8)
    parser.add_argument("--lda-invert-gripper", action="store_true")
    parser.add_argument("--lda-seed", type=int, default=None)
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


def main():
    lda_args, remaining_argv = parse_lda_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_args()
    os.environ["DISPLAY"] = ""

    model = LDAInference(
        checkpoint_path=lda_args.lda_checkpoint_path,
        dataset_path=lda_args.lda_dataset_path,
        device=lda_args.lda_device,
        exec_horizon=lda_args.lda_exec_horizon,
        invert_gripper=lda_args.lda_invert_gripper,
        seed=lda_args.lda_seed,
    )

    captured = []  # list of (episode_step_counter, 14,) normalized state arrays
    step_counter = {"n": 0}
    original_build_state = model._build_state

    def instrumented_build_state(ee_pose_proprio, gripper_proprio):
        out = original_build_state(ee_pose_proprio, gripper_proprio)
        captured.append((step_counter["n"], np.asarray(out).reshape(-1).copy()))
        step_counter["n"] += 1
        return out

    model._build_state = instrumented_build_state

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))

    if not captured:
        print("\nNo _build_state() calls captured -- checkpoint has state_dim: null?")
        return

    S = np.array([c[1] for c in captured])  # (N, 14)
    print(f"\n=== live _build_state() output over {len(S)} real closed-loop steps ===")
    print("(already q99-normalized -- training data lives in roughly [-1, 1];")
    print(" values well outside that band indicate reference-frame/OOD state)\n")
    print(f"{'col':10s} {'min':>8s} {'max':>8s} {'mean':>8s} {'std':>8s}  frac(|x|>0.9)  frac(|x|>1.5)")
    for i, name in enumerate(STATE_COLS):
        col = S[:, i]
        frac_near_clip = float((np.abs(col) > 0.9).mean())
        frac_extreme = float((np.abs(col) > 1.5).mean())
        print(f"{name:10s} {col.min():8.3f} {col.max():8.3f} {col.mean():8.3f} {col.std():8.3f}  "
              f"{frac_near_clip:11.2%}  {frac_extreme:6.2%}")

    rot_cols = S[:, [3, 4, 5]]
    frac_extreme_rot = float((np.abs(rot_cols) > 1.5).mean())
    print(f"\noverall |x|>1.5 fraction, left rotation columns only: {frac_extreme_rot:.2%}")
    if frac_extreme_rot > 0.1:
        print("VERDICT: rotation state frequently lands well outside the training "
              "range -> the reference-frame approximation is producing "
              "out-of-distribution state on a meaningful fraction of real "
              "closed-loop steps.")
    else:
        print("VERDICT: rotation state mostly stays in-range -> the reference-frame "
              "approximation is not obviously blowing up; look elsewhere for the "
              "closed-loop failure.")


if __name__ == "__main__":
    main()
