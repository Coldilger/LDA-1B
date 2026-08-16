"""Entry point for the soft-clip state-normalization experiment (see
lda_policy_softclip.py's module docstring for what/why). Mirrors
main_inference.py exactly, swapping in LDAInferenceSoftClip.
"""

import argparse
import os
import sys

import numpy as np

SIMPLER_ENV_ROOT = "/mnt/beegfsnew/scratch/3295540/F1-VLA/eval/bridge/SimplerEnv"
sys.path.insert(0, SIMPLER_ENV_ROOT)

from simpler_env.evaluation.argparse import get_args  # noqa: E402
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from eval.bridge.lda_policy_softclip import LDAInferenceSoftClip  # noqa: E402


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


if __name__ == "__main__":
    lda_args, remaining_argv = parse_lda_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + remaining_argv
    args = get_args()

    os.environ["DISPLAY"] = ""

    model = LDAInferenceSoftClip(
        checkpoint_path=lda_args.lda_checkpoint_path,
        dataset_path=lda_args.lda_dataset_path,
        device=lda_args.lda_device,
        exec_horizon=lda_args.lda_exec_horizon,
        invert_gripper=lda_args.lda_invert_gripper,
        seed=lda_args.lda_seed,
    )
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
