"""Entry point for LDA-1B closed-loop rollout eval in SimplerEnv.

Mirrors F1-VLA/eval/bridge/main_inference.py so the two models are driven by the
same evaluator, the same CLI, and therefore the same protocol -- which is the
whole point of the comparison. SimplerEnv's evaluator and arg parser are
policy-agnostic and are reused as-is rather than duplicated.

SimplerEnv itself is not vendored again here: F1-VLA already carries a working
copy (its RESULTS.md records completed 24-episode runs against it), so this
points at that one. Nothing is written there.

Environment note: this needs an interpreter that has BOTH the LDA stack and the
SimplerEnv/ManiSkill2/sapien stack importable. The `lda` conda env has the former
only. See slurm/eval_bridge_simpler.slurm for how the run is launched.

Example:
    python eval/bridge/main_inference.py \
        --policy-setup widowx_bridge --robot widowx \
        --env-name PutCarrotOnPlateInScene-v0 --scene-name bridge_table_1_v1 \
        --lda-checkpoint-path outputs/lda_bridge/bridge_finetune/checkpoints/steps_150000_pytorch_model.pt
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
from eval.bridge.lda_policy import LDAInference  # noqa: E402


def parse_lda_args(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--lda-checkpoint-path", type=str, required=True)
    parser.add_argument(
        "--lda-dataset-path",
        type=str,
        default="/mnt/beegfsnew/scratch/3295540/data/bridge_lda",
        help="Only read for its fitted transforms: unapply needs the same q99 "
        "statistics the policy's outputs are normalized against.",
    )
    parser.add_argument("--lda-device", type=str, default="cuda")
    parser.add_argument(
        "--lda-exec-horizon",
        type=int,
        default=8,
        help="Steps executed per predicted 16-step chunk. Fewer means replanning "
        "more often. Untuned -- worth sweeping.",
    )
    parser.add_argument(
        "--lda-invert-gripper",
        action="store_true",
        help="Apply the AgiBot-convention 1-gripper flip. Off by default: the "
        "converted Bridge data keeps Bridge's own polarity.",
    )
    parser.add_argument(
        "--lda-seed",
        type=int,
        default=None,
        help="Diffusion sampling is stochastic; set per run to average over seeds.",
    )
    known, remaining = parser.parse_known_args(argv)
    return known, remaining


if __name__ == "__main__":
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
    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
