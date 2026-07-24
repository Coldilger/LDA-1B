#!/usr/bin/env python
"""Open-loop probe: does LDA-pretrain already know Bridge/WidowX under the `oxe` tag?

Why this exists instead of lda/eval/eval_policy.py: that script's `evaluation.trajs`
controls how many *datasets* are sampled per embodiment tag, not how many
trajectories — with a single-dataset mix it always evaluates exactly one
trajectory, which is far too thin to decide a question this load-bearing. This
loops the same (correct) `eval_relative_eef` machinery over N trajectories.

It also reports the numbers that actually answer the question, which the built-in
metric does not:

  * LEFT-ARM-ONLY errors. WidowX is single-arm, so the right-arm dims are all zero
    in ground truth. The built-in calculate_mse averages both arms, so three of its
    six position numbers are a trivial zero-vs-prediction and the reported error is
    roughly halved.
  * A STAY-STILL baseline: the error you would get by predicting the arm never
    leaves its initial pose. Predicted deltas are integrated back to absolute poses
    by delta2abs, so a model that outputs ~zero deltas lands exactly on this
    baseline. Beating it is the minimum bar for "knows something".
  * Right-arm leakage: how much motion the model invents for an arm that does not
    exist. Large values mean the `oxe` slot is not emitting WidowX-shaped output.
  * Gripper under both polarities, since the harness applies an unconditional
    `1 - gripper` inversion (an AgiBot convention) that may be wrong for Bridge.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from omegaconf import OmegaConf

from lda.dataloader.gr00t_lerobot.data_config import OxeDataConfig
from lda.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from lda.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from lda.model.framework.base_framework import baseframework
from lda.utils.eval_relative_eef import calc_mse_for_single_trajectory

LEFT_POS = slice(0, 3)
LEFT_ROT = slice(3, 6)
LEFT_GRIP = 6
RIGHT_ALL = slice(7, 14)


def summarize(gt: np.ndarray, pred: np.ndarray) -> dict:
    """Per-trajectory metrics, left arm isolated from the zero-filled right arm."""
    stay_still = np.abs(gt[:, LEFT_POS] - gt[0, LEFT_POS]).mean()
    return {
        "steps": int(gt.shape[0]),
        "pos_l1": float(np.abs(gt[:, LEFT_POS] - pred[:, LEFT_POS]).mean()),
        "rot_l1": float(np.abs(gt[:, LEFT_ROT] - pred[:, LEFT_ROT]).mean()),
        "grip_l1": float(np.abs(gt[:, LEFT_GRIP] - pred[:, LEFT_GRIP]).mean()),
        "grip_l1_inverted": float(np.abs(gt[:, LEFT_GRIP] - (1 - pred[:, LEFT_GRIP])).mean()),
        "stay_still_baseline": float(stay_still),
        "gt_pos_spread": float(gt[:, LEFT_POS].std()),
        "right_arm_gt_max": float(np.abs(gt[:, RIGHT_ALL]).max()),
        "right_arm_pred_max": float(np.abs(pred[:, RIGHT_ALL]).max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="/mnt/beegfsnew/scratch/3295540/data/bridge_lda")
    ap.add_argument("--checkpoint", default="outputs/lda_pretrain_run/checkpoints/LDA-pretrain.pt")
    ap.add_argument("--config-yaml", default="checkpoints/LDA-pretrain/config.yaml")
    ap.add_argument("--n-trajs", type=int, default=12)
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=48, help="Cap steps per trajectory.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="outputs/bridge_openloop")
    args = ap.parse_args()

    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    cfg = OmegaConf.load(args.config_yaml)
    data_cfg = cfg.datasets.vla_data
    data_cfg.lerobot_version = "v2.0"

    dcfg = OxeDataConfig()
    dataset = LeRobotSingleDataset(
        dataset_path=args.dataset_path,
        modality_configs=dcfg.modality_config(),
        transforms=dcfg.transform(),
        embodiment_tag=EmbodimentTag.OXE,
        video_backend=dcfg.video_backend,
        img_interval=dcfg.img_interval,
        data_cfg=data_cfg,
    )

    policy = baseframework.from_pretrained(pretrained_checkpoint=args.checkpoint)
    policy.eval()
    policy.to("cuda")

    # Only trajectories long enough for a full chunk are informative.
    lengths = dataset.trajectory_lengths
    eligible = [i for i, n in enumerate(lengths) if n >= args.action_horizon + 2]
    chosen = np.random.choice(eligible, size=min(args.n_trajs, len(eligible)), replace=False)
    print(f"evaluating {len(chosen)} trajectories of {len(eligible)} eligible", flush=True)

    rows = []
    for traj_id in chosen:
        traj_id = int(traj_id)
        steps = int(min(lengths[traj_id], args.max_steps))
        try:
            calc_mse_for_single_trajectory(
                policy,
                dataset,
                traj_id,
                steps=steps,
                action_horizon=args.action_horizon,
                plot=False,
                plot_state=False,
                save_plot_path=args.out_dir,
                create_trajectory_video=False,
                video_output_path=None,
                original_video_path=None,
            )
        except Exception as e:  # keep going; one bad trajectory shouldn't sink the probe
            print(f"traj {traj_id}: FAILED {type(e).__name__}: {e}", flush=True)
            continue

        # save_npy_file writes directly to <save_plot_path>/traj_<id>. eval_policy.py
        # appears to nest under an embodiment name only because it passes a
        # save_plot_path that already ends in one.
        traj_dir = os.path.join(args.out_dir, f"traj_{traj_id}")
        gt = np.load(os.path.join(traj_dir, "gt_action_across_time.npy"))
        pred = np.load(os.path.join(traj_dir, "pred_action_across_time.npy"))
        row = summarize(gt, pred)
        row["traj_id"] = traj_id
        rows.append(row)
        print(
            f"traj {traj_id}: pos={row['pos_l1']:.5f} (stay-still {row['stay_still_baseline']:.5f}) "
            f"rot={row['rot_l1']:.5f} grip={row['grip_l1']:.3f} "
            f"right_pred_max={row['right_arm_pred_max']:.3f}",
            flush=True,
        )

    if not rows:
        print("no trajectories evaluated", flush=True)
        return

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    print("\n================ SUMMARY (left arm only) ================", flush=True)
    print(f"trajectories:            {len(rows)}", flush=True)
    print(f"position L1:             {avg('pos_l1'):.5f} m", flush=True)
    print(f"  stay-still baseline:   {avg('stay_still_baseline'):.5f} m", flush=True)
    ratio = avg("pos_l1") / max(avg("stay_still_baseline"), 1e-9)
    print(f"  model / baseline:      {ratio:.3f}   (<1 better than not moving)", flush=True)
    print(f"  gt position spread:    {avg('gt_pos_spread'):.5f} m", flush=True)
    print(f"rotation L1:             {avg('rot_l1'):.5f} rad", flush=True)
    print(f"gripper L1:              {avg('grip_l1'):.3f}  (0.5 = coin flip)", flush=True)
    print(f"  if polarity inverted:  {avg('grip_l1_inverted'):.3f}", flush=True)
    print(f"right-arm gt max:        {avg('right_arm_gt_max'):.3f}  (WidowX has none)", flush=True)
    print(f"right-arm predicted max: {avg('right_arm_pred_max'):.3f}  (invented motion)", flush=True)

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump({"per_traj": rows, "n_trajs": len(rows)}, f, indent=2)
    print(f"\nwrote {args.out_dir}/summary.json", flush=True)


if __name__ == "__main__":
    main()
