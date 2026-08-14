#!/usr/bin/env python
"""
Offline oracle probe for LDA-1B on real BridgeDataV2 episodes -- same protocol
family as F1-VLA/eval/bridge/oracle_offline_probe.py and
mimic-video/eval/bridge/oracle_offline_probe.py.

NOT RUN YET: the Bridge finetune this needs to point at is being retrained
(see conversation). Written now so all three scripts land together and use
the same comparison logic; point --checkpoint at the new finetune once it's
ready.

Unlike the other two, this does not need a new plumbing script for the data
side: dataset.get_step_data_with_transform() (used by the existing
lda/eval/eval_bridge_openloop.py probe) already returns a `future_image` key
per step -- the real next frame(s), in the exact tensor format training used
for the forward_dynamics/inverse_dynamics/video_gen tasks. So "real future
instead of imagined future" here is just "don't strip that key before calling
predict_action", not a new encode-and-inject path.

The actual oracle mechanism (non-destructive hook added to
GR00T_ActionHeader_single_t_concat_curr_obs.py:predict_action and threaded
through QwenGR00T_single_t_concat_curr_obs.py:predict_action, both default
None/no-op): if the example dict passed to policy.predict_action() carries a
"future_image" key, task_embedding switches from policy_embedding to
id_embedding and the next_obs conditioning tokens are encoded from that real
future frame (encode_future_img) instead of next_obs_learnable_tokens -- i.e.
this runs the SAME inverse_dynamics task the model was actually trained on,
not a bolted-on ablation. See GR00T_ActionHeader_single_t_concat_curr_obs.py
lines 640-660 (training-time next_obs construction) vs the new
predict_action() oracle_future_imgs branch (mirrors it exactly).

CAVEAT (must resolve before comparing numbers to F1/mimic): predict_action()
returns *normalized* actions (the model's own training-space units), not
physical units. eval_bridge_openloop.py's calc_mse_for_single_trajectory
unapplies the dataset transform before comparing (see its `unapply(...)`
calls) -- this script does not do that yet, so its L1 numbers are only
internally comparable (oracle vs policy-default here), not yet in the same
units as F1's/mimic's action-space L1. Port that unapply step before citing
this against the other two.

PERFORMANCE WARNING before first run: F1's and mimic-video's equivalent
scripts both hung silently for a full 1h SLURM wall-time and got killed with
zero output, because raw LeRobotDataset.__init__ does one Path.is_file() per
episode/video file, and each call is an uncached network round-trip on
beegfs -- ~53k episodes turns into hours. Both were fixed with a
directory-listing cache context manager (see _fast_path_is_file in
F1-VLA/eval/bridge/oracle_offline_probe.py). LeRobotSingleDataset here is a
different loader (LDA's own, not raw LeRobotDataset), so it is NOT confirmed
to have the same problem -- but check its startup time on a small --n-trajs
before trusting a long unattended run.
"""

from __future__ import annotations

import argparse

import numpy as np
from omegaconf import OmegaConf

from lda.dataloader.gr00t_lerobot.data_config import OxeDataConfig
from lda.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset
from lda.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from lda.model.framework.base_framework import baseframework

LEFT_POS = slice(0, 3)
LEFT_ROT = slice(3, 6)
LEFT_GRIP = 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-path", default="/mnt/beegfsnew/scratch/3295540/data/bridge_lda")
    ap.add_argument(
        "--checkpoint",
        default="outputs/lda_bridge_v2/bridge_finetune_v2/final_model/pytorch_model.pt",
        help="VERIFY once retraining finishes -- current path from LDA-1B/eval/bridge/RESULTS.md.",
    )
    ap.add_argument("--config-yaml", default="lda/config/training/LDA_bridge_v2.yaml")
    ap.add_argument("--n-trajs", type=int, default=24)
    ap.add_argument("--samples-per-traj", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--traj-ids",
        type=str,
        default=None,
        help="Comma-separated explicit trajectory ids, bypassing random sampling. "
        "For testing against a partial download where only some episodes have video yet.",
    )
    args = ap.parse_args()

    np.random.seed(args.seed)

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

    lengths = dataset.trajectory_lengths
    if args.traj_ids is not None:
        chosen = [int(x) for x in args.traj_ids.split(",")]
    else:
        eligible = [i for i, n in enumerate(lengths) if n >= 3]
        chosen = np.random.choice(eligible, size=min(args.n_trajs, len(eligible)), replace=False)

    oracle_l1s, policy_l1s = [], []

    for traj_id in chosen:
        traj_id = int(traj_id)
        max_step = int(lengths[traj_id]) - 2
        if max_step < 1:
            continue
        steps = np.random.choice(
            np.arange(max_step), size=min(args.samples_per_traj, max_step), replace=False
        )

        for step in steps:
            step = int(step)
            data_point = dataset.get_step_data_with_transform(
                traj_id, step, None, policy.config.framework.qwenvl.base_vlm, return_state=True
            )
            true_action = np.asarray(data_point["action"])[0]  # first step of the chunk

            # Oracle: keep future_image -> hits the id_embedding/inverse_dynamics
            # branch added to predict_action.
            oracle_out = policy.predict_action([data_point])
            oracle_pred = oracle_out["normalized_actions"][0][0]

            # Policy-default: same sample, future_image stripped -> original
            # next_obs_learnable_tokens / policy_embedding path, untouched.
            policy_only_point = {k: v for k, v in data_point.items() if k != "future_image"}
            policy_out = policy.predict_action([policy_only_point])
            policy_pred = policy_out["normalized_actions"][0][0]

            oracle_l1s.append(np.abs(oracle_pred[:7] - true_action[:7]).mean())
            policy_l1s.append(np.abs(policy_pred[:7] - true_action[:7]).mean())

    oracle_l1s, policy_l1s = np.array(oracle_l1s), np.array(policy_l1s)
    print(f"n samples: {len(oracle_l1s)}")
    print(f"action L1, oracle/inverse_dynamics (normalized space): {oracle_l1s.mean():.4f}")
    print(f"action L1, policy-default          (normalized space): {policy_l1s.mean():.4f}")
    print("NOTE: normalized-space units -- see module docstring before comparing to F1/mimic numbers.")


if __name__ == "__main__":
    main()
