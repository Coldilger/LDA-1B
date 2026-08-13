#!/usr/bin/env python
"""Verify the episode-level gripper presence fix in `datasets.py`.

Before the fix, `pad_action_state_with_key` judged whether the gripper modality
was "present" from the current ~16-step training window alone, so a window that
happened to be entirely gripper-closed (holding an object) was masked out of the
loss as if the gripper did not exist for this embodiment -- the same failure mode
as the per-timestep bug that was fixed earlier, recurring at window granularity.
Measured on real Bridge data: 16.5% of 16-step windows are constant-closed.

This script pulls real samples through the actual training-time dataset
(`get_vla_dataset`, the same call `train_LDA.py` makes) and checks that
constant-closed windows now come back with `action_mask` == True at the gripper
index, instead of False.
"""
from __future__ import annotations

import random
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, "/mnt/beegfsnew/scratch/3295540/LDA-1B")
from lda.dataloader.lerobot_datasets import get_vla_dataset  # noqa: E402

GRIPPER_INDEX = 6  # RAW_SLICES["action.left_gripper"] in eval/bridge/lda_policy.py


def main() -> None:
    cfg = OmegaConf.load(
        "/mnt/beegfsnew/scratch/3295540/LDA-1B/outputs/lda_bridge/bridge_finetune/config.yaml"
    )
    data_cfg = cfg.datasets.vla_data
    model_cfg = cfg.framework.action_model

    dataset = get_vla_dataset(
        data_cfg=data_cfg,
        model_cfg={"state_dim": model_cfg.get("state_dim", None), "action_dim": model_cfg.action_dim},
    )
    print(f"dataset length: {len(dataset)}", flush=True)

    random.seed(0)
    n_samples = 300
    n_const_closed = 0
    n_const_closed_masked_true = 0
    n_const_closed_masked_false = 0
    n_const_open = 0
    n_const_open_masked_true = 0

    for _ in range(n_samples):
        idx = random.randint(0, len(dataset) - 1)
        item = dataset[idx]
        action = np.asarray(item["action"])
        action_mask = np.asarray(item["action_mask"])
        gv = action[:, GRIPPER_INDEX]
        gm = action_mask[:, GRIPPER_INDEX]

        if np.all(gv == gv[0]):
            if gv[0] < 0.5:
                n_const_closed += 1
                if gm.all():
                    n_const_closed_masked_true += 1
                elif not gm.any():
                    n_const_closed_masked_false += 1
            else:
                n_const_open += 1
                if gm.all():
                    n_const_open_masked_true += 1

    print(f"samples checked: {n_samples}", flush=True)
    print(f"constant-CLOSED windows seen: {n_const_closed}", flush=True)
    print(f"  -> masked True  (fix working):     {n_const_closed_masked_true}", flush=True)
    print(f"  -> masked False (fix NOT working): {n_const_closed_masked_false}", flush=True)
    print(f"constant-OPEN windows seen: {n_const_open}", flush=True)
    print(f"  -> masked True: {n_const_open_masked_true}", flush=True)

    if n_const_closed > 0 and n_const_closed_masked_false == 0 and n_const_closed_masked_true == n_const_closed:
        print("VERIFY: PASS -- constant-closed windows now keep gripper supervision", flush=True)
    else:
        print("VERIFY: FAIL -- constant-closed windows are still being masked out", flush=True)


if __name__ == "__main__":
    main()
