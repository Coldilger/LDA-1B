#!/usr/bin/env python
"""Does the model's gripper prediction track ground truth, or its inverse?

The action-magnitude diagnostic showed correct motion scale but a gripper pinned
at "closed" on every step, which alone would explain is_src_obj_grasped being
false throughout the smoke eval. One frame cannot distinguish a polarity flip
from an undertrained head, so this samples many frames whose ground-truth gripper
differs and scores both readings.

Bridge encodes 0 = close, 1 = open, and the converted data keeps that convention.
lda/utils/eval_relative_eef.py applies an unconditional 1 - gripper, which is an
AgiBot convention -- if the pretrained head carries it, predictions will agree
with the inverted reading instead.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from eval.bridge.lda_policy import RAW_SLICES, LDAInference  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset-path", default="/mnt/beegfsnew/scratch/3295540/data/bridge_lda")
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    policy = LDAInference(checkpoint_path=args.checkpoint, dataset_path=args.dataset_path, seed=0)
    ds = policy._dataset

    lengths = ds.trajectory_lengths
    eligible = [i for i, n in enumerate(lengths) if n >= 20]
    picks = rng.choice(eligible, size=min(args.n_samples, len(eligible)), replace=False)

    # Sampling frames at random gives ~80% open, so a thresholded accuracy is
    # dominated by the majority class and a real change hides inside it. Collect a
    # balanced set instead: keep drawing until both classes are filled.
    want_per_class = max(1, args.n_samples // 2)
    gt_all, pred_all = [], []
    n_open = n_closed = 0
    attempts = 0
    while (n_open < want_per_class or n_closed < want_per_class) and attempts < args.n_samples * 25:
        attempts += 1
        tid = int(rng.choice(picks))
        base = int(rng.integers(0, max(1, lengths[tid] - 18)))
        data = ds.get_step_data_with_transform(tid, base, return_state=True)
        gt_peek = float(np.asarray(data["action"])[0, 6])
        if gt_peek > 0.5 and n_open >= want_per_class:
            continue
        if gt_peek <= 0.5 and n_closed >= want_per_class:
            continue
        example = {
            "image": np.stack([np.asarray(im) for im in data["image"][:2]], axis=0),
            "lang": data["lang"],
            "embodiment_id": policy.embodiment_id,
        }
        with torch.no_grad():
            raw = np.asarray(policy.policy.predict_action([example])["normalized_actions"][0])
        denorm = policy._transforms.unapply(
            {k: torch.as_tensor(raw[:, sl]) for k, sl in RAW_SLICES.items()}
        )
        pred = float(np.asarray(denorm["action.left_gripper"]).reshape(-1)[0])
        gt = gt_peek
        gt_all.append(gt)
        pred_all.append(pred)
        if gt > 0.5:
            n_open += 1
        else:
            n_closed += 1

    gt = np.array(gt_all)
    pred = np.array(pred_all)
    print(f"\nsamples: {len(gt)}")
    print(f"ground truth: open(1) {int((gt > 0.5).sum())}, closed(0) {int((gt <= 0.5).sum())}")
    print(f"prediction:   min {pred.min():.3f}  max {pred.max():.3f}  mean {pred.mean():.3f}")
    print()
    direct = ((pred > 0.5) == (gt > 0.5)).mean()
    inverted = ((1 - pred > 0.5) == (gt > 0.5)).mean()
    # On a balanced set both trivial baselines sit at 50%, so accuracy is readable
    # directly. Correlation is reported too because it moves before the threshold
    # does -- a head that has started to learn shifts its scores well before enough
    # of them cross 0.5 to change the accuracy.
    corr = float(np.corrcoef(pred, gt)[0, 1]) if pred.std() > 1e-9 else 0.0
    mean_open = pred[gt > 0.5].mean()
    mean_closed = pred[gt <= 0.5].mean()
    print(f"agreement, as-is      : {direct * 100:5.1f}%   (50% = chance on a balanced set)")
    print(f"agreement, inverted   : {inverted * 100:5.1f}%")
    print(f"correlation with truth: {corr:+.3f}")
    print(f"mean score | truth=open  : {mean_open:.3f}")
    print(f"mean score | truth=closed: {mean_closed:.3f}   (separation {mean_open - mean_closed:+.3f})")
    print()
    if pred.max() - pred.min() < 0.1:
        print("VERDICT: prediction barely varies -> the head is not discriminating yet;")
        print("         polarity cannot be judged from this, and neither reading is meaningful.")
    elif inverted > direct + 0.15:
        print("VERDICT: inverted reading wins clearly -> set invert_gripper=True.")
    elif direct > inverted + 0.15:
        print("VERDICT: as-is reading is correct -> leave invert_gripper=False.")
    else:
        print("VERDICT: neither polarity separates from the other -> undertrained, recheck later.")


if __name__ == "__main__":
    main()
