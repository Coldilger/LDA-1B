#!/usr/bin/env python
"""Decisive A/B test for the leading hypothesis behind LDA's 0% closed-loop
success: does gripper accuracy depend on history_action being GROUND TRUTH
specifically, or does the model's own self-generated history work just as well?

This reuses diag_gripper.py's exact, already-trusted evaluation methodology
(balanced open/closed sampling, accuracy + correlation against ground truth,
comparing denormalized predictions -- diag_gripper.py is what originally
produced the +0.86 correlation / 93% accuracy number, WITH ground-truth
history_action from get_step_data_with_transform). The only thing that
changes here is where history_action comes from: instead of pulling it from
the dataset, we replay the trajectory for a few real steps through
LDAInference.step() (exactly as real closed-loop would) and let
self.action_history accumulate the model's own predictions naturally, then
read it off via model._history_array() at the sample point.

Everything else -- image, task description, prediction call, denormalization,
scoring -- is identical to diag_gripper.py. If accuracy/correlation collapses
here relative to the --with-history run, that's strong, apples-to-apples
evidence for the "model needs TRUE history, its own guesses aren't good
enough" hypothesis. If it doesn't collapse, the hypothesis is wrong and the
real cause is elsewhere.
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
    ap.add_argument("--warmup-steps", type=int, default=6, help="Real steps replayed before sampling, so action_history is non-trivially self-generated (not just zero-padding).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    policy = LDAInference(checkpoint_path=args.checkpoint, dataset_path=args.dataset_path, seed=0)
    ds = policy._dataset

    lengths = ds.trajectory_lengths
    eligible = [i for i, n in enumerate(lengths) if n >= args.warmup_steps + 20]

    want_per_class = max(1, args.n_samples // 2)
    gt_all, pred_all = [], []
    n_open = n_closed = 0
    attempts = 0

    while (n_open < want_per_class or n_closed < want_per_class) and attempts < args.n_samples * 25:
        attempts += 1
        tid = int(rng.choice(eligible))
        base = int(rng.integers(0, max(1, lengths[tid] - args.warmup_steps - 18)))

        # Peek at the ground truth gripper we'd be sampling at, same balancing
        # logic as diag_gripper.py, before paying for the warmup replay.
        peek = ds.get_step_data_with_transform(tid, base + args.warmup_steps, return_state=True)
        gt_peek = float(np.asarray(peek["action"])[0, 6])
        if gt_peek > 0.5 and n_open >= want_per_class:
            continue
        if gt_peek <= 0.5 and n_closed >= want_per_class:
            continue

        # Real closed-loop replay: warmup_steps real steps through step(), so
        # action_history/image_history are populated the way they would be
        # mid-episode, with the model's OWN past predictions -- not ground truth.
        task_description = None
        policy.reset("")
        for k in range(args.warmup_steps):
            data_k = ds.get_step_data_with_transform(tid, base + k, return_state=True)
            if task_description is None:
                task_description = data_k["lang"]
                policy.task_description = task_description
            img_k = np.asarray(data_k["image"][-1])  # current-frame view (post expand2square+224 resize)
            policy.image_history.append(img_k)
            example_k = {
                "image": np.stack([np.asarray(im) for im in data_k["image"][:2]], axis=0),
                "lang": task_description,
                "embodiment_id": policy.embodiment_id,
                "history_action": policy._history_array(),
            }
            with torch.no_grad():
                raw_k = np.asarray(policy.policy.predict_action([example_k])["normalized_actions"][0])
            policy.action_history.append(raw_k[0].astype(np.float16))

        # Now take the actual sample: same call pattern as diag_gripper.py,
        # but history_action comes from policy._history_array() (self-generated)
        # instead of the dataset's ground truth.
        data = ds.get_step_data_with_transform(tid, base + args.warmup_steps, return_state=True)
        example = {
            "image": np.stack([np.asarray(im) for im in data["image"][:2]], axis=0),
            "lang": task_description,
            "embodiment_id": policy.embodiment_id,
            "history_action": policy._history_array(),
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
    print(f"\nsamples: {len(gt)}  (warmup_steps={args.warmup_steps}, self-generated history)")
    print(f"ground truth: open(1) {int((gt > 0.5).sum())}, closed(0) {int((gt <= 0.5).sum())}")
    print(f"prediction:   min {pred.min():.3f}  max {pred.max():.3f}  mean {pred.mean():.3f}")
    print()
    direct = ((pred > 0.5) == (gt > 0.5)).mean()
    inverted = ((1 - pred > 0.5) == (gt > 0.5)).mean()
    corr = float(np.corrcoef(pred, gt)[0, 1]) if pred.std() > 1e-9 else 0.0
    mean_open = pred[gt > 0.5].mean() if (gt > 0.5).any() else float("nan")
    mean_closed = pred[gt <= 0.5].mean() if (gt <= 0.5).any() else float("nan")
    print(f"agreement, as-is      : {direct * 100:5.1f}%   (50% = chance on a balanced set)")
    print(f"agreement, inverted   : {inverted * 100:5.1f}%")
    print(f"correlation with truth: {corr:+.3f}")
    print(f"mean score | truth=open  : {mean_open:.3f}")
    print(f"mean score | truth=closed: {mean_closed:.3f}   (separation {mean_open - mean_closed:+.3f})")
    print()
    print("Compare directly against diag_gripper.py --with-history (ground-truth history):")
    print("  93.0% accuracy / +0.863 correlation (documented in lda_policy.py).")
    print("If this run is close to that, self-generated history is NOT the problem.")
    print("If this run collapses toward 50% / ~0 correlation, self-generated history IS the problem.")


if __name__ == "__main__":
    main()
