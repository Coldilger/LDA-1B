#!/usr/bin/env python3
"""
Experiment 4 positive control: do the pooled features carry *anything*?

The future-pose probe came back at the level of trivial baselines for both
checkpoints (ridge L1 0.05292 finetuned / 0.05379 pretrained, vs 0.05326 for
the best constant and 0.05266 for predicting no motion). Four explanations fit
that equally well:

  1. pooling 19200 tokens down to mean+std destroyed the usable structure
  2. 400 samples from 40 episodes is too little
  3. the K=5 pose delta is mostly noise at Bridge's control rate
  4. the representation genuinely does not encode future pose linearly

Only (4) is a finding; the rest are measurement failures. This separates them
without re-extracting anything, by probing for something the features are
*guaranteed* to contain: which episode a sample came from. Episodes differ in
scene, objects and task text, all of which the video backbone sees directly.

If episode identity is recoverable and future pose is not, the instrument
works and the null result stands. If episode identity is *also* unrecoverable,
the pooled features are too lossy to support any conclusion about future
information, and the extraction needs to change before this experiment can say
anything.

Uses the same standardization and the same dual-form ridge as train_probe.py,
so nothing but the target differs.

CORRECTED 2026-08-18: the original version of this script selected alpha by
directly maximizing accuracy ON THE HELD-OUT VAL SET -- a leak (val used for
both hyperparameter selection and evaluation), which inflated the originally
reported "100%, 40x chance" result for both mimic-video and F1-VLA. Alpha is
now chosen by k-fold CV within the train split only, matching train_probe.py/
compare_targets.py's already-correct methodology, never touching val labels
before the final scoring pass. The qualitative finding survives (episode
identity is still well above chance), but the reported magnitude was wrong
and needed correcting, not just noting.
"""

import argparse

import numpy as np


def ridge_fit_predict(Xtr, Ytr, Xva, alpha: float):
    n = Xtr.shape[0]
    K = Xtr @ Xtr.T
    dual = np.linalg.solve(K + alpha * np.eye(n), Ytr)
    return Xva @ (Xtr.T @ dual)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["features"].astype(np.float64)
    ep = data["episode_ids"]
    variant = str(data["variant"])

    uniq = np.unique(ep)
    # One-hot episode identity as a regression target, scored by argmax --
    # ridge classification, so the probe family matches the pose probe exactly.
    onehot = (ep[:, None] == uniq[None, :]).astype(np.float64)

    # Split by SAMPLE here, not by episode: the question is whether the
    # features encode episode identity at all, which requires every episode to
    # appear on both sides. This is a deliberately easy task -- that is the
    # point of a positive control.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(X))
    n_val = int(round(len(X) * args.val_frac))
    va, tr = order[:n_val], order[n_val:]

    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    print(f"variant={variant}  samples={len(X)}  episodes={len(uniq)}")
    print(f"split: {len(tr)} train / {len(va)} val")
    chance = 1.0 / len(uniq)
    print(f"chance accuracy: {chance:.4f}\n")

    # alpha chosen by k-fold CV WITHIN the train split only -- val is never
    # touched until the single final scoring pass below.
    rng_cv = np.random.default_rng(args.seed)
    n_folds = 5
    cv_order = rng_cv.permutation(len(tr))
    folds = np.array_split(cv_order, n_folds)

    best_alpha, best_cv_acc = None, -1.0
    for alpha in np.logspace(-1, 6, 22):
        accs = []
        for i in range(n_folds):
            fold_va = folds[i]
            fold_tr = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            pred = ridge_fit_predict(Xn[tr][fold_tr], onehot[tr][fold_tr], Xn[tr][fold_va], alpha)
            accs.append(float((uniq[pred.argmax(1)] == ep[tr][fold_va]).mean()))
        cv_acc = float(np.mean(accs))
        if cv_acc > best_cv_acc:
            best_alpha, best_cv_acc = float(alpha), cv_acc

    pred = ridge_fit_predict(Xn[tr], onehot[tr], Xn[va], best_alpha)
    acc = float((uniq[pred.argmax(1)] == ep[va]).mean())
    alpha = best_alpha
    print(f"episode-identity accuracy: {acc:.4f}  (alpha={alpha:g}, CV-selected, train-only)")
    print(f"chance:                    {chance:.4f}")
    print(f"ratio over chance:         {acc / chance:.1f}x")
    print()
    if acc > 5 * chance:
        print("VERDICT: features carry strong episode information -- the probe")
        print("pipeline works, so the flat future-pose result is about the")
        print("target, not the instrument.")
    elif acc > 2 * chance:
        print("VERDICT: features carry some episode information, but weakly.")
        print("The future-pose null is suggestive but the pooling is a live")
        print("alternative explanation.")
    else:
        print("VERDICT: features barely beat chance on a task they should ace.")
        print("The pooled representation is too lossy; the future-pose result")
        print("cannot be interpreted until extraction changes.")


if __name__ == "__main__":
    main()
