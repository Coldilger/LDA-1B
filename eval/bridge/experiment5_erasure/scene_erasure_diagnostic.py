#!/usr/bin/env python3
"""
Experiment 5, Level A: does erasing scene identity make pose recoverable?

Both models' Experiment 4 decisive control (compare_targets.py) found the
probe ~3.5x WORSE than a constant predictor on CURRENT pose -- a quantity
unambiguously present in the input frame -- while an episode-identity probe
scored 100% (40x chance). Read together: the representation is dominated by
which-scene-is-this, and with ~30 training episodes a probe fit on it
extrapolates catastrophically to unseen scenes, regardless of how carefully
pooling preserves spatial detail (confirmed across two structurally
different extraction designs: mimic-video's spatial grid, F1-VLA's
individually-kept 30 tokens).

This asks the sharper question directly, using LEACE (leace.py) rather than
re-running with more data (expensive, ~1.5h/extraction, and the payoff was
unclear given Bridge episodes cluster into a handful of scene families):
fit a closed-form eraser that removes exactly the subspace of the frozen
features that linearly distinguishes among the TRAINING episodes, apply it
to both train and held-out val features, and re-run the SAME probes used in
Experiment 4 on the erased result.

Three readable outcomes:
  - Episode-identity accuracy collapses to ~chance AND current-pose becomes
    recoverable (beats the constant predictor) -> pose information was
    present all along, masked by scene dominance. "More diverse episodes"
    would likely have fixed this too, but this confirms it cheaply, on data
    already extracted, at zero GPU cost.
  - Episode-identity collapses but pose remains unrecoverable -> scene
    dominance wasn't (solely) the problem; extraction/pooling or sample
    count is still the binding constraint, independent of the confound.
  - Episode-identity doesn't collapse -> the erasure itself didn't work as
    intended on this data; a bug, not a finding (leace.py's self-test
    already validates the method itself on synthetic data with a known
    answer, so this would point at something in how this script applies it).

Erasure is fit ONLY on the train split and applied unchanged to held-out val
features -- never re-fit per point (the paper's own warning about the
oracle variant, O-LEACE, potentially INCREASING nonlinearly-recoverable
concept information; see leace.py's docstring).
"""

import argparse
import json

import numpy as np
import torch
from torch import nn

from leace import fit_leace


def ridge_fit_predict(Xtr, Ytr, Xva, alpha):
    n = Xtr.shape[0]
    K = Xtr @ Xtr.T
    dual = np.linalg.solve(K + alpha * np.eye(n), Ytr)
    return Xva @ (Xtr.T @ dual)


def ridge_cv(Xtr, Ytr, Xva, Yva, seed, n_folds=5, is_classification=False):
    """Same CV-alpha-selection ridge as train_probe.py/compare_targets.py.

    For classification (Z = one-hot episode id), alpha is selected by CV
    classification accuracy (argmax match), not CV L1 -- L1-on-one-hot
    systematically favors an over-regularized alpha relative to what
    actually maximizes accuracy (confirmed: using L1 here silently produced
    20% where the correct criterion gives 100%, matching positive_control.py's
    own from-scratch CV loop on the same data). For regression, alpha is
    selected by CV L1 as before.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(Xtr))
    order = rng.permutation(idx)
    folds = np.array_split(order, n_folds)

    if is_classification:
        best_alpha, best_acc = None, -1.0
        for alpha in np.logspace(-1, 6, 22):
            accs = []
            for i in range(n_folds):
                vi = folds[i]
                ti = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                pred = ridge_fit_predict(Xtr[ti], Ytr[ti], Xtr[vi], alpha)
                accs.append(float((pred.argmax(1) == Ytr[vi].argmax(1)).mean()))
            acc = float(np.mean(accs))
            if acc > best_acc:
                best_alpha, best_acc = float(alpha), acc
        pred = ridge_fit_predict(Xtr, Ytr, Xva, best_alpha)
        return pred, best_alpha

    best_alpha, best_err = None, np.inf
    for alpha in np.logspace(-1, 6, 22):
        errs = []
        for i in range(n_folds):
            vi = folds[i]
            ti = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            pred = ridge_fit_predict(Xtr[ti], Ytr[ti], Xtr[vi], alpha)
            errs.append(np.abs(pred - Ytr[vi]).mean())
        e = float(np.mean(errs))
        if e < best_err:
            best_alpha, best_err = float(alpha), e

    pred = ridge_fit_predict(Xtr, Ytr, Xva, best_alpha)
    l1 = float(np.abs(pred - Yva).mean())
    const_l1 = float(np.abs(Yva - Ytr.mean(0, keepdims=True)).mean())
    gain = 100.0 * (const_l1 - l1) / const_l1 if const_l1 > 0 else float("nan")
    return l1, const_l1, gain, best_alpha


def split_by_episode(episode_ids, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(episode_ids)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(uniq[:n_val].tolist())
    is_val = np.array([e in val_eps for e in episode_ids])
    return ~is_val, is_val, uniq[n_val:], uniq[:n_val]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--current-pose", default="current_pose.npz")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["features"].astype(np.float64)
    future = data["targets"].astype(np.float64)
    ep = data["episode_ids"]
    variant = str(data["variant"])

    cur = np.load(args.current_pose, allow_pickle=True)
    current = cur["current_pose"].astype(np.float64)
    if not np.array_equal(cur["episode_ids"], ep):
        raise SystemExit("episode order mismatch between features and current_pose")

    tr, va, tr_eps, va_eps = split_by_episode(ep, args.val_frac, args.seed)
    print(f"variant={variant}  n={len(X)}  train={tr.sum()} ({len(tr_eps)} episodes) "
          f"val={va.sum()} ({len(va_eps)} episodes)\n", flush=True)

    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    # The eraser is fit on the EPISODE-split's train partition only -- it
    # removes the subspace that best distinguishes among those 30 episodes,
    # as a fixed linear map, later applied identically to train and to
    # entirely-unseen val episodes.
    onehot_fit = (ep[tr][:, None] == tr_eps[None, :]).astype(np.float64)
    eraser = fit_leace(Xn[tr], onehot_fit)

    # Validating that the erasure actually worked must stay inside the SAME
    # 30-episode universe the eraser was fit to distinguish -- checking it
    # against the other 10 (held-out) episodes would test a different, higher
    # -rank task (up to 39 separating directions for 40 classes vs the 29 the
    # eraser targets for 30), which a partial erasure could fail even after
    # perfectly solving the problem it was actually built for. So this splits
    # FRESH samples of the SAME 30 train episodes (a sub-split within `tr`)
    # and fits a brand-new classifier post-erasure on those, matching
    # positive_control.py's own "every class seen on both sides" design,
    # just restricted to the eraser's own label universe.
    rng_sub = np.random.default_rng(args.seed)
    tr_idx = np.where(tr)[0]
    sub_order = rng_sub.permutation(len(tr_idx))
    n_sub_val = int(round(len(tr_idx) * args.val_frac))
    sub_va, sub_tr = tr_idx[sub_order[:n_sub_val]], tr_idx[sub_order[n_sub_val:]]
    onehot_sub = (ep[:, None] == tr_eps[None, :]).astype(np.float64)  # full-length, indexed below

    def episode_id_accuracy(Xn_full):
        pred, _ = ridge_cv(
            Xn_full[sub_tr], onehot_sub[sub_tr], Xn_full[sub_va], None, args.seed, is_classification=True
        )
        return float((tr_eps[pred.argmax(1)] == ep[sub_va]).mean())

    chance = 1.0 / len(tr_eps)

    print("=== BEFORE erasure ===", flush=True)
    ep_acc_before = episode_id_accuracy(Xn)
    cur_l1_b, cur_const_b, cur_gain_b, _ = ridge_cv(Xn[tr], current[tr], Xn[va], current[va], args.seed)
    fut_l1_b, fut_const_b, fut_gain_b, _ = ridge_cv(Xn[tr], future[tr], Xn[va], future[va], args.seed)
    print(f"episode-identity accuracy (fresh sub-split, same 30 episodes): {ep_acc_before:.4f}  (chance {chance:.4f})")
    print(f"current-pose  val L1: {cur_l1_b:.5f}  const: {cur_const_b:.5f}  gain: {cur_gain_b:+.1f}%")
    print(f"future-pose   val L1: {fut_l1_b:.5f}  const: {fut_const_b:.5f}  gain: {fut_gain_b:+.1f}%")

    Xn_tr_erased = eraser.erase(Xn[tr])
    Xn_va_erased = eraser.erase(Xn[va])  # held-out, never seen by fit_leace
    Xn_all_erased = eraser.erase(Xn)  # for the same-universe episode-id check
    distortion = float(np.linalg.norm(Xn_va_erased - Xn[va]) / np.linalg.norm(Xn[va]))
    print(f"\nrelative distortion on held-out features: {distortion:.4f}", flush=True)

    print("\n=== AFTER erasure ===", flush=True)
    ep_acc_after = episode_id_accuracy(Xn_all_erased)
    cur_l1_a, cur_const_a, cur_gain_a, _ = ridge_cv(Xn_tr_erased, current[tr], Xn_va_erased, current[va], args.seed)
    fut_l1_a, fut_const_a, fut_gain_a, _ = ridge_cv(Xn_tr_erased, future[tr], Xn_va_erased, future[va], args.seed)
    print(f"episode-identity accuracy (fresh sub-split, same 30 episodes): {ep_acc_after:.4f}  (chance {chance:.4f})")
    print(f"current-pose  val L1: {cur_l1_a:.5f}  const: {cur_const_a:.5f}  gain: {cur_gain_a:+.1f}%")
    print(f"future-pose   val L1: {fut_l1_a:.5f}  const: {fut_const_a:.5f}  gain: {fut_gain_a:+.1f}%")

    print("\n=== verdict ===", flush=True)
    erasure_worked = ep_acc_after < 3 * chance
    pose_recovered = cur_gain_a > 5.0
    print(f"erasure collapsed episode-identity to near chance: {erasure_worked}")
    print(f"current-pose became recoverable (beats constant by >5%): {pose_recovered}")
    if erasure_worked and pose_recovered:
        verdict = "POSE_WAS_MASKED_BY_SCENE"
        print("VERDICT: pose information was present, masked by scene dominance.")
    elif erasure_worked and not pose_recovered:
        verdict = "POSE_STILL_ABSENT"
        print("VERDICT: scene dominance removed, but pose is still not recoverable --")
        print("         extraction/pooling or sample count remains the binding constraint.")
    else:
        verdict = "ERASURE_DID_NOT_WORK"
        print("VERDICT: erasure did not collapse episode-identity as expected on this data --")
        print("         investigate before trusting the pose numbers above (leace.py's own")
        print("         self-test on synthetic data passes, so check this script's usage.")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                dict(
                    variant=variant,
                    n_train_episodes=len(tr_eps),
                    n_val_episodes=len(va_eps),
                    distortion=distortion,
                    episode_acc_before=ep_acc_before,
                    episode_acc_after=ep_acc_after,
                    current_gain_before=cur_gain_b,
                    current_gain_after=cur_gain_a,
                    future_gain_before=fut_gain_b,
                    future_gain_after=fut_gain_a,
                    verdict=verdict,
                ),
                f,
                indent=2,
            )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
