#!/usr/bin/env python3
"""
Experiment 4, the decisive control: current pose vs. future pose delta.

Runs the identical ridge probe, identical standardization and identical
episode-level split on two targets that differ only in what is being asked:

  current  -- the end-effector pose at time t (definitely present in the input
              frame, and spatially localized)
  future   -- the pose delta at t+K (the actual experiment)

Both are localized quantities, so mean+std pooling over 19200 tokens treats
them alike. That is what makes the pair informative, where the earlier
episode-identity control was not: episode identity is a *global* scene
property and survives pooling trivially (it came back at 100%, 40x chance),
so passing it did not rule out "pooling removed the localized signal".

Reading the outcome:

  current recovered, future not  -> pooling preserves localized information;
                                    the flat future result is a property of
                                    the representation, and Experiment 4 has
                                    a real finding.
  neither recovered              -> pooling (or n) is the binding constraint;
                                    no conclusion about future information is
                                    available until extraction changes.

Scores are reported as improvement over the best constant predictor fitted on
the train split, because the two targets have very different scales and raw
L1 cannot be compared across them.
"""

import argparse

import numpy as np


def ridge_fit_predict(Xtr, Ytr, Xva, alpha: float):
    n = Xtr.shape[0]
    K = Xtr @ Xtr.T
    dual = np.linalg.solve(K + alpha * np.eye(n), Ytr)
    return Xva @ (Xtr.T @ dual)


def split_by_episode(episode_ids, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(episode_ids)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(uniq[:n_val].tolist())
    is_val = np.array([e in val_eps for e in episode_ids])
    return ~is_val, is_val


def probe(Xn, Y, tr, va, seed, n_folds=5):
    """Ridge with alpha cross-validated inside the train split."""
    rng = np.random.default_rng(seed)
    idx = np.where(tr)[0]
    order = rng.permutation(len(idx))
    folds = np.array_split(order, n_folds)

    best_alpha, best_err = None, np.inf
    for alpha in np.logspace(-1, 6, 22):
        errs = []
        for i in range(n_folds):
            vi = idx[folds[i]]
            ti = idx[np.concatenate([folds[j] for j in range(n_folds) if j != i])]
            pred = ridge_fit_predict(Xn[ti], Y[ti], Xn[vi], alpha)
            errs.append(np.abs(pred - Y[vi]).mean())
        e = float(np.mean(errs))
        if e < best_err:
            best_alpha, best_err = float(alpha), e

    pred = ridge_fit_predict(Xn[tr], Y[tr], Xn[va], best_alpha)
    probe_l1 = float(np.abs(pred - Y[va]).mean())
    const_l1 = float(np.abs(Y[va] - Y[tr].mean(0, keepdims=True)).mean())
    gain = 100.0 * (const_l1 - probe_l1) / const_l1 if const_l1 > 0 else float("nan")
    return probe_l1, const_l1, gain, best_alpha


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--current-pose", default="current_pose.npz")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
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

    tr, va = split_by_episode(ep, args.val_frac, args.seed)
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    print(f"variant={variant}  n={len(X)}  train={tr.sum()} val={va.sum()}"
          f"  (episode-level split)\n")

    hdr = f"{'target':<10}{'probe L1':>12}{'const L1':>12}{'gain %':>10}{'alpha':>10}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name, Y in (("current", current), ("future", future)):
        p, c, g, a = probe(Xn, Y, tr, va, args.seed)
        results[name] = g
        print(f"{name:<10}{p:>12.5f}{c:>12.5f}{g:>10.1f}{a:>10g}")

    print()
    if results["current"] > 15.0 and results["future"] < 5.0:
        print("VERDICT: current pose is recoverable, future is not -> pooling keeps")
        print("localized information, so the flat future result is a real property")
        print("of the representation.")
    elif results["current"] < 5.0:
        print("VERDICT: even current pose is not recoverable -> the pooled features")
        print("(or the sample count) are the binding constraint. No conclusion about")
        print("future information can be drawn until extraction changes.")
    else:
        print("VERDICT: intermediate -- neither reading is clean; see the numbers.")


if __name__ == "__main__":
    main()
