#!/usr/bin/env python3
"""
LEACE: closed-form, provably-minimal linear concept erasure.

Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman (NeurIPS 2023,
arXiv:2306.03819). Given features X (n, d) and a concept Z (n, k) -- one-hot
for a categorical concept, raw values for a continuous/vector one -- produces
a fixed affine map r(x) such that no linear predictor can recover Z from
r(X) above chance/the-mean, and among all maps with that property, r is the
one minimizing E[||r(x) - x||^2].

Implementation notes (deviations from a naive reading of the paper, and why):

- Whitening uses the REGULARIZED inverse-square-root of Cov(X), not a raw
  pseudo-inverse. With d in the thousands and n in the hundreds (our case:
  d=960-4096, n~300 train samples), Cov(X) has rank <= n-1 -- most of its
  spectrum is exactly zero, and the nonzero-but-small end is noise, not
  signal. Un-regularized whitening would divide by near-zero eigenvalues and
  blow up whatever lives in those directions. Ridge-regularized eigenvalues
  (lambda proportional to the mean eigenvalue) are used instead, and
  eigenvalues below a numerical floor are dropped entirely (treated as
  "no data support," left untouched by the eraser rather than amplified).
  Same shrinkage instinct as the ridge probe's own alpha selection elsewhere
  in this experiment -- this is not a new discipline, just the same one
  applied to a different linear-algebra step.

- fit() only ever sees TRAIN-split data; erase() is then applied to both
  train and held-out val features using that one fixed map. This matches the
  paper's own explicit warning about the "oracle" variant (O-LEACE): fitting
  a per-point erasure directly from that point's own true label can
  paradoxically leave MORE nonlinear information recoverable than a proper
  train-fit/held-out-apply eraser, defeating the point of the test. Same
  train/val discipline already used for feature standardization and ridge
  alpha selection in this experiment.

Self-test at the bottom: synthetic X with a KNOWN injected concept
direction, confirms erasure drives a linear regressor's R^2 on Z to ~0 while
leaving unrelated structure in X largely intact, and that fitting on one
split and applying to a held-out split still erases perfectly (no leakage
requires re-fitting per point).
"""

import numpy as np


class LeaceEraser:
    def __init__(self, mean_x, mean_z, W, W_inv, P):
        self.mean_x = mean_x  # (d,)
        self.mean_z = mean_z  # (k,)
        self.W = W  # (d, d) whitening map (regularized)
        self.W_inv = W_inv  # (d, d) un-whitening map
        self.P = P  # (d, d) orthogonal projector onto whitened range(Cov(X,Z))

    def erase(self, X: np.ndarray) -> np.ndarray:
        """X: (n, d) -> (n, d), same shape, concept-erased."""
        Xc = X - self.mean_x
        # remove the whitened-then-projected component, then un-whiten
        removed = (Xc @ self.W) @ self.P @ self.W_inv.T
        return X - removed


def fit_leace(X: np.ndarray, Z: np.ndarray, shrinkage: float = 1e-2, eig_floor_ratio: float = 1e-8) -> LeaceEraser:
    """Fit a LEACE eraser on TRAIN-split (X, Z) only.

    X: (n, d) features, Z: (n, k) concept (one-hot for categorical, raw
    values for continuous). Returns an eraser fit purely from this data --
    apply it to held-out data separately, never re-fit per point.
    """
    n = X.shape[0]
    mean_x = X.mean(0)
    mean_z = Z.mean(0)
    Xc = X - mean_x
    Zc = Z - mean_z

    cov_x = (Xc.T @ Xc) / n  # (d, d)
    cov_xz = (Xc.T @ Zc) / n  # (d, k)

    eigval, eigvec = np.linalg.eigh(cov_x)  # ascending order
    eigval = np.clip(eigval, 0.0, None)
    floor = eigval.max() * eig_floor_ratio
    keep = eigval > floor
    # ridge shrinkage on the retained spectrum, same instinct as the probe's
    # own alpha selection -- avoids dividing by near-zero-but-kept eigenvalues
    reg = shrinkage * eigval[keep].mean() if keep.any() else 0.0
    ev_reg = eigval[keep] + reg

    Uk = eigvec[:, keep]  # (d, r)
    W = Uk @ np.diag(ev_reg**-0.5) @ Uk.T  # (d, d), zero outside kept subspace
    W_inv = Uk @ np.diag(ev_reg**0.5) @ Uk.T

    cov_xz_whitened = W @ cov_xz  # (d, k)
    # orthogonal projector onto range(cov_xz_whitened) via SVD (rank <= k)
    U, S, _ = np.linalg.svd(cov_xz_whitened, full_matrices=False)
    s_floor = S.max() * 1e-10 if S.size else 0.0
    Ur = U[:, S > s_floor]
    P = Ur @ Ur.T  # (d, d)

    return LeaceEraser(mean_x, mean_z, W, W_inv, P)


def _ridge_fit_predict(Xtr, Ytr, Xva, alpha):
    """Same dual-form ridge as train_probe.py/compare_targets.py, so the
    self-test measures erasure with the exact same probe family used for the
    real diagnostic."""
    n = Xtr.shape[0]
    K = Xtr @ Xtr.T
    dual = np.linalg.solve(K + alpha * np.eye(n), Ytr)
    return Xva @ (Xtr.T @ dual)


def _self_test():
    rng = np.random.default_rng(0)
    n, d, k = 400, 200, 5

    concept_dir = rng.normal(size=(k, d))
    concept_dir /= np.linalg.norm(concept_dir, axis=1, keepdims=True)
    Z = rng.normal(size=(n, k))
    noise = rng.normal(size=(n, d)) * 0.5
    X = Z @ concept_dir + noise  # concept is linearly present by construction

    tr = np.arange(n) < 300
    va = ~tr

    # sanity: concept IS linearly recoverable before erasure
    alpha = 10.0
    pred_before = _ridge_fit_predict(X[tr], Z[tr], X[va], alpha)
    r2_before = 1 - ((pred_before - Z[va]) ** 2).sum() / ((Z[va] - Z[tr].mean(0)) ** 2).sum()
    assert r2_before > 0.5, f"self-test setup broken: concept not recoverable before erasure (R2={r2_before:.3f})"

    eraser = fit_leace(X[tr], Z[tr])
    X_tr_erased = eraser.erase(X[tr])
    X_va_erased = eraser.erase(X[va])  # held-out, never seen by fit_leace

    pred_after = _ridge_fit_predict(X_tr_erased, Z[tr], X_va_erased, alpha)
    r2_after = 1 - ((pred_after - Z[va]) ** 2).sum() / ((Z[va] - Z[tr].mean(0)) ** 2).sum()

    distortion = np.linalg.norm(X_va_erased - X[va]) / np.linalg.norm(X[va])

    print(f"R^2 before erasure: {r2_before:.4f}  (should be well above 0)")
    print(f"R^2 after erasure:  {r2_after:.4f}  (should be near 0, ideally <= 0)")
    print(f"relative distortion on held-out X: {distortion:.4f}")
    assert r2_after < 0.05, f"SELF-TEST FAILED: concept still recoverable after erasure (R2={r2_after:.3f})"
    print("SELF_TEST_PASSED")


if __name__ == "__main__":
    _self_test()
