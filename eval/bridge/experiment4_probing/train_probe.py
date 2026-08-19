#!/usr/bin/env python3
"""
Experiment 4 (mimic-video), step 2: fit a probe head on frozen features.

Reads an .npz produced by extract_features.py and trains a small MLP to
predict the delta end-effector pose K steps ahead from the frozen
`crossattn_emb` summary. Reports held-out error against two references, so a
number is only ever read as "the representation carries future information"
if it beats both:

  no-motion : predict an all-zero delta (the arm doesn't move)
  mean      : predict the training split's mean delta (the best constant)

A probe that fails to beat the mean predictor has learned nothing from the
representation, regardless of how small its absolute error looks -- Bridge
deltas are small in absolute terms, which is exactly what made the absolute
pose target uninformative in Experiment 1's offline probe.

Train/val split is **by episode**, never by sample: consecutive frames of one
episode are near-duplicates, so a random per-sample split would put
near-copies of training frames in validation and inflate the score.

Note on the memorization confound: the val episodes are unseen by the *probe*,
but the finetuned backbone saw all of Bridge during its own training (no
held-out split exists for it). That is why the headline comparison is
pretrained-vs-finetuned on identical data and identical probe settings, rather
than any single variant's absolute number -- see README.md.
"""

import argparse
import json

import numpy as np
import torch
from torch import nn


class ProbeHead(nn.Module):
    """Deliberately small: this measures what the representation already
    carries, not what a large head can compute from it. Same architecture is
    intended for all three models, so the numbers stay comparable."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def ridge_fit_predict(Xtr, Ytr, Xva, alpha: float):
    """Closed-form ridge in the dual (kernel) form.

    Features are far wider than the sample count here (4096 vs a few hundred),
    so solving the n-by-n system is both cheaper and better conditioned than
    the d-by-d one: w = X^T (X X^T + alpha I)^-1 Y.
    """
    n = Xtr.shape[0]
    K = Xtr @ Xtr.T
    dual = np.linalg.solve(K + alpha * np.eye(n), Ytr)
    return Xva @ (Xtr.T @ dual)


def ridge_probe(Xtr, Ytr, Xva, Yva, seed: int, n_folds: int = 5):
    """Linear probe with alpha chosen by cross-validation on the train split.

    This is the primary probe measurement, not the MLP. With d >> n, an MLP
    head has enough capacity to fit noise and its held-out error then reflects
    how badly it overfit rather than what the representation encodes -- the
    first run of this experiment produced exactly that, both variants scoring
    worse than a constant predictor. A regularized linear probe is the
    standard tool for the question actually being asked ("is future pose
    linearly recoverable from this representation"), and its capacity is
    controlled rather than assumed.
    """
    alphas = np.logspace(-1, 6, 22)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xtr))
    folds = np.array_split(order, n_folds)

    best_alpha, best_err = None, np.inf
    for alpha in alphas:
        errs = []
        for i in range(n_folds):
            va_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            pred = ridge_fit_predict(Xtr[tr_idx], Ytr[tr_idx], Xtr[va_idx], alpha)
            errs.append(np.abs(pred - Ytr[va_idx]).mean())
        err = float(np.mean(errs))
        if err < best_err:
            best_alpha, best_err = float(alpha), err

    pred = ridge_fit_predict(Xtr, Ytr, Xva, best_alpha)
    return float(np.abs(pred - Yva).mean()), best_alpha


def split_by_episode(episode_ids: np.ndarray, val_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    uniq = np.unique(episode_ids)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(uniq[:n_val].tolist())
    is_val = np.array([e in val_eps for e in episode_ids])
    return ~is_val, is_val, len(uniq) - n_val, n_val


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    data = np.load(args.features, allow_pickle=True)
    X = data["features"].astype(np.float32)
    Y = data["targets"].astype(np.float32)
    ep = data["episode_ids"]
    variant = str(data["variant"])
    horizon = int(data["horizon"])

    tr, va, n_tr_eps, n_va_eps = split_by_episode(ep, args.val_frac, args.seed)
    print(f"variant={variant} horizon={horizon}")
    print(f"features {X.shape}, targets {Y.shape}")
    print(f"split: {tr.sum()} train samples ({n_tr_eps} eps) / {va.sum()} val samples ({n_va_eps} eps)")

    # Standardize using train-split statistics only. Without this, a raw scale
    # difference between the two checkpoints' activations would show up as a
    # probe-accuracy difference that has nothing to do with information content.
    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    Xtr = torch.from_numpy(Xn[tr])
    Ytr = torch.from_numpy(Y[tr])
    Xva = torch.from_numpy(Xn[va])
    Yva = torch.from_numpy(Y[va])

    model = ProbeHead(X.shape[1], Y.shape[1], hidden=args.hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Xtr), Ytr)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xva)
            val_l1 = (val_pred - Yva).abs().mean().item()
        if val_l1 < best_val:
            best_val = val_l1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  train_mse {loss.item():.6f}  val_l1 {val_l1:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mlp_l1 = (model(Xva) - Yva).abs().mean().item()

    ridge_l1, best_alpha = ridge_probe(Xn[tr], Y[tr], Xn[va], Y[va], args.seed)

    nomotion_l1 = np.abs(Y[va]).mean()
    mean_pred = Y[tr].mean(0, keepdims=True)
    mean_l1 = np.abs(Y[va] - mean_pred).mean()

    # The ridge probe is the headline number; the MLP is reported alongside as
    # a capacity check, not as the measurement.
    probe_l1 = ridge_l1

    print()
    print(f"ridge probe val L1: {ridge_l1:.6f}   (alpha={best_alpha:g}, CV-selected)  <- primary")
    print(f"MLP probe   val L1: {mlp_l1:.6f}   (capacity check only)")
    print(f"mean pred   val L1: {mean_l1:.6f}   (best constant from train split)")
    print(f"no-motion   val L1: {nomotion_l1:.6f}   (predict zero delta)")
    beats = probe_l1 < mean_l1
    print(f"ridge probe beats mean predictor: {beats}")
    if mean_l1 > 0:
        print(f"relative improvement over mean predictor: {100 * (mean_l1 - probe_l1) / mean_l1:.1f}%")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(
                dict(
                    variant=variant,
                    horizon=horizon,
                    n_samples=int(X.shape[0]),
                    feature_dim=int(X.shape[1]),
                    n_train_episodes=int(n_tr_eps),
                    n_val_episodes=int(n_va_eps),
                    probe_val_l1=probe_l1,
                    ridge_val_l1=ridge_l1,
                    ridge_alpha=best_alpha,
                    mlp_val_l1=mlp_l1,
                    mean_pred_val_l1=float(mean_l1),
                    nomotion_val_l1=float(nomotion_l1),
                    beats_mean_predictor=bool(beats),
                ),
                f,
                indent=2,
            )
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
