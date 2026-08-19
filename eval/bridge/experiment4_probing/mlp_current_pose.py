#!/usr/bin/env python3
"""
Experiment 4, testing the "nonlinear, not absent" hypothesis on the
decisive current-pose control.

compare_targets.py only ever ran a *linear* (ridge) probe on current pose,
which came back worse than a constant predictor for all three models
(F1-VLA, mimic-video, LDA-1B) despite current pose being trivially present
in the input frame. train_probe.py already trains an MLP probe alongside
ridge, but only for the future-pose target -- current pose was never given
the same nonlinear-capacity shot.

If current pose *is* linearly inaccessible but nonlinearly present, an MLP
(same architecture train_probe.py already uses for the future target)
should recover it far better than ridge did. If the MLP also fails to beat
a constant predictor, linearity isn't the explanation -- something more is
wrong with the extraction/pooling itself (matching the reasoning
compare_targets.py's own verdict already gestures at).

Generic over any {current_pose, episode_ids} + {features, episode_ids}
pair, run against LDA-1B, F1-VLA and mimic-video's own already-extracted
feature files.
"""

import argparse

import numpy as np
import torch
from torch import nn


class ProbeHead(nn.Module):
    """Identical architecture to train_probe.py's -- same capacity for all
    three models and for both the future-pose (already tested) and
    current-pose (this script) targets, so results stay comparable."""

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
    ap.add_argument("--current-pose", default=None, help="defaults to --features if that file already has current_pose")
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    feat_data = np.load(args.features, allow_pickle=True)
    X = feat_data["features"].astype(np.float32)
    ep = feat_data["episode_ids"]
    variant = str(feat_data["variant"]) if "variant" in feat_data else args.features

    cp_path = args.current_pose or args.features
    cp_data = np.load(cp_path, allow_pickle=True)
    Y = cp_data["current_pose"].astype(np.float32)
    if not np.array_equal(cp_data["episode_ids"], ep):
        raise SystemExit(f"episode order mismatch between {args.features} and {cp_path}")

    tr, va, n_tr_eps, n_va_eps = split_by_episode(ep, args.val_frac, args.seed)
    print(f"variant={variant}  features={X.shape}  current_pose={Y.shape}")
    print(f"split: {tr.sum()} train ({n_tr_eps} eps) / {va.sum()} val ({n_va_eps} eps)")

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
            val_l1 = (model(Xva) - Yva).abs().mean().item()
        if val_l1 < best_val:
            best_val = val_l1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 50 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:4d}  train_mse {loss.item():.6f}  val_l1 {val_l1:.6f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mlp_l1 = (model(Xva) - Yva).abs().mean().item()

    mean_pred = Y[tr].mean(0, keepdims=True)
    const_l1 = float(np.abs(Y[va] - mean_pred).mean())
    gain = 100.0 * (const_l1 - mlp_l1) / const_l1 if const_l1 > 0 else float("nan")

    print()
    print(f"MLP current-pose val L1: {mlp_l1:.5f}")
    print(f"const   current-pose val L1: {const_l1:.5f}")
    print(f"MLP gain over constant: {gain:.1f}%")
    if gain > 15.0:
        print("VERDICT: MLP recovers current pose where ridge failed -- the")
        print("information is present but not LINEARLY accessible.")
    elif gain > -20.0:
        print("VERDICT: MLP is roughly at the constant-predictor level -- still")
        print("not clearly recovering current pose, nonlinearity alone doesn't")
        print("explain ridge's failure.")
    else:
        print("VERDICT: MLP is worse than ridge already was -- with d >> n this")
        print("is consistent with overfitting rather than genuine recovery (see")
        print("train_probe.py's own note on why ridge, not MLP, is the primary")
        print("probe for the future-pose target).")


if __name__ == "__main__":
    main()
