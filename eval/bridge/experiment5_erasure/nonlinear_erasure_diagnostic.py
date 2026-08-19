#!/usr/bin/env python3
"""
Experiment 5, LDA-1B-specific extension: is the pose signal MLP recovers
genuinely nonlinear, or a nonlinear readout of a linear signal LEACE can
strip out?

LEACE is a LINEAR erasure tool -- it can only remove a linear subspace of
the features. Experiment 4 found LDA's current/future pose recoverable by
an MLP (+60.2%/+64.9%) but NOT by ridge (-229%/+40.6% -- current fails
outright, future barely beats mean). This raises a real question LEACE can
answer directly: fit an eraser that removes exactly the LINEAR direction(s)
correlated with current pose (LEACE natively supports continuous vector
targets, not just one-hot categorical ones -- see leace.py), then check
whether the MLP can *still* recover pose from what's left.

Two readable outcomes:
  - MLP recoverability collapses after erasing the linear pose direction ->
    the "nonlinear" finding was largely a nonlinear readout of what was,
    in fact, a linear signal all along (ridge's failure would then be more
    about regularization/optimization than a real absence of a linear
    direction) -- a real correction to Experiment 4's reading.
  - MLP recoverability survives (mostly unchanged) -> the pose information
    lives through a genuinely nonlinear pathway, independent of the best
    single linear predictor of pose -- a clean, decisive positive result,
    not just "ridge under-performed."

Also runs the complementary check from scene_erasure_diagnostic.py's own
Level A (erase episode identity), but with the MLP probe added alongside
ridge -- scene_erasure_diagnostic.py only checked ridge, and Experiment 4's
whole point for LDA is that ridge is the wrong instrument here.

Erasure fit ONLY on the train split, applied unchanged to held-out val --
same discipline as leace.py's own docstring and scene_erasure_diagnostic.py.
"""

import argparse
import json

import numpy as np
import torch
from torch import nn

from leace import fit_leace


class ProbeHead(nn.Module):
    """Identical to experiment4_probing/train_probe.py's -- same capacity,
    comparable numbers before and after erasure."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def ridge_fit_predict(Xtr, Ytr, Xva, alpha):
    n = Xtr.shape[0]
    K = Xtr @ Xtr.T
    dual = np.linalg.solve(K + alpha * np.eye(n), Ytr)
    return Xva @ (Xtr.T @ dual)


def ridge_gain(Xtr, Ytr, Xva, Yva, seed, n_folds=5):
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(Xtr))
    folds = np.array_split(order, n_folds)
    best_alpha, best_err = None, np.inf
    for alpha in np.logspace(-1, 6, 22):
        errs = []
        for i in range(n_folds):
            vi, ti = folds[i], np.concatenate([folds[j] for j in range(n_folds) if j != i])
            pred = ridge_fit_predict(Xtr[ti], Ytr[ti], Xtr[vi], alpha)
            errs.append(np.abs(pred - Ytr[vi]).mean())
        e = float(np.mean(errs))
        if e < best_err:
            best_alpha, best_err = float(alpha), e
    pred = ridge_fit_predict(Xtr, Ytr, Xva, best_alpha)
    l1 = float(np.abs(pred - Yva).mean())
    const_l1 = float(np.abs(Yva - Ytr.mean(0, keepdims=True)).mean())
    gain = 100.0 * (const_l1 - l1) / const_l1 if const_l1 > 0 else float("nan")
    return l1, const_l1, gain


def mlp_gain(Xtr, Ytr, Xva, Yva, seed, epochs=300, hidden=256, lr=1e-3, wd=1e-4):
    torch.manual_seed(seed)
    Xtr_t, Ytr_t = torch.from_numpy(Xtr).float(), torch.from_numpy(Ytr).float()
    Xva_t, Yva_t = torch.from_numpy(Xva).float(), torch.from_numpy(Yva).float()
    model = ProbeHead(Xtr.shape[1], Ytr.shape[1], hidden=hidden)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(Xtr_t), Ytr_t)
        loss.backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            val_l1 = (model(Xva_t) - Yva_t).abs().mean().item()
        if val_l1 < best_val:
            best_val, best_state = val_l1, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        l1 = (model(Xva_t) - Yva_t).abs().mean().item()
    const_l1 = float(np.abs(Yva - Ytr.mean(0, keepdims=True)).mean())
    gain = 100.0 * (const_l1 - l1) / const_l1 if const_l1 > 0 else float("nan")
    return l1, const_l1, gain


def split_by_episode(episode_ids, val_frac, seed):
    rng = np.random.default_rng(seed)
    uniq = np.unique(episode_ids)
    rng.shuffle(uniq)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_eps = set(uniq[:n_val].tolist())
    is_val = np.array([e in val_eps for e in episode_ids])
    return ~is_val, is_val


def report(name, l1, const_l1, gain):
    print(f"  {name:<10} L1={l1:.5f}  const={const_l1:.5f}  gain={gain:+.1f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", required=True)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    data = np.load(args.features, allow_pickle=True)
    X = data["features"].astype(np.float64)
    future = data["targets"].astype(np.float64)
    current = data["current_pose"].astype(np.float64)
    ep = data["episode_ids"]
    variant = str(data["variant"])

    tr, va = split_by_episode(ep, args.val_frac, args.seed)
    print(f"variant={variant}  n={len(X)}  train={tr.sum()} val={va.sum()}\n")

    mu, sd = X[tr].mean(0, keepdims=True), X[tr].std(0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd

    results = {}

    print("=== BASELINE (no erasure) ===")
    for name, Y in (("current", current), ("future", future)):
        rl1, rc, rg = ridge_gain(Xn[tr], Y[tr], Xn[va], Y[va], args.seed)
        ml1, mc, mg = mlp_gain(Xn[tr].astype(np.float32), Y[tr].astype(np.float32),
                                Xn[va].astype(np.float32), Y[va].astype(np.float32), args.seed)
        report(f"{name} ridge", rl1, rc, rg)
        report(f"{name} MLP", ml1, mc, mg)
        results[f"{name}_baseline"] = dict(ridge_gain=rg, mlp_gain=mg)

    print("\n=== ERASE EPISODE IDENTITY (LEACE, one-hot, Level A's own target) ===")
    tr_eps = np.unique(ep[tr])
    onehot = (ep[tr][:, None] == tr_eps[None, :]).astype(np.float64)
    eraser_ep = fit_leace(Xn[tr], onehot)
    Xtr_e1, Xva_e1 = eraser_ep.erase(Xn[tr]), eraser_ep.erase(Xn[va])
    for name, Y in (("current", current), ("future", future)):
        rl1, rc, rg = ridge_gain(Xtr_e1, Y[tr], Xva_e1, Y[va], args.seed)
        ml1, mc, mg = mlp_gain(Xtr_e1.astype(np.float32), Y[tr].astype(np.float32),
                                Xva_e1.astype(np.float32), Y[va].astype(np.float32), args.seed)
        report(f"{name} ridge", rl1, rc, rg)
        report(f"{name} MLP", ml1, mc, mg)
        results[f"{name}_after_episode_erasure"] = dict(ridge_gain=rg, mlp_gain=mg)

    print("\n=== ERASE CURRENT-POSE'S LINEAR COMPONENT (LEACE, continuous 116-D target) ===")
    eraser_pose = fit_leace(Xn[tr], current[tr])
    Xtr_e2, Xva_e2 = eraser_pose.erase(Xn[tr]), eraser_pose.erase(Xn[va])
    # Sanity check: ridge current-pose recoverability should now be at/below
    # its already-bad baseline (confirms the eraser targeted the right thing,
    # even though there wasn't much linear signal to remove in the first
    # place -- see the printed baseline ridge gain above).
    rl1_check, rc_check, rg_check = ridge_gain(Xtr_e2, current[tr], Xva_e2, current[va], args.seed)
    print(f"  [sanity] current ridge after pose-erasure: gain={rg_check:+.1f}% "
          f"(baseline was {results['current_baseline']['ridge_gain']:+.1f}%)")
    for name, Y in (("current", current), ("future", future)):
        ml1, mc, mg = mlp_gain(Xtr_e2.astype(np.float32), Y[tr].astype(np.float32),
                                Xva_e2.astype(np.float32), Y[va].astype(np.float32), args.seed)
        report(f"{name} MLP", ml1, mc, mg)
        results[f"{name}_after_pose_erasure"] = dict(mlp_gain=mg)

    print("\n=== VERDICT ===")
    mlp_before = results["current_baseline"]["mlp_gain"]
    mlp_after_pose_erasure = results["current_after_pose_erasure"]["mlp_gain"]
    drop = mlp_before - mlp_after_pose_erasure
    print(f"current-pose MLP gain: {mlp_before:+.1f}% before -> {mlp_after_pose_erasure:+.1f}% "
          f"after erasing pose's own linear component (drop: {drop:.1f}pp)")
    if mlp_after_pose_erasure < 15.0:
        verdict = "LINEAR_COMPONENT_WAS_SUFFICIENT"
        print("VERDICT: MLP recoverability collapses once the linear pose direction is")
        print("removed -- the earlier nonlinear finding was substantially a nonlinear")
        print("readout of an underlying linear signal, not independent of one.")
    else:
        verdict = "GENUINELY_NONLINEAR"
        print("VERDICT: MLP still recovers pose after removing its best linear")
        print("predictor -- the signal lives through a genuinely nonlinear pathway,")
        print("not just a linear direction ridge failed to fit well.")
    results["verdict"] = verdict

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
