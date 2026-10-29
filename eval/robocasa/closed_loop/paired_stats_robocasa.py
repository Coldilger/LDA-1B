"""Paired comparison of two RoboCasa closed-loop conditions (thesis protocol,
Section 4.4), on the per-episode logs written by launcher/seeded_episodes.py:
episodes/<condition>_<seed base>.txt, lines "<k> <scene seed> <success>".
Episodes are paired by scene seed (same scene in both conditions). Same
statistics as F1-VLA's / mimic-video's eval/bridge/paired_stats.py: Wilson
CIs, paired bootstrap stratified by seed base, exact McNemar.

    python paired_stats_robocasa.py baseline wm_on [--boot 10000]
"""

import argparse
import math
import pathlib
from collections import defaultdict

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def load(condition: str) -> dict[int, tuple[int, bool]]:
    out = {}
    for f in sorted((HERE / "episodes").glob(f"{condition}_*.txt")):
        base = int(f.stem.rsplit("_", 1)[1])
        for line in f.read_text().split("\n"):
            if line.strip():
                _, seed, ok = line.split()
                out[int(seed)] = (base, ok == "1")
    return out


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("condition")
    ap.add_argument("--boot", type=int, default=10000)
    a = ap.parse_args()
    base, cond = load(a.baseline), load(a.condition)
    seeds = sorted(set(base) & set(cond))
    x = np.array([base[s][1] for s in seeds], float)
    y = np.array([cond[s][1] for s in seeds], float)
    strata = np.array([base[s][0] for s in seeds])
    n = len(seeds)
    rng = np.random.default_rng(0)
    groups = [np.flatnonzero(strata == g) for g in np.unique(strata)]
    diffs = [y[i].mean() - x[i].mean() for i in
             (np.concatenate([rng.choice(g, len(g)) for g in groups]) for _ in range(a.boot))]
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    print(f"paired episodes: {n} (only in one condition: {len(set(base) ^ set(cond))})")
    per = defaultdict(lambda: [0, 0, 0])
    for s in seeds:
        g = base[s][0]
        per[g][0] += 1
        per[g][1] += base[s][1]
        per[g][2] += cond[s][1]
    print(f"| scene set | n | {a.baseline} | {a.condition} |\n|---|---|---|---|")
    for g, (m, k1, k2) in sorted(per.items()):
        print(f"| {g} | {m} | {k1}/{m} | {k2}/{m} |")
    for name, v in ((a.baseline, x), (a.condition, y)):
        lo, hi = wilson(int(v.sum()), n)
        print(f"{name}: {int(v.sum())}/{n} = {100 * v.mean():.1f}% [Wilson 95%: {100 * lo:.1f}, {100 * hi:.1f}]")
    print(f"paired difference: {100 * (y.mean() - x.mean()):+.1f} pp "
          f"[stratified bootstrap 95%: {100 * np.percentile(diffs, 2.5):+.1f}, {100 * np.percentile(diffs, 97.5):+.1f}]")
    print(f"discordant pairs: {a.baseline}-only {b}, {a.condition}-only {c}; exact McNemar p = {mcnemar_exact(b, c):.4f}")


if __name__ == "__main__":
    main()
