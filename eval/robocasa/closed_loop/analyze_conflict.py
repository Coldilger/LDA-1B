"""Experiment 8 (LDA-1B): summarise the conflict probe
(conflict_records/probe_*.jsonl, server_policy_conflict.py --mode probe) and
the closed-loop cells (episodes/cfl<text><forecast><alt>_<base>.txt, paired
by scene seed with the world-model-on policy wm_on_<base>.txt, which is the
text A / forecast A cell).

Probe: mean over decisions with a 95% bootstrap over decisions (decisions
within an episode are correlated, so the interval is optimistic). d = mean
over the 16-step chunk of the L2 distance between normalised action rows;
follow(AB) = d(AB,AA) / (d(AB,AA) + d(AB,BB)): 0 = the action follows the
text the head reads, 1 = it follows the forecast's instruction.

    python analyze_conflict.py [--alt can]
"""

import argparse
import collections
import json
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def boot(x, n=4000):
    x = np.asarray(x, float)
    m = x[np.random.default_rng(0).integers(0, len(x), (n, len(x)))].mean(1)
    return f"{x.mean():.4f} [{np.percentile(m, 2.5):.4f}, {np.percentile(m, 97.5):.4f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", default="can")
    ap.add_argument("--glob", default="probe_*.jsonl")
    a = ap.parse_args()
    recs = [json.loads(l) for f in sorted((HERE / "conflict_records").glob(a.glob)) for l in f.read_text().splitlines() if l.strip()]
    if recs:
        print(f"### Single decisions ({len(recs)} decisions)\n")
        print(f"noise floor d(AA2, AA) = {boot([r['d_AA2_AA'] for r in recs])}; forecast change between seeds = {boot([r['fc_seed_rel'] for r in recs])}\n")
        print("| B | forecast change A→B | d(BB, AA) | d(AB, AA) | d(BA, BB) | follow(AB) | follow(BA) |")
        print("|---|---|---|---|---|---|---|")
        by = collections.defaultdict(lambda: collections.defaultdict(list))
        for r in recs:
            for n, x in r["alts"].items():
                for k, v in x.items():
                    by[n][k].append(v)
        for n, x in by.items():
            print(f"| {n}: \"{x['instruction'][0]}\" | {boot(x['fc_rel_AB'])} | {boot(x['d_BB_AA'])} | {boot(x['d_AB_AA'])} "
                  f"| {boot(x['d_BA_BB'])} | {boot(x['follow_AB'])} | {boot(x['follow_BA'])} |")
    labels = [f"cfl{t}{f}{a.alt}" for t, f in (("A", "B"), ("B", "A"), ("B", "B"))]
    present = [l for l in labels if list((HERE / "episodes").glob(f"{l}_*.txt"))]
    if present:
        print(f"\n### Closed loop (B = {a.alt}); text A / forecast A = wm_on\n")
        for l in present:
            out = subprocess.run([sys.executable, str(HERE / "paired_stats_robocasa.py"), "wm_on", l], capture_output=True, text=True).stdout
            print(f"#### wm_on vs {l}\n```\n{out.strip()}\n```")
        if "cflBB" + a.alt in present and "cflBA" + a.alt in present:
            out = subprocess.run([sys.executable, str(HERE / "paired_stats_robocasa.py"), "cflBB" + a.alt, "cflBA" + a.alt], capture_output=True, text=True).stdout
            print(f"#### cflBB{a.alt} vs cflBA{a.alt}\n```\n{out.strip()}\n```")


if __name__ == "__main__":
    main()
