"""Experiment 12 closed loop: LDA-1B Bridge v4 checkpoint with the corrected
wrapper (LDA_BRIDGE_FRAME=1, tag suffix _e12fix) vs the old one (=0, _e12old).

Reads the SimplerEnv logs of the jobs listed in jobs.txt and counts, per
wrapper and task, the episodes whose final episode_stats show each partial
progress flag (SimplerEnv keeps these flags as "ever happened in the episode").

    python analyze_closed_loop.py   # prints a table, writes closed_loop_summary.json
"""
from __future__ import annotations

import collections
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
LOGS = HERE.parents[2] / "slurm" / "logs"
FLAGS = ["moved_correct_obj", "moved_wrong_obj", "is_src_obj_grasped", "consecutive_grasp", "src_on_target"]


def job_ids() -> list[str]:
    ids = []
    for line in (HERE / "jobs.txt").read_text().splitlines():
        if line.startswith("lda12_"):
            ids += line.split(":", 1)[1].split()
    return ids


def main() -> None:
    agg: dict = collections.defaultdict(collections.Counter)
    for j in job_ids():
        text = (LOGS / f"eval-{j}.out").read_text()
        mode = "fix" if "[lda] bridge_frame=True" in text else "old"
        assert re.search(r"EVAL_EXIT=0", text), j
        task = re.search(r"EVAL_EXIT=0 TASK=(\S+)-v0", text).group(1)
        steps = re.findall(r"^(\d+) (\{'elapsed_steps'.*)$", text, re.M)
        finals = [d for i, (_, d) in enumerate(steps) if i + 1 == len(steps) or steps[i + 1][0] == "0"]
        c = agg[(mode, task)]
        c["episodes"] += len(finals)
        c["success"] += sum("'success': True" in d for d in finals)
        for k in FLAGS:
            c[k] += sum(f"('{k}', True)" in d for d in finals)
    out = {f"{m}/{t}": dict(c) for (m, t), c in sorted(agg.items())}
    for m in ("fix", "old"):
        tot = collections.Counter()
        for (mm, _), c in agg.items():
            if mm == m:
                tot.update(c)
        out[f"{m}/all"] = dict(tot)
    cols = ["episodes", "success"] + FLAGS
    print("| wrapper / task | " + " | ".join(cols) + " |\n|" + "---|" * (len(cols) + 1))
    for k, c in out.items():
        print(f"| {k} | " + " | ".join(str(c.get(x, 0)) for x in cols) + " |")
    (HERE / "closed_loop_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
