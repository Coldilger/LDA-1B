#!/usr/bin/env python3
"""Compares qualitative closed-loop episode behavior between two existing
eval logs (already-run SimplerEnv jobs, no new rollout needed) by parsing
each episode's final per-step `episode_stats` dict.

Written to answer a specific question before committing to a mean_std
retrain: does v3's real (if clipped) proprioception produce ANY
qualitative closed-loop improvement over v2's state_dim: null (no
proprioception at all)? Both score 0% task success, so the raw success
rate alone can't distinguish "completely broken" from "partially working,
fails to finish" -- the per-episode interaction stats can. See
RESULTS.md's "Does giving the model real state help at all, even
qualitatively?" section for the result and interpretation.

Usage: edit LOGS below to point at whichever two (or more) eval-*.out logs
you want to compare -- these are plain stdout captures from
main_inference.py's maniskill2_evaluator print loop, not a stored artifact
with a stable schema, so this reads them as text rather than assuming a
particular file format.
"""

import re

LOGS = [
    ("v2 (no state)", "slurm/logs/eval-626928.out"),
    ("v3 (state, clipped)", "slurm/logs/eval-628404.out"),
]


def parse_log(path):
    """Returns one dict per episode: the LAST per-step line's 'episode_stats'
    (cumulative over that episode), keyed on the step-index column resetting
    to 0 as the episode-boundary marker."""
    episodes = []
    cur_last = None
    with open(path, errors="replace") as f:
        for line in f:
            m = re.match(r"^(\d+) (\{.*\})$", line.strip())
            if not m:
                continue
            step_idx = int(m.group(1))
            try:
                # The printed dicts contain OrderedDict([...]) reprs, which
                # ast.literal_eval can't parse (it's not a literal) --
                # eval() with OrderedDict mapped to plain dict and no
                # builtins is safe here since this is our own trusted,
                # self-generated log output, not external input.
                d = eval(m.group(2), {"OrderedDict": dict, "__builtins__": {}})
            except Exception:
                continue
            if step_idx == 0 and cur_last is not None:
                episodes.append(cur_last)
            cur_last = d
    if cur_last is not None:
        episodes.append(cur_last)
    return episodes


def main():
    for name, path in LOGS:
        eps = parse_log(path)
        n = len(eps)
        if n == 0:
            print(f"{name}: NO EPISODES PARSED from {path}")
            continue
        moved = sum(1 for e in eps if e["episode_stats"]["moved_correct_obj"])
        grasped = sum(1 for e in eps if e["episode_stats"]["is_src_obj_grasped"])
        consec = sum(1 for e in eps if e["episode_stats"]["consecutive_grasp"])
        on_target = sum(1 for e in eps if e["episode_stats"]["src_on_target"])
        wrong = sum(1 for e in eps if e["episode_stats"]["moved_wrong_obj"])
        print(f"{name}: n={n} episodes ({path})")
        print(f"  moved_correct_obj (ever):   {moved}/{n}")
        print(f"  is_src_obj_grasped (final): {grasped}/{n}")
        print(f"  consecutive_grasp (ever):   {consec}/{n}")
        print(f"  src_on_target (success):    {on_target}/{n}")
        print(f"  moved_wrong_obj (ever):     {wrong}/{n}")
        print()


if __name__ == "__main__":
    main()
