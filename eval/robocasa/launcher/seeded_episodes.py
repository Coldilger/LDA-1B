"""Deterministic per-episode scene seeding for RoboCasa evals (thesis paired
protocol, 2026-10-29).

Why: the authors' client (examples/Robocasa_tabletop/eval_files/
simulation_env.py) defines a `seed` argument but never uses it, and
RoboCasaEnv.reset (robocasa/utils/gym_utils/gymnasium_basic.py) calls
np.random.seed(None). The scene -- layout, objects, placements, textures --
comes from the robosuite env's own generator `env.rng`, created once per
process with seed=None (robosuite/environments/base.py). So every eval run
draws a different random set of scenes, and two conditions (baseline vs
world-model-on) cannot be compared episode by episode.

What: patch RoboCasaEnv.reset so that episode k (k-th reset in this
process) starts from seed BASE + k: `env.rng` is re-seeded IN PLACE (its
bit-generator state is replaced, so every object already holding a
reference to it -- placement samplers etc. -- sees the new stream), plus
numpy's and Python's global RNGs. Episode k then gets the same scene in
every run and condition, whatever the policy did in earlier episodes.
Enabled only when EPISODE_SEED_BASE is set; otherwise nothing changes.

Also records each finished episode's outcome (success at any step, same rule
as the client's own `current_successes |= success`) as one line
"<k> <seed> <0|1>" in EPISODE_LOG, if set: the authors' client prints only
the aggregate success rate, and paired statistics need per-episode outcomes.
An episode is logged when the next reset starts, so the extra reset the
client does after its last episode logs that episode, and the unfinished
episode it starts is never logged.
"""

from __future__ import annotations

import os
import random

import numpy as np


def install() -> int | None:
    base = os.environ.get("EPISODE_SEED_BASE")
    if base is None:
        return None
    base = int(base)
    from robocasa.utils.gym_utils.gymnasium_basic import RoboCasaEnv

    orig_reset = RoboCasaEnv.reset
    orig_step = RoboCasaEnv.step
    log_path = os.environ.get("EPISODE_LOG")

    def step(self, action_dict):
        out = orig_step(self, action_dict)
        self._seeded_success = getattr(self, "_seeded_success", False) or bool(out[-1].get("success", False))
        return out

    def reset(self, seed=None, options=None):
        k = getattr(self, "_seeded_episode_k", 0)
        if log_path and k > 0:
            with open(log_path, "a") as f:
                f.write(f"{k - 1} {base + k - 1} {int(getattr(self, '_seeded_success', False))}\n")
        self._seeded_success = False
        self._seeded_episode_k = k + 1
        s = base + k
        rng = self.env.rng
        rng.bit_generator.state = np.random.default_rng(s).bit_generator.state
        random.seed(s)
        print(f"[seeded_episodes] reset #{k} -> seed {s}", flush=True)
        return orig_reset(self, seed=s, options=options)  # orig also does np.random.seed(s)

    RoboCasaEnv.reset = reset
    RoboCasaEnv.step = step
    return base
