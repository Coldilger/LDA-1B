"""Test that seeded_episodes.py makes RoboCasa episode scenes reproducible.

Builds the eval env twice in one process (as two separate eval runs would),
resets each twice, and compares the initial MuJoCo state (qpos) of every
episode: with the patch, episode k must match across the two builds and
episodes 0 and 1 must differ; without it (EPISODE_SEED_BASE unset) the two
builds' episode 0 should differ (the current, unpaired behaviour).
Run in the `robocasa` env with PYTHONPATH as in the eval slurm scripts.
"""

import os
import sys

import gymnasium as gym
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import robocasa.utils.gym_utils.gymnasium_groot  # noqa: F401,E402  (registers gr1_unified/* envs)
import seeded_episodes  # noqa: E402

ENV = "gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env"


def initial_states(n_episodes=2):
    env = gym.make(ENV, enable_render=True)
    out = []
    for _ in range(n_episodes):
        env.reset()
        out.append(env.unwrapped.env.sim.data.qpos.copy())
    env.close()
    return out


def same(x, y) -> bool:
    # different object sets give qpos of different length: that means "different"
    return x.shape == y.shape and np.allclose(x, y)


def gap(x, y) -> str:
    return f"{np.abs(x - y).max():.3e}" if x.shape == y.shape else f"shapes {x.shape} vs {y.shape}"


def main():
    mode = "patched" if seeded_episodes.install() is not None else "unpatched"
    a = initial_states()
    b = initial_states()
    same0 = same(a[0], b[0])
    same1 = same(a[1], b[1])
    diff01 = not same(a[0], a[1])
    print(f"[{mode}] episode 0 identical across builds: {same0} ({gap(a[0], b[0])})")
    print(f"[{mode}] episode 1 identical across builds: {same1} ({gap(a[1], b[1])})")
    print(f"[{mode}] episodes 0 and 1 differ: {diff01}")
    ok = (same0 and same1 and diff01) if mode == "patched" else (not same0)
    print(f"SEED_TEST_{mode.upper()}_{'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
