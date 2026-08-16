#!/usr/bin/env python
"""Direct comparison: SimplerEnv's raw ee_pose_at_base range vs LDA's own
q99-normalization training window for state.left_eef_position.

Modeled on F1-VLA/eval/bridge/diag_state.py's methodology (idle actions,
read env.unwrapped.agent.controllers[...].ee_pose_at_base directly) but
compared against LDA's ACTUAL q01/q99 statistics (meta/stats_gr00t.json in
the training dataset), not eyeballed against a hardcoded docstring range --
LDA's normalization hard-clips outside [q01, q99] (torch.clamp in
transform/state_action.py:134), collapsing every out-of-range value to the
identical -1.0 or +1.0, unlike F1's softer mean/std scheme. So what matters
here isn't whether the ranges "roughly overlap" but whether SimplerEnv's
actual reachable positions fall inside LDA's specific window.
"""
import json

import gymnasium as gym
import mani_skill2_real2sim.envs  # noqa: F401
import numpy as np
from scipy.spatial.transform import Rotation

CONTROL_MODE = "arm_pd_ee_target_delta_pose_align2_gripper_pd_joint_pos"

with open("/mnt/beegfsnew/scratch/3295540/data/bridge_lda/meta/stats_gr00t.json") as f:
    stats = json.load(f)
q01 = np.array(stats["state.left_eef_position"]["q01"])
q99 = np.array(stats["state.left_eef_position"]["q99"])
train_mean = np.array(stats["state.left_eef_position"]["mean"])

env = gym.make(
    "PutCarrotOnPlateInScene-v0",
    obs_mode="rgbd",
    robot="widowx",
    sim_freq=500,
    control_freq=5,
    max_episode_steps=60,
    scene_name="bridge_table_1_v1",
    control_mode=CONTROL_MODE,
)

rows = []
init_rows = []
# Match the real eval sweep's fixed robot init pose exactly (eval_bridge_simpler.slurm's
# INIT_X/INIT_Y for the Carrot/Spoon/Stack tasks) -- the env's own default reset
# pose is a different, uncontrolled starting point and isn't representative.
ROBOT_INIT_X, ROBOT_INIT_Y = 0.147, 0.028
for ep in range(6):
    obs, _ = env.reset(options={
        "robot_init_options": {
            "init_xy": np.array([ROBOT_INIT_X, ROBOT_INIT_Y]),
            "init_rot_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        },
        "obj_init_options": {"episode_id": ep},
    })
    ctrl = env.unwrapped.agent.controllers[CONTROL_MODE]
    pose0 = ctrl.controllers["arm"].ee_pose_at_base
    init_rows.append(np.asarray(pose0.p, dtype=np.float64).copy())
    for t in range(40):
        pose = ctrl.controllers["arm"].ee_pose_at_base
        rows.append(np.asarray(pose.p, dtype=np.float64).copy())
        # idle-ish small random action so the arm drifts around a bit, like a
        # real (if untrained) rollout would, rather than sitting frozen
        env.step(np.zeros(7, dtype=np.float32))
env.close()

P = np.array(rows)          # (N, 3) every step, all episodes
P0 = np.array(init_rows)    # (n_episodes, 3) episode-start pose only

names = ["x", "y", "z"]
np.set_printoptions(precision=4, suppress=True)

print("=== LDA training data window (state.left_eef_position) ===")
print(f"{'dim':4s} {'q01':>8s} {'q99':>8s} {'mean':>8s}")
for i, n in enumerate(names):
    print(f"{n:4s} {q01[i]:8.4f} {q99[i]:8.4f} {train_mean[i]:8.4f}")

print("\n=== SimplerEnv episode-START pose (ee_pose_at_base.p), 6 episodes ===")
print(f"{'dim':4s} {'min':>8s} {'max':>8s} {'mean':>8s}")
for i, n in enumerate(names):
    print(f"{n:4s} {P0[:,i].min():8.4f} {P0[:,i].max():8.4f} {P0[:,i].mean():8.4f}")

print("\n=== SimplerEnv pose over a 40-step idle rollout, 6 episodes (240 steps) ===")
print(f"{'dim':4s} {'min':>8s} {'max':>8s} {'mean':>8s}  frac(<q01 or >q99)")
for i, n in enumerate(names):
    col = P[:, i]
    frac_oob = float(((col < q01[i]) | (col > q99[i])).mean())
    print(f"{n:4s} {col.min():8.4f} {col.max():8.4f} {col.mean():8.4f}  {frac_oob:6.1%}")

print("\nrobot_init_x used by the real eval sweep (eval_bridge_simpler.slurm): 0.147")
print(f"LDA's own q01 for x: {q01[0]:.4f} -- init x {'IS' if 0.147 < q01[0] else 'is NOT'} below the training q01 bound")
print("DIAG_DONE")
