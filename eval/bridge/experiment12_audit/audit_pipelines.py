"""Experiment 12 (advisor's point 9): audit of the Bridge data pipelines.

CPU only, reads the two local copies of BridgeData V2:
  bridge_orig_lerobot  (IPEC-COMMUNITY, LeRobot v2.0; F1-VLA trains on it)
  bridge_lda           (our conversion for LDA-1B, data_preprocessing/convert_bridge_to_lda.py)
mimic-video trains on the authors' own export of the raw Bridge data
(mimic-video/data_preprocessing/action/process_bridge.py), which is not on disk
here; it is covered in the README from its code.

Checks (a sample of episodes, seeded):
  A. episode counts and empty instructions (LDA's loader drops those)
  B. the LDA copy reproduces bridge_orig (state, next-state action, gripper)
  C. what bridge_orig's action columns are (position = state difference,
     rotation = difference of Euler angles)
  D. the simulator's orientation convention: SimplerEnv's own ground-truth
     replay script places the tool at euler2mat(state rpy) @ T_SIM (below), so
     R_sim = R_data @ T_SIM. What F1-VLA's and LDA-1B's eval wrappers make of the
     simulator pose (rotation relative to the episode's first pose) is compared
     with the state the models were trained on.
  E. LDA-1B's action decoding at eval: gripper-frame deltas integrated from the
     simulator pose (lda_policy.py, delta2abs) vs the true motion.
  F. how often the four SimplerEnv instructions' objects occur in training.

    python audit_pipelines.py [--n 300]   # prints markdown, writes audit_summary.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

HERE = pathlib.Path(__file__).resolve().parent
DATA = pathlib.Path("/mnt/beegfsnew/scratch/3295540/data")
ORIG, LDA = DATA / "bridge_orig_lerobot", DATA / "bridge_lda"
# SimplerEnv/simpler_env/utils/debug/widowx_test_dataset_inference_rollout_gt_traj_in_sim.py
T_SIM = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
CHUNK = 16  # LDA action horizon


def parquet(root: pathlib.Path, ep: int) -> pd.DataFrame:
    return pd.read_parquet(root / "data" / f"chunk-{ep // 1000:03d}" / f"episode_{ep:06d}.parquet")


def stack(df, col):
    return np.stack(df[col].values).astype(np.float64)


def gripper_frame_deltas(pos, rot):
    """LDA's calculate_delta_eef: dT_i = T_i^-1 T_{i+1} (lda/utils/rotation_convert.py)."""
    dp = np.einsum("nji,nj->ni", rot[:-1].as_matrix(), pos[1:] - pos[:-1])
    dr = rot[:-1].inv() * rot[1:]
    return dp, dr


def integrate(dp, dr, p0, r0):
    """LDA's delta2abs followed by lda_policy.py's differencing to base-frame deltas."""
    ps, rs = [p0], [r0]
    for d, q in zip(dp, dr):
        ps.append(ps[-1] + rs[-1].apply(d))
        rs.append(rs[-1] * q)
    ps = np.array(ps)
    return ps[1:] - ps[:-1], [rs[i + 1] * rs[i].inv() for i in range(len(dr))]


def angle_deg(a, b):
    na, nb = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
    ok = (na > 1e-3) & (nb > 1e-3)  # moves of at least 1 mm
    c = np.sum(a[ok] * b[ok], axis=1) / (na[ok] * nb[ok])
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = {}

    # A. episodes and instructions
    eps = [json.loads(l) for l in (ORIG / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()]
    empty = [e["episode_index"] for e in eps if not e["tasks"] or e["tasks"][0].strip() == ""]
    lda_eps = len([l for l in (LDA / "meta" / "episodes.jsonl").read_text().splitlines() if l.strip()])
    out["A"] = {"episodes": len(eps), "empty_instruction": len(empty), "kept_by_lda_loader": len(eps) - len(empty),
                "lda_copy_episodes_listed": lda_eps}
    print(f"## A. Episodes\n\n{len(eps)} episodes; {len(empty)} with an empty instruction "
          f"({100 * len(empty) / len(eps):.1f}%), dropped by LDA's loader -> {len(eps) - len(empty)} "
          f"(F1-VLA trains on all {len(eps)}).\n")

    rng = np.random.default_rng(args.seed)
    nonempty = np.array(sorted(set(range(len(eps))) - set(empty)))
    sample = rng.choice(nonempty, size=args.n, replace=False)

    B = {"state_pos": 0.0, "state_rot": 0.0, "state_grip": 0.0, "action_pos_next_state": 0.0, "action_grip": 0.0}
    C = {"pos_vs_state_diff": [], "rot_vs_euler_diff": [], "rot_vs_world_delta": []}
    D = {"rel_err": [], "train_state_rot": [], "ref_err_deg": []}
    E = {"angle_lda": [], "relerr_lda": [], "angle_fixed": [], "rot_err_lda_deg": [], "rot_err_fixed_deg": []}
    for ep in sample:
        a, b = parquet(ORIG, ep), parquet(LDA, ep)
        st, ac = stack(a, "observation.state"), stack(a, "action")
        n = len(st)
        if n < 3:
            continue
        # B
        B["state_pos"] = max(B["state_pos"], np.abs(stack(b, "state.left_eef_position") - st[:, :3]).max())
        B["state_rot"] = max(B["state_rot"], np.abs(stack(b, "state.left_eef_rotation") - st[:, 3:6]).max())
        B["state_grip"] = max(B["state_grip"], np.abs(stack(b, "state.left_gripper")[:, 0] - st[:, 7]).max())
        B["action_pos_next_state"] = max(B["action_pos_next_state"],
                                         np.abs(stack(b, "action.left_eef_position")[:-1] - st[1:, :3]).max())
        B["action_grip"] = max(B["action_grip"], np.abs(stack(b, "action.left_gripper")[:, 0] - ac[:, 6]).max())
        # C
        rot = R.from_euler("xyz", st[:, 3:6])
        C["pos_vs_state_diff"].append(np.abs(ac[:-1, :3] - (st[1:, :3] - st[:-1, :3])).max())
        C["rot_vs_euler_diff"].append(np.abs(ac[:-1, 3:6] - (st[1:, 3:6] - st[:-1, 3:6])).max())
        C["rot_vs_world_delta"].append(np.abs(ac[:-1, 3:6] - (rot[1:] * rot[:-1].inv()).as_euler("xyz")).max())
        # D: simulator pose and the wrappers' episode-relative rotation
        rot_sim = rot * R.from_matrix(T_SIM)
        rel = (rot_sim[0].inv() * rot_sim).as_euler("xyz")  # f1_vla_policy.py / lda_policy.py _build_state
        D["rel_err"].append(rel - st[:, 3:6])
        D["train_state_rot"].append(st[:, 3:6])
        D["ref_err_deg"].append(np.degrees((rot_sim[0].inv() * rot_sim * rot.inv()).magnitude()))
        # E: LDA decoding, one chunk from every step
        pos = st[:, :3]
        dp, dr = gripper_frame_deltas(pos, rot)
        true_dp = pos[1:] - pos[:-1]
        true_dr = [rot[i + 1] * rot[i].inv() for i in range(n - 1)]
        for t in range(0, n - 1, CHUNK):
            s = slice(t, min(t + CHUNK, n - 1))
            for key, r0 in (("lda", rot_sim[t]), ("fixed", rot_sim[t] * R.from_matrix(T_SIM.T))):
                got_dp, got_dr = integrate(dp[s], dr[s], pos[t], r0)
                E[f"angle_{key}"].extend(angle_deg(got_dp, true_dp[s]))
                if key == "lda":
                    nt = np.linalg.norm(true_dp[s], axis=1)
                    ok = nt > 1e-3
                    E["relerr_lda"].extend(np.linalg.norm(got_dp - true_dp[s], axis=1)[ok] / nt[ok])
                E[f"rot_err_{key}_deg"].extend(
                    np.degrees([(g * q.inv()).magnitude() for g, q in zip(got_dr, true_dr[s])]))

    out["B"] = {k: float(v) for k, v in B.items()}
    print("## B. LDA copy vs bridge_orig (max abs difference over the sample)\n")
    for k, v in out["B"].items():
        print(f"- {k}: {v:.2e}")
    out["C"] = {k: float(np.max(v)) for k, v in C.items()}
    print("\n## C. bridge_orig action columns (max abs difference)\n")
    for k, v in out["C"].items():
        print(f"- action {k}: {v:.2e}")

    rel_err, trs = np.concatenate(D["rel_err"]), np.concatenate(D["train_state_rot"])
    ref = np.concatenate(D["ref_err_deg"])
    sd = trs.std(0)
    out["D"] = {"train_rot_std": sd.tolist(), "rel_abs_err_mean": np.abs(rel_err).mean(0).tolist(),
                "rel_abs_err_in_std": (np.abs(rel_err).mean(0) / sd).tolist(),
                "orientation_err_deg_median": float(np.median(ref)), "orientation_err_deg_p90": float(np.percentile(ref, 90))}
    print("\n## D. Rotation state at eval (episode-relative approximation vs the trained value)\n")
    print("| | roll | pitch | yaw |\n|---|---|---|---|")
    print("| training state std (rad) | " + " | ".join(f"{x:.3f}" for x in sd) + " |")
    print("| mean abs error of the approximation (rad) | " + " | ".join(f"{x:.3f}" for x in out["D"]["rel_abs_err_mean"]) + " |")
    print("| error / std | " + " | ".join(f"{x:.2f}" for x in out["D"]["rel_abs_err_in_std"]) + " |")
    print(f"\norientation error of the approximation: median {np.median(ref):.1f} deg, 90th pct {np.percentile(ref, 90):.1f} deg; "
          f"exact conversion R_data = R_sim @ T_SIM^T has zero error by construction.")

    E2 = {k: np.array(v) for k, v in E.items()}
    out["E"] = {k: {"median": float(np.median(v)), "p90": float(np.percentile(v, 90))} for k, v in E2.items()}
    print("\n## E. LDA action decoding at eval (one chunk of ground-truth deltas from every 16th step)\n")
    print("| | median | 90th pct |\n|---|---|---|")
    for k, lab in (("angle_lda", "position step direction error, current wrapper (deg)"),
                   ("relerr_lda", "position step relative error, current wrapper"),
                   ("rot_err_lda_deg", "rotation step error, current wrapper (deg)"),
                   ("angle_fixed", "position step direction error, simulator pose converted (deg)"),
                   ("rot_err_fixed_deg", "rotation step error, simulator pose converted (deg)")):
        print(f"| {lab} | {out['E'][k]['median']:.3g} | {out['E'][k]['p90']:.3g} |")

    # F. instructions
    tasks = [e["tasks"][0].lower() if e["tasks"] else "" for e in eps]
    pats = {"Carrot (carrot & plate)": (r"carrot", r"plate"), "Spoon (spoon & towel|cloth)": (r"spoon", r"towel|cloth"),
            "Stack (green & yellow & block|cube)": (r"green", r"yellow", r"block|cube"),
            "Eggplant (eggplant & basket)": (r"eggplant", r"basket")}
    out["F"] = {}
    print("\n## F. Training episodes whose instruction names the eval task's objects\n")
    print("| task | episodes | share |\n|---|---|---|")
    for k, ps in pats.items():
        c = sum(all(re.search(p, t) for p in ps) for t in tasks)
        out["F"][k] = c
        print(f"| {k} | {c} | {100 * c / len(tasks):.2f}% |")

    (HERE / "audit_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
