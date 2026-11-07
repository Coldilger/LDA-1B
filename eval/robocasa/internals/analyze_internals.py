"""Summarize server_policy_internals.py records (CPU only):

    python analyze_internals.py records-<job>.npz

Per condition, over all decisions: next-obs token statistics, the next-obs
share of obs_merger's output, attention of action queries (mass on register /
obs / action-stream keys, normalized entropy, kernel-vs-manual error), and
the action chunk relative to the policy path on the same state, split into
steps inside vs beyond future_obs_index (inverse_dynamics is trained only on
the first future_obs_index steps).
"""

import sys

import numpy as np

CONDITIONS = ("policy", "wm_on", "oracle", "zero", "noise", "shuffle")


def main(path):
    recs = list(np.load(path, allow_pickle=True)["records"])
    print(f"decisions: {len(recs)}  rng_untouched: {sum(r.get('rng_untouched', False) for r in recs)}/{len(recs)}")
    r0 = recs[0]
    fi = r0["future_obs_index"]
    print(f"future_obs_index: {fi}  imagined shape: {r0['imagined_shape']}  curr: {r0['curr_stats']['shape']}")
    print("merger input:", r0["conditions"]["policy"]["merger"]["input_shape"],
          "curr/next channels:", r0["conditions"]["policy"]["merger"]["curr_channels"],
          r0["conditions"]["policy"]["merger"]["next_channels"])
    for k in ("cos_wm_vs_oracle", "cos_wm_vs_curr", "cos_oracle_vs_curr", "rel_l2_wm_vs_oracle"):
        x = np.array([r[k] for r in recs])
        print(f"{k}: {x.mean():.3f} ± {x.std():.3f}")
    c = recs[0]["curr_stats"]
    print(f"curr_obs tokens: mean {c['mean']:.3f} std {c['std']:.3f} norm {c['token_norm']:.2f}\n")

    cols = ["next_norm", "next_std", "next_share", "mass_reg", "mass_obs", "mass_act", "entropy",
            "kernel_err", "l1_first", "l1_rest", "act_absmax", "nan"]
    print("| condition | n | " + " | ".join(cols) + " |")
    print("|---|---|" + "---|" * len(cols))
    for name in CONDITIONS:
        rows = []
        for r in recs:
            if name not in r["conditions"]:
                continue
            c = r["conditions"][name]
            pol = r["conditions"]["policy"]["action"]
            a = c["action"]
            attn = c["attn"]
            rows.append({
                "next_norm": c["next_stats"]["token_norm"],
                "next_std": c["next_stats"]["std"],
                "next_share": c["merger"]["next_share"],
                "mass_reg": np.mean([x["mass_reg"] for x in attn]),
                "mass_obs": np.mean([x["mass_obs"] for x in attn]),
                "mass_act": np.mean([x["mass_act"] for x in attn]),
                "entropy": np.mean([x["entropy"] for x in attn]),
                "kernel_err": np.max([x["kernel_err"] for x in attn]),
                "l1_first": np.abs(a[:fi] - pol[:fi]).mean(),
                "l1_rest": np.abs(a[fi:] - pol[fi:]).mean() if a.shape[0] > fi else np.nan,
                "act_absmax": np.abs(a).max(),
                "nan": c["action_nan"],
                "mask_ok": all(x["mask_all_true"] for x in attn),
            })
        if not rows:
            continue
        vals = []
        for k in cols:
            x = np.array([d[k] for d in rows], dtype=float)
            vals.append(f"{int(x.sum())}" if k == "nan" else f"{np.nanmean(x):.4f} ± {np.nanstd(x):.4f}")
        print(f"| {name} | {len(rows)} | " + " | ".join(vals) + f" |  masks all-true: {all(d['mask_ok'] for d in rows)}")

    print("\nper-block mass on obs keys (wm_on vs policy):")
    for name in ("policy", "wm_on", "zero"):
        blocks = {}
        for r in recs:
            for x in r["conditions"][name]["attn"]:
                blocks.setdefault(x["block"], []).append(x["mass_obs"])
        print(name, " ".join(f"{np.mean(v):.3f}" for _, v in sorted(blocks.items())))

    print("\nper-step |wm_on - policy| (mean over decisions and dims):")
    d = np.stack([np.abs(r["conditions"]["wm_on"]["action"] - r["conditions"]["policy"]["action"]).mean(-1) for r in recs])
    print(" ".join(f"{v:.3f}" for v in d.mean(0)))
    print("per-step |oracle - policy|:")
    d = np.stack([np.abs(r["conditions"]["oracle"]["action"] - r["conditions"]["policy"]["action"]).mean(-1) for r in recs])
    print(" ".join(f"{v:.3f}" for v in d.mean(0)))

    print("\nperturbation test (max |action change| per perturbed next-obs token position):")
    for r in recs:
        if "perturb_effect_per_pos" in r:
            print(f"decision {r['decision']}: {r['perturb_effect_per_pos']}  repeat noise: {r['perturb_repeat_noise']:.2e}")


if __name__ == "__main__":
    main(sys.argv[1])
