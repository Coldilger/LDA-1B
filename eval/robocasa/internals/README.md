# Experiment 1 internals: does LDA-1B's action head use the forecast? (2026-11-07)

Advisor point 2: for each ablation, check shapes, masks, attention, activation
values, and whether the action branch still runs correctly. This is the
LDA-1B counterpart of mimic-video's `experiment1_ablation/ablation_sanity/`
and F1-VLA's `experiment1_ablation/internals/`.

Closed-loop Experiment 1 result being explained
(`../closed_loop/paired_baseline_vs_wm_on.md`): default policy 67/150, world
model on 62/150, paired difference -3.3 pp [-12.7, +6.0], exact McNemar
p = 0.58.

## Setup

- Job 745495, `internals.slurm` + `../server_policy_internals.py` (analysis:
  `analyze_internals.py`, output: `summary-745495.txt`).
- RoboCasa `PnPCupToDrawerClose`, scene seeds 1000-1002 (3 episodes,
  0/3 successes), `n_action_steps=12`, 180 decisions in total.
- The robot is driven by the default policy path. At every decision the
  previous decision's input is re-run through the action head with
  different contents of the next-observation slot, with the same seed (same
  initial action noise) for all conditions:

| condition | next-observation slot | task embedding |
|---|---|---|
| policy | learnable placeholder tokens (default) | `policy_embedding` |
| wm_on | the model's own `video_gen()` forecast (Experiment 1) | `id_embedding` |
| oracle | the real next frame, encoded like the current frame | `id_embedding` |
| zero | zeros, forecast shape | `id_embedding` |
| noise | Gaussian with each forecast token's mean/std | `id_embedding` |
| shuffle | forecast from 60 decisions earlier (another episode) | `id_embedding` |

## Checks

- **Config.** `future_obs_index = 16`, `action_horizon = 16`. The inverse-
  dynamics loss therefore covers the whole 16-step chunk, including the 12
  executed steps; the code default of 5 does not apply to this checkpoint.
- **Shapes.** Forecast `[1, 201, 384]`, current observation `[201, 768]`
  (2 history frames folded into channels as `(c t)`), `obs_merger` input
  `[1, 201, 1152]` = 768 + 384, as the architecture expects.
- **Masks.** `JointAttention` receives no padding mask; the recorded masks
  are all-true in every block, decision and condition.
- **Attention kernel.** Attention weights recomputed from the kernel's q/k/v
  reproduce its output (relative error 0.0000 everywhere).
- **RNG.** The side computations leave the driving policy's CUDA RNG
  untouched in 180/180 decisions.
- **Activations.** The forecast lands in the input range: token norm 6.7
  (forecast) vs 7.1 (real next frame); cosine to the real next frame 0.88.
  After `obs_merger`, the next-observation half contributes 29% of the output
  norm for the forecast and 30% for the real frame (5% for the placeholder).
  No NaN in any condition.
- **Attention.** Action queries put ~16% of attention on register tokens,
  ~41% on observation tokens and ~43% on the action stream, with normalized
  entropy ~0.55, the same in all six conditions to within 0.01 (per-block
  values also match).

## Action branch: the forecast slot barely changes the action

Relative L2 change of the 16x138 action chunk vs the `policy` path on the
same state (normalized action units, mean |a| = 0.75):

| condition | mean | max |
|---|---|---|
| wm_on | 0.10% | 5.8% |
| oracle (real next frame) | 0.29% | 17.8% |
| zero | 0.06% | 2.2% |
| noise | 0.11% | 9.6% |
| shuffle | 0.12% | 7.4% |
| reference: consecutive decisions | 133% (median) | |

Replacing a single forecast token with noise changes the action by at most
0.002 (repeat of an identical input: exactly 0).

So the path runs correctly (right shapes, no masks, valid activations that
reach the attention), but the action head's output is almost independent of
what is in the next-observation slot, including the real future frame and
the switch from `policy_embedding` to `id_embedding`. This is consistent with
the non-significant closed-loop difference.

Not tested yet: whether the head is sensitive to the *current* observation
tokens (a positive control). Without it, "ignores the forecast" and "ignores
the visual tokens in general" cannot be told apart.
