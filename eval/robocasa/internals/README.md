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

## Positive control: the current observation (job 745580, 2026-11-07)

Question: is the action head insensitive to the forecast in particular, or
to visual tokens in general? Same harness, same 3 scene seeds, 180
decisions, now also replacing the **current** observation on the default
policy path (`summary-745580.txt`):

- `curr_*`: only the current-frame tokens at `obs_merger`'s input, the layer
  the forecast also enters. Their relative change of the input is given.
- `img_*`: the whole image input of the decision, which feeds both the
  Qwen VLM and the action head.
- `curr_self`: the current-frame tokens replaced by themselves (checks the
  replacement hook).

| condition | input change | action change, median | 90th pct | max |
|---|---|---|---|---|
| curr_self (hook check) | 0 | **0 (exactly, 180/180)** | 0 | 0 |
| forecast slot, wm_on / oracle / zero / noise / shuffle (this run) | — | 0.04–0.05% | ≤0.06% | 1.7–14% |
| curr_zero | 100% | 0.24% | 0.33% | 10% |
| curr_noise | 141% | 0.24% | 0.36% | 10% |
| curr_stale (5 decisions earlier, same episode) | 28% | 0.01% | 0.03% | 7% |
| curr_other (another episode) | 56% | 0.02% | 0.03% | 1.7% |
| img_stale (whole image, 5 decisions earlier) | — | 0.05% | **36%** | **46%** |
| img_other (whole image, another episode) | — | 0.10% | **20%** | **44%** |

The whole-image effect depends on where in the episode the decision is
(median action change):

| decision in episode | 0–4 | 5–14 | 15–29 | 30–59 |
|---|---|---|---|---|
| img_stale | — | 28% | 0.9% | 0.02% |
| img_other | 0.7% | 10% | 0.09% | 0.08% |
| curr_zero | 0.29% | 0.26% | 0.25% | 0.24% |
| wm_on | 0.07% | 0.05% | 0.05% | 0.05% |

**What this means.**
1. The measurement can see an effect: changing the image changes the action
   by up to 46%, and the hook check is exact. The positive control passes.
2. The action head barely uses its own visual-token input. Replacing the
   current frame's tokens with zeros or noise changes the action by ~0.25%,
   with another episode's frame by 0.02%. The forecast enters the same input
   and is ignored the same way. So "LDA ignores its forecast" is a special
   case of "the action head ignores the visual tokens it receives through
   `obs_merger`".
3. By elimination, the image reaches the action through the Qwen VLM
   features (the whole-image swap changes both paths, the token swap alone
   does almost nothing). A VLM-only swap was not run.
4. Even the whole image matters mainly early in the episode (decisions
   5–14); from decision 30 on, swapping it changes the action by ~0.02%.

**Side observation.** With the same scene seeds this run succeeded in 2/3
episodes, job 745495 in 0/3. The decision-0 action is identical in both runs;
from decision 1 the observations already differ, so the rollout itself is not
bit-reproducible (likely GPU or simulator nondeterminism, not investigated).
The paired closed-loop protocol pairs scenes, not trajectories, so it is not
affected. Episodes always run the full 720 steps (180 = 3 × 60 decisions in
both runs, despite the successes), so the "same episode" / "another episode"
lags above are correct.
