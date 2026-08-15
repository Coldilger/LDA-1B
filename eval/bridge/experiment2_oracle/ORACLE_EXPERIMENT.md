# Experiment 2: Oracle injection — LDA-1B

## What this is and why it's grounded, not invented

This experiment is not something we made up. It directly extends the
method from Section III / Fig. 2 of the mimic-video paper
(arXiv:2512.15692, "Case Study: How Does Video Generation Quality Affect
Robot Policy Performance?"). There, the authors give the action decoder
either a predicted future or the real ("oracle") one, and measure how
much that changes performance. We applied the same method to all three
models under comparison (F1-VLA, mimic-video, LDA-1B), each time through
that specific model's own natively-trained future-prediction mechanism.

## Pre-training background

LDA-1B is not trained from scratch on Bridge. It starts from a released
`LDA-pretrain` checkpoint, pre-trained on a broad open-x-embodiment mix
(embodiment tag `oxe`), then we finetune it on BridgeData V2 specifically
(this project's `bridge_finetune` / `bridge_finetune_v2` runs). This
mirrors F1-VLA's setup (pre-trained by the original authors, stage-3
finetuned here on Bridge).

Pre-training scale, per the architecture reference: 48x NVIDIA H800 GPUs,
400k iterations, 4,608 total GPU-hours. The VLM and DINO encoder are kept
frozen throughout pre-training; only the MM-DiT and action encoder/decoder
are updated, to preserve the frozen models' generalization and visual
representation quality. This is a useful data point for the "confounds
across models" comparison (pre-training scale/compute), alongside F1's
330k episodes and mimic-video's 200h + internet video figures.

## How exactly LDA "sees" the real future — this one's structurally different

Unlike F1 and mimic, LDA by default has **no imagination of the future at
all**. The model uses a fixed, learned placeholder — a constant vector
carrying no information about the specific moment
(`next_obs_learnable_tokens`). This follows from the checkpoint being
trained with `state_dim: null` — the model never receives its own arm
position as input, only the image.

In the oracle condition we aren't replacing "imagination" with "reality"
(there's no imagination to replace) — we're replacing a **placeholder**
with reality: the real next frame is passed through the same DINOv3
encoder used for the current frame, and the model is additionally
switched into `inverse_dynamics` mode — a second, also natively-trained
mode (alongside the default `policy` mode) the model was specifically
trained to use when the real next observation is given directly.

**Important for interpretation:** since LDA's default mode computes
nothing about the future at all, "oracle doesn't help" here isn't the
same claim as for mimic (which computes *something* by default). For LDA
the sharper question is: "if the model is given a channel carrying real
future information, which it was trained to use, does that help?"

## How the metric is computed

Same as F1/mimic: take a real logged moment, compare the model's
predicted action to what actually happened. **L1** = mean
`|predicted − real|` across the action vector's numbers.

**Difference for LDA:** the number is currently in *normalized* space
(the model's own units, after q99 normalization), not physical units
like F1/mimic — direct magnitude comparison across models isn't possible
yet, only direction (better/worse).

## Current results

| n samples | oracle/inverse_dynamics | policy-default |
|---|---|---|
| 30 (10 ep. × 3) | 0.4123 | 0.3725 |
| 320 (80 ep. × 4) | 0.3567 | 0.3501 |

**How to read it:** at the small sample size the gap looked meaningful
(~10.7%); at 10x the sample size it nearly vanished (~1.9%) but didn't
flip sign — oracle is consistently slightly worse than default. This no
longer looks like pure noise (a direct, self-generated illustration of
why 24-episode benchmarks aren't enough), but it isn't a final
conclusion either — LDA's `inverse_dynamics` head is documented as
having trained with difficulty (see `RESULTS.md`), so "doesn't help"
could be an architectural finding or a property of this specific
undertrained checkpoint — not yet distinguishable.

## Side finding: probable cause of 0% closed-loop success

Alongside the oracle test we investigated why LDA v2 (retrained from
scratch with the gripper fix present from step 0) still gets 0% success
rate on all 4 SimplerEnv-Bridge tasks, despite:
- the gripper being separately verified and working (+0.86 correlation,
  `grip-*.out`);
- the open-loop probe showing a real position signal (L1 beats the
  "arm doesn't move" baseline, `RESULTS.md`).

Checked and **ruled out** (math and conventions match exactly, including
a numeric check against scipy):
- gripper/rotation/image-aspect fixes — same as F1's;
- delta-conversion math (`calculate_delta_eef` / `delta2abs`);
- rotation convention (`batched_rpy_to_R` vs
  `scipy.Rotation.from_euler("xyz", ...)`);
- frame order `[past, current]` — matches `observation_indices = [-5, 0]`
  in the config.

**Candidate found by directly comparing against the training pipeline**
(`eval/bridge/diag_policy_wrapper.py`, tests `LDAInference` — the actual
closed-loop wrapper — on real Bridge frames, no simulator involved):
`lda_policy.py` builds `history_action` (the past-action history fed to
the model) from the **model's own past predictions**
(`self.action_history`), while the training pipeline
(`get_step_data_with_transform`) builds it from **real, ground-truth
past actions** from the dataset. Since the checkpoint has
`state_dim: null`, `history_action` is the model's only channel for
knowing its own current state (gripper especially — without it, gripper
accuracy drops to chance, per `lda_policy.py`'s own comments). If
slightly imperfect self-predictions land there instead of ground truth,
error could compound over time (exposure bias, a classic receding-horizon
control problem).

**Growth (exposure-bias) hypothesis tested and NOT supported for position
or rotation** (`diag_policy_wrapper_rollout.py`, 12 episodes × 20 steps,
real closed-loop via `LDAInference.step()`, history built from the
wrapper's own predictions): neither position nor rotation error grows
over the rollout.

| | overall | first half | second half |
|---|---|---|---|
| position L1 | 0.0091 | 0.0097 | 0.0084 |
| rotation L1 | 0.0326 | 0.0363 | 0.0285 |
| gripper error | **0.4293** | 0.4771 | 0.3771 |

Position and rotation look reasonable throughout and, if anything,
improve slightly across the rollout — not the growing-error signature
exposure bias would predict.

**Gripper is a different story, and this is very likely the real cause
of 0% closed-loop success.** 0.43 mean error on a ~[0,1]-scale prediction
is close to what pure guessing would produce — the wrapper's gripper
channel carries essentially no usable open/close signal, from step 0
onward, not growing worse over time but never good either. This
reconciles the two apparently contradictory measurements: the earlier
+0.86-correlation gripper result (`grip-*.out`) was measured with
**ground-truth** `history_action` from the training pipeline
(`get_step_data_with_transform`); this test uses the wrapper's own
**self-generated** `history_action`, exactly as real closed-loop control
must (you never have ground truth of your own past actions at
deployment). Per `lda_policy.py`'s own documented measurement, gripper
accuracy without `history_action` at all collapses to chance (51.5%
accuracy / +0.026 correlation) versus 93.0% / +0.863 with the
ground-truth version. This test shows that self-generated history does
**not** recover that benefit — it behaves close to the "without" case
even once several steps of self-predicted history have accumulated.

**Read together, this points to a training/deployment mismatch rather
than a wrapper bug**: the model appears to have learned to rely on
`history_action` being accurate (i.e. implicitly on it being ground
truth) for gripper state, which is information that is structurally
unavailable in real closed-loop control. If so, this isn't something a
code fix to `lda_policy.py` can resolve on its own — it would need
either retraining with self-generated (noisy) history in the loop
(scheduled sampling / DAgger-style), or removing the model's reliance on
history_action for gripper by giving it a more direct channel (e.g.
lifting `state_dim: null`, if the pre-trained backbone supports state
conditioning at all). Worth checking with someone closer to the
architecture before committing to either fix.

## Caveats

1. **Memorization** — training had no held-out split, like all three
   models.
2. **Units** — normalized space, not directly comparable in magnitude to
   F1/mimic (see above).
3. **One run, no seed repeats** (aside from the 30-vs-320-sample
   comparison, which already partly addresses statistical power).
