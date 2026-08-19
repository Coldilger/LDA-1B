# Experiment 5 — Concept erasure (LEACE) — LDA-1B

**Status: done (2026-08-19). Decisive result, and it corrects Experiment
4's own interpretation: current pose's "nonlinear" recoverability turns
out to be substantially a nonlinear readout of a LINEAR signal — removing
that linear direction collapses it. Future pose keeps a smaller, more
genuinely uncertain nonlinear residual. See "The decisive test" below.**

## Motivation

Experiment 4 found LDA's current/future pose recoverable by an MLP
(+59-65% over constant) but not by ridge (current: −229%; future: +40.6%,
weaker) — read at the time as "the information is present but only
nonlinearly accessible," a genuinely different story from F1-VLA and
mimic-video (where neither probe family recovers pose at all). That
reading deserved a harder test before being trusted: LEACE is a *linear*
erasure tool, so it can directly ask whether the MLP's advantage survives
once the best *linear* predictor of pose is surgically removed.

## Method: LEACE, not INLP

Same as F1-VLA/mimic-video's own Exp5 (`leace.py`, copied verbatim —
Belrose et al., NeurIPS 2023, arXiv:2306.03819, chosen for closed-form
continuous-target support and provably minimal distortion). Fit only on
the train split, applied unchanged to held-out val features. Self-test on
synthetic data with a known injected direction passes
(`R^2` 0.67 → −0.005 after erasure) before touching real features.

## Level A (replicated as-is): does erasing episode identity help?

`scene_erasure_diagnostic.py`, unmodified from F1/mimic. **Erasure only
partially worked here**: episode-identity accuracy dropped from 98.7% to
55.3% (chance 8.3%) — a real reduction, but nowhere near F1/mimic's
collapse to near-chance, so this run doesn't clear the script's own
"erasure worked" bar. Read together with Experiment 4's own finding, this
is itself informative rather than a failure: if LDA's representation
really does encode information (pose, and apparently some of episode
identity too) through structure a purely linear tool can only partially
reach, a linear eraser leaving substantial residual classifiable signal is
exactly what that predicts, not a bug. Current/future pose gains barely
moved either way (current: −229.2% → −227.3%; future: +40.6% → +37.0%) —
consistent with there not being much of a scene-identity confound to
remove in the first place for this data.

## The decisive test: erase pose's own linear component, re-check the MLP

`nonlinear_erasure_diagnostic.py` (new, LDA-specific). Instead of only
erasing episode identity, fit a *second* LEACE eraser directly against
**current pose** as a continuous 116-D target (LEACE natively supports
this — no one-hot needed), then re-run the MLP probe (same `ProbeHead`
architecture as Experiment 4's) on what's left.

| condition | current MLP gain | future MLP gain |
|---|---|---|
| baseline (no erasure) | **+59.1%** | **+65.7%** |
| after erasing episode identity | +56.9% (≈unchanged) | +61.1% (≈unchanged) |
| **after erasing current-pose's linear component** | **−11.4%** | **+13.4%** |

**Current pose's MLP recoverability collapses** (+59.1% → −11.4%, a 70.5pp
drop, crossing from clearly-better-than-constant to worse-than-constant)
once its own best linear direction is removed. Since LEACE is a linear
map, only a *linear* eraser was needed to destroy essentially all of what
the *nonlinear* MLP was using — which is the direct, operational
definition of "this signal was linear all along." Episode-identity erasure,
by contrast, barely moved either number, ruling out a scene-identity
confound as the explanation for either result.

**Reading this against Experiment 4's own ridge result is the interesting
part.** Ridge already had every opportunity to use this same linear
direction (it's a linear regression with CV-selected L2 regularization)
and still scored −229%. Two closed-form linear estimators, same
underlying data, opposite readable conclusions about "is there a linear
pose signal" — LEACE's closed-form covariance-based fit (with its own,
different regularization: eigenvalue-floor + shrinkage on `Cov(X)`, not
weight-space L2) evidently found and used a direction the ridge probe's
specific regularization path missed. **This means Experiment 4's own
ridge-vs-MLP comparison was measuring probe-family sensitivity as much as
representation content** — "ridge fails, MLP succeeds" does not reliably
mean "nonlinear," at least not here.

**Future pose tells a less clean story**: +65.7% → +13.4%, a large drop
but not a full collapse. Some of the future-pose MLP signal is explained
by the same linear direction as current pose (unsurprising — future is
`current + delta`, correlated with current by construction), but a
smaller, genuinely uncertain residual (+13.4%, modest but still on the
positive side of the ±15%-ish noise band used elsewhere in this
experiment) may reflect real nonlinear content specific to the *delta*.
Not strong enough to call decisively either way without a repeat.

## Revised reading of Experiment 4

Not retracted, but narrowed: LDA-1B's `vl_embs` does encode pose
information unavailable to F1-VLA's/mimic-video's own world-model-specific
representations — that contrast still stands (LEACE's Level A, and the
plain fact that *some* probe recovers pose here at all, unlike the other
two models). What's revised is *why* ridge failed: not because the
information is fundamentally nonlinear, but because ridge's specific
regularization path underperformed a different linear estimator (LEACE)
on the same question, at least for current pose. `experiment4_probing/README.md`
updated to point here.

## Level B

Not applicable in the form F1/mimic scoped it (erase the pose direction,
re-inject into the frozen action decoder, compare closed-loop success
against Experiment 1's ablation) — LDA-1B's Experiment 1 already ran a
directly analogous real closed-loop test (turn the video_gen/inverse_dynamics
path on) and found it inert (44% vs. 48% baseline), independent of this
representation-level analysis. Re-injecting an erased/restored `vl_embs`
into a decoder that was never conditioned on `vl_embs` at that hook point
in the first place doesn't map onto this architecture's own control flow
the way it does for F1/mimic's ablation hook. Not scoped further.

## Files

- `leace.py` — LEACE fit/erase implementation + synthetic self-test
  (copied verbatim from F1-VLA/mimic-video)
- `scene_erasure_diagnostic.py` — Level A (copied verbatim)
- `nonlinear_erasure_diagnostic.py` — the decisive test (new)

## Not yet done

- [ ] Independent repeat (different seed) of the decisive test before
      treating the −11.4%/+13.4% split as final rather than
      preliminary-but-clear.
- [ ] Understand *why* LEACE's covariance-based fit finds a linear current-
      pose direction ridge's CV-selected L2 regularization misses — a real,
      somewhat surprising methodological finding in its own right, not
      just a footnote.
