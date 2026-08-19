# Experiment 4 — Representation probing (LDA-1B)

**Status: done (2026-08-19), live extraction via RoboCasa. Current AND
future pose are both recoverable — but only with a nonlinear (MLP) probe;
the linear (ridge) probe both models F1-VLA and mimic-video rely on as
their primary measurement fails badly here. See "Results" below.**

## What this tests

The direct test of the "training-time representational effect" side of the
central research question: does LDA's frozen backbone already encode useful
future information in its representations, independent of whether the
`policy` inference path (the only one used by default) actually consumes
that information?

## Method

- **Extraction point: `vl_embs`** — the shared Qwen3-VL backbone output
  feeding both the `policy` path and the normally-unused
  `forward_dynamics`/`inverse_dynamics`/`video_gen` heads. Chosen over
  `video_gen()`'s own output because it's available from a single forward
  pass (no 4-step diffusion sampling needed per sample) and is the shared
  representation upstream of all task-specific branching — see
  mimic-video's own `experiment4_probing/README.md` for why
  `dynamics_loss`-adjacent candidates were ruled out. Pooled mean+std over
  the token axis (`server_policy_extract_features.py`), matching F1/mimic's
  own pooling convention: `(B, L, 2560) -> (B, 5120)`.
- **No static RoboCasa dataset exists on this cluster** (same constraint as
  Experiments 1/2), so extraction runs the *normal, unmodified* policy live
  over real RoboCasa episodes and captures `vl_embs` as a side effect —
  robot behavior is completely unaffected, mechanically equivalent to
  F1/mimic's offline extraction, just sourced from live simulation instead
  of a saved file.
- **Pose target**: `state` — concat of
  left_arm/right_arm/left_hand/right_hand/waist proprioception (116-D,
  this humanoid's own available channels), analogous to F1/mimic's 9-D
  end-effector pose. **Future** target is the *delta* K=5 replan-steps
  ahead (`state[t+K] - state[t]`), matching F1/mimic's own reasoning for
  avoiding the "barely moves" degeneracy at an absolute-pose target.
  **Current** pose (`state[t]`) is saved alongside for the decisive control.
- **Probe**: same architecture/protocol as F1/mimic — ridge (closed-form,
  alpha cross-validated within the train split) as the primary linear
  probe, plus an MLP (`ProbeHead`, 2 hidden layers of 256) as a capacity
  check. Episode-level train/val split (never per-sample — see
  `train_probe.py`'s own docstring for why).
- Scripts `train_probe.py`, `positive_control.py`, `compare_targets.py` are
  copied **verbatim** from F1-VLA/mimic-video's own `experiment4_probing/`
  — fully generic over any `{features, targets, current_pose,
  episode_ids, horizon, variant}` .npz, no LDA-specific code in them.
  `server_policy_extract_features.py` and `mlp_current_pose.py` are new,
  LDA/RoboCasa-specific.

## Results (2026-08-19, n=800, 16 episodes × 50 samples)

**Episode identity (sanity control):** 97.0% accuracy vs. 6.25% chance
(15.5x) — the extraction pipeline is not too lossy at the coarse level;
the instrument works.

**Current pose, ridge (the control `compare_targets.py` treats as
decisive):** probe L1 0.575 vs. constant 0.175 — **−229% gain, badly
worse than trivial.** Read naively per `compare_targets.py`'s own verdict
logic (copied from F1/mimic), this would mean "no conclusion about future
information is available" — same failure mode F1-VLA and mimic-video both
hit.

**But unlike F1/mimic, an MLP recovers it.** User's hypothesis, tested
directly (`mlp_current_pose.py`, same `ProbeHead` architecture
`train_probe.py` already uses for the future target): MLP current-pose val
L1 0.0695 vs. constant 0.175 — **+60.2% gain.** Run identically against
F1-VLA's and mimic-video's own already-extracted features as a cross-check:
F1 −0.1%, mimic −7.7% — both still at/below the constant-predictor level.
**The nonlinearity hypothesis is confirmed for LDA specifically, and
refuted for F1/mimic.**

**Future pose delta**, now readable with both controls in hand:

| probe | val L1 | vs. constant/mean | vs. no-motion |
|---|---|---|---|
| ridge | 0.0564 | **+40.6%** | (mean 0.0951, no-motion 0.0808) |
| MLP | 0.0333 | **+64.9%** | |

**Reading:** for LDA-1B's `vl_embs`, both current and future pose are
substantially recoverable — but only nonlinearly. Since the current-pose
control now passes (under the probe family that actually works here), the
future-pose result is credible, not an artifact of pooling or sample count.
This is a genuinely different story from F1-VLA and mimic-video, where
neither current nor future pose is recoverable under *either* probe family
— their world-model-specific representations (`gen_out` tokens,
`crossattn_emb`) appear not to encode pose information at all, while LDA's
shared, general-purpose VLM backbone does, just not along directions a
linear probe can read.

**Caveat on the MLP result's own reliability:** with feature dim 5120 >>
600 train samples, an MLP has ample capacity to overfit — this is exactly
why `train_probe.py`'s own docstring designates ridge, not MLP, as the
*primary* measurement elsewhere in this experiment ("the first run...
produced exactly that, both variants scoring worse than a constant
predictor"). The current-pose MLP number here is taken with early-stopping
on held-out (episode-split) validation loss, same selection rule as
`train_probe.py` already uses, and the gain (+60.2%) is well outside the
±20% band that would suggest pure noise-fitting — but a fully independent
repeat (different seed, ideally more episodes) would strengthen this
further before treating it as final.

## Not yet done

- [x] Decide the extraction point (`vl_embs`).
- [x] Implement frozen-feature extraction + probe head training.
- [x] Run live via RoboCasa, report probe accuracy (ridge + MLP) against
      current and future pose.
- [ ] Independent repeat (different seed / more episodes) of the MLP
      current-pose and future-pose results, before citing +60.2%/+64.9% as
      final rather than preliminary-but-well-outside-noise.
- [ ] Cross-model write-up: fold the "MLP recovers pose for LDA but not
      F1/mimic" contrast into F1-VLA's and mimic-video's own
      `experiment4_probing/README.md` (their probing methodology's own
      linear-vs-nonlinear question, answered on their own data too, now
      that the test has been built).
