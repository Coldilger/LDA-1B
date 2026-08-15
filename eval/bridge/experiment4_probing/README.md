# Experiment 4 — Representation probing (LDA-1B)

**Status: planned, not yet run.**

## What this tests

The direct test of the "training-time representational effect" side of the
central research question: does LDA's frozen backbone already encode useful
future information in its representations, independent of whether the
`policy` inference path (the only one used by default) actually consumes
that information? If a small probe can recover future end-effector pose
from a single frozen hidden state, that's evidence the *training* process
(which does include `video_gen`/`inverse_dynamics` as co-objectives, even
though inference doesn't use them) already built the representation.

## Method

- Freeze the backbone, take one hidden state at the same relative position
  in all three models (candidate for LDA: the shared MM-DiT backbone's
  output before the task-specific head branches off into
  `policy`/`inverse_dynamics`/`video_gen`/`forward_dynamics` — not yet
  confirmed as the right comparable point against F1/mimic's own extraction
  points).
- Train a small probe head (same size/architecture for all three models) on
  top of the frozen features.
- Probe target: **future end-effector pose** — deliberately not what the
  `policy` path is trained to predict directly.
- Use the same Bridge-finetuned checkpoint as Experiments 1–3 (the v3
  checkpoint once it's ready, per `../EXPERIMENTS.md`).
- Cost: forward pass only, train just the small probe head — hours, no
  retraining of the backbone itself.

## Not yet done

- [ ] Decide the exact extraction point in LDA's architecture, comparable
      to the equivalent choice in F1 and mimic-video.
- [ ] Implement frozen-feature extraction + probe head training script.
- [ ] Run on real logged Bridge trajectories, report probe accuracy against
      future end-effector pose.
