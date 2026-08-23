# Experiments — index

## Central research question

Does the world-model computation performed at inference time carry causal
weight for action selection — or is the benefit entirely a training-time
representational effect?

Each experiment below attacks this question from a different angle, across
all three models under comparison (F1-VLA, mimic-video, LDA-1B). This repo
is LDA-1B's fork; the same five experiments also live in the F1-VLA and
mimic-video forks, each with a model-specific implementation.

**LDA-1B-specific note (updated 2026-08-19):** Bridge closed-loop is still
0% (root-caused 2026-08-14 to a missing proprioception channel, retrain with
real state input landed but didn't fix it — see `RESULTS.md`), and the
leading explanation is now that LDA-1B's representation doesn't transfer
across camera viewpoint at all (egocentric-trained, Bridge is third-person;
supported by a real Bridge 0% result plus a RoboCasa camera-swap test, the
latter with a self-occlusion caveat as of 2026-08-22 — see
`../robocasa/RESULTS.md`'s "Decision" section) — not
a bug waiting on a fix. **Decision: Experiments 1/2/4(/5) below now run
against the confirmed-working RoboCasa checkpoint instead of waiting on
Bridge**, same as Experiment 3 already does, accepting the dataset mismatch
as a stated tradeoff. Bridge's 0% stays documented, just no longer blocks
the rest of this repo's experiments.

## Experiments

- [`experiment1_ablation/`](experiment1_ablation/) — **Ablation of the
  world-model signal.** One hypothesis, two opposite interventions:
  suppress the foresight signal where the model normally uses it (F1,
  mimic-video), or turn it on where the model normally doesn't (LDA-1B).
  **Done (2026-08-19), real closed-loop, via RoboCasa** — turning the
  dormant world-model path on doesn't help (44% vs 48% baseline, within
  this thesis's own noise floor for a single 50-episode run). See
  `experiment1_ablation/README.md`.
- [`experiment2_oracle/`](experiment2_oracle/) — **Oracle injection.**
  Replace the predicted future with the ground-truth future, encoded
  through each model's own pipeline. **Done, live probe via RoboCasa** —
  both oracle and world-model conditions are worse than a trivial
  zero-action baseline (0.90 vs 0.75 L1) at matching what the real policy
  actually does. Confirmed 2026-08-21 at ~10x scale (n=595, up from n=62)
  with a genuine successful-episodes-only filter (n=419) — barely moves the
  numbers, ruling out "comparing against a mediocre policy" as the
  explanation. Four alternative explanations also checked and ruled out
  (task undertraining, frame ordering, dead action dims, chunk-length
  mismatch) — reads as a genuine property of this checkpoint's
  `inverse_dynamics` pathway. **The correctness question, resolved
  2026-08-23:** a direct `L1(oracle_action, world_model_action)` (job
  634481, n=595) comes out ~0.90 — as large as either condition's distance
  to the real policy's action — so oracle and world-model actions are
  *not* close to each other. **Opposite finding from F1-VLA**: correctness
  of the fed-in future genuinely changes this pathway's output. See
  `ORACLE_EXPERIMENT.md`'s "Resolved 2026-08-23" section.
- [`experiment3_cost/`](experiment3_cost/) — **Cost per decision.**
  Characterizes how much inference-time compute each model actually spends
  on its world-model computation. LDA-1B's default path has no extra
  inference-time cost (the visual-forecasting head is a training-time
  co-objective, unused at inference by default) — but activating it (Exp1's
  world-model-on condition) costs ~68% more per decision (428.2ms vs.
  254.7ms median, measured 2026-08-23, previously never instrumented).
  **Latency measured 2026-08-19 / 2026-08-23** via the confirmed-working
  RoboCasa checkpoint (Bridge still has nothing working to time) — see
  `experiment3_cost/README.md`.
- [`experiment4_probing/`](experiment4_probing/) — **Representation
  probing.** Freezes each model's backbone and trains a small probe head to
  predict future end-effector pose from a single frozen hidden state.
  **Done (2026-08-19), live extraction via RoboCasa, extraction point
  `vl_embs`.** Unlike F1-VLA and mimic-video, both current and future pose
  ARE recoverable by an MLP where ridge fails. Experiment 5 then found this
  is substantially a linear signal ridge's own regularization missed, not
  a genuinely nonlinear encoding — see `experiment4_probing/README.md`.
- [`experiment5_erasure/`](experiment5_erasure/) — **Concept erasure
  (LEACE).** Level A (erase episode identity, same as F1-VLA/mimic-video):
  only partially erased (98.7%→55.3%, not to chance) — consistent with the
  representation not being purely linear in how it encodes things.
  **Decisive test (LDA-specific): erase current pose's own linear
  component, re-check the MLP.** Current pose's MLP recoverability
  collapses (+59.1%→−11.4%) — a linear direction ridge missed accounted
  for nearly all of it. Future pose keeps a smaller, less certain
  nonlinear residual (+65.7%→+13.4%). See `experiment5_erasure/README.md`.
