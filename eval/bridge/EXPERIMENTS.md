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
confirmed directly, see `../robocasa/RESULTS.md`'s "Decision" section) — not
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
  through each model's own pipeline. **Done (2026-08-19), live probe via
  RoboCasa** — oracle is worse than a trivial zero-action baseline (0.90 vs
  0.75 L1), indistinguishable from feeding the model its own imagined
  future. Four alternative explanations checked and ruled out (task
  undertraining, frame ordering, dead action dims, chunk-length mismatch) —
  reads as a genuine property of this checkpoint's `inverse_dynamics`
  pathway. See `experiment2_oracle/ORACLE_EXPERIMENT.md`.
- [`experiment3_cost/`](experiment3_cost/) — **Cost per decision.**
  Characterizes how much inference-time compute each model actually spends
  on its world-model computation. LDA-1B is the interesting baseline case
  here: no extra inference-time cost at all (the visual-forecasting head is
  a training-time co-objective, unused at inference by default). **Latency
  measured 2026-08-19** via the confirmed-working RoboCasa checkpoint
  (Bridge still has nothing working to time) — see
  `experiment3_cost/README.md`.
- [`experiment4_probing/`](experiment4_probing/) — **Representation
  probing.** Freezes each model's backbone and trains a small probe head to
  predict future end-effector pose from a single frozen hidden state.
  **Status: planned, not yet run** — extraction point in LDA's architecture
  not yet decided (F1-VLA/mimic-video's own docs suggest `vl_embs` as the
  current candidate). Will run against RoboCasa, per the priority-shift
  decision above.
- [`experiment5_erasure/`](experiment5_erasure/) — **Concept erasure
  (LEACE).** Follow-up to Experiment 4's decisive control in F1-VLA and
  mimic-video: erase the scene/episode-identity direction and check whether
  pose becomes recoverable. **Status: not started** — scoped pending
  Experiment 4's extraction point being resolved for LDA-1B first.
