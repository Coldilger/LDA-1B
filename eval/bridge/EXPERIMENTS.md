# Experiments — index

## Central research question

Does the world-model computation performed at inference time carry causal
weight for action selection — or is the benefit entirely a training-time
representational effect?

Each experiment below attacks this question from a different angle, across
all three models under comparison (F1-VLA, mimic-video, LDA-1B). This repo
is LDA-1B's fork; the same four experiments also live in the F1-VLA and
mimic-video forks, each with a model-specific implementation.

**LDA-1B-specific note:** all four experiments are currently blocked on
fixing LDA's closed-loop pipeline first (0% success on real SimplerEnv-Bridge
runs, root-caused 2026-08-14 to a missing proprioception channel — see
`RESULTS.md` and the `diag_*.py` scripts in this directory for the
investigation). A retrain with real state input is in progress
(`lda/config/training/LDA_bridge_v3.yaml`, wandb:
lda-bridge-v3/bridge_finetune_v3). Experiments 1/2/4 below need a working
checkpoint to run against; re-run once v3 finishes.

## Experiments

- [`experiment1_ablation/`](experiment1_ablation/) — **Ablation of the
  world-model signal.** One hypothesis, two opposite interventions:
  suppress the foresight signal where the model normally uses it (F1,
  mimic-video), or turn it on where the model normally doesn't (LDA-1B).
- [`experiment2_oracle/`](experiment2_oracle/) — **Oracle injection.**
  Replace the predicted future with the ground-truth future, encoded
  through each model's own pipeline. **Status: results so far are
  invalid** — the oracle hook was added to a class the checkpoint doesn't
  actually use (`QwenGR00T`/`GR00T_ActionHeader_single_t_concat_curr_obs.py`)
  instead of the real one (`QwenMMDiT`/`MMDiT_ActionHeader.py`). Needs
  re-implementing in the right file, then re-running once the v3 checkpoint
  is ready. See `experiment2_oracle/ORACLE_EXPERIMENT.md` for the full
  (currently-invalid) write-up, kept for the mechanism explanation, not the
  numbers.
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
