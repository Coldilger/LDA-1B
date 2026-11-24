# Experiment 11 — Action-only comparison (LDA-1B, RoboCasa)

**Question (advisor's point 5).** Train a comparable model on actions only:
does co-training with forecasting help the policy? Same point as F1-VLA's
`eval/bridge/experiment11_nogenloss` (forecasting-loss weight 0) and
mimic-video's `eval/bridge/experiment11_backbone`. For LDA-1B the question is
sharp: the deployed policy does not forecast at inference (the "forecast on"
condition of Experiments 8–9 is our own composition), so any benefit of
forecasting must come through training. The authors compare co-training with
policy-only training by action error (paper Fig. 10); this experiment
measures closed-loop success.

## Design

Two finetunes of the authors' `LDA-pretrain` checkpoint, identical except for
one flag (`lda/config/training/LDA_robocasa_exp11_{full,policyonly}.yaml`):

| run | training tasks | loss |
|---|---|---|
| `full` | policy, forward dynamics, inverse dynamics, visual forecasting (authors' recipe) | action + inverse-dynamics + latent-forecast losses |
| `policyonly` | `only_policy: true`: every sample is a policy sample (`lda/model/modules/action_model/MMDiT_ActionHeader.py` L650) | action loss only (L829 skips the others) |

Common settings: the evaluated task's data only
(`gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000`, mixture
`robocasa_cup_two_history`), model block copied from the authors' RoboCasa run
(`checkpoints/LDA-robocasa_run/config.yaml`), seed 42, 100k micro-batches of
16 with gradient accumulation 5 (effective batch 80 = the authors' 8 GPUs ×
10), i.e. 1.6M samples, the same sample budget as F1-VLA's Experiment 11.

**Deviations from the authors' RoboCasa run, applied to both runs** (one GPU
per job): the Qwen-VL interface is frozen (authors: lr 1e-5), as in our
Bridge finetune; one task instead of 24; 100k instead of 300k steps. The
authors' released checkpoint is therefore not the comparison point — the
`full` run is. Waves resume from model weights only (`train_LDA.py` does not
restore optimizer state), identically for both runs.

Training: `slurm/exp11_robocasa_selfchain.slurm` (`VARIANT=full|policyonly`),
self-chaining 6 h waves on one H100, keeps the two newest checkpoints.

## Evaluation (after training)

`closed_loop/closed_loop.slurm`, world model off (the deployed policy mode),
150 paired episodes per run, as in Experiments 8–9; paired difference
`policyonly − full` with Wilson and bootstrap CIs and exact McNemar.

## Results

Training finished at 100k for both runs (2026-11-22). The evaluated
weights are `outputs/lda_robocasa_exp11/{full,policyonly}/final_model/
pytorch_model.pt`, written by the trainer at the end of training (the
`full` run's `steps_100000` file was deleted by a pruning bug, fixed; see
`../../bridge/experiment12_audit/README.md`). Evaluation submitted
2026-11-23: `closed_loop/closed_loop.slurm` with `CONDITION=baseline`,
`CKPT=<final_model>`, `LABEL=exp11full|exp11policyonly`, scene seeds
1000/2000/3000 × 50 (the Experiment 1 episodes), jobs in `jobs_eval.txt`.

**Result (2026-11-24).** All six jobs finished (150 paired episodes per
run; `python closed_loop/paired_stats_robocasa.py exp11full exp11policyonly`):

| run | success | per scene set (1000 / 2000 / 3000) |
|---|---|---|
| `full` (co-trained with forecasting) | 24/150 = 16.0% [11.0, 22.7] | 5 / 10 / 9 |
| `policyonly` (action loss only) | 42/150 = 28.0% [21.4, 35.7] | 12 / 14 / 16 |
| authors' RoboCasa checkpoint (Experiment 1 baseline, reference) | 67/150 = 44.7% | 24 / 19 / 24 |

`policyonly − full` = **+12.0 pp** [bootstrap 95%: +4.0, +20.0]; discordant
pairs 12 (`full` only) / 30 (`policyonly` only), exact McNemar p = 0.008.

**What this shows, and what it does not.**
1. At this budget, training LDA-1B on actions only gives a better policy
   than co-training with forecasting, in every scene set.
2. **Budget per objective is not matched.** In co-training every sample is
   assigned to one of four tasks with equal weight
   (`training_task_weights: [1, 1, 1, 1]`; `MMDiT_ActionHeader.py`, task
   assignment before L686): policy, forward dynamics, inverse dynamics,
   visual forecasting. So `full` saw about a quarter of its 1.6M samples as
   policy samples (plus a quarter as inverse dynamics, which also predicts
   actions but from the future frame), `policyonly` all of them. The result
   is "at equal total compute, forecasting objectives cost policy
   updates", not "forecasting hurts at equal policy updates". The second
   would need `full` trained about 4× longer.
3. Both runs are far below the authors' checkpoint (44.7%): one task instead
   of 24, frozen Qwen-VL, 100k instead of 300k steps; neither run is
   converged, one training seed each.

**Control for the sample split (queued 2026-11-24).** `policyonly25k`
(`lda/config/training/LDA_robocasa_exp11_policyonly25k.yaml`): action-only
training for 25k micro-batches, i.e. as many policy samples as `full` got in
100k, with its own complete cosine schedule. Then the same 150-episode
evaluation. Reading: ≈ 24/150 → the 12 pp gap is the sample split and the
forecasting objectives neither help nor hurt the policy; clearly above →
co-training costs more than its share of samples; below → the forecasting
objectives help per policy sample.

**Offline action error (the authors' metric) is not possible here.** The
dataset has 1,000 demonstrations of this task and LDA's loader trains on all
of them (`_get_trajectories` reads every line of `episodes.jsonl`; the
`splits` field is ignored), so there are no held-out demonstrations. Errors
on training episodes measure memorisation and are not reported (protocol
rule). Decision (2026-11-24): skipped; the comparison rests on closed-loop
success and the `policyonly25k` control. Retraining all three runs with a
held-out set was judged not worth its cost.
