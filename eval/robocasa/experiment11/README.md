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
