# Experiment 2 — Oracle injection (LDA-1B)

**Status: results so far are INVALID. Needs re-implementing before the
numbers can be trusted.**

## What went wrong

The oracle hook (`oracle_future_imgs` parameter, switching `task_embedding`
to `id_embedding`/inverse-dynamics mode when a real future frame is given)
was added to `lda/model/modules/action_model/GR00T_ActionHeader_single_t_concat_curr_obs.py`
and `lda/model/framework/QwenGR00T_single_t_concat_curr_obs.py`. The
checkpoint actually used for Bridge (`framework.name: QwenMMDiT` in every
`LDA_bridge*.yaml`) loads a *different* class,
`lda/model/framework/QwenMMDiT.py` → `Qwen_MMDiT`, whose `predict_action`
never references `future_image` at all — confirmed by direct code trace
(`QwenMMDiT.py` imports `MMDiT_ActionHeader.py`'s `FlowmatchingActionHead`,
not the GR00T-family action head the hook was added to). The hook lives in
dead code for this checkpoint; every reported oracle-vs-baseline number in
`ORACLE_EXPERIMENT.md` reflects the *unmodified* default path in both
conditions, not a real oracle intervention.

## What this experiment tests (once fixed)

Replace the model's future-observation channel with the real ground-truth
future, encoded through the model's own pipeline, and compare
predicted-action L1 error against the unmodified default. Extends the
method from mimic-video's own paper (arXiv:2512.15692, Section III / Fig. 2).
For LDA specifically, this isn't "replacing imagination with reality" — by
default LDA's `policy` inference path has no imagination of the future at
all (a constant placeholder). It's "replacing a placeholder with reality":
switching into the trained-but-normally-unused `inverse_dynamics` mode and
giving it the real next frame (DINOv3-encoded) instead of nothing.

## Fix needed before re-running

1. Add the equivalent `oracle_future_imgs` hook to the *actually-used*
   files: `lda/model/modules/action_model/MMDiT_ActionHeader.py` (confirmed
   real action-model class — has `TRAINING_TASKS` including
   `inverse_dynamics`, and `id_embedding` support in `forward()`, but
   `predict_action()` is hardcoded to the `policy` path) and
   `lda/model/framework/QwenMMDiT.py`.
2. Wait for the v3 retrain (`LDA_bridge_v3.yaml`, real proprioception) —
   running this against v2 would still produce a real number, but v3 is the
   checkpoint the project is moving forward with, so re-running once against
   the right checkpoint avoids doing this twice.
3. Re-run the offline probe (`lda/eval/oracle_offline_probe.py`,
   launchers in `slurm/oracle_offline_probe.slurm` / `slurm/oracle_smoke.slurm`
   / `slurm/oracle_smoke_big.slurm` — these live outside `eval/bridge/` by
   this repo's existing convention, not moved here to avoid touching
   already-fragile, pending-fix code) and rewrite
   `ORACLE_EXPERIMENT.md`'s results table with real numbers.

The mechanism explanation and pretraining-background sections of
`ORACLE_EXPERIMENT.md` remain accurate; only the results table and its
interpretation need redoing.
