# Experiment 1 — Ablation of the world-model signal (LDA-1B)

**Status: planned, not yet run. Blocked on the v3 retrain (real
proprioception) landing — see `../EXPERIMENTS.md`.**

## What this tests

One hypothesis, two opposite interventions, applied per-model depending on
whether the model's world-model computation normally runs at inference:

- Where it **normally runs** (F1, mimic-video): turn it **off**.
- Where it **normally doesn't run** (**LDA-1B**): turn it **on**.

This is the complement to Experiment 2 (oracle injection): E2 asks "does a
*perfect* future help", E1 asks "does the model's *own, actually-computed*
future matter at all, in either direction."

## LDA-1B's mechanism: turn ON the inference path it doesn't normally use

LDA's checkpoint was trained on four tasks (`MMDiT_ActionHeader.py`,
`TRAINING_TASKS = ["policy", "forward_dynamics", "inverse_dynamics",
"video_gen"]`), but at inference `predict_action()` always hardcodes the
**policy** path: `task_embedding = self.policy_embedding.unsqueeze(0)...` —
direct policy `p(a|o)`, no future observation involved at all. The other
three trained-but-unused-at-inference tasks include `inverse_dynamics`,
`p(a|o,o')` — action conditioned on both the current AND a future
observation — and `video_gen`, which can actually *produce* a predicted
future frame.

The ablation: **same weights, different inference path.**

1. Run `video_gen` to get the model's own predicted future frame (this is
   real, imagined content — not the oracle ground truth E2 uses).
2. Feed that self-generated frame into the `inverse_dynamics` path (same
   mechanism already built for E2's oracle hook — switching
   `task_embedding` to `id_embedding` and encoding the future frame via
   `encode_future_img` — but with the model's *own* prediction instead of
   the real next frame).
3. Compare the resulting action prediction against the default `policy`
   path's prediction, on the same real logged moments.

If routing through the model's own imagined future via `inverse_dynamics`
changes/improves the prediction relative to the always-used `policy` path,
that's evidence the trained-but-dormant world-model pathway carries real
information the default path doesn't use. If it doesn't, that pathway is
inert at inference regardless of what future it's given (self-generated or,
per E2, oracle).

## Not yet done

- [ ] Move/re-verify the oracle-style `inverse_dynamics` hook in the
      correct file (`MMDiT_ActionHeader.py` / `QwenMMDiT.py`) — same
      prerequisite fix E2 needs.
- [ ] Wire up a two-stage call: `video_gen` to predict a future frame, then
      `inverse_dynamics` fed that self-generated frame.
- [ ] Run offline against real logged Bridge moments, compare against the
      default `policy`-path baseline.
