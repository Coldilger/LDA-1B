# Experiment 2: Oracle injection — LDA-1B

**Status: done (2026-08-19), live probe via RoboCasa. Oracle is worse than
a trivial zero-action baseline — not just "doesn't help," actively worse,
and indistinguishable from feeding the model its own imagined future
(Experiment 1's condition). See "Current results" below.**

## What this is and why it's grounded, not invented

This experiment is not something we made up. It directly extends the
method from Section III / Fig. 2 of the mimic-video paper
(arXiv:2512.15692, "Case Study: How Does Video Generation Quality Affect
Robot Policy Performance?"). There, the authors give the action decoder
either a predicted future or the real ("oracle") one, and measure how
much that changes performance. We applied the same method to all three
models under comparison (F1-VLA, mimic-video, LDA-1B), each time through
that specific model's own natively-trained future-prediction mechanism.

## Pivoted to RoboCasa, not Bridge (2026-08-19)

Bridge closed-loop is still 0% (see `../RESULTS.md`), and the leading
explanation is now that LDA-1B's representation doesn't transfer across
camera viewpoint at all (confirmed directly, see
`../../robocasa/RESULTS.md`'s "Decision" section) — not a bug waiting on a
fix. Per the priority-shift decision in `../EXPERIMENTS.md`, this experiment
now runs against the confirmed-working RoboCasa checkpoint instead of
waiting on Bridge, same as Experiments 1 and 3.

No static RoboCasa dataset is available on this cluster either (the
checkpoint's own training config points at `/data/RobotData/robocasa_1k_20hz`,
which doesn't exist here), so this uses a **live** probe instead of an
offline one: real (curr, next, real-action) triples are generated during an
actual RoboCasa rollout (the "next" observation and the "real" action are
just what genuinely happened one step later in the live simulation), rather
than read from a saved file. Mechanically equivalent to F1-VLA/mimic-video's
offline probes, just sourced online instead of from disk. See
`../../robocasa/server_policy_oracle_probe.py` / `server_policy_diag_masking.py`.

## How exactly LDA "sees" the real future — this one's structurally different

Unlike F1 and mimic, LDA by default has **no imagination of the future at
all**. The model uses a fixed, learned placeholder — a constant vector
carrying no information about the specific moment
(`next_obs_learnable_tokens`). The checkpoint was trained on four tasks
(`MMDiT_ActionHeader.py`: `policy`, `forward_dynamics`, `inverse_dynamics`,
`video_gen`), but `predict_action()` at inference always hardcoded the
**policy** path — direct `p(a|o)`, no future observation involved.

In the oracle condition we aren't replacing "imagination" with "reality"
(there was no imagination to replace) — we're replacing a **placeholder**
with reality: the real next frame is encoded the same way the current frame
is, and the model is switched into `inverse_dynamics` mode (`p(a|o,o')`) —
a second, natively-trained mode the checkpoint was specifically trained to
use when a real next observation is given directly. Implementation: added
`oracle_future_imgs` / `inverse_dynamics_next_obs_tokens` params to
`MMDiT_ActionHeader.predict_action` (previously hardcoded to policy only),
threaded through `QwenMMDiT.predict_action`'s kwargs — see that file's git
history for the two rearrange-shape bugs found and fixed along the way.

**An earlier version of this doc reported results from the wrong model
class** (`QwenGR00T`/`GR00T_ActionHeader_single_t_concat_curr_obs.py` —
the checkpoint actually uses `QwenMMDiT`/`MMDiT_ActionHeader.py`, confirmed
by checking the RoboCasa checkpoint's own `config.yaml`: `name: QwenMMDiT`).
That investigation (gripper/exposure-bias work on Bridge v2, further down
this doc) is kept for the record but is about a different checkpoint on a
different dataset — not the results below.

## How the metric is computed

Take a real moment from a live rollout: the "now" frame, the frame that
resulted one step later, and the action the (unmodified, default) policy
actually took at that step (this rollout's own analogue of a logged
dataset's `action` field). Compare:
- **oracle** — `inverse_dynamics`, given the real next frame.
- **world-model** — `inverse_dynamics`, given the model's own `video_gen()`
  imagined next frame instead (this is Experiment 1's condition, computed
  by the same probe for efficiency — see that experiment's own README for
  its own, decisive closed-loop result; this doc only concerns Oracle).
- **zero** — trivial "don't move" baseline.

**L1** = mean `|predicted − real| ` across the action vector's numbers,
normalized-space units (not physical units like F1/mimic — direct magnitude
comparison across models isn't meaningful, only direction).

## Current results (2026-08-19, n=62 live samples, 2 episodes)

| condition | L1 |
|---|---|
| oracle (real future) | **0.901** |
| world-model (imagined future) | **0.897** |
| zero (trivial, no movement) | **0.753** |

**Oracle and world-model are statistically indistinguishable from each
other, and both are clearly worse than doing nothing.** This is a stronger
and stranger result than mimic-video's own "oracle barely beats a trivial
baseline" (mimic's oracle L1 0.0092 vs baseline 0.0097, ~1.05x) — LDA's
oracle isn't merely unhelpful, it's actively worse, by a wide and stable
margin.

**Four alternative explanations checked and ruled out, each by a direct
test, not just plausible reasoning** (see [[feedback_verify_before_confirming]]):

1. **`inverse_dynamics` undertrained relative to `policy`?** No —
   `id_embedding`'s norm (8.37) is essentially identical to
   `policy_embedding`'s (8.26), both ~10x their random-init scale (~0.78).
   Loaded the checkpoint directly and compared task-embedding parameter
   norms (`diag_task_embeddings.slurm`).
2. **Wrong frame taken as "the real next frame" (frame-history ordering
   bug)?** No — logged per-frame means across 5 consecutive real steps
   (`server_policy_diag_frames.py`); the assumed-newest index is
   consistently on the same side of the run's overall brightness trend as
   the next step's frames, confirming the ordering assumption.
3. **Unmasked L1 penalizing structurally-dead action dimensions** (this
   embodiment has no legs/head/base)? No — empirically, all 138/138 action
   dimensions move by more than a trivial threshold across real samples;
   masking to "active" dims only (which is all of them) doesn't change the
   number at all (`server_policy_diag_masking.py`).
4. **Chunk-length mismatch** (the "real action" reference is the full
   16-step predicted chunk, but the client only ever executes 12 of those
   16 before replanning)? No — restricting the L1 comparison to just the
   first, definitely-executed step (or 4, 8, 12) gives the same picture as
   the full 16-step chunk; the oracle/world-model-vs-zero gap doesn't shift
   with the cutoff.

**Reading, given all four are ruled out:** this looks like a genuine
property of this checkpoint's `inverse_dynamics` pathway, not an artifact
of how the probe is built. Combined with Experiment 1's own closed-loop
finding (turning the world-model path on doesn't help, 44% vs 48% baseline)
this is a consistent, mechanistically-linked story: `inverse_dynamics`'
action decoding is poorly calibrated regardless of what future observation
conditions it, real or imagined — which is exactly what would make the
closed-loop world-model-on condition unable to improve on the default
policy, independent of whether the underlying world-model signal (video_gen)
itself is any good.

## Side finding: probable cause of 0% closed-loop success (Bridge, historical)

Alongside the original (Bridge, wrong-class) oracle test we investigated why
LDA v2 got 0% success rate on all 4 SimplerEnv-Bridge tasks. Kept for the
record; superseded as an explanation by the camera-viewpoint finding in
`../../robocasa/RESULTS.md`, but the mechanism described here (gripper
history_action mismatch between training and closed-loop deployment) may
still be a real, independent contributing issue.

Checked and **ruled out** (math and conventions match exactly, including a
numeric check against scipy): gripper/rotation/image-aspect fixes (same as
F1's); delta-conversion math (`calculate_delta_eef` / `delta2abs`); rotation
convention; frame order `[past, current]` (matches
`observation_indices = [-5, 0]` in the config).

**Candidate found by directly comparing against the training pipeline**
(`eval/bridge/diag_policy_wrapper.py`): `lda_policy.py` builds
`history_action` from the **model's own past predictions**, while training
builds it from **real, ground-truth past actions**. Since the checkpoint
has `state_dim: null`, `history_action` is the model's only channel for its
own current state (gripper especially). Growth (exposure-bias) hypothesis
tested and **not supported** for position/rotation
(`diag_policy_wrapper_rollout.py`, 12 episodes × 20 steps) — neither error
grows over the rollout. **Gripper is different**: 0.43 mean error on a
~[0,1]-scale prediction, close to chance, not growing worse over time but
never good either — consistent with a training/deployment mismatch (model
learned to rely on `history_action` being ground truth for gripper state,
which is structurally unavailable at real deployment).

## Caveats

1. **Memorization** — training had no held-out split, like all three
   models.
2. **Units** — normalized space, not directly comparable in magnitude to
   F1/mimic.
3. **n=62, 2 episodes, one run** — no seed repeats yet. The pattern was
   already stable across the four ablation checks above (all drawn from
   overlapping samples), but a fully independent repeat would strengthen
   this further.
4. **Closed-loop success rate for Oracle specifically is not attempted,
   and isn't just an "outstanding" item — it's not a coherent construction**
   for this experiment. Showing a policy the true future *before* it acts
   requires that future to be caused by something other than the policy's
   own action (else it's not really "the future" yet) — the paper's own
   solution is a live human teleoperator (`main_inference_hil.py`), who
   really does cause the future shown to the (non-acting) model, and the
   metric there is action-similarity against the human, not model-driven
   task success. This is unlike Experiment 1, where both conditions
   (baseline foresight vs. ablated/world-model-on) are self-consistent
   counterfactuals a policy can actually act on, so closed-loop success
   rate is the right metric there.
