# Experiment 1 — Ablation of the world-model signal (LDA-1B)

**Status: done, real closed-loop (2026-08-19). Turning the world-model path
on does not help — 44% (0.44, 50 episodes) vs. the confirmed 48%
policy-only baseline, both via RoboCasa (Bridge is still 0%, see
`../RESULTS.md` and `../../robocasa/RESULTS.md`'s camera-viewpoint
decision).**

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

## Metric: real closed-loop success rate, not offline/live L1

Corrected mid-session (2026-08-19) after building the wrong thing first: an
offline or live L1 probe (predicted action vs. the action actually taken)
is at best a fast first pass, never Experiment 1's decisive number — same
reasoning F1-VLA's and mimic-video's own copies of this experiment already
follow (their "real closed-loop" result is explicitly the one that
matters, not the offline probe). L1 has a memorization confound: a model
can reproduce "the right" action because it memorized the trajectory, not
because it's genuinely using the injected/imagined signal. The only way to
actually test whether turning the world-model path on helps is to let it
really drive the robot for full episodes and measure task success.

## Results

No static RoboCasa dataset is available on this cluster (see
`../../robocasa/server_policy_oracle_probe.py`'s docstring), and Bridge is
still 0% (nothing to compare against there either) — so, per the priority
shift in `../EXPERIMENTS.md`, this ran against the confirmed-working
RoboCasa checkpoint instead, same task as Experiment 3's own LDA-1B row.

**Real closed-loop (decisive):** `server_policy_worldmodel_on.py` — every
decision calls `video_gen()` then feeds that imagined frame into
`inverse_dynamics` via `predict_action(..., inverse_dynamics_next_obs_tokens=...)`,
and *that* action is what actually drives the robot for the whole episode
(no side-channel, no comparison against a buffered baseline — the
world-model path is the real policy for this run).

| stage | n episodes | success rate |
|---|---|---|
| smoke (job 631219) | 6 | 50.0% (3/6) — too small to trust on its own |
| **full** (job 631244) | **50** | **44.0%** |
| baseline (policy-only, confirmed) | 50 | 48.0% |

**Turning the world-model path on does not help — if anything, slightly
hurts (-4pp).** Given this thesis's own established noise floor ("two
models/conditions differing by less than ~10pp cannot be separated at this
sample size," from F1-VLA's `RESULTS.md`), a single 50-episode run at -4pp
is **not distinguishable from no effect** — this reads as "the dormant
pathway is inert at inference," not "it actively hurts," until repeated
with more seeds.

**Preliminary/live L1 probe (not decisive, kept for the record):**
`server_policy_oracle_probe.py` ran alongside this (same job family,
different script) and computed live L1 against the policy's own action at
each step, for both this world-model-on condition and Experiment 2's
oracle condition. n=181 (3 episodes): oracle L1 ≈0.900, world-model L1
≈0.899, both *worse* than a trivial zero-action baseline (≈0.749) — an
unresolved anomaly (real oracle input performing worse than "predict
nothing" is not expected), not yet debugged, and explicitly not to be
read as a real finding until it is. See `../experiment2_oracle/ORACLE_EXPERIMENT.md`.

## Not yet done

- [x] Move/re-verify the oracle-style `inverse_dynamics` hook in the
      correct file (`MMDiT_ActionHeader.py` / `QwenMMDiT.py`).
- [x] Wire up a two-stage call: `video_gen` to predict a future frame, then
      `inverse_dynamics` fed that self-generated frame.
- [x] Run real closed-loop, compare against the default `policy`-path
      baseline (RoboCasa, not Bridge — see "Results" above).
- [ ] Debug the L1 probe anomaly (oracle/world-model L1 worse than the
      trivial zero-action baseline) before citing those numbers anywhere.
- [ ] Repeat the closed-loop run with more seeds before treating -4pp as
      more than noise.
