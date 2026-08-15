# Experiment 3 — Cost per decision (LDA-1B)

**Status: planned, not yet run.**

## What this tests

Context for interpreting Experiments 1 and 2: how much inference-time
compute does each model actually spend on its world-model computation, and
at what latency / success-rate trade-off?

| Category | Model | World-model compute @ inference | Latency / control step (median · p95) | Success rate, SimplerEnv-Bridge (95% CI) |
|---|---|---|---|---|
| 1 | F1-VLA | VAR foresight loop, re-run every control step | TBD | TBD |
| 2 | mimic-video | One video-backbone forward pass per action chunk (amortised over the chunk) | TBD | TBD |
| 3 | **LDA-1B** | **None** beyond the shared MM-DiT — the visual-forecasting head is a training-time co-objective, unused at inference | TBD | TBD |

LDA-1B is the cheap baseline in this comparison by construction: the
`policy` inference path (the only one used by default) never runs
`video_gen`/`inverse_dynamics` at all, so there's no extra world-model
compute to measure beyond the shared backbone every path uses anyway. This
makes LDA's success-rate column the more interesting number here — cheap
compute is only informative alongside whether it actually works
(`RESULTS.md` currently: 0% on real closed-loop, root cause under
investigation, v3 retrain in progress).

## Not yet done

- [ ] Instrument `lda_policy.py`'s `step()`/`_predict_chunk()` to record
      per-step wall-clock latency.
- [ ] Run on real hardware across a real SimplerEnv-Bridge eval sweep once
      the v3 checkpoint gets a non-zero success rate, report median/p95.
- [ ] Cross-reference against `RESULTS.md`.
