# Experiment 3 — Cost per decision (LDA-1B)

**Status: latency measured (2026-08-19), via RoboCasa rather than Bridge —
see "Results" below. The Bridge-specific success-rate cell stays 0% until
the closed-loop investigation in `RESULTS.md` resolves.**

## What this tests

Context for interpreting Experiments 1 and 2: how much inference-time
compute does each model actually spend on its world-model computation, and
at what latency / success-rate trade-off?

| Category | Model | World-model compute @ inference | Latency / control step (median · p95) | Success rate, SimplerEnv-Bridge |
|---|---|---|---|---|
| 1 | F1-VLA | VAR foresight loop, re-run every control step | 215.7 ms · 263.2 ms | 48.6% (3-seed × 4-task average, see F1-VLA's `RESULTS.md`) |
| 2 | mimic-video | One video-backbone forward pass per action chunk (amortised over the chunk) | 10226.9 ms · 10416.5 ms | 11.5% (baseline at the matched `stop=23` setting, see mimic-video's `experiment1_ablation/README.md`) |
| 3 | **LDA-1B** | **None** beyond the shared MM-DiT — the visual-forecasting head is a training-time co-objective, unused at inference | 254.7 ms · 256.8 ms (RoboCasa checkpoint — see caveat below) | 0% (0/12, still under investigation — see `RESULTS.md`) |

LDA-1B is the cheap baseline in this comparison by construction: the
`policy` inference path (the only one used by default) never runs
`video_gen`/`inverse_dynamics` at all, so there's no extra world-model
compute to measure beyond the shared backbone every path uses anyway. This
makes LDA's success-rate column the more interesting number here — cheap
compute is only informative alongside whether it actually works.

## Results (2026-08-19, via RoboCasa)

Measured against LDA-1B's own confirmed-working RoboCasa checkpoint
(`LDA-robocasa.pt`, 48% on the authors' own benchmark — see
`../../robocasa/RESULTS.md`) rather than Bridge, which is still 0%
closed-loop and has nothing to time yet. This is a legitimate substitute
specifically for *this* experiment: Exp3 asks which computations the
architecture runs at inference, a property of the model/checkpoint's
weights and code path, not of which eval suite is pointed at it — the same
`predict_action` call runs regardless of task. What it does *not* give is a
Bridge-specific number; if a future Bridge checkpoint runs meaningfully
more (or less) compute per decision than this one, this measurement
wouldn't catch that.

Instrumented `deployment.model_server.tools.websocket_policy_server`'s
`policy.predict_action` call directly (`eval/robocasa/server_policy_timed.py`,
a thin wrapper — `server_policy.py`, `websocket_policy_server.py`, and
`base_framework` itself are untouched), timing every real websocket `infer`
request. Unlike Bridge's SimplerEnv wrapper (`lda_policy.py`'s `step()`),
no gating on "is this actually a replan" was needed: RoboCasa's client
already does its own action-chunking (`--args.n_action_steps 12`) and only
sends a request when it needs a new chunk, so every request timed here
already is one real per-decision cost.

3 episodes, `PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env`, job 631084:

| n (replans) | median | p95 | mean | min | max |
|---|---|---|---|---|---|
| 182 | 254.7 ms | 256.8 ms | 279.2 ms | 175.8 ms | 4818.8 ms |

Cheapest of the three models by a wide margin on the metric this experiment
actually measures (median ~40x faster than F1-VLA, ~40x faster than
mimic-video), consistent with the "no extra world-model compute" structural
claim above — there's nothing here for an ablation to even remove. The
33% success rate incidentally observed on these same 3 episodes is **not**
a new RoboCasa result — n=3 is far too small (the eval/robocasa/RESULTS.md
smoke run showed exactly this kind of swing, 67% on 6 episodes vs. the
real 48% on 50) — the confirmed number stays the 48%/50-episode one.

## Not yet done

- [x] Instrument the real inference call to record per-decision wall-clock
      latency (`server_policy_timed.py`, done via RoboCasa's websocket
      server rather than `lda_policy.py`'s `step()`, since Bridge has
      nothing working to instrument yet).
- [ ] Re-measure once a working Bridge checkpoint exists, to get a genuine
      Bridge-specific number (architecturally should match, per the
      "Results" caveat above, but not yet directly confirmed).
- [x] Cross-reference against `RESULTS.md` and `../../robocasa/RESULTS.md`.
