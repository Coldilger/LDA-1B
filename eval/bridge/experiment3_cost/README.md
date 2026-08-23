# Experiment 3 — Cost per decision (LDA-1B)

**Status: latency measured (2026-08-19), via RoboCasa rather than Bridge —
see "Results" below. The Bridge-specific success-rate cell stays 0% until
the closed-loop investigation in `RESULTS.md` resolves. Updated 2026-08-23
with the world-model-on ("with foresight") cost, previously never
instrumented — see "Update 2026-08-23" below: turning the dormant path on
costs ~68% more per decision (428.2ms vs. 254.7ms median).**

## What this tests

Context for interpreting Experiments 1 and 2: how much inference-time
compute does each model actually spend on its world-model computation, and
at what latency / success-rate trade-off?

| Category | Model | World-model compute @ inference | Latency / control step (median · p95) | Success rate, SimplerEnv-Bridge |
|---|---|---|---|---|
| 1 | F1-VLA | VAR foresight loop, re-run every control step | 215.7 ms · 263.2 ms | 48.6% (3-seed × 4-task average, see F1-VLA's `RESULTS.md`) |
| 2 | mimic-video | One video-backbone forward pass per action chunk (amortised over the chunk) | 10226.9 ms · 10416.5 ms | 50.0% (July best-of-sweep over `--vam-stop-video-denoising-step`; the fixed `stop=23` used for Experiment 1's own matched comparison measured 11.5% — see mimic-video's `experiment1_ablation/README.md`) |
| 3 | **LDA-1B** | **None by default** — dormant unless explicitly activated; costs ~68% more (428.2ms vs. 254.7ms median) when turned on, see "Update 2026-08-23" | 254.7 ms · 256.8 ms default / **428.2 ms · 432.2 ms with foresight on** (RoboCasa checkpoint — see caveat below) | 0% (0/12, still under investigation — see `RESULTS.md`) |

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

## Update 2026-08-23 — world-model-on cost: the missing "with" number

Everything above measured the *default* path only — `policy`, world-model
dormant. There was never a timing number for the *other* condition Exp1
already tests (activating `video_gen → inverse_dynamics`, 44% vs. 48%
baseline) — `server_policy_worldmodel_on.py` existed for that closed-loop
success-rate run but had zero timing instrumentation. Added
`server_policy_worldmodel_on_timed.py` (purely compositional: reuses
`add_worldmodel_on` and `add_timing` from the two existing scripts
unmodified, wraps the whole `video_gen`-then-`inverse_dynamics` call).

Same task/protocol as the 254.7ms measurement above, job 634482
(`CLIENT_EXIT=0`):

| n (replans) | median | p95 | mean | min | max |
|---|---|---|---|---|---|
| 182 | 428.2 ms | 432.2 ms | 441.0 ms | 349.0 ms | 2579.9 ms |

**Foresight costs ~173ms, ~68% more than the default path (254.7ms → 428.2ms
median).** The `max` (2579.9ms) is a one-time first-call cost (model/CUDA
warmup right after the eval process starts, not a recurring per-decision
tax) — same pattern documented for F1-VLA's and mimic-video's own timing
runs; median/p95 are the numbers to cite. LDA-1B is still the cheapest of
the three models with foresight *on* (428.2ms vs. F1-VLA's 215.7ms and
mimic-video's 1060ms) — but no longer "no extra cost at all": the dormant
path is only free because it's dormant by default, not because activating
it is free.

## Not yet done

- [x] Instrument the real inference call to record per-decision wall-clock
      latency (`server_policy_timed.py`, done via RoboCasa's websocket
      server rather than `lda_policy.py`'s `step()`, since Bridge has
      nothing working to instrument yet).
- [x] World-model-on ("with foresight") cost — see "Update 2026-08-23" above.
- [ ] Re-measure once a working Bridge checkpoint exists, to get a genuine
      Bridge-specific number (architecturally should match, per the
      "Results" caveat above, but not yet directly confirmed).
- [x] Cross-reference against `RESULTS.md` and `../../robocasa/RESULTS.md`.
