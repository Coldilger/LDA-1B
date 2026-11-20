# LDA-1B on RoboCasa (PnPCupToDrawerClose, GR1): Experiments 8 and 9

Same paired protocol as Experiment 1 (`paired_baseline_vs_wm_on.md`):
150 episodes, scene seeds 1000–1049 / 2000–2049 / 3000–3049 fixed per
episode, `paired_stats_robocasa.py` (Wilson, stratified bootstrap, exact
McNemar). Server variants in `../server_policy_wm_steps.py` and
`../server_policy_conflict.py`, conditions in `closed_loop.slurm`, jobs in
`jobs_exp9.txt` and `~/slurm_util/queue_done.txt`. The world-model-on mode
(video_gen → inverse dynamics) is our composition of two trained tasks,
not an inference mode of the authors.

## Experiment 9 — compute vs. performance (advisor's point 4), 2026-11-20

The forecast's denoising steps are varied (the action is always denoised
with the checkpoint's 4 steps). Latency: every decision of every episode,
H100 NVL (`LATENCY_MS` in the server logs).

| condition | forecast steps | success | paired diff vs no forecast [95% CI] | vs 4 steps [95% CI] | latency median (p95), ms |
|---|---|---|---|---|---|
| no forecast (authors' policy) | 0 | 67/150 = 44.7% | — | — | 256.9 (258.5) |
| world model on | 1 | 57/150 = 38.0% | −6.7 pp [−16.0, +2.0], p = 0.19 | −3.3 pp [−12.7, +6.0], p = 0.57 | 333.8 (336.1) |
| world model on (Exp 1) | 4 | 62/150 = 41.3% | −3.3 pp [−12.7, +6.0], p = 0.58 | — | 430.1 (432.3) |
| world model on | 16 | 63/150 = 42.0% | −2.7 pp [−12.0, +6.0], p = 0.68 | +0.7 pp [−8.0, +9.3], p = 1.00 | 802.6 (806.2) |

Latencies of the 0- and 4-step rows come from 5-episode timing runs on the
same GPU type (`timevg0`, `timevg4`; 304 decisions each); their success
numbers are the 150-episode Experiment 1 runs.

**What this shows.** Generating the forecast costs 77–546 ms per decision
(+30% to +212%) and no amount of it changes success measurably; every
world-model-on variant is within noise of the policy that does not forecast
at all. Consistent with Experiment 1's internals (the action head barely
reacts to the forecast slot).

## Experiment 8 — instruction–foresight conflict (advisor's point 7), 2026-11-20

Design as in F1-VLA's `eval/bridge/experiment8_conflict`. The instruction
reaches LDA's action head directly (the Qwen-VL features of image +
instruction are the MM-DiT's text tokens) and through the forecast
(video_gen is conditioned on the same features). A = the env's
instruction, "unlocked_waist: pick up the cup, place it into the drawer and
close the drawer"; B keeps the prefix and changes only the task words.
Forecast seed and action-noise seed are shared by all cells of a decision.

### Single decisions (1,219 decisions, 20 episodes, seed base 1000)

`conflict_records/probe_1000_777251.jsonl`; `analyze_conflict.py`
(bootstrap over decisions). Noise floor: another forecast seed under A
changes the forecast by 30% and the action by d = 0.0019 [0.0016, 0.0024].

| B | forecast change A→B | d(BB, AA) | d(AB, AA) | d(BA, BB) | follow(AB) | follow(BA) |
|---|---|---|---|---|---|---|
| "…pick up the **can**, place it into the drawer and close the drawer" | 0.165 | 0.099 | 0.0017 | 0.0019 | 0.014 [0.013, 0.016] | 0.016 [0.015, 0.017] |
| "…close the drawer" | 0.128 | 0.040 | 0.0012 | 0.0014 | 0.029 [0.027, 0.030] | 0.030 [0.029, 0.032] |

(d in normalised action units, mean over the 16-step chunk. A first smoke
run whose B lacked the env's prefix is kept apart in
`conflict_records/smoke_noprefix/` and not used.)

### Closed loop (B = can, 150 episodes per cell)

| text read by the action head | forecast generated under | success (task A) | vs text A / forecast A [95% CI] |
|---|---|---|---|
| A | A (= world model on, Exp 1) | 62/150 = 41.3% | — |
| A | B | 62/150 = 41.3% | +0.0 pp [−8.7, +8.7], p = 1.00 |
| B | A | 36/150 = 24.0% | −17.3 pp [−27.3, −7.3], p = 0.0016 |
| B | B | 39/150 = 26.0% | −15.3 pp [−24.7, −6.0], p = 0.0032 |

text B / forecast B → text B / forecast A: −2.0 pp [−8.0, +4.0], p = 0.66.

**What this shows.** The text the action head reads decides the behaviour:
changing it to "can" costs ~16 pp of cup-task success whichever forecast
is used, and swapping the forecast changes nothing (single decisions: a
forecast made under B moves the action as much as a re-drawn forecast
under A; follow 0.01–0.03). The same picture as F1-VLA, the opposite of
mimic-video. RoboCasa scores only task A, so the closed loop shows that A
is carried out less often under B, not where the robot goes instead.
