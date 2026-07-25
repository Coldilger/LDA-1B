# LDA-1B on SimplerEnv Bridge — setup, findings, and eval log

Companion to `F1-VLA/eval/bridge/RESULTS.md`. Same benchmark, same protocol, so
the two (and mimic-video) land in one table. Everything below is measured unless
explicitly marked as pending.

**Status: finetune running, closed-loop eval not yet run.** The success-rate table
at the bottom is empty on purpose.

## The question this run answers

Whether LDA-1B, finetuned on BridgeData V2, matches F1-VLA and mimic-video on the
four SimplerEnv WidowX tasks. The comparison is head-to-head and fair: identical
finetuning dataset, identical evaluator, identical 4-task x 24-episode x 3-seed
protocol. What it does *not* isolate is *why* the numbers differ — pretraining
corpora differ between the three models, so a gap cannot be attributed to
architecture alone.

## Did pretraining already know Bridge? No.

The released `LDA-pretrain` mixes in `open-x-embodiment` under embodiment tag
`oxe` (index 5), which made it tempting to assume BridgeData was already in there
and that the finetune would only need to polish an existing WidowX projector.
That assumption was wrong, and reading configs could not settle it — measurement
could.

Open-loop probe (`lda/eval/eval_bridge_openloop.py`), 12 held-out Bridge
trajectories, left arm only, tag `oxe`:

| metric | value | reading |
|---|---|---|
| position L1 | 0.051 m (sd 0.009) | — |
| "arm never moves" baseline | 0.046 m | model is *worse* than not moving |
| model / baseline | 1.24 (sd 0.50) | beats the baseline on 3 of 12 trajectories |
| rotation L1 | 0.147 rad | ~8 degrees |
| gripper L1 | 0.413 | 0.5 is chance; both polarities checked |
| right-arm motion predicted | 1.000, every trajectory | WidowX has no right arm |

The last row is the most telling and is independent of any normalization
question: the `oxe` slot emits bimanual output for a single-arm robot. So the
finetune is teaching this embodiment slot Bridge more or less from scratch, and
the honest reading of the eventual number is "how fast LDA adapts to a new
embodiment", not "how much its pretraining helped".

`oxe` is still the right tag to finetune into: it is an in-range trained
embodiment. `new_embodiment` (index 32) is out of bounds for this checkpoint's
32-slot embodiment tables and faults on the first forward pass — which is also
why the authors' own `demo_data` cannot smoke-test this checkpoint.

## Bugs found, and what each would have cost

### 1. Normalization statistics computed over the wrong quantity (ours)

The converter wrote `action.*_eef_position/rotation` statistics from the stored
**absolute poses**, but the loader normalizes what `calculate_delta_eef` derives
from them — a ~0.3 m spread standing in for ~0.009 m of real per-step motion.

Caught because the probe's numbers were implausible rather than merely bad:
position error 0.576 m, near-constant at 0.540–0.612 across 12 trajectories whose
own motion varied fourfold. **Error that ignores what the robot is doing is the
signature of a scale bug, not of a model that does not know the data.**

After the fix, q99 for `action.left_eef_position` went from
`[0.453, 0.236, 0.193]` to `[0.029, 0.039, 0.040]` — delta scale, symmetric about
zero — and the probe's position error fell 11x to 0.051 m.

This would have hurt training too, and silently: targets squeezed into a sliver
of `[-1, 1]`, loss curve looking perfectly healthy, final policy inexplicably
poor. `lda/dataloader/calculate_delta_min_max.py` exists for exactly this
computation.

### 2. `eval_wo_postprocess.py`'s generic path was never executed upstream

Its bimanual branch reads `modality_keys` without ever assigning it
(`UnboundLocalError` on the first trajectory), and compares denormalized *delta*
predictions against raw *absolute* ground truth. The correct harness for
`use_delta_action: true` is `eval_relative_eef.py`, which maps predicted deltas
back to absolute poses via `delta2abs` and whose `calculate_mse` has a correct
`D == 14` branch (`pos [0,1,2,7,8,9]`, `gripper [6,13]`).

### 3. `eval_policy.py`'s `evaluation.trajs` counts datasets, not trajectories

On a single-dataset mix it evaluates exactly one trajectory regardless of the
value. The first "decisive" reading came from n=1; per-trajectory variance turned
out to be large, so the probe was rewritten to loop properly.

### 4. `mse_score` in the training loop compares incompatible quantities

`train_LDA.py:400-417` scores `predict_action`'s `normalized_actions` against
`example["action"]`, but that label comes from `raw_data` — absolute poses, before
normalization and before the delta conversion. Normalized predictions against raw
absolute labels, the same mismatch as bug 2.

Practical consequence: the W&B `mse_score` curve rose steadily through the first
12k steps, which reads as "the model is getting worse" and is not evidence of
anything. Ignore that panel.

What the model is actually doing has to be read off the training loss, and only
after smoothing: a single diffusion-loss sample has sd 0.10 against a mean of
0.27, so the raw curve looks like flat noise. Averaged in quarters over the first
415 logged values:

| | Q1 | Q2 | Q3 | Q4 | change |
|---|---|---|---|---|---|
| `action_dit_loss` | 0.2999 | 0.2683 | 0.2659 | 0.2415 | −24% |
| total loss | 0.6744 | 0.5924 | 0.5866 | 0.5295 | −21% |

`dynamics_loss` stays near flat (0.046 → 0.041), which fits: the visual-forecasting
head is far less embodiment-specific than the action head.

Sanity check on the freeze set: 2.23B of 6.69B parameters are trainable, i.e. the
VLM is frozen as intended and the MM-DiT plus heads are not.

### 5. Launch-path breakage

`accelerate` resolved to a Python 3.9 install in `~/.local` without DeepSpeed;
`deepspeed_zero2.yaml` pointed at `/mnt/home/liukai/...` on the authors' machine;
`ds_config.yaml` hardcoded a batch contradicting the run's.

## Conversion: three things that had to be right

Source is `IPEC-COMMUNITY/bridge_orig_lerobot` (53,192 episodes / 1,893,026
frames / 5 fps / 256x256). Target layout is pinned by `OxeDataConfig`, which is
`BaseDataConfig` verbatim: a 14-dim compact bimanual layout, right arm zero-filled.

1. **The gripper is at index 7 of `observation.state`, not 6.** Index 6 is a pad
   column that is identically zero. `state[:7]` — the obvious reading of "7-dof
   arm" — feeds the model a constant zero where the gripper belongs.
2. **`action.*_eef_*` must hold absolute poses.** With `use_delta_action: true`
   the loader differentiates them itself, in the gripper frame. Bridge's `action`
   column is already a delta, but a **world-frame** one: verified that
   `action_t[0:3] == state_{t+1}[0:3] - state_t[0:3]` to machine precision. Passing
   it through would differentiate twice *and* mix two frames.
3. **Zero-filled right-arm columns are safe** — the q99 normalizer masks the
   `q01 == q99` case rather than dividing by zero.

## Known asymmetry vs F1-VLA

LDA trains on **38,660 episodes**; F1-VLA trained on **all 53,192**
(`Num examples = 1,893,026` in its training log). The 14,532-episode difference is
Bridge episodes whose language instruction is an empty string, which LDA's loader
drops and F1's did not.

Neither choice is wrong — F1 fed empty strings as the task, LDA skips those
episodes — but it is a real difference in training data and it disadvantages LDA
by roughly a quarter of the corpus. If LDA wins anyway the conclusion is stronger;
if it loses, this belongs in the footnote.

## Configuration

Architecture is copied from the released checkpoint's own `config.yaml`, not from
this repo's `LDA_pretrain.yaml` — the latter describes an older model
(QwenGR00T + Qwen2.5-VL + dinov2, `action_dim` 29) that would not match the weights.

Training hyperparameters follow the paper's Table V:

| Table V | ours | note |
|---|---|---|
| batch 12*8 (finetune) | 16 x 6 accumulation | 12 GPUs x 8/device = effective 96; we have one GPU |
| LR 1e-4 | 1e-4 | `qwen_vl_interface` is frozen, so every trainable parameter is in `action_model` |
| weight decay 1e-5 | 1e-5 | the trainer reads `trainer.optimizer.weight_decay`; the sibling `trainer.weight_decay` key is inert |
| betas, eps, cosine w/ min lr | same | |
| Hidden Size 1536 | config says `hidden_size: 2560` | **not a mismatch** — see below |
| Action Chunk 16 | 16 | |

The apparent hidden-size conflict is two different quantities sharing a name.
Checkpoint tensors show MM-DiT blocks 1536 wide with cross-attention keys/values
taking 2560 from the VLM side: 1536 is the DiT width (set by
`action_model_type: DiT-L`), while the config's `hidden_size: 2560` is the
category-MLP width. Setting it to 1536 "per the paper" would break weight loading.

**`max_train_steps` counts micro-batches, not optimizer steps.** Under DeepSpeed
the engine owns accumulation, so `accelerator.sync_gradients` is true every
iteration and `completed_steps` ticks per micro-batch (`train_LDA.py:356-358`).
The effective batch is still 96. Budget: 150,000 micro-batches = 2.4M samples over
a 447,733-step dataset, ~5.4 epochs (25,000 real optimizer steps). Measured
throughput on one H100 is 2.55 micro-batches/s, so ~3 six-hour waves.

## Eval harness

`eval/bridge/lda_policy.py` carries over the fixes F1-VLA paid for in measured
success rate: gripper binarized to `[-1, +1]`, rotation euler -> axis-angle, frame
squared to 256x256 before the model sees it. F1's fourth fix (episode-relative
state rotation) has no analogue here — this checkpoint has `state_dim: null`, so
no proprioception reaches the policy.

The new piece is the frame conversion. LDA predicts deltas in the **gripper's own
frame**; the widowx controller wants **base-frame** deltas. The
alignment-to-frame-0 inside `calculate_delta_eef` cancels algebraically, so those
deltas carry no frame of their own — integrating them from the robot's live pose
via `delta2abs` yields absolute poses in SimplerEnv's base frame, and differencing
consecutive poses gives what the controller wants. Verified numerically: the
round-trip is exact, and seeding from a different initial pose preserves per-step
motion magnitude to 0.0, i.e. the trajectory is rigidly rotated, not distorted.

Every run gets a unique `--additional-env-save-tags`. The evaluator silently skips
episodes whose output video already exists, so without it a rerun reports a number
nothing measured — this produced a phantom 0.0% during the F1-VLA work.

## Results

Pending. Protocol: 4 tasks x 24 episodes x 3 seeds, means of three seeds, via
`slurm/submit_eval_sweep.sh`.

| task | LDA-1B | F1-VLA | mimic-video | paper (F1) |
|---|---|---|---|---|
| Put Carrot on Plate | — | 38.9% | — | 70.8% |
| Put Spoon on Towel | — | 47.2% | — | 50.0% |
| Stack Green Cube | — | 37.5% | — | 50.0% |
| Put Eggplant in Basket | — | 69.4% | — | 66.7% |
| **average** | — | **48.2%** | — | **59.4%** |

F1-VLA column is from `F1-VLA/eval/bridge/RESULTS.md` (3-seed means, `chunk_size: 4`
checkpoint). Fill the mimic-video column from `mimic-video-project` before
publishing any comparison.
