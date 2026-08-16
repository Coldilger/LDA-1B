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

## Closed-loop bring-up (checkpoints 50k / 70k, mid-training)

The smoke run was deliberately done on intermediate weights: its job is to find
wiring bugs, and those do not care how good the policy is. Three surfaced, each
only visible once the whole loop ran, and each hidden behind the previous one:

1. The saved config stores the VLM path relative to the repo root, while the eval
   runs from SimplerEnv's directory; huggingface_hub then read it as a Hub repo id.
2. `lda_eval` had picked up numpy 2, under which transforms3d's
   `np.array(..., copy=False)` raises and every episode dies in `quat2euler`.
3. The wrapper fed 256x256, but `predict_action` builds `curr_imgs` — the tensor
   reaching DINOv3 — from the raw image *before* `resize_images`. Training always
   delivered 224 (a 14x14 token grid, the paper's stated latent shape). Feeding 256
   raises nothing and simply produces features the model never trained on.

### The pipeline is correct; the model was not ready

Both runs scored 0.0% with `moved_correct_obj` false on all 1440 steps, so the
question was whether the loop was broken or the policy merely undertrained. Three
measurements settle it:

| check | result | reference |
|---|---|---|
| motion reaching the controller | 0.00965 m/step | Bridge's own ~0.009 |
| arm motion in the rendered rollout | 1.19 per frame, 3.29 end-to-end | F1 at 38.9% success: 2.31 / 7.49 |
| episodes actually run | 24, none skipped | — |

The arm moves, at roughly the right per-step scale, about half as decisively as a
policy that succeeds ~39% of the time. That is an undertrained policy whose step
directions partly cancel, not a broken conversion.

### Finetuning is measurably working

The same open-loop probe used on the pretrained checkpoint, rerun on 70k steps
(47% of the run) — identical trajectories, seed and protocol:

| | pretrained | 70k steps |
|---|---|---|
| position L1 | 0.051 m | **0.0384 m** |
| **ratio to stay-still baseline** | **1.24** | **0.837** |
| rotation L1 | 0.147 rad | 0.101 rad |
| gripper L1 | 0.413 | 0.398 |

Crossing 1.0 on that ratio is the threshold that separates "knows something" from
"knows nothing": the pretrained model was worse than an arm that never moves, and
the finetuned one is better. Halfway through training.

### Open concern: the gripper head is not learning

Gripper L1 moved 0.413 -> 0.398 against 0.5 for a coin flip. A dedicated test over
40 frames with differing ground truth agrees: 45% as-is, 55% inverted, against an
**80% always-open baseline** — uncorrelated, not merely flipped, so `invert_gripper`
stays off.

For pick-and-place this matters more than reaching accuracy: a gripper that never
closes on cue caps success at zero however well the arm is aimed.

Four candidate explanations were tested, and three are ruled out.

**Loss masking (real bug, fixed).** `pad_action_state_with_key` decided whether a
timestep entered the loss with `not np.all(action_state[i] == 0)`. For a binary
gripper, 0.0 is the *close* command, not padding, so every closed-gripper step was
dropped — 41% of the supervision and all of one class, leaving a head that could
only learn "open". The same rule also dropped any exactly-stationary motion step.
Fixed by testing the modality across the whole trajectory instead, which still
masks out genuinely absent modalities such as WidowX's zero-filled right arm (the
`single_arm` shortcut does not cover it: that fires only for the franka and ur
tags, not oxe). Verified live in the training path — the loss now sees 7 valid
dims of 138, gripper included on every step.

**Gradient dilution — ruled out.** With the mask corrected the gripper is 1 of 7
supervised dimensions, not 1 of 138.

**Wrong target — ruled out.** The value the model trains against is exactly ±1;
q99 normalization is applied.

**Sampling resolution — ruled out.** A binary target under only 4 denoising steps
could plausibly collapse toward the midpoint, which is what the predictions look
like. Raising the step count makes it marginally *worse*, not better:

| denoising steps | correlation | class separation |
|---|---|---|
| 4 (config default) | +0.104 | +0.095 |
| 20 | +0.048 | +0.045 |
| 50 | +0.038 | +0.036 |

**What remains is training difficulty**, and there is a structural reason it is
harder here than for F1-VLA: this checkpoint has `state_dim: null`, so no
proprioception reaches the policy at all. The model cannot be told its own gripper
state — it has to read it out of a 224px frame while also judging whether the
fingers are around the object. F1 was handed gripper state directly. Position and
rotation, which need no such readout, improved normally over the same steps.

Measured with balanced classes (both trivial baselines then sit at 50%):

| checkpoint | n | correlation | accuracy |
|---|---|---|---|
| 70k (pre-fix) | 40 | +0.091 | 55.0% |
| 90k (10k post-fix) | 40 | −0.124 | 42.5% |
| 100k (20k post-fix) | 200 | +0.104 | 55.5% |

At n=40 the standard error is ~0.16, so the first two say nothing; the n=200
reading is ~1.5 SE from zero. A weak positive signal that is not yet growing.

## Results

Protocol: 4 tasks x 24 episodes x 3 seeds, means of three seeds, via
`slurm/submit_eval_sweep.sh`. Run 2026-08-16 on the v3 checkpoint
(`outputs/lda_bridge_v3/bridge_finetune_v3/final_model/pytorch_model.pt`,
full 150,000-step training, real gripper history + `state_dim: 14` fix —
see `LDA_bridge_v3.yaml`'s header and the eval harness section above for
what v3 changed relative to earlier checkpoints).

| task | LDA-1B | F1-VLA | mimic-video | paper (F1) |
|---|---|---|---|---|
| Put Carrot on Plate | **0.0%** | 38.9% | 41.7% | 70.8% |
| Put Spoon on Towel | **0.0%** | 47.2% | 45.8% | 50.0% |
| Stack Green Cube | **0.0%** | 37.5% | 16.7% | 50.0% |
| Put Eggplant in Basket | **0.0%** | 69.4% | 95.8% | 66.7% |
| **average** | **0.0%** | **48.2%** | **50.0%** | **59.4%** |

**0 of 12 runs (all 4 tasks x 3 seeds) succeeded.** This is not an eval-harness
bug: all 12 jobs completed cleanly (`EVAL_EXIT=0`, 24 episodes each, none
skipped), and the config-loading path was checked directly —
`read_mode_config` resolves `run_dir` from the checkpoint path via
`parents[1]`, which for `final_model/pytorch_model.pt` correctly lands on
`bridge_finetune_v3/config.yaml` (`state_dim: 14`, confirmed present); a
resolution failure there raises `AssertionError` and the jobs would not have
completed. So the v3 checkpoint's real-proprioception fix *was* exercised in
this eval, and it still scores zero.

Per-episode stats show the arm is not inert — job 628404 (carrot, seed 0)
logged `moved_correct_obj: True` with `is_src_obj_grasped` toggling true/false
across the rollout, i.e. some interaction happens — but `src_on_target` never
triggered on any of the 288 episodes run (24 episodes x 12 jobs).

### Re-diagnosing the gripper head at v3/150k: it is NOT the bottleneck anymore

The obvious hypothesis was the same gripper-head weakness already flagged
above (55.5% accuracy at 100k steps, barely above chance). Re-running
`diag_gripper.py --with-history` against the v3/150k checkpoint (n=200,
balanced open/closed, 2026-08-16) refutes it:

| checkpoint | n | correlation | accuracy | separation |
|---|---|---|---|---|
| 100k (pre-fix, no real state) | 200 | +0.104 | 55.5% | — |
| **v3/150k (real state, this run)** | **200** | **+0.849** | **92.0%** | **+0.828** |

That required fixing `diag_gripper.py` itself first: it never passed a
`state` field, which crashes checkpoints with `state_dim: not null`
(`MMDiT_ActionHeader.predict_action` calls `self.state_encoder(state, ...)`
unconditionally whenever the *model* has a configured `state_dim`, regardless
of what's passed — `state=None` reaches `torch.bmm` and throws). Fixed by
passing `data["state"]` straight through — already in the model's expected
`[1, state_dim]` shape via `get_step_data_with_transform`'s own transform
pipeline (the same one training used), no closed-loop-style reference-frame
correction needed for a single independent offline step.

**This flips the standing hypothesis.** Given real proprioceptive state, the
v3 gripper head discriminates open/closed cleanly — nowhere near the
"undertrained head" story that explained the pre-v3 zero. The closed-loop
zero must have a different cause. The prime suspect now is a mismatch between
*this* probe's state (read directly from the logged dataset) and what
closed-loop eval actually feeds the model: `lda_policy.py`'s `_build_state()`
reconstructs state from the live simulator at each step (position passthrough,
rotation *relative to a reference captured at episode start*), a
hand-rebuilt path this probe never exercises. If that reconstruction diverges
from the format the model was trained on — even subtly, e.g. in the reference
frame, units, or which columns are populated — every closed-loop episode
would be conditioning the gripper (and everything else) on a state the model
has never actually seen, while this offline probe's ground-truth state looks
correct in isolation.

### The reference-frame suspicion, checked directly — rotation is fine, position is not

`_build_state()`'s rotation reference (episode-start pose in whichever
simulator/robot is running) is only equivalent to Bridge's own convention if
real Bridge trajectories also happen to start at a consistent pose — i.e. if
raw rotation at t=0 is tightly clustered across trajectories. Checked with a
new offline diagnostic, `diag_state_reference.py` (no model forward pass —
just reads `state.left_eef_rotation` at t=0 straight from the converted
dataset), n=60 trajectories, 2026-08-16:

| | roll | pitch | yaw |
|---|---|---|---|
| t=0 rotation, mean (rad) | 0.005 | −0.088 | 0.074 |
| t=0 rotation, std (rad) | 0.086 | 0.164 | **0.379** |
| \|rotation change\|, t=0→mid-trajectory, mean (rad) | 0.059 | 0.119 | 0.239 |

Bridge trajectories do NOT start from one fixed pose — the spread at t=0
(std up to 0.38 rad / ~22° on yaw) is comparable to the typical
within-trajectory rotation signal itself (0.24 rad by mid-trajectory). This
looked like a strong lead. But **F1-VLA uses the exact same
episode-start-reference approximation** (`f1_vla_policy.py`'s `_build_state`,
copied near-verbatim, including F1's own note that Bridge's rotation has
"|angle| < 1.6" i.e. real, non-trivial spread) and still gets ~48% average
closed-loop success, not 0% — so this approximation being imperfect can't be
the whole story on its own, and needed a direct check rather than an
inference from the offline dataset alone.

Built `diag_live_state_range.py`: runs the real v3 model through actual
closed-loop SimplerEnv episodes (real predicted actions, not idle), with
`_build_state()` monkey-patched to log every value it actually produces.
360 real steps across 6 Carrot episodes, 2026-08-16:

| col | min | max | mean | std | frac(\|x\|>0.9) |
|---|---|---|---|---|---|
| left_x | **−1.000** | 0.045 | −0.545 | 0.327 | **22.5%** |
| left_y | −0.340 | 0.331 | −0.112 | 0.164 | 0.0% |
| left_z | −0.421 | **1.000** | 0.443 | 0.441 | **19.2%** |
| left_roll | −0.204 | 0.322 | 0.067 | 0.149 | 0.0% |
| left_pitch | 0.027 | 1.000 | 0.560 | 0.240 | 8.3% |
| left_yaw | −0.350 | 0.243 | −0.100 | 0.120 | 0.0% |
| left_grip | −1.000 | 0.978 | 0.260 | 0.848 | 73.1%† |

†gripper is expected to sit near the extremes most of the time (it's close
to a binary open/closed signal) — not itself a red flag.

**Rotation does not saturate** (roll/yaw: 0% of steps beyond 0.9, well inside
the trained range) — the reference-frame concern above, while a real
imprecision, is not pushing rotation state out of distribution in practice,
matching F1's experience of the same approximation working well enough.

**Position does.** `left_x` and `left_z` — this checkpoint's q99 normalization
is fit so only ~1% of *training* data should exceed ±1 — spend 22.5% and
19.2% of real closed-loop steps beyond 0.9, with `left_x` bottoming out
exactly at −1.000 and `left_z` topping out exactly at 1.000 (both consistent
with normalization clipping, not merely "near the edge"). That's roughly a
20x higher out-of-range rate than a correctly calibrated q99 transform should
produce, on the two position axes with the most physical range-of-motion
during a pick-and-place task (reaching sideways and lifting/lowering). The
existing "position passes through unchanged" claim (module docstring, citing
F1's own `diag_state.py`) was verified for F1's model/normalization, not
re-checked for LDA's q99 scheme specifically — plausible that F1's
mean/std-based state normalization tolerates the same absolute-frame offset
comfortably while LDA's tighter q99 bounds do not.

### Not a static coordinate-frame offset — a compounding drift, worsened by hard clipping

Two more checks (2026-08-16) settle what kind of problem this is.

**It isn't a calibration offset baked into the scene setup.** `diag_position_range.py`
directly compares SimplerEnv's raw `ee_pose_at_base.p` against LDA's own q01/q99
window. Important correction along the way: `robot_init_x`/`robot_init_y` (the
CLI knobs `eval_bridge_simpler.slurm` sets to 0.147/0.028) place the robot's
*chassis* in the scene — `ee_pose_at_base` is already expressed in the robot's
own base frame, so it's unaffected by where the chassis sits. With an idle
policy (zero actions), the end-effector's resting pose is `x=0.292, y=-0.006,
z=0.135` — comfortably inside LDA's own training window (`x: [0.171, 0.453]`,
`z: [-0.056, 0.195]`). So the arm starts in-distribution; nothing about the
static scene setup is offset from what training saw. (Position is a straight
`state[:, 0:3]` passthrough at conversion time too —
`data_preprocessing/convert_bridge_to_lda.py`'s `STATE_SRC` — no transform is
applied there that could introduce a silent frame shift.)

**It grows over the episode.** Re-instrumented `diag_live_state_range.py` to
also bucket `_build_state()`'s output by within-episode step index (n=360
steps, 6 real closed-loop episodes):

| episode step | n | frac \|left_x\|>0.9 | frac \|left_z\|>0.9 |
|---|---|---|---|
| 0–15 | 90 | **0.0%** | 17.8% |
| 15–30 | 90 | 21.1% | 33.3% |
| 30–45 | 90 | 35.6% | 17.8% |
| 45–60 | 90 | 33.3% | 7.8% |

`left_x` clipping is **zero in the first quarter of every episode and climbs
monotonically afterward** — the textbook signature of compounding rollout
drift, not a fixed offset (a fixed offset would clip from step 0). Put
together with the resting-pose check above: the arm starts fine, and drifts
out of the trained x-range as the (still-imperfect) policy's own actions
accumulate.

**Why this caps success at zero instead of just degrading it, the way it does
for F1:** LDA's state normalization *hard-clips* (`torch.clamp(normalized,
-1, 1)`, `transform/state_action.py:134`) — every value beyond `[q01, q99]`
collapses to the identical `-1.0` or `+1.0`, regardless of how far out of
range the real position actually is. F1's `_build_state` normalizes with
plain `(state - mean) / std`, no clamp — so even when F1's proprioception
also drifts outside its "typical" range (its own docstring accepts this:
rotation "|angle| < 1.6"), the signal stays *monotonic*, just less precise.
Once LDA's `left_x` crosses out of `[q01, q99]`, every subsequent position
in that direction is numerically indistinguishable from every other — the
model loses the one signal that could tell it "you've drifted, correct
back," and the drift compounds with no way to self-arrest. This is
consistent with everything measured so far: the gripper head is excellent
given correct state (92% offline), rotation state stays in-range and doesn't
explain it, and position clipping is absent at episode start but grows
exactly as the policy's own uncorrected errors accumulate.

Checked the architecture to confirm the blast radius: `state_features` (the
`state_encoder`'s output) isn't consumed narrowly — `MMDiT_ActionHeader.py`
concatenates it as a token onto the action sequence
(`action_features = torch.cat([state_features, action_features], dim=1)`,
e.g. line 782) before the shared DiT/self-attention stack, so every
predicted action dimension attends to it, gripper included. A clipped,
uninformative `left_x`/`left_z` doesn't just leave position predictions
guessing — it can degrade the whole action chunk, which fits the erratic
`is_src_obj_grasped` toggling seen in the closed-loop episode logs (a
gripper head that's excellent offline given clean state, but has to share
attention with a corrupted position signal once the rollout drifts).

### Tried the cheap fix first — a no-retrain soft-clip doesn't rescue it

Before committing to a retrain, tested whether the hard clamp specifically
(rather than the tighter q99 window itself) is what's costing success.
`eval/bridge/lda_policy_softclip.py`: a standalone `LDAInferenceSoftClip`
subclass (doesn't touch `lda_policy.py` or `transform/state_action.py`)
replicating the exact same q01/q99 linear normalization for the in-range
region, but replacing `torch.clamp(normalized, -1, 1)` with a C1-continuous
soft saturation, `f(x) = x` for `|x|<=1`, `f(x) = sign(x)*(1+tanh(|x|-1))`
beyond it — so two different real positions that used to both clip to an
identical `-1.0` now map to two different (if still compressed) values,
without changing anything about values already inside `[-1, 1]`.

Full 24-episode run, v3/150k checkpoint, Carrot task, seed 0, 2026-08-16:
**0/24 — still exactly zero.** No improvement over the unmodified
checkpoint's own 0/24-per-seed baseline on this task.

This doesn't prove the clipping mechanism is irrelevant — the model's
weights were still trained having *never* seen anything past exactly ±1 in
any shape, so a smooth-but-still-novel tail is plausibly just as
out-of-distribution to it as a flat one, only differently so. But it does
rule out "the hard clamp specifically is a cheap, patchable-at-eval-time
mistake" the way F1's bugs were. **Conclusion: no shortcut here** — a
genuine fix needs the normalization scheme baked into the weights from
training, not adjusted after the fact. Next step, if pursued, is a retrain
with `mean_std` normalization for state (already supported as a `mode` in
`transform/state_action.py`) or a widened q01/q99 fit, not further eval-time
patching.

F1-VLA is from `F1-VLA/eval/bridge/RESULTS.md`: 3-seed means on the `chunk_size: 4`
checkpoint.

**The three columns are not equally earned, and the table should not be read as a
straight ranking.** mimic-video's figures are its best `stop-step` out of eleven
that were swept, chosen on the evaluation itself, and each is a single 24-episode
run. F1-VLA's are means of three seeds with no such selection. The selection
matters a lot here — mimic's average ranges from 50.0% at stop-step 1 down to
29.2% at stop-step 10, and the eggplant task alone swings 100% to 20.8%. Seed
noise matters too: the same F1 config scored 29.2% and 45.8% on the carrot task on
different seeds.

The same trap applies to LDA: `exec_horizon` is an untuned knob. It is fixed at 8
a priori, and if it is ever swept the whole sweep goes in this table, not its
maximum.
