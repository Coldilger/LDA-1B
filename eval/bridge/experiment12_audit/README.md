# Experiment 12 — BridgeData pipeline audit (all three models)

**Question (advisor's point 9).** Do the three models see the same Bridge
data, in the same form, at training and at evaluation? Episode overlap,
language annotations, frame sampling, action coordinates and orientation
conventions, for F1-VLA, mimic-video and LDA-1B. The motivating case is
LDA-1B's 0% on SimplerEnv Bridge (`../RESULTS.md`), which earlier checks did
not explain (`thesis/notes/lda_bridge_viewpoint.md`).

Data sources:

| model | training copy of BridgeData V2 | built by |
|---|---|---|
| F1-VLA | `bridge_orig_lerobot` (IPEC-COMMUNITY LeRobot export of OXE `bridge_orig`; 53,192 episodes, 5 fps, 256×256) | as downloaded |
| LDA-1B | `bridge_lda`, converted from `bridge_orig_lerobot` | `data_preprocessing/convert_bridge_to_lda.py` |
| mimic-video | authors' export of the raw Bridge data (480×640, `eef_transform` pose) | `mimic-video/data_preprocessing/action/process_bridge.py` (authors') |

`audit_pipelines.py` (CPU, 300 random episodes with an instruction, seed 0)
checks the two local copies; `audit_summary.json` holds its numbers.
mimic-video's copy is not on disk, so it is audited from its code.

## Results (2026-11-22)

### 1. Orientation convention at evaluation — a bug in LDA-1B's Bridge wrapper

The Bridge state stores the tool orientation in a frame rotated by a fixed
90° relative to the simulator's tool frame. SimplerEnv's own ground-truth
replay script (`SimplerEnv/simpler_env/utils/debug/
widowx_test_dataset_inference_rollout_gt_traj_in_sim.py`, L69–75) places the
tool at `euler2mat(state rpy) @ T`, `T = [[0,0,1],[0,1,0],[-1,0,0]]`, i.e.
`R_sim = R_data · T`; mimic-video's processing script documents the same
rotation (`process_bridge.py`, docstring). Check against our own
measurement: SimplerEnv's rest pose (`f1_vla_policy.py` `_build_state`
comment, euler xyz (−3.06, 1.51, −3.08)) is 99° from Bridge's mean
starting orientation, and 11° after multiplying by `Tᵀ` (Bridge's own spread
at t = 0 is up to 22° on yaw).

Neither F1-VLA's nor LDA-1B's wrapper applies `T`. Both report the
rotation state relative to the episode's first pose instead (F1's "fix #4",
copied into LDA). For the **action**, LDA-1B is affected much more:

- LDA-1B predicts motion in the tool's own frame (`calculate_delta_eef`,
  `dT = T_i⁻¹ T_{i+1}`, computed on Bridge poses). `lda_policy.py`
  (`_predict_chunk`) integrates these deltas from the simulator's pose with
  `delta2abs`, i.e. in the simulator's tool frame, which is rotated by 90°.
- Offline test (section E of the script): the *ground-truth* tool-frame
  deltas of the sampled episodes, pushed through the wrapper's
  integration from the simulator pose, give a position step whose direction
  is off by **77° (median; 90th percentile 90°)** from the true motion
  (relative error 1.25). Converting the simulator pose to the Bridge
  convention first (`R_sim · Tᵀ`) makes the error 0 (numerical precision).
  Rotation steps are off by 2.4° (median).

So even a perfect LDA-1B policy would move the arm in the wrong direction on
SimplerEnv. This matches what was seen in summer: the arm starts in range
and drifts out of the trained position range as the episode goes on
(`../RESULTS.md`, "a compounding drift"). The summer note "seeding from a
different initial pose ... the trajectory is rigidly rotated, not
distorted" saw the rotation but not its size.

**Status: found offline, from the wrapper code and the simulator's own
convention; not yet confirmed in the closed loop.** The LDA-1B Bridge
checkpoints were deleted in the October scratch purge, so the closed-loop
test needs a new Bridge finetune (the bug is in evaluation, so the
training recipe does not change) and the corrected wrapper.

F1-VLA's actions are not affected: it predicts Bridge's world-frame
position deltas and Euler-angle differences (section C: `action[0:3]` equals
the state difference to 4e-9 and `action[3:6]` the difference of Euler
angles to 1e-8), converted exactly like SimplerEnv's replay script and its
Octo reference policy (`euler2axangle`). Its **state** is approximate
(section D): the episode-relative rotation differs from the trained value by
28° (median; 90th percentile 82°), i.e. 2.0 / 1.0 / 0.7 training standard
deviations on roll / pitch / yaw. F1-VLA still reaches 48% with it; whether
the exact conversion changes F1-VLA's success is an evaluation-only test of
the existing checkpoint (not run yet).

mimic-video is consistent: it trains on `eef_transform`, which is the
simulator's convention (the authors' docstring says so), and receives the
rotation as the first two rows of the rotation matrix, so there is no Euler
or frame conversion at all.

### 2. Episodes and language

| | F1-VLA | LDA-1B | mimic-video |
|---|---|---|---|
| episodes used | all 53,192 | 38,660: the 14,532 (27.3%) with an empty instruction are dropped by LDA's loader (`lda/dataloader/gr00t_lerobot/datasets.py` L724) | authors' export of the raw data (they report ~13% without a label; count not reproducible here) |
| training anchors | every frame | every 3rd frame (`img_interval = 3`, `BaseDataConfig`) | uniform in time |

The LDA copy reproduces `bridge_orig_lerobot` exactly (section B: state,
next-state action targets and gripper identical, 0 difference); videos are
symlinks to the same `image_0` files.

How often the evaluation tasks' objects occur in the training instructions
(all 53,192 episodes):

| SimplerEnv task | training episodes naming its objects |
|---|---|
| Carrot (carrot & plate) | 551 (1.0%) |
| Spoon (spoon & towel/cloth) | 444 (0.8%) |
| Stack (green & yellow & block/cube) | 96 (0.2%) |
| Eggplant (eggplant & basket) | 0 (eggplant alone: 418, mostly "pot or pan", "plate") |

The same for all three models (mimic-video's export comes from the same raw
episodes).

### 3. Frames and timing (code)

| | F1-VLA | LDA-1B | mimic-video |
|---|---|---|---|
| camera | `image_0`, 256×256 | `image_0`, 256×256 → 224 | raw primary camera, 480×640 |
| history at training | 4 frames at −9, −6, −3, 0 (every 0.6 s; `n_obs_img_steps 12`, `obs_img_stride 3`) | 2 frames at −5, 0 (1.0 s) | 5 frames at 5 Hz |
| forecast target at training | frame +3 (0.6 s) | frame +5 (1.0 s) | 56 frames at 5 Hz |
| history at evaluation | **4 consecutive frames (0.2 s apart)** — chosen on a 24-episode Carrot run (7/24 against 1/24 with the training spacing, `f1_vla_policy.py` L116–127) | 2 frames 5 steps apart (= training) | 5 consecutive frames at 5 Hz (= training) |
| actions | chunk 4 of 7-D world-frame deltas, mean/std | chunk 16 of tool-frame deltas, q01/q99 with clipping, 8 executed | chunk 15 of poses relative to the current one (6-D rotation), mean/std |

F1-VLA's evaluation history spacing differs from training. With
consecutive frames the "next frame of the sequence" is +1 step (0.2 s), not
+3, which matters for the horizon conditions of Experiment 7
(`F1-VLA/eval/bridge/experiment6_7_swap`). The 7/24 vs 1/24 choice was made
on an earlier checkpoint and one seed.

## Follow-ups (running, 2026-11-22)

1. **LDA-1B.** `lda_policy.py` now converts the simulator pose
   (`bridge_frame`, on by default; `LDA_BRIDGE_FRAME=0` restores the old
   wrapper): the integration seed and the rotation state are in Bridge's
   absolute convention (`R_sim · Tᵀ`). Check with the wrapper's own code
   (`verify_wrapper_fix.py`: the model replaced by a stub returning the true
   tool-frame deltas, 2,390 steps of 100 episodes): position direction error
   77.7° (median) with the old wrapper, 9e-7° with the fix; rotation step
   error 2.7° → 5e-8°. New Bridge finetune with the v3 recipe
   (`lda/config/training/LDA_bridge_v4.yaml`, 150k micro-batches,
   `slurm/bridge_v4_selfchain.slurm`, first wave job 786408), then closed
   loop on the four tasks with the corrected wrapper. Job ids: `jobs.txt`.
2. **F1-VLA**, evaluation only, 100k checkpoint, 4 tasks × seeds 0–2, paired
   with `exp1_baseline_seed{0,1,2}`: `F1_STATE_FRAME=bridge` (exact rotation
   state, tag `exp1_baseline_seed<S>_e12statebridge`) and
   `F1_HISTORY_SPACING=3` (history frames at t−9, −6, −3, 0 as in training,
   tag `..._e12hist3`), options of `F1-VLA/eval/bridge/f1_vla_policy.py`
   (defaults unchanged). Labels `f1e12_*` in `jobs.txt`.
