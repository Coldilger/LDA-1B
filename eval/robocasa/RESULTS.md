# RoboCasa (LDA-1B)

Moved out of `eval/bridge/RESULTS.md` 2026-08-18 -- this is not a Bridge
result, it's a control run on the benchmark LDA-1B's own paper actually
validates the architecture on (RoboCasa-GR1 tabletop tasks), used to
diagnose why the Bridge finetune scores 0% closed-loop (see
`eval/bridge/RESULTS.md` for the full Bridge investigation this complements).

Sim/client-side code (robosuite, robocasa-gr1-tabletop-tasks, the launcher
that works around an upstream import bug) lives outside this repo at
`/mnt/beegfsnew/scratch/3295540/robocasa-eval/` -- third-party clones, kept
out of git for size. The checkpoint (`Wayer2/LDA-robocasa`) is symlinked into
this repo's own `checkpoints/LDA-robocasa_run/`, alongside `LDA-pretrain`.

## RoboCasa cross-check: the codebase works, the Bridge finetune is what doesn't

**Result: LDA-1B's own published RoboCasa checkpoint scores 48% on the
authors' own benchmark, run through this copy of the codebase** (2026-08-18,
`Wayer2/LDA-robocasa`, `gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env`,
50 episodes, 2595s). Not zero — not even close.

A 6-episode smoke run of the same configuration scored 67%; the 50-episode
figure supersedes it. Recorded here because it is a concrete reminder that
small-n closed-loop numbers on these tasks swing widely, which is the same
reason the mimic-video baseline re-run is quoted per task rather than pooled.

Motivation: every diagnosis above (gripper head, rotation reference frame,
position clipping, soft-clip, exec_horizon) tested one candidate mechanism
inside the Bridge finetune and came back negative or inconclusive. None of
them could distinguish "our Bridge finetune is broken" from "something in our
copy of this codebase, environment, or hardware is broken, and Bridge merely
happens to be where we noticed." That distinction needed a control that shares
the codebase but not the finetune.

This is that control, and it is cheap: no dataset download and no training
run. The authors publish a RoboCasa-finetuned checkpoint and the full eval
pipeline lives in `examples/Robocasa_tabletop/`, so the model, its weights,
and its benchmark are all theirs — only the machine, the environment, and this
checkout are ours.

**What it rules out.** The model architecture, checkpoint loading, the
action head, the websocket policy-server path, the GPU, and this checkout of
the repo all work well enough to solve two thirds of a real manipulation task.
The Bridge 0% cannot be attributed to any of them.

**What it leaves.** Whatever is wrong is specific to the Bridge finetune
itself — its data pipeline, normalization, state construction, or training
setup — or to the SimplerEnv/Bridge evaluation path, which the RoboCasa run
does not exercise. That is a much smaller search space than before, and it
puts a retrain back on the table as a reasonable next step rather than a
shot in the dark.

### Setup notes (both were real obstacles, neither is a finding)

- The published `config.yaml` hardcodes `vision_encoder_path:
  /World-Action-Model/pretrained`, an absolute path on the authors' machine.
  Repointed at this repo's own `pretrained/` (which already symlinks the
  DINOv3 and Qwen3-VL snapshots). Config only; no weights touched.
- `lda/model/framework/__init__.py` auto-imports its submodules inside one
  try/except wrapping the whole loop, and the except branch calls
  `logger.log(...)`, which `PureOverwatch` does not define. So one failing
  submodule takes the entire import down *and* hides its own cause. Worked
  around in `eval/robocasa/launcher/run_client.py` without editing the rest
  of this repo.
  **This does not affect any Bridge result**: `slurm/check_framework_imports.slurm`
  imports all 15 submodules cleanly in the `lda_eval` env (ok=15, fail=0), so
  the framework registry is complete there. The failures are specific to the
  freshly-built `robocasa` client env's package versions.

## Camera viewpoint mismatch (egocentric vs third-person) — hypothesis, since confirmed directly below

The RoboCasa cross-check proves the codebase works, but doesn't say why
Bridge specifically fails. One candidate the diagnoses above never
considered: **RoboCasa-GR1 and Bridge don't just differ in embodiment, they
differ in camera viewpoint** — and LDA-1B's own paper and training pipeline
appear to be built around one of those viewpoints exclusively.

**Evidence, not speculation:**

- Code: `robocasa/utils/gym_utils/gymnasium_groot.py`'s `GrootRoboCasaEnv` —
  the class `examples/Robocasa_tabletop/eval_files/simulation_env.py` actually
  uses, i.e. what our RoboCasa cross-check ran through — maps its camera
  observation to the key `video.ego_view_pad_res256_freq20`. Egocentric by
  construction, not an evaluation choice we made.
- Paper (arXiv:2602.12215, fetched 2026-08-18): "The benchmark provides
  challenging and realistic settings that require high-DoF dexterous
  manipulation from **egocentric RGB observations captured by a head-mounted
  camera**" (RoboCasa-GR1 section), and for real-robot experiments: "Across
  all configurations, the policy receives only **egocentric RGB observations
  from a head-mounted camera**."
- **Bridge is not mentioned anywhere in the paper.** Not a lesser-emphasized
  benchmark — absent. Every camera-viewpoint claim the paper makes is about
  egocentric, head-mounted observations; Bridge's fixed external side/
  over-the-shoulder camera (BridgeData V2's standard WidowX setup) is a
  viewpoint the published architecture was never shown to have been
  validated on, on top of the embodiment change already documented above.

**What this would mean if confirmed.** Not just "different robot" (already
known) but "different visual input distribution the vision backbone/DiT was
never trained to handle" — a much more specific and mechanistically
plausible failure mode than a generic domain-gap story, and one that a
Bridge retrain would NOT fix if the frozen/pretrained visual components
(DINOv3, or whatever upstream visual representation LDA's pretraining relied
on) simply never learned to represent third-person Bridge-style scenes well.

**Not yet tested.** This is a documented, evidence-backed hypothesis, not a
confirmed cause — no experiment here isolates camera viewpoint from
embodiment. Two ways to test it without a full retrain:

- [x] Check whether RoboCasa's third-person camera option, used instead of
      `ego_view` with everything else held constant, degrades the
      already-confirmed 48% success rate -- **done, see "Camera-viewpoint
      test: confirmed directly" below: 48% -> 0%.**
- [ ] Check whether any dataset in the *pretraining* mixture (`LDA-pretrain`,
      `data_mix: all_dataset`) includes third-person/external-camera
      manipulation data (e.g. a slice of OXE) — if pretraining never saw a
      third-person view at all, that's a much stronger claim than "Bridge
      finetuning alone couldn't overcome it." Still open -- the confirmed
      result above already shows the *finetuned* checkpoint can't handle
      third-person views; this would clarify whether that's a pretraining
      -level or finetuning-level gap. Not prioritized (see decision below).

## Camera-viewpoint test: collapses success, but see the occlusion caveat below

**Result: switching only the camera (`egoview` -> `robot0_agentview_center`,
same robot, same task, same checkpoint, same episode protocol) collapses
success from 48% to 0% (0/50 episodes).** 2026-08-18,
`gr1_unified/PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_Env`, same
`Wayer2/LDA-robocasa` checkpoint as the main cross-check.

Method: `GR1ArmsAndWaistKeyConverter.get_camera_config()` -- the class our
task's robot name (`GR1ArmsAndWaistFourierHands`) resolves to via
`make_key_converter` -- hardcodes `camera_names=["egoview"]`. Patched at
runtime (`eval/robocasa/launcher/run_client_agentview.py`, no file on disk
touched) to return `camera_names=["robot0_agentview_center"]` instead, while
leaving `mapped_names` (`"video.ego_view_pad_res256_freq20"`, the
observation-dict key the model-facing client reads) untouched. So the
*only* thing that changes is which physical MuJoCo camera the pixels placed
under that key come from -- state construction, action mapping, robot,
task, and checkpoint are all identical to the 48% run.

`robot0_agentview_center` (not the bare `"agentview"` -- that name doesn't
exist for this robot; the live error listed the actual registered names)
is also `tabletop.py`'s own default `render_camera`, i.e. the environment's
own canonical third-person view, not an arbitrary pick.

Confirmed on a real sample size, not just a smoke run: 0/6 on the initial
smoke test, 0/50 on the full run -- getting exactly 0/6 by chance alone if
the true rate matched the 48% baseline has under 2% probability, so the
smoke result alone was already suggestive; the full run removes any
small-sample doubt.

**What this establishes.** Not a fine-grained "egocentric vs. third-person"
architectural claim -- swapping cameras changes the entire pixel
distribution (field of view, framing, everything), a qualitative jump, not
a small nudge. What it does establish: the model's failure mode on
Bridge-like third-person views is not merely "different embodiment" -- the
vision backbone specifically cannot handle a camera framing this far outside
what it was ever shown, independent of embodiment, state space, or action
space (all held constant here). See the caveat immediately below for what
this does *not*, on its own, establish.

**Caveat, done 2026-08-22 -- visual inspection reveals a real confound.**
The "not yet done" visual spot-check below was finally run: a sample frame
from `robocasa-eval/videos/agentview_full_630800/` shows the GR1 humanoid's
own head and both shoulders filling roughly two-thirds of the frame, with
only narrow slivers of the countertop visible at the left/right edges --
not the "sensible, well-framed third-person image" the smoke-test timing
argued for. This is not a rendering artifact (the timing/`CLIENT_EXIT=0`
evidence for a working pipeline still stands), but it does mean the test
has an unaddressed confound: **severe self-occlusion**, not (only) viewpoint
mismatch. A model with a perfectly viewpoint-invariant vision backbone could
still fail on this specific camera, simply because most of the workspace
isn't visible in it -- so the 48%->0% collapse can no longer be read as
*clean* evidence isolating "trained-viewpoint mismatch" from "not enough
usable pixels regardless of viewpoint." For direct comparison: a same-day
check of a real Bridge/SimplerEnv rollout frame (the actual benchmark this
finding is meant to explain) shows a close, fully unobstructed view of the
workspace with no robot-body occlusion at all beyond the gripper fingertips
at the top edge -- confirming Bridge's own camera is not analogous to
`robot0_agentview_center` in this respect, and this specific swap test is a
weaker proxy for "Bridge-style third-person" than assumed when it was
designed.

Root cause of the occlusion, traced in source: `robot0_agentview_center`
(and its siblings `robot0_agentview_left`/`_right`/`_frontview`) are defined
in `robocasa/utils/camera_utils.py`'s `CAM_CONFIGS` with
`parent_body="mobilebase0_support"` -- i.e. mounted to the robot's own
mobile-base body at a close offset (`pos=[-0.6, 0.0, 1.15]` for `_center`;
`_left`/`_right` are +-0.35 laterally, `_frontview` is nearly the same
offset again), not to anything fixed in the scene. Every currently-active
non-wrist camera option for this robot shares that same mount point, so
switching to `_left`/`_right`/`_frontview` instead would likely still be
substantially self-occluded (not verified -- no frame from those has been
pulled). The classic robosuite scene-fixed cameras (`frontview`, `birdview`,
`agentview`, `sideview`, the ones that would actually sit somewhere in the
room rather than on the robot) exist in
`models/assets/arenas/empty_tabletop_arena.xml` but are **commented out** --
none is currently registered or usable without code changes. `sideview`
(`pos=[-0.057, 1.276, 1.488]`, roughly countertop height, off to the side)
is the one that looks most analogous to Bridge's actual external camera
placement if someone wanted to build a cleaner version of this test --
requires uncommenting + registering it in `tabletop.py`'s `set_cameras()`
and a real re-run, not done here.

**Bottom line.** The original Bridge finding this section is meant to
support -- LDA-1B scores 0% on real Bridge (0/12) despite scoring 48% on its
own native RoboCasa benchmark -- is untouched by any of this; that's a
standalone fact about a real benchmark, not something this synthetic swap
test could invalidate. What's weakened is specifically this swap test's
strength *as corroboration* for the viewpoint-transfer explanation over a
simpler "not enough visible workspace" explanation -- worth being honest
about that gap rather than citing 48%->0% as decisive on its own.

## Decision: framing and next steps (2026-08-19)

**Framing adopted for the thesis: "LDA-1B's world-model representation does
not transfer across camera viewpoint,"** not "the Bridge finetune is
bugged." The 48% -> 0% result above is treated as the LDA-1B contribution to
the thesis's central research question, on its own terms — not as an
unresolved blocker waiting on a fix. This framing still rests mainly on the
real Bridge 0% fact and LDA-1B's paper training exclusively on egocentric
views, not on the RoboCasa camera-swap test alone — **see the occlusion
caveat above (2026-08-22)**, which weakens the swap test specifically as
corroboration, without touching the real-Bridge fact it was meant to
support.

**Explicitly not pursuing now:** running F1-VLA's and mimic-video's own
checkpoints through RoboCasa on both camera options, to check whether the
egocentric-vs-third-person sensitivity is LDA-specific or a property of all
three architectures. Would strengthen the framing (LDA-specific vs.
general), but is real additional engineering + compute for a cross-model
comparison that isn't this thesis's priority right now. Noted here as a
deliberate scope decision, not an oversight — revisit if time allows after
the rest of Experiments 1/2/4/5 are closed out.

**Priority shift: close out Experiments 1/2/4(/5) for LDA-1B against the
working RoboCasa checkpoint, not Bridge.** `eval/bridge/EXPERIMENTS.md`'s
blocker note ("Experiments 1/2/4 need a working checkpoint... re-run once v3
finishes") assumed the checkpoint would be a fixed Bridge one. Given the
framing decision above, that's no longer the plan — LDA-1B's remaining
experiments will run against RoboCasa instead, same as Experiment 3 already
does, accepting the dataset mismatch (RoboCasa, not Bridge) as an explicit,
stated tradeoff in exchange for actually having LDA-1B's rows filled in
across the thesis's shared experiment framework. Bridge stays documented as
a real, unresolved 0% in `eval/bridge/RESULTS.md`, not silently dropped —
just no longer the blocking dependency for the rest of this repo's
experiments.
