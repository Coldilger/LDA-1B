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
  around in `robocasa-eval/launcher/run_client.py` without editing this repo.
  **This does not affect any Bridge result**: `slurm/check_framework_imports.slurm`
  imports all 15 submodules cleanly in the `lda_eval` env (ok=15, fail=0), so
  the framework registry is complete there. The failures are specific to the
  freshly-built `robocasa` client env's package versions.

## New hypothesis: camera viewpoint mismatch (egocentric vs third-person), not yet tested

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

- [ ] Check whether any dataset in the *pretraining* mixture (`LDA-pretrain`,
      `data_mix: all_dataset`) includes third-person/external-camera
      manipulation data (e.g. a slice of OXE) — if pretraining never saw a
      third-person view at all, that's a much stronger claim than "Bridge
      finetuning alone couldn't overcome it."
- [ ] A cheap diagnostic: run the diag scripts already built for the Bridge
      investigation (e.g. `diag_gripper.py`'s pattern) on frames cropped/
      warped to approximate an egocentric framing, or conversely check
      whether RoboCasa's `agentview` (third-person) camera option, if used
      instead of `ego_view`, degrades the already-confirmed 48% success rate
      -- that would isolate viewpoint from embodiment directly, using
      infrastructure that already exists (`GrootRoboCasaEnv`'s camera_names
      config).
