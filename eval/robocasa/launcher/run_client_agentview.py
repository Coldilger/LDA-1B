#!/usr/bin/env python3
"""
Camera-viewpoint test: same task, same checkpoint, same everything -- except
the pixels the model sees come from RoboCasa's third-person `agentview`
camera instead of the egocentric head-mounted one.

Motivation: LDA-1B's own published RoboCasa checkpoint scores 48% through
this codebase (eval/robocasa/RESULTS.md), while the Bridge finetune scores
0%. The leading, still-unverified explanation (same doc, "New hypothesis")
is that RoboCasa-GR1 and Bridge differ in camera viewpoint, not just
embodiment -- LDA's paper documents only egocentric, head-mounted-camera
setups and never mentions Bridge (third-person, fixed external camera) at
all. This isolates viewpoint from embodiment directly: same robot, same
task, same trained weights, only the camera the model is shown changes.

How the swap works, and why it's a clean intervention rather than a hack:

- `robocasa.models.robots.GR1ArmsAndWaistKeyConverter.get_camera_config()`
  is what our task (`GR1ArmsAndWaistFourierHands`) uses. It hardcodes
  `camera_names=["egoview"]` -- the MuJoCo camera the simulator renders
  from -- paired with `mapped_names=["video.ego_view_pad_res256_freq20"]`
  -- the KEY the resulting image gets stored under in the observation dict
  the model-facing client reads (`model2robocasa_interface_with_history.py`
  reads `observations["video.ego_view"]`, a further-renamed form of that
  same key produced downstream, unaffected by this patch since only
  `camera_names` changes here, not `mapped_names`).
- `robocasa.environments.tabletop.tabletop.py`'s `set_cameras()` confirms
  `agentview` is a SCENE-level camera (attached to the arena via
  `mujoco_arena.set_camera`, no `parent_body`), unlike `egoview` (mounted to
  the robot's own head, `parent_body="robot0_head_pitch"`). So `agentview`
  genuinely exists in this exact task's scene already -- this isn't asking
  the simulator to render something that was never set up for this
  robot/task combination, just reading from a camera that's already there.
- Patching `get_camera_config` to return `camera_names=["robot0_agentview_center"]`
  while leaving `mapped_names` untouched means the observation dict key the
  client reads never changes -- only which physical camera the pixels
  placed under that key come from. No other file in this repo or in LDA-1B
  is modified. (First attempt used the bare name `"agentview"`, matching
  `GR1FixedLowerBodyKeyConverter`'s own hardcoded string -- robosuite
  rejected it: the actual registered name for this robot/task is
  `robot0_agentview_center`, one of several `robot0_agentview_*` variants,
  confirmed from the live error message listing every valid camera name.
  `robot0_agentview_center` is also `tabletop.py`'s own default
  `render_camera`, i.e. the camera the environment's own renderer already
  treats as "the" canonical third-person view.)

If success rate drops sharply, that's direct evidence the model's failure
mode on third-person views (like Bridge's) isn't merely "different robot" --
it's specifically that the visual input distribution the vision backbone was
never trained to handle causes trouble, independent of embodiment. If
success rate holds up, viewpoint isn't the (whole) story and the
embodiment-only explanation gets more weight back.
"""

import runpy
import sys

from lda.training.trainer_utils.overwatch import PureOverwatch

if not hasattr(PureOverwatch, "log"):
    PureOverwatch.log = lambda self, msg, *args, **kwargs: self.info(msg)

from robocasa.models.robots import GR1ArmsAndWaistKeyConverter  # noqa: E402

_original_get_camera_config = GR1ArmsAndWaistKeyConverter.get_camera_config.__func__


def _agentview_camera_config(cls):
    mapped_names, _camera_names, camera_widths, camera_heights = _original_get_camera_config(cls)
    return mapped_names, ["robot0_agentview_center"], camera_widths, camera_heights


GR1ArmsAndWaistKeyConverter.get_camera_config = classmethod(_agentview_camera_config)
print(
    "PATCHED: GR1ArmsAndWaistKeyConverter now renders from "
    "'robot0_agentview_center' (third-person) instead of 'egoview' "
    "(egocentric); observation key unchanged.",
    flush=True,
)

sys.argv[0] = "examples/Robocasa_tabletop/eval_files/simulation_env.py"
runpy.run_module(
    "examples.Robocasa_tabletop.eval_files.simulation_env",
    run_name="__main__",
    alter_sys=True,
)
