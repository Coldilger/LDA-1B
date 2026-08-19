#!/bin/bash
#SBATCH --job-name=install_robocasa
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/install-%j.out

# Sets up a standalone conda env for evaluating LDA-1B's own official
# RoboCasa-finetuned checkpoint (Wayer2/LDA-robocasa) on RoboCasa-GR1
# Tabletop Tasks, using LDA's client-server eval design: this env only needs
# robosuite + robocasa-gr1-tabletop-tasks (the sim/client side) -- LDA's own
# policy server (server_policy.py) runs separately in the existing lda_eval
# conda env and talks to the sim over a local websocket, so no torch/model
# deps are needed here at all (confirmed via grep: LDA's own eval wrappers
# under examples/Robocasa_tabletop/eval_files/ never import gr00t/Isaac-GR00T,
# only robocasa/robosuite + a plain websocket client).
#
# robosuite and robocasa-gr1-tabletop-tasks are cloned into
# /mnt/beegfsnew/scratch/3295540/robocasa-eval/, not into this repo --
# together they're ~9.3GB of vendored simulator + asset files with their own
# upstream git history (ARISE-Initiative/robosuite,
# robocasa/robocasa-gr1-tabletop-tasks), which would bloat this repo's git
# history irreversibly if committed. This script is what makes that
# reproducible without committing them: rerun it and the checkout rebuilds
# from scratch. Only the driver code we actually wrote
# (eval/robocasa/launcher/, the eval_robocasa_*.slurm scripts, this file)
# lives under version control.

set -euo pipefail

ROOT=/mnt/beegfsnew/scratch/3295540/robocasa-eval
mkdir -p "$ROOT"
cd "$ROOT"

if [ ! -d "$ROOT/robosuite" ]; then
    git clone https://github.com/ARISE-Initiative/robosuite.git "$ROOT/robosuite"
fi
if [ ! -d "$ROOT/robocasa-gr1-tabletop-tasks" ]; then
    git clone https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git "$ROOT/robocasa-gr1-tabletop-tasks"
fi

source /software/miniconda3/etc/profile.d/conda.sh

if ! conda env list | grep -q "^robocasa "; then
    conda create -y -c conda-forge -n robocasa python=3.10
fi
conda activate robocasa

pip install --upgrade pip setuptools wheel

# robosuite (sim/control backend) -- upstream ARISE-Initiative repo, already
# shallow-cloned, per robocasa-gr1-tabletop-tasks's own install guide.
pip install -e "$ROOT/robosuite"

# robocasa-gr1-tabletop-tasks (env/task definitions + assets), already
# shallow-cloned. Pins mujoco==3.2.6 exactly, which will downgrade whatever
# robosuite pulled in -- installed last, matching the upstream README's own
# install order, so if there's a real runtime incompatibility it'll show up
# when we actually run demo_task.py below, not as a pip resolution error.
pip install -e "$ROOT/robocasa-gr1-tabletop-tasks"

# client-side deps needed by LDA's own robocasa eval wrapper
# (examples/Robocasa_tabletop/eval_files/), which are not part of either
# robosuite's or robocasa's own install_requires.
pip install tyro websockets pydantic gymnasium matplotlib opencv-python typing_extensions msgpack

cd "$ROOT/robocasa-gr1-tabletop-tasks"
python robocasa/scripts/download_tabletop_assets.py -y

echo "INSTALL_EXIT=$?"

# Smoke test: import the env class LDA's simulation_env.py actually imports,
# and instantiate nothing yet -- just confirm the install is import-clean.
python -c "
import robocasa
import robosuite
from robocasa.utils.gym_utils import GrootRoboCasaEnv
print('robocasa/robosuite import OK')
print('robocasa version:', robocasa.__file__)
"

echo "SMOKE_TEST_EXIT=$?"
