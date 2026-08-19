#!/bin/bash
#SBATCH --job-name=robocasa_deps
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/deps-%j.out

# Resolves the robocasa (client) env's remaining imports. LDA's own robocasa
# eval wrapper imports `lda.model.framework.share_tools`, which transitively
# drags in most of the LDA training stack, so the sim-side env needs those
# deps too even though it never runs the model (the model runs in lda_eval,
# behind the websocket).
#
# Deliberately NOT solved by installing robocasa into the working lda_eval
# env instead: robocasa pins numpy==1.26.4 and mujoco==3.2.6, which risks
# breaking the environment our Bridge evals depend on. Two envs stay separate.
#
# Runs as a job, not on the login node: each failed-import probe takes
# minutes, and the login node has 4 cores shared across users.

set -u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate robocasa

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
export PYTHONPATH="/mnt/beegfsnew/scratch/3295540/LDA-1B:/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks:${PYTHONPATH:-}"

# import-name -> pip-name where they differ
pipname() {
  case "$1" in
    cv2) echo "opencv-python" ;;
    PIL) echo "pillow" ;;
    yaml) echo "pyyaml" ;;
    sklearn) echo "scikit-learn" ;;
    decord) echo "decord" ;;
    *) echo "$1" ;;
  esac
}

for i in $(seq 1 25); do
  err=$(python -c "import examples.Robocasa_tabletop.eval_files.simulation_env" 2>&1)
  missing=$(echo "$err" | grep -oP "No module named '\K[^']+" | head -1)
  if [ -z "$missing" ]; then
    echo "=== IMPORTS RESOLVED after $((i-1)) installs ==="
    echo "$err" | tail -5
    break
  fi
  # only the top-level package name is installable
  top="${missing%%.*}"
  pkg="$(pipname "$top")"
  echo ">>> [$i] missing '$missing' -> installing '$pkg'"
  pip install "$pkg" 2>&1 | tail -2
done

echo ""
echo "=== final import check ==="
python -c "
import examples.Robocasa_tabletop.eval_files.simulation_env
print('SIMULATION_ENV_IMPORTS_OK')
" 2>&1 | tail -12
echo "DEPS_EXIT=$?"
