#!/bin/bash
#SBATCH --job-name=robocasa_probe_import
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/robocasa-eval/probe-import-%j.out

# Surfaces the real import error that lda/model/framework/__init__.py hides,
# and tests whether that failure even matters for the sim client.
#
# The module auto-imports its submodules inside a try/except whose handler
# calls `logger.log(...)`. PureOverwatch implements debug/info/warning/error/
# critical but not `log`, so the handler raises AttributeError and the real
# ImportError is never printed -- an upstream bug, not a local setup problem.
#
# Adding the missing `log` method (aliased to `info`) at runtime lets the
# handler do its job: print the warning and continue. If the import then
# succeeds, the submodule failure is irrelevant to the client, which only
# needs `read_mode_config` -- the submodules exist for model construction,
# which happens server-side in the lda_eval env where it already works.
# Nothing in the LDA repo is edited.

set -u
source /software/miniconda3/etc/profile.d/conda.sh
conda activate robocasa

cd /mnt/beegfsnew/scratch/3295540/LDA-1B
export PYTHONPATH="/mnt/beegfsnew/scratch/3295540/LDA-1B:/mnt/beegfsnew/scratch/3295540/robocasa-eval/robocasa-gr1-tabletop-tasks:${PYTHONPATH:-}"

python - <<'PY'
import traceback

from lda.training.trainer_utils.overwatch import PureOverwatch

if not hasattr(PureOverwatch, "log"):
    PureOverwatch.log = lambda self, msg, *a, **kw: self.info(msg)
    print("patched PureOverwatch.log -> info", flush=True)

print("\n=== importing the client entry point ===", flush=True)
try:
    import examples.Robocasa_tabletop.eval_files.simulation_env  # noqa: F401
    print("SIMULATION_ENV_IMPORTS_OK", flush=True)
except Exception:
    print("STILL_FAILS:", flush=True)
    traceback.print_exc()

print("\n=== which framework submodules actually fail, and why ===", flush=True)
import importlib
import pkgutil

import lda.model.framework as fw

for _, name, _ in pkgutil.iter_modules(fw.__path__):
    try:
        importlib.import_module(f"lda.model.framework.{name}")
        print(f"  OK    {name}", flush=True)
    except Exception as e:
        print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)
PY

echo "PROBE_EXIT=$?"
