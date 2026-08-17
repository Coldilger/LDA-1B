#!/bin/bash
#SBATCH --job-name=lda_check_imports
#SBATCH --partition=medium_cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/beegfsnew/scratch/3295540/LDA-1B/slurm/logs/check-imports-%j.out

# Does the framework-registration bug found in the fresh robocasa client env
# also affect lda_eval -- the env every Bridge closed-loop eval runs in?
#
# In the robocasa env, lda/model/framework/__init__.py's auto-import loop dies
# on the first failing submodule and its except branch then raises
# (PureOverwatch has no `log`). If the same happened in lda_eval, submodules
# after the failure would never register, so build_framework could not resolve
# QwenMMDiT -- which would mean every Bridge eval had been running on a
# partially-registered framework registry.
#
# Read-only check: import the package, list which submodules load, and confirm
# the registry actually contains the framework the Bridge configs name.

set -u
export HF_HOME=/mnt/beegfsnew/scratch/3295540/hf_cache
export TOKENIZERS_PARALLELISM=false
cd /mnt/beegfsnew/scratch/3295540/LDA-1B

/home/3295540/.conda/envs/lda_eval/bin/python - <<'PY'
import importlib
import pkgutil
import traceback

print("=== importing lda.model.framework in lda_eval ===", flush=True)
try:
    import lda.model.framework as fw
    print("PACKAGE_IMPORT_OK", flush=True)
except Exception:
    print("PACKAGE_IMPORT_FAILED", flush=True)
    traceback.print_exc()
    raise SystemExit(1)

print("\n=== per-submodule status ===", flush=True)
ok, fail = [], []
for _, name, _ in pkgutil.iter_modules(fw.__path__):
    try:
        importlib.import_module(f"lda.model.framework.{name}")
        ok.append(name)
        print(f"  OK    {name}", flush=True)
    except Exception as e:
        fail.append(name)
        print(f"  FAIL  {name}: {type(e).__name__}: {e}", flush=True)

print(f"\nsubmodules ok={len(ok)} fail={len(fail)}", flush=True)

print("\n=== is the framework the Bridge/RoboCasa configs name actually registered? ===", flush=True)
from lda.model.tools import FRAMEWORK_REGISTRY

keys = sorted(getattr(FRAMEWORK_REGISTRY, "_registry", {}).keys())
print(f"registry entries ({len(keys)}): {keys}", flush=True)
print(f"QwenMMDiT registered: {'QwenMMDiT' in keys}", flush=True)
PY

echo "CHECK_EXIT=$?"
