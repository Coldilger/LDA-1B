#!/usr/bin/env python3
"""
Client-side launcher for LDA-1B's RoboCasa evaluation.

Exists to work around an upstream bug in LDA rather than editing the LDA repo.

`lda/model/framework/__init__.py` auto-imports its submodules inside a single
try/except wrapping the whole loop, and the except branch calls
`logger.log(...)`. PureOverwatch implements debug/info/warning/error/critical
but not `log`, so the handler itself raises AttributeError and takes the
import down -- while also hiding whatever actually failed.

Three of that package's submodules fail to import *in this env*: two need
`x_transformers`, and `QwenGR00T_single_t_concat_curr_obs` raises TypeError
("non-default argument 'training_task_weights' follows default argument").
Because the try wraps the entire loop, the first failure -- alphabetically
the QwenGR00T one -- triggers the broken handler, so installing the missing
package is not sufficient on its own.

These are environment-specific, not defects in the LDA code: check job 630063
imported all 15 submodules cleanly in the lda_eval env (ok=15 fail=0),
including the QwenGR00T one. The TypeError therefore depends on the
dataclass/typing behaviour of this env's package versions, not on the source.
By the same token the framework registry is complete in lda_eval, so this
never affected the Bridge closed-loop evals, which run there.

None of it matters for the sim client either: it needs only
`read_mode_config` from `lda.model.framework.share_tools`. The framework
submodules exist to construct the model, which happens in the *server*
process (lda_eval env), where it loads successfully. Verified by probe job
630058: with `log` present, the client entry point imports cleanly and the
3 failures are only warnings.

So: supply the missing `log` method, then hand off to the real, unmodified
`simulation_env` entry point with the arguments untouched.

Kept in its own `launcher/` directory on purpose. Python puts the script's
own directory first on sys.path, and the parent (`robocasa-eval/`) holds the
`robosuite` git clone -- from there `import robosuite` resolves to the repo
root as a namespace package instead of the installed one, so robocasa's
`assert robosuite.__version__ in [...]` dies with AttributeError (job 630059).
Anything added beside this file must not share a name with an installed
package.
"""

import runpy
import sys

from lda.training.trainer_utils.overwatch import PureOverwatch

if not hasattr(PureOverwatch, "log"):
    PureOverwatch.log = lambda self, msg, *args, **kwargs: self.info(msg)

# Run the real script as __main__ so its `tyro.cli(...)` block executes and
# parses sys.argv exactly as it would when invoked directly.
sys.argv[0] = "examples/Robocasa_tabletop/eval_files/simulation_env.py"
runpy.run_module(
    "examples.Robocasa_tabletop.eval_files.simulation_env",
    run_name="__main__",
    alter_sys=True,
)
