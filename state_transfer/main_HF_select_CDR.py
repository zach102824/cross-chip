#!/usr/bin/env python3
"""Run ``main_HF.py`` with selected-CDR fitting.

The baseline files are not modified.  This entrypoint installs
``shot_measurement_select_CDR`` as the drop-in measurement module and executes
the existing HF pipeline.  Results default to ``data_select_CDR/`` so they can
be compared directly with the baseline ``data/`` results.

Default strategy (per-term affine refit): the ORIGINAL 30 training circuits
are kept.  Each Pauli term's CDR line ``exact ~= a * noisy + b`` is refit
using only its informative training points (exact |<P>| > tolerance); the
intercept ``b`` is free.  Terms with no informative data use the identity map
(a=1, b=0) so the measured target value passes through instead of being
zeroed out.

Configuration:

    CDR_PER_TERM_FIT=affine              # affine | through_origin | none
    CDR_PER_TERM_EXACT_TOL=0.05
    CDR_PER_TERM_MIN_POINTS=2
    CDR_PER_TERM_SLOPE_MAX=10.0
    CDR_SELECTION_METHOD=none            # weighted_maxmin to re-enable pooling
    CDR_SELECTION_POOL_SIZE=300
    CDR_SELECTION_LOCAL_COUNT=10
    CDR_SELECTION_VERBOSE=0
    CDR_SELECTED_DATA_DIR=data_select_CDR

All environment variables accepted by ``main_HF.py`` remain available.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from typing import Any


STATE_TRANSFER_DIR = Path(__file__).resolve().parent
REPO_ROOT = STATE_TRANSFER_DIR.parent
JUNE_MAIN_DIR = REPO_ROOT / "June_main"

os.chdir(STATE_TRANSFER_DIR)
# Insert broad paths first and the local state_transfer path last so the local
# modules have highest precedence.
for path in (str(REPO_ROOT), str(JUNE_MAIN_DIR), str(STATE_TRANSFER_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# Import the baseline module inside the selected variant first, then redirect
# subsequent ``import shot_measurement`` statements in main_HF.py and
# main_cursor_lib.py's local CDR import to the selected implementation.
import shot_measurement_select_CDR as selected_measurement

sys.modules["shot_measurement"] = selected_measurement

# Keep selected-CDR artifacts separate without changing main_HF.py's checkpoint
# calls.  Since Python caches modules, main_HF.py imports this wrapped function.
import cloud_results


_BASE_SAVE_CHECKPOINT = cloud_results.save_checkpoint
_SELECTED_DATA_DIR = Path(
    os.environ.get("CDR_SELECTED_DATA_DIR", "data_select_CDR")
)


def _save_selected_checkpoint(
    *,
    data_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    del data_dir
    selected_metadata = dict(metadata or {})
    selected_metadata.update(
        {
            "cdr_per_term_fit": os.environ.get("CDR_PER_TERM_FIT", "affine"),
            "cdr_per_term_exact_tol": float(
                os.environ.get("CDR_PER_TERM_EXACT_TOL", "0.05")
            ),
            "cdr_per_term_min_points": int(
                os.environ.get("CDR_PER_TERM_MIN_POINTS", "2")
            ),
            "cdr_per_term_slope_max": float(
                os.environ.get("CDR_PER_TERM_SLOPE_MAX", "10.0")
            ),
            "cdr_selection_method": os.environ.get("CDR_SELECTION_METHOD", "none"),
            "cdr_selection_pool_size": int(
                os.environ.get("CDR_SELECTION_POOL_SIZE", "300")
            ),
            "cdr_selection_local_count": int(
                os.environ.get("CDR_SELECTION_LOCAL_COUNT", "10")
            ),
            "baseline_entrypoint": "main_HF.py",
            "selected_entrypoint": "main_HF_select_CDR.py",
        }
    )
    return _BASE_SAVE_CHECKPOINT(
        data_dir=_SELECTED_DATA_DIR,
        metadata=selected_metadata,
        **kwargs,
    )


cloud_results.save_checkpoint = _save_selected_checkpoint

_selection_method = os.environ.get("CDR_SELECTION_METHOD", "none")
print(
    "[CDR-select] per-term robust refit: "
    f"mode={os.environ.get('CDR_PER_TERM_FIT', 'affine')}, "
    f"exact_tol={os.environ.get('CDR_PER_TERM_EXACT_TOL', '0.05')}, "
    f"min_points={os.environ.get('CDR_PER_TERM_MIN_POINTS', '2')}, "
    f"slope_max={os.environ.get('CDR_PER_TERM_SLOPE_MAX', '10.0')}"
)
if _selection_method == "none":
    print(
        "[CDR-select] circuit selection: none "
        f"(original {os.environ.get('CDR_NUM_TRAINING_CIRCUITS', '30')} "
        "training circuits)"
    )
else:
    print(
        f"[CDR-select] circuit selection: {_selection_method}, "
        f"pool={os.environ.get('CDR_SELECTION_POOL_SIZE', '300')}, "
        f"selected={os.environ.get('CDR_NUM_TRAINING_CIRCUITS', '30')}, "
        f"local={os.environ.get('CDR_SELECTION_LOCAL_COUNT', '10')}"
    )
print(f"[CDR-select] results directory: {_SELECTED_DATA_DIR}")

runpy.run_path(str(STATE_TRANSFER_DIR / "main_HF.py"), run_name="__main__")
