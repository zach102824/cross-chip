#!/usr/bin/env python3
"""Run an existing cloud export with additive process-parallel CDR training."""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Callable

from parallel_cdr import install_parallel_cdr


def run_existing_main(
    filename: str,
    *,
    source_transform: Callable[[str], str] | None = None,
) -> None:
    """Install parallel CDR, then execute an existing main.

    ``source_transform`` is used only by the additive Cl2 wrapper to replace
    its local CMX measurement loop; the original file on disk is not changed.
    """
    export_dir = Path(__file__).resolve().parent
    original = export_dir / filename
    if not original.is_file():
        raise FileNotFoundError(original)

    # Prevent every worker process from creating its own BLAS thread team.  The
    # Slurm scripts set these before Python starts; these defaults also protect
    # direct local runs.
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

    install_parallel_cdr()
    if source_transform is None:
        runpy.run_path(str(original), run_name="__main__")
        return

    source = original.read_text(encoding="utf-8")
    transformed = source_transform(source)
    namespace = {
        "__name__": "__main__",
        "__file__": str(original),
        "__package__": None,
    }
    exec(compile(transformed, str(original), "exec"), namespace)
