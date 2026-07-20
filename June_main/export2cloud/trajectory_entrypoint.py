#!/usr/bin/env python3
"""Run an existing cloud export with trajectory noisy simulation (+ optional parallel CDR)."""

from __future__ import annotations

import os
import runpy
from pathlib import Path
from typing import Callable

from parallel_cdr import install_parallel_cdr
from trajectory_backend import install_trajectory_backend


def run_existing_main_trajectory(
    filename: str,
    *,
    source_transform: Callable[[str], str] | None = None,
    install_parallel: bool = True,
) -> None:
    """Install trajectory backend (and usually parallel CDR), then execute a main."""
    export_dir = Path(__file__).resolve().parent
    original = export_dir / filename
    if not original.is_file():
        raise FileNotFoundError(original)

    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ.setdefault(variable, "1")

    install_trajectory_backend()
    if install_parallel:
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
