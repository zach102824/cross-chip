#!/usr/bin/env python3
"""Trajectory noisy-sim entrypoint for the unchanged ``main_Br2.py`` workflow."""

from parallel_cmx import transform_br2_source
from trajectory_entrypoint import run_existing_main_trajectory


if __name__ == "__main__":
    # Parallel CMX transform is applied; CMX itself remains disabled inside main_Br2.py
    # until ``_run_cmx_disabled()`` is uncommented.
    run_existing_main_trajectory("main_Br2.py", source_transform=transform_br2_source)
