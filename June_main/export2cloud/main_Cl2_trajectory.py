#!/usr/bin/env python3
"""Trajectory noisy-sim entrypoint for the unchanged ``main_Cl2.py`` workflow."""

from parallel_cmx import transform_cl2_source
from trajectory_entrypoint import run_existing_main_trajectory


if __name__ == "__main__":
    run_existing_main_trajectory("main_Cl2.py", source_transform=transform_cl2_source)
