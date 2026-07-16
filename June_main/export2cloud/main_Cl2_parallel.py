#!/usr/bin/env python3
"""Process-parallel CDR entrypoint for the unchanged ``main_Cl2.py`` workflow."""

from parallel_cmx import transform_cl2_source
from parallel_entrypoint import run_existing_main


if __name__ == "__main__":
    run_existing_main("main_Cl2.py", source_transform=transform_cl2_source)
