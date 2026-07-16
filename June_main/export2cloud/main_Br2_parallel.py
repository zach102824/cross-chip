#!/usr/bin/env python3
"""Process-parallel CDR entrypoint for the unchanged ``main_Br2.py`` workflow."""

from parallel_entrypoint import run_existing_main


if __name__ == "__main__":
    run_existing_main("main_Br2.py")
