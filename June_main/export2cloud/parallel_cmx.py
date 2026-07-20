#!/usr/bin/env python3
"""Additive hybrid parallelism for the Cl2 CMX moment sweep.

CMX has two independent dimensions:

* outer: repeat × moment (H, H2, H3);
* inner: near-Clifford CDR training circuits for one moment.

The outer level uses threads so it can call the local ``_measure_moment``
function defined by the unchanged Cl2 script.  Each thread's CDR call uses the
existing process-parallel trainer with a smaller worker count.  Inner pools use
``spawn`` because forking from an outer worker thread is unsafe.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Callable


def _positive_env_int(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def measure_moment_grid_parallel(
    *,
    measure_moment: Callable[[str, int], dict],
    shots_by_key: dict[str, int],
    repeats: int,
) -> dict[str, list[dict]]:
    """Measure one multiplier's repeat × moment grid concurrently.

    Results are returned in the same per-key repeat order as the original
    nested loops, independent of completion order.
    """
    repeats = int(repeats)
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    keys = ("H", "H2", "H3")
    missing = [key for key in keys if key not in shots_by_key]
    if missing:
        raise KeyError(f"Missing CMX shot budgets for: {missing}")

    outer_workers = _positive_env_int("PARALLEL_CMX_WORKERS", 7)
    inner_workers = _positive_env_int("PARALLEL_CMX_CDR_WORKERS", 4)
    outer_workers = min(outer_workers, repeats * len(keys))

    previous_workers = os.environ.get("PARALLEL_CDR_WORKERS")
    previous_start_method = os.environ.get("PARALLEL_CDR_START_METHOD")
    os.environ["PARALLEL_CDR_WORKERS"] = str(inner_workers)
    os.environ["PARALLEL_CDR_START_METHOD"] = "spawn"

    tasks = [
        (rep, key, int(shots_by_key[key]))
        for rep in range(repeats)
        for key in keys
    ]

    def run_task(task: tuple[int, str, int]) -> tuple[int, str, dict]:
        rep, key, shots = task
        print(f"[CMX-parallel] repeat {rep + 1}/{repeats}, moment={key}")
        return rep, key, measure_moment(key, shots)

    try:
        with ThreadPoolExecutor(
            max_workers=outer_workers,
            thread_name_prefix="cmx",
        ) as executor:
            rows = list(executor.map(run_task, tasks))
    finally:
        if previous_workers is None:
            os.environ.pop("PARALLEL_CDR_WORKERS", None)
        else:
            os.environ["PARALLEL_CDR_WORKERS"] = previous_workers
        if previous_start_method is None:
            os.environ.pop("PARALLEL_CDR_START_METHOD", None)
        else:
            os.environ["PARALLEL_CDR_START_METHOD"] = previous_start_method

    grouped = {key: [None] * repeats for key in keys}
    for rep, key, result in rows:
        grouped[key][rep] = result
    return {key: list(grouped[key]) for key in keys}


_CL2_SERIAL_BLOCK = """            moment_replicates = {key: [] for key in ("H", "H2", "H3")}
            for rep in range(CME_VARIANCE_REPEATS):
                print(f"[CMX] variance repeat {rep + 1}/{CME_VARIANCE_REPEATS}")
                moment_replicates["H"].append(_measure_moment("H", CME_H_NUM_SHOTS))
                moment_replicates["H2"].append(_measure_moment("H2", h2_h3_shots))
                moment_replicates["H3"].append(_measure_moment("H3", h2_h3_shots))
"""

_CL2_PARALLEL_BLOCK = """            from parallel_cmx import measure_moment_grid_parallel
            moment_replicates = measure_moment_grid_parallel(
                measure_moment=_measure_moment,
                shots_by_key={
                    "H": CME_H_NUM_SHOTS,
                    "H2": h2_h3_shots,
                    "H3": h2_h3_shots,
                },
                repeats=CME_VARIANCE_REPEATS,
            )
"""


def transform_cl2_source(source: str) -> str:
    """Replace only the Cl2 CMX serial measurement grid."""
    occurrences = source.count(_CL2_SERIAL_BLOCK)
    if occurrences != 1:
        raise RuntimeError(
            "Expected exactly one Cl2 CMX serial loop to replace, "
            f"found {occurrences}. The original script layout may have changed."
        )
    return source.replace(_CL2_SERIAL_BLOCK, _CL2_PARALLEL_BLOCK, 1)


# Br2's CMX serial grid text matches Cl2; reuse the same rewrite when CMX is enabled.
_BR2_SERIAL_BLOCK = _CL2_SERIAL_BLOCK
_BR2_PARALLEL_BLOCK = _CL2_PARALLEL_BLOCK


def transform_br2_source(source: str) -> str:
    """Replace only the Br2 CMX serial measurement grid (same layout as Cl2)."""
    occurrences = source.count(_BR2_SERIAL_BLOCK)
    if occurrences != 1:
        raise RuntimeError(
            "Expected exactly one Br2 CMX serial loop to replace, "
            f"found {occurrences}. The original script layout may have changed."
        )
    return source.replace(_BR2_SERIAL_BLOCK, _BR2_PARALLEL_BLOCK, 1)
