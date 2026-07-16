#!/usr/bin/env python3
"""Tests for additive Cl2 CMX outer-loop parallelism."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import cirq
import numpy as np
import sympy

REPO_ROOT = Path(__file__).resolve().parents[2]
JUNE_MAIN = REPO_ROOT / "June_main"
EXPORT_DIR = REPO_ROOT / "June_main" / "export2cloud"
for path in (EXPORT_DIR, JUNE_MAIN):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import parallel_cdr  # noqa: E402
import shot_measurement as sm  # noqa: E402
from parallel_cmx import measure_moment_grid_parallel, transform_cl2_source  # noqa: E402


def test_parallel_grid_preserves_repeat_order_and_shot_budgets(monkeypatch) -> None:
    monkeypatch.setenv("PARALLEL_CMX_WORKERS", "3")
    monkeypatch.setenv("PARALLEL_CMX_CDR_WORKERS", "2")
    monkeypatch.setenv("PARALLEL_CDR_WORKERS", "11")
    monkeypatch.setenv("PARALLEL_CDR_START_METHOD", "fork")

    lock = threading.Lock()
    active = 0
    max_active = 0
    call_index = 0

    def fake_measure(key: str, shots: int) -> dict:
        nonlocal active, max_active, call_index
        with lock:
            active += 1
            max_active = max(max_active, active)
            index = call_index
            call_index += 1
        try:
            time.sleep(0.03)
            return {"key": key, "shots": shots, "call_index": index}
        finally:
            with lock:
                active -= 1

    result = measure_moment_grid_parallel(
        measure_moment=fake_measure,
        shots_by_key={"H": 10, "H2": 20, "H3": 30},
        repeats=4,
    )

    assert max_active > 1
    assert list(result) == ["H", "H2", "H3"]
    assert all(len(result[key]) == 4 for key in result)
    assert [row["shots"] for row in result["H"]] == [10] * 4
    assert [row["shots"] for row in result["H2"]] == [20] * 4
    assert [row["shots"] for row in result["H3"]] == [30] * 4
    assert [row["key"] for row in result["H"]] == ["H"] * 4
    assert os.environ["PARALLEL_CDR_WORKERS"] == "11"
    assert os.environ["PARALLEL_CDR_START_METHOD"] == "fork"


def test_cl2_transform_replaces_only_measurement_grid() -> None:
    original_path = EXPORT_DIR / "main_Cl2.py"
    original = original_path.read_text(encoding="utf-8")
    transformed = transform_cl2_source(original)

    assert transformed != original
    assert "measure_moment_grid_parallel(" in transformed
    assert 'moment_replicates["H"].append(_measure_moment(' not in transformed
    compile(transformed, str(original_path), "exec")


def test_outer_threads_can_launch_inner_spawn_pools(monkeypatch) -> None:
    monkeypatch.setenv("PARALLEL_CMX_WORKERS", "2")
    monkeypatch.setenv("PARALLEL_CMX_CDR_WORKERS", "2")

    qubits = list(cirq.LineQubit.range(2))
    theta = sympy.Symbol("th_0")
    circuit = cirq.Circuit(cirq.ry(theta).on(qubits[0]), cirq.CZ(*qubits))
    observable = 0.7 * cirq.Z(qubits[0]) + 0.3 * cirq.X(qubits[0]) * cirq.X(qubits[1])
    resolvers = [{theta: 0.0}, {theta: np.pi / 2}]
    parallel_cdr._ORIGINAL_TRAIN_CF = sm.train_cf_models_per_pauli

    def measure(key: str, shots: int) -> dict:
        model = parallel_cdr.train_cf_models_per_pauli_parallel(
            circuit,
            observable,
            qubits,
            resolvers,
            noise_params={
                "two_qubit_depol_prob": 0.01,
                "one_qubit_depol_prob": 0.001,
                "cross_chip_two_qubit_depol_prob": 0.02,
            },
            num_shots=shots,
            measurement_scheme="direct_pauli",
            apply_readout_noise=False,
            sampling_seed=1234,
            simulator_seed=1234,
        )
        return {"key": key, "coeffs": model["coeffs_rem_to_exact_per_term"]}

    result = measure_moment_grid_parallel(
        measure_moment=measure,
        shots_by_key={"H": 16, "H2": 16, "H3": 16},
        repeats=1,
    )
    assert [result[key][0]["key"] for key in ("H", "H2", "H3")] == ["H", "H2", "H3"]


def test_cl2_wrapper_enables_transform() -> None:
    wrapper = (EXPORT_DIR / "main_Cl2_parallel.py").read_text(encoding="utf-8")
    assert "transform_cl2_source" in wrapper
    assert 'run_existing_main("main_Cl2.py", source_transform=transform_cl2_source)' in wrapper
