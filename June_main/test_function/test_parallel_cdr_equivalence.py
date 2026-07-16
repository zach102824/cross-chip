#!/usr/bin/env python3
"""Serial/parallel equivalence tests for the additive cloud entrypoints.

The numerical test uses an eight-qubit, three-parameter HF-sized toy problem.
It compares the original serial CDR training function against the new process
parallel function with identical circuits, resolvers, noise and random seeds.

Run locally (macOS uses multiprocessing ``spawn``):

    .venv_py311/bin/python -m pytest -q \
        June_main/test_function/test_parallel_cdr_equivalence.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import cirq
import numpy as np
import sympy

REPO_ROOT = Path(__file__).resolve().parents[2]
JUNE_MAIN = REPO_ROOT / "June_main"
EXPORT_DIR = JUNE_MAIN / "export2cloud"
for path in (EXPORT_DIR, JUNE_MAIN, REPO_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import parallel_cdr  # noqa: E402
import shot_measurement as sm  # noqa: E402


def _hf_sized_problem():
    qubits = list(cirq.LineQubit.range(8))
    symbols = list(sympy.symbols("th_0:3"))
    circuit = cirq.Circuit()
    circuit.append(cirq.X(qubits[i]) for i in range(3))
    circuit.append(cirq.ry(symbols[0]).on(qubits[1]))
    circuit.append(cirq.rz(symbols[1]).on(qubits[4]))
    circuit.append(cirq.rx(symbols[2]).on(qubits[6]))
    circuit.append(cirq.CZ(qubits[i], qubits[i + 1]) for i in range(7))
    circuit.append(cirq.CNOT(qubits[6], qubits[2]))

    observable = (
        -98.0 * cirq.PauliString()
        + 0.7 * cirq.Z(qubits[0])
        - 0.4 * cirq.Z(qubits[3])
        + 0.2 * cirq.X(qubits[1]) * cirq.X(qubits[2])
        + 0.1 * cirq.Y(qubits[5]) * cirq.Y(qubits[6])
    )
    resolvers = [
        {symbols[0]: 0.0, symbols[1]: 0.0, symbols[2]: 0.0},
        {symbols[0]: np.pi / 2, symbols[1]: 0.23, symbols[2]: 0.47},
        {symbols[0]: np.pi, symbols[1]: 0.61, symbols[2]: 0.91},
    ]
    return circuit, observable, qubits, resolvers


def _kwargs():
    return {
        "noise_params": {
            "two_qubit_depol_prob": 0.01,
            "one_qubit_depol_prob": 0.001,
            "cross_chip_two_qubit_depol_prob": 0.02,
        },
        "simulator_seed": 1234,
        "num_shots": 128,
        "measurement_scheme": "direct_pauli",
        "p_0_success": np.full(8, 0.97),
        "p_1_success": np.full(8, 0.90),
        "apply_readout_noise": True,
        "sampling_seed": 4321,
    }


def _assert_models_equal(serial: dict, parallel: dict) -> None:
    assert serial.keys() == parallel.keys()
    for key in serial:
        left = serial[key]
        right = parallel[key]
        if isinstance(left, np.ndarray):
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12)
        elif isinstance(left, list) and left and isinstance(left[0], (list, float)):
            np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12)
        else:
            assert left == right


def test_hf_sized_parallel_training_matches_original_serial(monkeypatch) -> None:
    circuit, observable, qubits, resolvers = _hf_sized_problem()
    serial = sm.train_cf_models_per_pauli(
        circuit,
        observable,
        qubits,
        resolvers,
        **_kwargs(),
    )

    monkeypatch.setenv("PARALLEL_CDR_WORKERS", "2")
    monkeypatch.setenv(
        "PARALLEL_CDR_START_METHOD",
        "spawn" if sys.platform == "darwin" else "fork",
    )
    parallel_cdr._ORIGINAL_TRAIN_CF = sm.train_cf_models_per_pauli
    parallel = parallel_cdr.train_cf_models_per_pauli_parallel(
        circuit,
        observable,
        qubits,
        resolvers,
        **_kwargs(),
    )
    _assert_models_equal(serial, parallel)


def test_one_worker_delegates_to_original(monkeypatch) -> None:
    circuit, observable, qubits, resolvers = _hf_sized_problem()
    monkeypatch.setenv("PARALLEL_CDR_WORKERS", "1")
    parallel_cdr._ORIGINAL_TRAIN_CF = sm.train_cf_models_per_pauli

    serial = sm.train_cf_models_per_pauli(
        circuit,
        observable,
        qubits,
        resolvers[:1],
        **_kwargs(),
    )
    delegated = parallel_cdr.train_cf_models_per_pauli_parallel(
        circuit,
        observable,
        qubits,
        resolvers[:1],
        **_kwargs(),
    )
    _assert_models_equal(serial, delegated)


def test_run_mitigation_dispatch_uses_parallel_hook(monkeypatch) -> None:
    circuit, observable, qubits, resolvers = _hf_sized_problem()
    target = resolvers[0]
    common = {
        "ansatz_circuit": circuit,
        "observable_h": observable,
        "qubits": qubits,
        "target_resolver": target,
        "target_params": target,
        "symbols": list(target),
        "base_noise_cfg": _kwargs()["noise_params"],
        "shot_cfg": {
            "num_shots": 128,
            "measurement_scheme": "direct_pauli",
            "apply_readout_noise": True,
            "sampling_seed": 4321,
        },
        "readout_cal": {
            "p_0_success": np.full(8, 0.97),
            "p_1_success": np.full(8, 0.90),
        },
        "cdr_cfg": {
            "num_circuits": 2,
            "t_max": 2,
            "seed": 42,
            "cdr_training": "random_clifford",
            "cdr_fit_scope": "per_pauli",
        },
        "simulator_seed": 1234,
    }
    original = sm.train_cf_models_per_pauli
    serial = sm.run_mitigation("cdr", **common)

    monkeypatch.setenv("PARALLEL_CDR_WORKERS", "2")
    monkeypatch.setenv(
        "PARALLEL_CDR_START_METHOD",
        "spawn" if sys.platform == "darwin" else "fork",
    )
    parallel_cdr._ORIGINAL_TRAIN_CF = original
    sm.train_cf_models_per_pauli = parallel_cdr.train_cf_models_per_pauli_parallel
    try:
        parallel = sm.run_mitigation("cdr", **common)
    finally:
        sm.train_cf_models_per_pauli = original

    for key in ("unmit_target", "rem_target", "cdr_unmit_corrected", "cdr_rem_corrected"):
        np.testing.assert_allclose(serial[key], parallel[key], rtol=0.0, atol=1e-12)
    _assert_models_equal(serial["cdr_models"], parallel["cdr_models"])


def test_parallel_entrypoints_are_additive_wrappers() -> None:
    expected = {
        "main_HF_parallel.py": "main_HF.py",
        "main_Cl2_parallel.py": "main_Cl2.py",
        "main_Br2_parallel.py": "main_Br2.py",
    }
    for wrapper_name, original_name in expected.items():
        wrapper = (EXPORT_DIR / wrapper_name).read_text(encoding="utf-8")
        assert f'run_existing_main("{original_name}"' in wrapper
        assert (EXPORT_DIR / original_name).is_file()


if __name__ == "__main__":
    # Keep direct execution useful on machines without pytest.
    os.environ.setdefault("PARALLEL_CDR_WORKERS", "2")
    os.environ.setdefault(
        "PARALLEL_CDR_START_METHOD",
        "spawn" if sys.platform == "darwin" else "fork",
    )
    problem = _hf_sized_problem()
    serial_result = sm.train_cf_models_per_pauli(*problem[:3], problem[3], **_kwargs())
    parallel_cdr._ORIGINAL_TRAIN_CF = sm.train_cf_models_per_pauli
    parallel_result = parallel_cdr.train_cf_models_per_pauli_parallel(
        *problem[:3], problem[3], **_kwargs()
    )
    _assert_models_equal(serial_result, parallel_result)
    print("PASS: HF-sized serial and parallel CDR results match")
