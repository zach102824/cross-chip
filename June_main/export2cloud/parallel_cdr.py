#!/usr/bin/env python3
"""Additive process parallelism for the CDR training-circuit loop.

The existing ``main_*.py``, ``shot_measurement.py`` and ``main_cursor_lib.py``
files remain unchanged.  Parallel entrypoints call :func:`install_parallel_cdr`
before executing an existing main script.  The installer replaces only
``shot_measurement.train_cf_models_per_pauli``; all resolver generation,
target-energy evaluation, fitting, VQE updates, checkpointing and output code
continue to use the original implementation.

Each near-Clifford training resolver is independent, making this loop a safe
CPU process boundary.  ``executor.map`` preserves resolver order, so fixed
seeds produce the same arrays and fits as the serial implementation.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np

_WORKER_CONTEXT: dict[str, Any] = {}
_ORIGINAL_TRAIN_CF = None


def _ensure_shared_modules_importable() -> None:
    export_dir = Path(__file__).resolve().parent
    # export2cloud first so trajectory_sampling / parallel_* imports resolve.
    export_value = str(export_dir)
    if export_value not in sys.path:
        sys.path.insert(0, export_value)
    candidates = (export_dir / "June_main", export_dir.parent, export_dir)
    for candidate in candidates:
        if (candidate / "shot_measurement.py").is_file():
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
            return
    raise ImportError("Could not locate June_main/shot_measurement.py")


def configured_worker_count(num_tasks: int | None = None) -> int:
    """Return the requested process count, capped by the number of tasks."""
    raw = os.environ.get(
        "PARALLEL_CDR_WORKERS",
        os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count() or 1)),
    )
    workers = max(1, int(raw))
    if num_tasks is not None:
        workers = min(workers, max(1, int(num_tasks)))
    return workers


def configured_start_method() -> str:
    """Use spawn on macOS; use fork on Linux unless explicitly overridden."""
    requested = os.environ.get("PARALLEL_CDR_START_METHOD")
    if requested:
        if requested not in mp.get_all_start_methods():
            raise ValueError(
                f"Unsupported PARALLEL_CDR_START_METHOD={requested!r}; "
                f"available={mp.get_all_start_methods()}"
            )
        return requested
    return "spawn" if sys.platform == "darwin" else "fork"


def _init_training_worker(context: dict[str, Any]) -> None:
    global _WORKER_CONTEXT
    import cirq

    qubits = context["qubits"]
    paulis = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    observable_h = cirq.PauliSum()
    offset = float(context["hamiltonian_offset"])
    if offset:
        observable_h += offset * cirq.PauliString()
    for row, weight in zip(context["observables_int"], context["weights"]):
        term = cirq.PauliString(
            {
                qubits[index]: paulis[int(code)]
                for index, code in enumerate(row)
                if int(code) != 0
            },
            coefficient=float(weight),
        )
        observable_h += term
    _WORKER_CONTEXT = dict(context)
    _WORKER_CONTEXT["observable_h"] = observable_h


def _evaluate_training_resolver(task: tuple[int, dict]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, int]:
    """Evaluate one resolver using the original serial primitives."""
    _ensure_shared_modules_importable()
    import shot_measurement as sm

    index, resolver = task
    ctx = _WORKER_CONTEXT
    circuit = ctx["ansatz_circuit"]
    observable_h = ctx["observable_h"]
    qubits = ctx["qubits"]
    observables_int = ctx["observables_int"]

    state = sm._simulate_noiseless_state_for_resolver(
        circuit,
        resolver,
        qubits,
        simulator_seed=ctx["simulator_seed"],
    )
    exact = np.asarray(
        [
            sm.exact_pauli_expectation_from_int_row(state, row, qubits)
            for row in observables_int
        ],
        dtype=float,
    )

    estimate = sm.estimate_noisy_shots_for_resolver(
        circuit,
        resolver,
        observable_h,
        qubits,
        ctx["noise_params"],
        simulator_seed=ctx["simulator_seed"],
        num_shots=ctx["num_shots"],
        measurement_scheme=ctx["measurement_scheme"],
        p_0_success=ctx["p_0_success"],
        p_1_success=ctx["p_1_success"],
        apply_rem=True,
        apply_readout_noise=ctx["apply_readout_noise"],
        sampling_seed=ctx["sampling_seed"],
        epsilon=ctx["epsilon"],
        ogm_file=ctx["ogm_file"],
        shadowgrouping_root=ctx["shadowgrouping_root"],
        return_per_term=True,
    )
    return (
        int(index),
        exact,
        np.asarray(estimate["per_term_unmitigated"], dtype=float),
        np.asarray(estimate["per_term_rem"], dtype=float),
        int(sm.count_non_clifford_ops(circuit, resolver)),
    )


def train_cf_models_per_pauli_parallel(
    ansatz_circuit,
    observable_h,
    qubits,
    resolvers,
    *,
    noise_params,
    simulator_seed: int = 1234,
    num_shots: int = 8192,
    measurement_scheme: str = "ogm",
    p_0_success: Iterable[float] | None = None,
    p_1_success: Iterable[float] | None = None,
    apply_readout_noise: bool = True,
    sampling_seed: int = 1234,
    epsilon: float = 0.1,
    ogm_file: str | Path | None = None,
    shadowgrouping_root: str | Path | None = None,
) -> dict[str, object]:
    """Parallel equivalent of ``train_cf_models_per_pauli``."""
    _ensure_shared_modules_importable()
    import shot_measurement as sm

    if not resolvers:
        raise ValueError("At least one resolver is required to train per-Pauli CF models.")

    observables_int, weights, offset = sm.pauli_sum_to_int_observables(observable_h, qubits)
    n_terms = len(weights)
    if n_terms == 0:
        return {
            "fit_scope": "per_pauli",
            "hamiltonian_offset": float(offset),
            "weights": [],
            "coeffs_unmit_to_exact_per_term": [],
            "coeffs_rem_to_exact_per_term": [],
            "r2_rem_to_exact_per_term": [],
            "training_t_remaining": [0] * len(resolvers),
            "training_exact_per_term": np.zeros((len(resolvers), 0), dtype=float),
            "training_unmit_per_term": np.zeros((len(resolvers), 0), dtype=float),
            "training_rem_per_term": np.zeros((len(resolvers), 0), dtype=float),
        }

    workers = configured_worker_count(len(resolvers))
    if workers == 1:
        assert _ORIGINAL_TRAIN_CF is not None
        return _ORIGINAL_TRAIN_CF(
            ansatz_circuit,
            observable_h,
            qubits,
            resolvers,
            noise_params=noise_params,
            simulator_seed=simulator_seed,
            num_shots=num_shots,
            measurement_scheme=measurement_scheme,
            p_0_success=p_0_success,
            p_1_success=p_1_success,
            apply_readout_noise=apply_readout_noise,
            sampling_seed=sampling_seed,
            epsilon=epsilon,
            ogm_file=ogm_file,
            shadowgrouping_root=shadowgrouping_root,
        )

    context = {
        "ansatz_circuit": ansatz_circuit,
        "qubits": list(qubits),
        "observables_int": observables_int,
        "weights": np.asarray(weights, dtype=float),
        "hamiltonian_offset": float(offset),
        "noise_params": dict(noise_params),
        "simulator_seed": int(simulator_seed),
        "num_shots": int(num_shots),
        "measurement_scheme": str(measurement_scheme),
        "p_0_success": None if p_0_success is None else np.asarray(p_0_success, dtype=float),
        "p_1_success": None if p_1_success is None else np.asarray(p_1_success, dtype=float),
        "apply_readout_noise": bool(apply_readout_noise),
        "sampling_seed": int(sampling_seed),
        "epsilon": float(epsilon),
        "ogm_file": ogm_file,
        "shadowgrouping_root": shadowgrouping_root,
    }
    tasks = list(enumerate(resolvers))
    mp_context = mp.get_context(configured_start_method())
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp_context,
        initializer=_init_training_worker,
        initargs=(context,),
    ) as executor:
        rows = list(executor.map(_evaluate_training_resolver, tasks, chunksize=1))

    rows.sort(key=lambda item: item[0])
    tex_exact = np.stack([item[1] for item in rows], axis=0)
    tunmit = np.stack([item[2] for item in rows], axis=0)
    trem = np.stack([item[3] for item in rows], axis=0)
    t_rem_list = [int(item[4]) for item in rows]

    coeffs_unmit: list[list[float]] = []
    coeffs_rem: list[list[float]] = []
    r2_rem: list[float] = []
    for k in range(n_terms):
        xu = tunmit[:, k]
        xr = trem[:, k]
        y = tex_exact[:, k]
        if len(resolvers) >= 2 and float(np.std(xu)) > 0.0:
            cu = np.polyfit(xu, y, deg=1)
        else:
            cu = np.array([1.0, float(np.mean(y - xu))])
        if len(resolvers) >= 2 and float(np.std(xr)) > 0.0:
            cr = np.polyfit(xr, y, deg=1)
        else:
            cr = np.array([1.0, float(np.mean(y - xr))])
        coeffs_unmit.append([float(cu[0]), float(cu[1])])
        coeffs_rem.append([float(cr[0]), float(cr[1])])
        y_pred = float(cr[0]) * xr + float(cr[1])
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2_rem.append(sm.stable_r2_from_sums(ss_res, ss_tot, ss_res_tol=1e-4))

    return {
        "fit_scope": "per_pauli",
        "hamiltonian_offset": float(offset),
        "weights": [float(w) for w in weights],
        "coeffs_unmit_to_exact_per_term": coeffs_unmit,
        "coeffs_rem_to_exact_per_term": coeffs_rem,
        "r2_rem_to_exact_per_term": r2_rem,
        "training_t_remaining": t_rem_list,
        "training_exact_per_term": tex_exact,
        "training_unmit_per_term": tunmit,
        "training_rem_per_term": trem,
    }


def install_parallel_cdr() -> None:
    """Patch the imported dispatcher module once, leaving source files untouched."""
    global _ORIGINAL_TRAIN_CF
    _ensure_shared_modules_importable()
    import shot_measurement as sm

    if getattr(sm.train_cf_models_per_pauli, "_parallel_cdr_impl", False):
        return
    _ORIGINAL_TRAIN_CF = sm.train_cf_models_per_pauli
    train_cf_models_per_pauli_parallel._parallel_cdr_impl = True  # type: ignore[attr-defined]
    sm.train_cf_models_per_pauli = train_cf_models_per_pauli_parallel
    print(
        "[parallel-cdr] installed: "
        f"workers={configured_worker_count()}, start_method={configured_start_method()}, "
        "BLAS threads should be set to 1"
    )
