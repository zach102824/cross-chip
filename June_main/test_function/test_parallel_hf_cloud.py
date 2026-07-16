#!/usr/bin/env python3
"""Opt-in serial/parallel equivalence test using the real HF cloud assets.

This executes the unchanged setup portion of ``main_HF.py`` (through the CDR
setup, before VQE/CMX), then runs one identical CDR mitigation call through the
original serial trainer and the additive parallel trainer.

It is intentionally excluded from normal pytest runs because the real 8-qubit
density-matrix calculation is much slower than a unit test.

Cloud/local command:

    RUN_FULL_HF_PARALLEL_TEST=1 \
    GLOBAL_NUM_SHOTS=128 \
    CDR_NUM_TRAINING_CIRCUITS=2 \
    PARALLEL_CDR_WORKERS=2 \
    python -u June_main/test_function/test_parallel_hf_cloud.py

Increase workers/training circuits after this equivalence check when benchmarking.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
JUNE_MAIN = REPO_ROOT / "June_main"
EXPORT_DIR = JUNE_MAIN / "export2cloud"
for path in (EXPORT_DIR, JUNE_MAIN, REPO_ROOT):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)

import parallel_cdr  # noqa: E402
import shot_measurement as sm  # noqa: E402


def _assert_close(left, right, path: str = "result") -> None:
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            _assert_close(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, (list, tuple, np.ndarray)):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12, err_msg=path)
    elif isinstance(left, (float, np.floating)):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-12, err_msg=path)
    else:
        assert left == right, path


def _load_real_hf_setup() -> dict:
    source_path = EXPORT_DIR / "main_HF.py"
    source = source_path.read_text(encoding="utf-8")
    marker = "# %% Notebook cell 15"
    if marker not in source:
        raise RuntimeError(f"Could not find setup cutoff marker in {source_path}")
    setup_source = source.split(marker, 1)[0]
    namespace = {
        "__name__": "hf_parallel_equivalence_setup",
        "__file__": str(source_path),
    }
    exec(compile(setup_source, str(source_path), "exec"), namespace)
    return namespace


def run_real_hf_equivalence() -> None:
    if os.environ.get("RUN_FULL_HF_PARALLEL_TEST") != "1":
        raise SystemExit(
            "Set RUN_FULL_HF_PARALLEL_TEST=1 to run the real HF integration test."
        )

    os.environ.setdefault("GLOBAL_NUM_SHOTS", "128")
    os.environ.setdefault("CDR_NUM_TRAINING_CIRCUITS", "2")
    os.environ.setdefault("MEASUREMENT_SCHEME", "direct_pauli")
    os.environ.setdefault("PARALLEL_CDR_WORKERS", "2")
    os.environ.setdefault(
        "PARALLEL_CDR_START_METHOD",
        "spawn" if sys.platform == "darwin" else "fork",
    )

    old_cwd = Path.cwd()
    original_train = sm.train_cf_models_per_pauli
    try:
        ns = _load_real_hf_setup()
        params = np.zeros(int(ns["n_params"]), dtype=float)
        resolver = ns["resolver_from_params"](params)
        shot_cfg = dict(ns["shot_cfg"])
        shot_cfg["sampling_seed"] = int(ns["GLOBAL_SAMPLING_SEED"])
        cdr_cfg = dict(ns["cdr_cfg_base"])
        cdr_cfg["num_circuits"] = int(os.environ["CDR_NUM_TRAINING_CIRCUITS"])

        kwargs = {
            "ansatz_circuit": ns["circuit"],
            "observable_h": ns["pauli_sum"],
            "qubits": ns["qubits"],
            "target_resolver": resolver,
            "target_params": resolver,
            "symbols": ns["symbols"],
            "base_noise_cfg": dict(ns["base_noise_cfg"]),
            "shot_cfg": shot_cfg,
            "readout_cal": dict(ns["readout_cal"]),
            "cdr_cfg": cdr_cfg,
            "simulator_seed": int(ns["GLOBAL_RANDOM_SEED"]),
        }

        sm.train_cf_models_per_pauli = original_train
        serial_started = time.perf_counter()
        serial = sm.run_mitigation("cdr", **kwargs)
        serial_seconds = time.perf_counter() - serial_started

        parallel_cdr._ORIGINAL_TRAIN_CF = original_train
        sm.train_cf_models_per_pauli = parallel_cdr.train_cf_models_per_pauli_parallel
        parallel_started = time.perf_counter()
        parallel = sm.run_mitigation("cdr", **kwargs)
        parallel_seconds = time.perf_counter() - parallel_started

        for key in (
            "unmit_target",
            "rem_target",
            "cdr_unmit_corrected",
            "cdr_rem_corrected",
        ):
            _assert_close(serial[key], parallel[key], key)
        _assert_close(serial["cdr_models"], parallel["cdr_models"], "cdr_models")
    finally:
        sm.train_cf_models_per_pauli = original_train
        os.chdir(old_cwd)

    print(
        "PASS: real HF serial and parallel CDR results match; "
        f"serial={serial_seconds:.2f}s, parallel={parallel_seconds:.2f}s, "
        f"speedup={serial_seconds / parallel_seconds:.2f}x"
    )


if __name__ == "__main__":
    run_real_hf_equivalence()
