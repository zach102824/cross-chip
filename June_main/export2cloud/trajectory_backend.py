#!/usr/bin/env python3
"""Enable Monte Carlo trajectory noisy simulation for cloud export mains."""

from __future__ import annotations

import os


def _install_skip_dm_diagnostics() -> None:
    """Replace leftover demo DensityMatrixSimulator calls with a cheap stub.

    VQE/CMX objectives use ``estimate_noisy_shots_for_resolver`` (trajectory /
    Stim). Early notebook cells and CMX ``exact`` diagnostics still call
    ``DensityMatrixSimulator``; those are skipped when
    ``SKIP_DM_DIAGNOSTICS`` is truthy (default on for trajectory entrypoints).
    """
    raw = str(os.environ.get("SKIP_DM_DIAGNOSTICS", "1")).strip().lower()
    if raw in ("0", "false", "no", "off"):
        return

    import cirq
    import numpy as np

    if getattr(cirq.DensityMatrixSimulator, "_cross_chip_skip_dm_patched", False):
        return

    _Orig = cirq.DensityMatrixSimulator

    class _SkipDMDiagnosticsSimulator(_Orig):  # type: ignore[misc,valid-type]
        _cross_chip_skip_dm_patched = True

        def simulate(self, program, *args, qubit_order=None, **kwargs):
            qubits = list(qubit_order) if qubit_order is not None else sorted(program.all_qubits())
            n = len(qubits)
            dim = 2**n
            rho = np.zeros((dim, dim), dtype=np.complex128)
            rho[0, 0] = 1.0

            class _Result:
                final_density_matrix = rho

            return _Result()

    _SkipDMDiagnosticsSimulator._cross_chip_skip_dm_patched = True
    cirq.DensityMatrixSimulator = _SkipDMDiagnosticsSimulator  # type: ignore[misc]
    print("[trajectory] SKIP_DM_DIAGNOSTICS=1 (demo DensityMatrixSimulator stubbed)")


def install_trajectory_backend() -> None:
    """Select the trajectory noisy-sim backend via ``NOISY_SIM_BACKEND``.

    Existing ``main_*.py`` scripts keep calling ``run_mitigation`` /
    ``estimate_noisy_shots_for_resolver``; this only switches the inner
    density-matrix path to independent statevector trajectories that realize
    the same ``GateArityDepolarizingNoise`` channel.
    """
    os.environ["NOISY_SIM_BACKEND"] = "trajectory"
    os.environ.setdefault("SKIP_DM_DIAGNOSTICS", "1")
    os.environ.setdefault("USE_STIM_CLIFFORD", "1")
    _install_skip_dm_diagnostics()
    print("[trajectory] NOISY_SIM_BACKEND=trajectory (no DensityMatrixSimulator on hot path)")
    if str(os.environ.get("USE_STIM_CLIFFORD", "1")).strip() not in ("0", "false", "no", "off"):
        try:
            import stim  # noqa: F401

            print("[trajectory] USE_STIM_CLIFFORD=1 (Stim near-Clifford trainer path enabled)")
        except ImportError:
            print("[trajectory] Stim not installed; using NumPy trajectories only")
