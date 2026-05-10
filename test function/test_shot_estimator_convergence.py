from __future__ import annotations

import numpy as np
import cirq
import sys
from pathlib import Path
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shot_measurement import estimate_energy_from_noisy_rho_shots


def test_direct_pauli_converges_to_trace_without_readout_noise() -> None:
    rng = np.random.default_rng(21)
    qubits = [cirq.LineQubit(i) for i in range(2)]
    # random pure state
    psi = rng.normal(size=4) + 1j * rng.normal(size=4)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())

    observable_h = 0.5 * cirq.X(qubits[0]) * cirq.Y(qubits[1]) - 0.3 * cirq.Z(qubits[0]) + 0.2
    hmat = observable_h.matrix(qubits=qubits)
    exact = float(np.trace(hmat @ rho).real)

    est = estimate_energy_from_noisy_rho_shots(
        rho,
        observable_h,
        qubits,
        num_shots=200_000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=101,
    )

    assert_allclose(est["energy_unmitigated"], exact, atol=8e-3, rtol=0.0)


def main() -> None:
    test_direct_pauli_converges_to_trace_without_readout_noise()
    print("PASS: finite-shot direct-pauli estimator converges to trace energy.")


if __name__ == "__main__":
    main()
