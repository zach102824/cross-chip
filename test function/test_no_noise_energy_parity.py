from __future__ import annotations

from pathlib import Path

import cirq
import numpy as np
import sys
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_cursor_lib import (
    load_observable_h,
    ordered_parameter_symbols,
    prepare_decomposed_ansatz_cirq,
    trace_energy,
)
from shot_measurement import estimate_energy_from_noisy_rho_shots


def test_no_gate_noise_local_vs_shot_estimator() -> None:
    workspace = Path(__file__).resolve().parents[1]
    num_spatial = 4
    num_layers = 1

    circuit, qubits = prepare_decomposed_ansatz_cirq(num_spatial, num_layers)
    symbols = ordered_parameter_symbols(num_spatial, num_layers)
    params = np.linspace(-0.5, 0.9, len(symbols))
    resolver = cirq.ParamResolver(dict(zip(symbols, params)))
    state = cirq.Simulator(seed=5).simulate(
        cirq.resolve_parameters(circuit, resolver), qubit_order=qubits
    ).final_state_vector
    rho = np.outer(state, state.conj())

    obs = load_observable_h(workspace, qubits, h_atom=4, bond_length=2.0)
    exact = trace_energy(obs.matrix(qubits=qubits), rho)

    est = estimate_energy_from_noisy_rho_shots(
        rho,
        obs,
        qubits,
        num_shots=120_000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=99,
    )

    assert_allclose(est["energy_unmitigated"], exact, atol=1.5e-2, rtol=0.0)


def main() -> None:
    test_no_gate_noise_local_vs_shot_estimator()
    print("PASS: no-noise local vs shot-estimator energy parity.")


if __name__ == "__main__":
    main()
