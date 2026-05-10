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
    prepare_original_fsim_ansatz_cirq,
)


def exact_energy(circuit: cirq.Circuit, qubits: list[cirq.Qid], resolver: cirq.ParamResolver, hmat: np.ndarray) -> float:
    state = cirq.Simulator(seed=123).simulate(
        cirq.resolve_parameters(circuit, resolver), qubit_order=qubits
    ).final_state_vector
    return float(np.vdot(state, hmat @ state).real)


def test_exact_energy_is_deterministic_and_matches_baseline() -> None:
    workspace = Path(__file__).resolve().parents[1]
    num_spatial = 4
    num_layers = 3

    dec_circuit, qubits = prepare_decomposed_ansatz_cirq(num_spatial, num_layers)
    base_circuit, base_qubits = prepare_original_fsim_ansatz_cirq(num_spatial, num_layers)
    assert qubits == base_qubits

    symbols = ordered_parameter_symbols(num_spatial, num_layers)
    params = np.linspace(-0.9, 0.8, len(symbols))
    resolver = cirq.ParamResolver(dict(zip(symbols, params)))

    obs = load_observable_h(workspace, qubits, h_atom=4, bond_length=2.0)
    hmat = obs.matrix(qubits=qubits)

    dec_e_1 = exact_energy(dec_circuit, qubits, resolver, hmat)
    dec_e_2 = exact_energy(dec_circuit, qubits, resolver, hmat)
    base_e = exact_energy(base_circuit, qubits, resolver, hmat)

    assert_allclose(dec_e_1, dec_e_2, atol=1e-6, rtol=0.0)
    assert_allclose(dec_e_1, base_e, atol=1e-5, rtol=0.0)


def main() -> None:
    test_exact_energy_is_deterministic_and_matches_baseline()
    print("PASS: exact noiseless deterministic parity test succeeded.")


if __name__ == "__main__":
    main()
