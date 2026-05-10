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
    trace_energy,
)


def test_decomposed_matches_original_fsim_noiseless_energy() -> None:
    workspace = Path(__file__).resolve().parents[1]
    num_spatial = 4
    num_layers = 2

    dec_circuit, dec_qubits = prepare_decomposed_ansatz_cirq(num_spatial, num_layers)
    orig_circuit, orig_qubits = prepare_original_fsim_ansatz_cirq(num_spatial, num_layers)
    assert dec_qubits == orig_qubits
    qubits = dec_qubits

    symbols = ordered_parameter_symbols(num_spatial, num_layers)
    params = np.linspace(-1.2, 1.2, len(symbols))
    resolver = cirq.ParamResolver(dict(zip(symbols, params)))

    sim = cirq.Simulator(seed=10)
    dec_state = sim.simulate(cirq.resolve_parameters(dec_circuit, resolver), qubit_order=qubits).final_state_vector
    orig_state = sim.simulate(cirq.resolve_parameters(orig_circuit, resolver), qubit_order=qubits).final_state_vector

    obs = load_observable_h(workspace, qubits, h_atom=4, bond_length=2.0)
    hmat = obs.matrix(qubits=qubits)
    dec_energy = trace_energy(hmat, np.outer(dec_state, dec_state.conj()))
    orig_energy = trace_energy(hmat, np.outer(orig_state, orig_state.conj()))

    assert_allclose(dec_energy, orig_energy, atol=1e-5, rtol=0.0)


def main() -> None:
    test_decomposed_matches_original_fsim_noiseless_energy()
    print("PASS: decomposed and original FSim paths match noiseless energy.")


if __name__ == "__main__":
    main()
