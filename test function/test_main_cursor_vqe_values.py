from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import cirq
import numpy as np
import sympy
from numpy.testing import assert_allclose

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_cursor_lib import load_observable_h


EXPECTED_VQE_ENERGY = -3.061944645714
EXPECTED_GROUND_ENERGY = -3.062277999937
EXPECTED_ENERGY_DIFF = 3.333542222768e-04
EXPECTED_STATE_FIDELITY = 0.998270269221


def load_main_namespace(root: Path) -> dict:
    """Execute setup/build cells from main_cursor.ipynb without running noise simulation."""
    notebook = json.loads((root / "main_cursor.ipynb").read_text())
    namespace = {"__name__": "__main__"}

    # Execute config, Hamiltonian helpers, and clean ansatz construction.
    for cell_idx in (1, 2, 3):
        source = "".join(notebook["cells"][cell_idx]["source"])
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, f"main_cursor.ipynb cell {cell_idx}", "exec"), namespace)
        if cell_idx == 1:
            namespace["print_circuit"] = False

    return namespace


def prepare_original_fsim_ansatz_cirq(
    num_spatial_orbitals: int, num_layers: int = 1
) -> tuple[cirq.Circuit, list[cirq.GridQubit]]:
    qubits = [cirq.GridQubit(0, i) for i in range(num_spatial_orbitals)] + [
        cirq.GridQubit(1, i) for i in range(num_spatial_orbitals)
    ]
    circuit = cirq.Circuit()
    p_idx = 0
    circuit.append([cirq.X(qubits[i]) for i in range(1, len(qubits), 2)])

    for layer in range(num_layers):
        even_odd_moments = []
        for i in range(0, num_spatial_orbitals - 1, 2):
            theta = sympy.Symbol(f"th_{layer}_{p_idx}")
            even_odd_moments.append(cirq.FSimGate(theta, 0).on(qubits[i], qubits[i + 1]))
            even_odd_moments.append(
                cirq.FSimGate(theta, 0).on(
                    qubits[i + num_spatial_orbitals],
                    qubits[i + 1 + num_spatial_orbitals],
                )
            )
            p_idx += 1
        circuit.append(even_odd_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        odd_even_moments = []
        for i in range(1, num_spatial_orbitals - 1, 2):
            theta = sympy.Symbol(f"th_{layer}_{p_idx}")
            odd_even_moments.append(cirq.FSimGate(theta, 0).on(qubits[i], qubits[i + 1]))
            odd_even_moments.append(
                cirq.FSimGate(theta, 0).on(
                    qubits[i + num_spatial_orbitals],
                    qubits[i + 1 + num_spatial_orbitals],
                )
            )
            p_idx += 1
        circuit.append(odd_even_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        onsite_moments = []
        for i in range(num_spatial_orbitals):
            phi = sympy.Symbol(f"ph_{layer}_{p_idx}")
            onsite_moments.append(
                cirq.FSimGate(0, phi).on(qubits[i], qubits[i + num_spatial_orbitals])
            )
            p_idx += 1
        circuit.append(onsite_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit, qubits


def state_fidelity(state: np.ndarray, reference: np.ndarray) -> float:
    return float(abs(np.vdot(reference, state)) ** 2)


def run_value_check(root: Path, atol: float = 1e-7) -> dict[str, float]:
    ns = load_main_namespace(root)

    num_spatial = ns["num_spatial"]
    num_layers = ns["ansatz_layers"]
    vqe_parameters = ns["vqe_parameters"]
    ansatz_qubits = ns["ansatz_qubits"]
    decomposed_circuit = ns["ansatz_circuit"]
    symbols = ns["ordered_parameter_symbols"](num_spatial, num_layers)

    if num_layers != 8:
        raise AssertionError(f"Expected 8 layers from the 56-value vector, got {num_layers}.")
    if len(vqe_parameters) != 56:
        raise AssertionError(f"Expected 56 VQE parameters, got {len(vqe_parameters)}.")
    if len(symbols) != len(vqe_parameters):
        raise AssertionError(f"Expected {len(symbols)} symbols, got {len(vqe_parameters)} values.")

    original_circuit, original_qubits = prepare_original_fsim_ansatz_cirq(num_spatial, num_layers)
    if original_qubits != ansatz_qubits:
        raise AssertionError("Original and decomposed ansatz qubit order differs.")

    resolver = cirq.ParamResolver(dict(zip(symbols, vqe_parameters)))
    simulator = cirq.Simulator()

    original_state = simulator.simulate(
        cirq.resolve_parameters(original_circuit, resolver), qubit_order=ansatz_qubits
    ).final_state_vector
    decomposed_state = simulator.simulate(
        cirq.resolve_parameters(decomposed_circuit, resolver), qubit_order=ansatz_qubits
    ).final_state_vector
    if not np.isclose(abs(np.vdot(original_state, decomposed_state)), 1.0, atol=atol):
        raise AssertionError("Decomposed ansatz does not match original FSim ansatz.")

    observable_h = load_observable_h(root, ansatz_qubits, h_atom=4, bond_length=2.0)
    hamiltonian_matrix = observable_h.matrix(qubits=ansatz_qubits)
    vqe_energy = float(np.vdot(decomposed_state, hamiltonian_matrix @ decomposed_state).real)

    eigvals, eigvecs = np.linalg.eigh(hamiltonian_matrix)
    ground_idx = int(np.argmin(eigvals))
    ground_energy = float(eigvals[ground_idx].real)
    fidelity = state_fidelity(decomposed_state, eigvecs[:, ground_idx])
    energy_diff = vqe_energy - ground_energy

    assert_allclose(vqe_energy, EXPECTED_VQE_ENERGY, atol=atol, rtol=0.0)
    assert_allclose(ground_energy, EXPECTED_GROUND_ENERGY, atol=atol, rtol=0.0)
    assert_allclose(energy_diff, EXPECTED_ENERGY_DIFF, atol=atol, rtol=0.0)
    assert_allclose(fidelity, EXPECTED_STATE_FIDELITY, atol=atol, rtol=0.0)

    return {
        "vqe_energy": vqe_energy,
        "ground_energy": ground_energy,
        "energy_diff": energy_diff,
        "state_fidelity": fidelity,
    }


def test_main_cursor_vqe_values_regression() -> None:
    root = Path(__file__).resolve().parents[1]
    run_value_check(root, atol=1e-4)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the fixed main_cursor VQE parameter vector against expected values."
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-4,
        help="Absolute tolerance. Default reflects the rounded precision of the provided vector.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    results = run_value_check(root, atol=args.atol)

    print("PASS: Fixed VQE parameter vector agrees with expected reference values.")
    print(f"VQE Final Energy:    {results['vqe_energy']:.12f}")
    print(f"Exact Ground Energy: {results['ground_energy']:.12f}")
    print(f"Energy Difference:   {results['energy_diff']:.12e}")
    print(f"State Fidelity:      {results['state_fidelity']:.12f}")


if __name__ == "__main__":
    main()
