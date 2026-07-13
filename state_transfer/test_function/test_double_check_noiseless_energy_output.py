"""Verbose noiseless-energy cross-check for the saved tapered HF circuit.

Run from the repository root with output enabled:

    .venv_h4_tencirchem/bin/python -m pytest \
        state_transfer/test_function/test_double_check_noiseless_energy_output.py \
        -s -v

This test intentionally prints every BFGS iteration. It compares:

1. the optimized statevector energy of the SAVED 6-qubit RZX circuit;
2. an independent product of the tapered Pauli exponentials used by
   ``state_transfer/tapered_doubles.ipynb`` (the blue ``e_tap_ss_Ha`` curve);
3. the notebook's displayed TenCirChem top-3 result (orange curve), for context.

The circuit uses ``pair=False``, so it must agree with (2) to numerical
precision. It is expected to be very close, but not identical, to (3), because
TenCirChem uses its spin-adapted excitation treatment.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

from conftest import STATE_TRANSFER, hf_state_vector


CIRCUIT_JSON = (
    STATE_TRANSFER / "circuits2read" / "HF_tapered_6q_3doubles_rzx.json"
)

# Values displayed in tapered_doubles.ipynb at d=1.0 A.
# The direct single-string value is independently recomputed below at full
# precision; these rounded values are included so terminal output aligns with
# the notebook table the user sees.
NOTEBOOK_TAPERED_SINGLE_STRING_DISPLAY = -98.596939
NOTEBOOK_TENCIRCHEM_TOP3_DISPLAY = -98.597022
NOTEBOOK_FCI_DISPLAY = -98.597486

CIRCUIT_VS_DIRECT_TOL_HA = 1e-8
DISPLAY_ROUNDING_TOL_HA = 5e-7


def _load_numbered_hamiltonian(path: Path, n_qubits: int) -> QubitOperator:
    code_to_pauli = {1: "X", 2: "Y", 3: "Z"}
    operator = QubitOperator()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        coefficient = float(parts[0])
        codes = [int(value) for value in parts[1:]]
        assert len(codes) == n_qubits
        term = tuple(
            (q, code_to_pauli[code])
            for q, code in enumerate(codes)
            if code != 0
        )
        operator += QubitOperator(term, coefficient)
    return operator


def _qiskit_to_openfermion_statevector(circuit) -> np.ndarray:
    """Convert Qiskit's little-endian statevector to qubit-0-as-MSB ordering."""
    from qiskit.quantum_info import Statevector

    qiskit_vector = Statevector(circuit).data
    n_qubits = circuit.num_qubits
    output = np.empty_like(qiskit_vector)
    for index, amplitude in enumerate(qiskit_vector):
        reversed_index = int(format(index, f"0{n_qubits}b")[::-1], 2)
        output[reversed_index] = amplitude
    return output


def test_double_check_noiseless_optimized_energy(taper_lib, gen):
    """Print circuit optimization and compare it with the notebook objective."""
    from qiskit import QuantumCircuit

    circuit_data = json.loads(CIRCUIT_JSON.read_text(encoding="utf-8"))
    n_qubits = int(circuit_data["num_qubits"])
    bond = float(circuit_data["bond_length"])
    occupied = [int(q) for q in circuit_data["hf_occupied_qubits"]]

    ham_path = (
        STATE_TRANSFER
        / "Pauli_Ham"
        / f"HF_tapered_bond_{bond:.10g}.txt"
    )
    hamiltonian = _load_numbered_hamiltonian(ham_path, n_qubits)
    h_sparse = get_sparse_operator(hamiltonian, n_qubits=n_qubits)
    h_dense = h_sparse.toarray()

    ansatz = gen.qc_from_logical_gates(circuit_data["gates"], n_qubits)
    parameters = sorted(ansatz.parameters, key=lambda parameter: parameter.name)
    prep = QuantumCircuit(n_qubits)
    for qubit in occupied:
        prep.x(qubit)

    evaluation_count = 0

    def circuit_energy(theta) -> float:
        nonlocal evaluation_count
        evaluation_count += 1
        bound = ansatz.assign_parameters(
            {parameter: float(value) for parameter, value in zip(parameters, theta)}
        )
        state = _qiskit_to_openfermion_statevector(prep.compose(bound))
        return float(np.real(np.vdot(state, h_dense @ state)))

    def circuit_parameter_shift_gradient(theta) -> np.ndarray:
        """Exact gradient for the single-occurrence Pauli rotations."""
        theta = np.asarray(theta, dtype=float)
        gradient = np.zeros_like(theta)
        for index in range(len(theta)):
            plus = theta.copy()
            minus = theta.copy()
            plus[index] += np.pi / 2.0
            minus[index] -= np.pi / 2.0
            gradient[index] = 0.5 * (
                circuit_energy(plus) - circuit_energy(minus)
            )
        return gradient

    initial_theta = np.zeros(len(parameters), dtype=float)
    initial_energy = circuit_energy(initial_theta)
    previous_energy = initial_energy
    iteration = 0

    print("\n=== Saved tapered RZX circuit: noiseless BFGS optimization ===")
    print(f"bond length              : {bond:.1f} A")
    print(f"qubits / parameters      : {n_qubits} / {len(parameters)}")
    print(f"RZX pairs                : {circuit_data['bridge_pair']} (all 3)")
    print(
        f"iter {iteration:02d}  E={initial_energy:.12f} Ha"
        f"  dE={0.0:+.3e}  theta={initial_theta.tolist()}"
    )

    def print_iteration(theta):
        nonlocal iteration, previous_energy
        iteration += 1
        energy = circuit_energy(theta)
        delta = energy - previous_energy
        print(
            f"iter {iteration:02d}  E={energy:.12f} Ha"
            f"  dE={delta:+.3e}  theta={np.asarray(theta).round(8).tolist()}"
        )
        previous_energy = energy

    result = minimize(
        circuit_energy,
        initial_theta,
        method="BFGS",
        jac=circuit_parameter_shift_gradient,
        callback=print_iteration,
        options={"gtol": 1e-7, "maxiter": 100},
    )
    circuit_optimum = float(result.fun)

    # Independent notebook objective: apply the tapered Pauli exponentials
    # directly, without using the generated Qiskit/RZX circuit.
    reference = hf_state_vector(n_qubits, occupied)
    strings = list(circuit_data["tapered_strings"])
    signs = [float(sign) for sign in circuit_data["signs"]]
    pauli_matrices = [
        get_sparse_operator(
            taper_lib.pauli_string_to_qubit_operator(string),
            n_qubits=n_qubits,
        )
        for string in strings
    ]

    def direct_notebook_energy(theta) -> float:
        state = reference.copy()
        for value, sign, pauli in zip(theta, signs, pauli_matrices):
            state = expm_multiply(-0.5j * float(value) * sign * pauli, state)
        return float(np.real(np.vdot(state, h_sparse @ state)))

    direct_result = minimize(
        direct_notebook_energy,
        initial_theta,
        method="BFGS",
        options={"gtol": 1e-7, "maxiter": 100},
    )
    direct_optimum = float(direct_result.fun)
    fci_energy = float(np.linalg.eigvalsh(h_dense)[0].real)

    print("\n=== Energy comparison ===")
    print(f"circuit optimized energy          : {circuit_optimum:.12f} Ha")
    print(f"direct tapered-Pauli optimum      : {direct_optimum:.12f} Ha")
    print(
        "circuit - direct               : "
        f"{(circuit_optimum - direct_optimum):+.3e} Ha"
    )
    print(
        "notebook displayed e_tap_ss     : "
        f"{NOTEBOOK_TAPERED_SINGLE_STRING_DISPLAY:.6f} Ha"
    )
    print(
        "notebook displayed TenCirChem   : "
        f"{NOTEBOOK_TENCIRCHEM_TOP3_DISPLAY:.6f} Ha"
    )
    print(f"exact tapered FCI                : {fci_energy:.12f} Ha")
    print(
        "circuit - TenCirChem top-3      : "
        f"{(circuit_optimum - NOTEBOOK_TENCIRCHEM_TOP3_DISPLAY) * 1000:+.6f} mHa"
    )
    print(
        "circuit - FCI                   : "
        f"{(circuit_optimum - fci_energy) * 1000:+.6f} mHa"
    )
    print(
        f"optimizer status                : success={result.success}, "
        f"nit={result.nit}, function_evals={evaluation_count}"
    )

    assert abs(circuit_optimum - direct_optimum) < CIRCUIT_VS_DIRECT_TOL_HA
    assert (
        abs(circuit_optimum - NOTEBOOK_TAPERED_SINGLE_STRING_DISPLAY)
        < DISPLAY_ROUNDING_TOL_HA
    )
    assert (
        abs(fci_energy - NOTEBOOK_FCI_DISPLAY)
        < DISPLAY_ROUNDING_TOL_HA
    )
