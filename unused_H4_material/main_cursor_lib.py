from __future__ import annotations

import pickle
from pathlib import Path
from typing import Iterable

import cirq
import numpy as np
import sympy

CZ_NORMAL_TAG = "cz_normal"
CZ_HIGH_TAG = "cz_high"
CZ_ONSITE_TAG = "cz_onsite_normal"

PAULI_CHAR_TO_GATE = {"I": None, "X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}

# Heuristic defaults loosely scaled to Google Weber/Sycamore typical medians (~0.1% 1Q RB,
# ~0.9% isolated 2Q XEB); see docs/noise_model.md. Not a certified hardware calibration.
DEFAULT_AMP_DAMP_GAMMA = 0
DEFAULT_PHASE_DAMP_GAMMA = 0
TWO_QUBIT_GATE_DEPOL_PROB = 0.005
ONE_QUBIT_GATE_DEPOL_PROB = 0.0001
DEFAULT_LEAKAGE_APPROX_PROB = 0
DEFAULT_HIGH_CZ_MULTIPLIER = 1.0


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
                    qubits[i + num_spatial_orbitals], qubits[i + 1 + num_spatial_orbitals]
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
                    qubits[i + num_spatial_orbitals], qubits[i + 1 + num_spatial_orbitals]
                )
            )
            p_idx += 1
        circuit.append(odd_even_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        onsite_moments = []
        for i in range(num_spatial_orbitals):
            phi = sympy.Symbol(f"ph_{layer}_{p_idx}")
            onsite_moments.append(cirq.FSimGate(0, phi).on(qubits[i], qubits[i + num_spatial_orbitals]))
            p_idx += 1
        circuit.append(onsite_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    return circuit, qubits


def operation_has_tag(operation: cirq.Operation, tag: str) -> bool:
    return tag in getattr(operation, "tags", ())


class LocationAwareDecomposedNoise(cirq.NoiseModel):
    def __init__(
        self,
        amp_damp_gamma: float = DEFAULT_AMP_DAMP_GAMMA,
        phase_damp_gamma: float = DEFAULT_PHASE_DAMP_GAMMA,
        depol_prob: float = TWO_QUBIT_GATE_DEPOL_PROB,
        one_qubit_depol_prob: float = ONE_QUBIT_GATE_DEPOL_PROB,
        high_cz_multiplier: float = DEFAULT_HIGH_CZ_MULTIPLIER,
        leakage_approx_prob: float = DEFAULT_LEAKAGE_APPROX_PROB,
    ):
        self.amp_damp_gamma = amp_damp_gamma
        self.phase_damp_gamma = phase_damp_gamma
        self.depol_prob = depol_prob
        self.high_cz_multiplier = high_cz_multiplier
        self.leakage_approx_prob = leakage_approx_prob

    def noisy_operation(self, operation: cirq.Operation):
        if isinstance(operation.gate, cirq.MeasurementGate):
            yield operation
            return

        if isinstance(operation.gate, cirq.CZPowGate):
            yield operation
            multiplier = self.high_cz_multiplier if operation_has_tag(operation, CZ_HIGH_TAG) else 1.0
            extra_depol = self.leakage_approx_prob if operation_has_tag(operation, CZ_HIGH_TAG) else 0.0
            for q in operation.qubits:
                yield cirq.amplitude_damp(min(1.0, self.amp_damp_gamma * multiplier)).on(q)
                yield cirq.phase_damp(min(1.0, self.phase_damp_gamma * multiplier)).on(q)
                total_depol = min(1.0, (self.depol_prob * multiplier) + extra_depol)
                yield cirq.depolarize(total_depol).on(q)
            return

        if len(operation.qubits) == 1:
            yield operation
            for q in operation.qubits:
                yield cirq.amplitude_damp(self.amp_damp_gamma / 10.0).on(q)
                yield cirq.phase_damp(self.phase_damp_gamma / 10.0).on(q)
                yield cirq.depolarize(self.one_qubit_depol_prob).on(q)
            return

        yield operation


def load_hamiltonian_paths(workspace: Path, h_atom: int, bond_length: float | int) -> tuple[Path, Path, Path]:
    local_folder = workspace / "Pauli_Ham"
    colab_folder = Path("/content/drive/My Drive/Quantum_chemistry/pauli_Ham")
    save_folder = local_folder if local_folder.exists() else colab_folder
    bond_token = f"{bond_length}".rstrip("0").rstrip(".") if isinstance(bond_length, float) else str(bond_length)
    pkl_path = save_folder / f"H{h_atom}_bond_{bond_token}.pkl"
    text_path = save_folder / f"H{h_atom}_bond_{bond_token}_pauli_convention.txt"
    return save_folder, pkl_path, text_path


def pauli_text_to_pauli_sum(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if len(lines) % 2 != 0:
        raise ValueError(f"Expected alternating Pauli/coeff lines in {path}")

    pauli_sum = cirq.PauliSum()
    for pauli_word, coeff_text in zip(lines[0::2], lines[1::2]):
        if len(pauli_word) != len(qubits):
            raise ValueError(
                f"Pauli word length {len(pauli_word)} does not match {len(qubits)} qubits: {pauli_word}"
            )
        coefficient = complex(coeff_text)
        pauli_string = cirq.PauliString(coefficient)
        for idx, pauli_char in enumerate(pauli_word):
            gate = PAULI_CHAR_TO_GATE[pauli_char]
            if gate is not None:
                pauli_string *= gate(qubits[idx])
        pauli_sum += pauli_string
    return pauli_sum


def qubit_operator_to_pauli_sum(qubit_operator, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    pauli_map = {"X": cirq.X, "Y": cirq.Y, "Z": cirq.Z}
    pauli_sum = cirq.PauliSum()
    for term, coefficient in qubit_operator.terms.items():
        pauli_string = cirq.PauliString(coefficient)
        for qubit_idx, operator_str in term:
            pauli_string *= pauli_map[operator_str](qubits[qubit_idx])
        pauli_sum += pauli_string
    return pauli_sum


def load_observable_h(
    workspace: Path, ansatz_qubits: list[cirq.Qid], h_atom: int, bond_length: float | int
) -> cirq.PauliSum:
    save_folder, pkl_path, text_path = load_hamiltonian_paths(workspace, h_atom, bond_length)
    if pkl_path.exists():
        try:
            import openfermion as of  # noqa: F401

            with pkl_path.open("rb") as f:
                h_qubit_loaded = pickle.load(f)
            return qubit_operator_to_pauli_sum(h_qubit_loaded, ansatz_qubits)
        except ModuleNotFoundError:
            pass

    if text_path.exists():
        return pauli_text_to_pauli_sum(text_path, ansatz_qubits)

    raise FileNotFoundError(
        f"Could not load Hamiltonian from {pkl_path} or {text_path} under {save_folder}."
    )


def trace_energy(hamiltonian: np.ndarray, rho: np.ndarray) -> float:
    return np.trace(hamiltonian @ rho).real


def states_equal_up_to_global_phase(lhs: np.ndarray, rhs: np.ndarray, atol: float = 1e-7) -> bool:
    return np.isclose(abs(np.vdot(lhs, rhs)), 1.0, atol=atol)


def is_cz_plus_single_qubit(operations: Iterable[cirq.Operation]) -> bool:
    for op in operations:
        if len(op.qubits) == 1:
            continue
        if len(op.qubits) == 2 and isinstance(op.gate, cirq.CZPowGate):
            if np.isclose(float(op.gate.exponent), 1.0):
                continue
        return False
    return True


# ---------------------------------------------------------------------------
# Clifford Data Regression (CDR) helpers
# ---------------------------------------------------------------------------
