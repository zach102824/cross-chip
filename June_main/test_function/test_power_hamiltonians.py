#!/usr/bin/env python3
"""Verify the generated Hamiltonian powers (H^2, H^3) against the base Hamiltonian.

The check is independent of how the power files were produced: it builds dense
matrices directly from the numbered-Pauli files (via Kronecker products of the
single-qubit Pauli matrices) and confirms two things for every bond length:

  1. Matrix identity:   H @ H == H_square,   H @ H @ H == H_triple.
  2. Eigenvalue identity (the property the connected-moments method relies on):
     spectrum(H_square) == spectrum(H) ** 2,
     spectrum(H_triple) == spectrum(H) ** 3.

Run directly (``python test_power_hamiltonians.py``) or via pytest.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
PAULI_DIR = REPO_ROOT / "Pauli_Ham"

MOLECULE = "HF"
POWER_LABELS = {2: "square", 3: "triple"}

# I, X, Y, Z indexed by 0, 1, 2, 3.
_SINGLE_QUBIT = [
    np.array([[1, 0], [0, 1]], dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
]

ATOL = 1e-7


def read_pauli_file(path: Path) -> tuple[list[tuple[float, tuple[int, ...]]], int]:
    rows: list[tuple[float, tuple[int, ...]]] = []
    num_qubits: int | None = None
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            coeff = float(tokens[0])
            paulis = tuple(int(token) for token in tokens[1:])
            num_qubits = len(paulis) if num_qubits is None else num_qubits
            rows.append((coeff, paulis))
    assert num_qubits is not None, f"{path}: no terms found"
    return rows, num_qubits


def pauli_term_matrix(term: tuple[int, ...]) -> np.ndarray:
    matrix = np.array([[1.0 + 0j]])
    for pauli in term:
        matrix = np.kron(matrix, _SINGLE_QUBIT[pauli])
    return matrix


def build_dense(path: Path) -> np.ndarray:
    rows, num_qubits = read_pauli_file(path)
    dimension = 2 ** num_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    for coeff, term in rows:
        matrix += coeff * pauli_term_matrix(term)
    return matrix


def base_bond_tokens(molecule: str) -> list[str]:
    pattern = re.compile(rf"^{re.escape(molecule)}_bond_(?P<token>[^_]+)\.txt$")
    tokens: list[str] = []
    for path in sorted(PAULI_DIR.glob(f"{molecule}_bond_*.txt")):
        match = pattern.match(path.name)
        if match:
            tokens.append(match.group("token"))
    return tokens


def verify_bond(molecule: str, token: str) -> None:
    base_matrix = build_dense(PAULI_DIR / f"{molecule}_bond_{token}.txt")
    base_spectrum = np.sort(np.linalg.eigvalsh(base_matrix))

    for power in sorted(POWER_LABELS):
        expected_matrix = np.linalg.matrix_power(base_matrix, power)
        label = POWER_LABELS[power]
        power_path = PAULI_DIR / f"{molecule}_{label}_bond_{token}.txt"
        assert power_path.exists(), (
            f"Missing {power_path.name}; run generate_power_hamiltonians.py first."
        )
        power_matrix = build_dense(power_path)

        max_matrix_error = float(np.max(np.abs(power_matrix - expected_matrix)))
        assert max_matrix_error < ATOL, (
            f"{molecule} bond {token} H^{power}: matrix mismatch (max abs error {max_matrix_error:.2e})."
        )

        power_spectrum = np.sort(np.linalg.eigvalsh(power_matrix))
        expected_spectrum = np.sort(base_spectrum ** power)
        max_eig_error = float(np.max(np.abs(power_spectrum - expected_spectrum)))
        assert max_eig_error < ATOL, (
            f"{molecule} bond {token} H^{power}: eigenvalue mismatch (max abs error {max_eig_error:.2e})."
        )
        print(
            f"  H^{power} ({label}): matrix_err={max_matrix_error:.2e}, "
            f"eig_err={max_eig_error:.2e}  OK"
        )


def test_power_hamiltonians() -> None:
    tokens = base_bond_tokens(MOLECULE)
    assert tokens, f"No base Hamiltonians found for {MOLECULE} in {PAULI_DIR}."
    for token in tokens:
        print(f"{MOLECULE} bond {token}:")
        verify_bond(MOLECULE, token)


if __name__ == "__main__":
    test_power_hamiltonians()
    print("\nAll power-Hamiltonian checks passed.")
