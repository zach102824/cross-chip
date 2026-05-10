"""Exact diagonalization of the LiH Pauli Hamiltonian (six qubits) on disk.

Loads the same `Pauli_Ham/LiH_bond_*_pauli_convention.txt` as the FIG.13 pipeline,
builds the dense 64×64 complex Hermitian matrix, and reports the ground-state
(minimum) eigenvalue in Hartree.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cirq
import numpy as np


def _find_workspace_with_pauli_ham(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "Pauli_Ham").is_dir():
            return p
    return start


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ground-state energy of LiH Pauli Hamiltonian (exact diagonalization)."
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=None,
        help="Directory that contains Pauli_Ham/ (default: search upward from CWD).",
    )
    parser.add_argument("--bond", type=float, default=2.2, help="Internuclear distance (Angstrom).")
    args = parser.parse_args()

    case_dir = Path(__file__).resolve().parent
    if str(case_dir) not in sys.path:
        sys.path.insert(0, str(case_dir))

    from main_cursor_lib_test_LiH import load_hamiltonian_paths, load_observable_h

    workspace = (args.workspace or _find_workspace_with_pauli_ham(Path.cwd())).resolve()
    h_atom = 2
    bond = args.bond
    hamiltonian_basename = "LiH"

    qubits = cirq.LineQubit.range(6)
    observable_h = load_observable_h(
        workspace,
        qubits,
        h_atom,
        bond,
        hamiltonian_basename=hamiltonian_basename,
    )
    _, pkl_path, text_path = load_hamiltonian_paths(
        workspace, h_atom, bond, hamiltonian_basename=hamiltonian_basename
    )

    hmat = observable_h.matrix(qubits=qubits)
    hmat = 0.5 * (hmat + np.conjugate(hmat.T))
    if not np.allclose(hmat, np.conjugate(hmat.T), atol=1e-10):
        raise ValueError("Hamiltonian matrix is not numerically Hermitian.")

    evals = np.linalg.eigvalsh(hmat)
    gs_energy = float(evals[0])
    max_energy = float(evals[-1])
    print(f"Workspace: {workspace}")
    print(f"Hamiltonian file: {text_path.name} (exists={text_path.is_file()}) | pkl: {pkl_path.name}")
    print(f"Hilbert space dimension: {hmat.shape[0]}")
    print(f"Ground-state energy (min eigenvalue): {gs_energy:.12f} Ha")
    print(f"Maximum eigenvalue:                    {max_energy:.12f} Ha")
    print(f"Spectral width (max - min):            {max_energy - gs_energy:.12f} Ha")
    print(f"First 5 eigenvalues (lowest): {np.array2string(evals[:5], precision=10, floatmode='fixed')}")


if __name__ == "__main__":
    main()
