#!/usr/bin/env python3
"""Validate generated Hamiltonians against TenCirChem active-space FCI.

Default check:
    HF at 1.0 Angstrom, active_space=(6, 4)

The generated Hamiltonian is exported in spin-block Pauli-label order:

    [alpha spatial orbitals..., beta spatial orbitals...]

OpenFermion's sparse matrix convention is little-endian with respect to those
labels. TenCirChem state vectors use the opposite bit display order, so the FCI
vector is bit-reversed before comparing overlaps.
"""

from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from openfermion.linalg import get_sparse_operator
from scipy.sparse.linalg import eigsh
from tencirchem import M, UCCSD
from tencirchem.static.ci_utils import get_ci_strings

REPO_ROOT = Path(__file__).resolve().parents[2]


GeometryFactory = Callable[[float], list[tuple[str, tuple[float, float, float]]]]


@dataclass(frozen=True)
class MoleculePreset:
    name: str
    active_space: tuple[int, int]
    geometry_factory: GeometryFactory
    basis: str = "sto-3g"
    charge: int = 0


def _diatomic(symbol_a: str, symbol_b: str) -> GeometryFactory:
    def geometry(bond: float) -> list[tuple[str, tuple[float, float, float]]]:
        return [
            (symbol_a, (0.0, 0.0, -bond / 2.0)),
            (symbol_b, (0.0, 0.0, bond / 2.0)),
        ]

    return geometry


MOLECULE_PRESETS: dict[str, MoleculePreset] = {
    "HF": MoleculePreset("HF", (6, 4), _diatomic("H", "F")),
    "F2": MoleculePreset("F2", (10, 6), _diatomic("F", "F")),
    "Cl2": MoleculePreset("Cl2", (14, 8), _diatomic("Cl", "Cl")),
    "Br2": MoleculePreset("Br2", (14, 8), _diatomic("Br", "Br")),
}


def bond_token(bond: float) -> str:
    return f"{bond:.10g}".rstrip("0").rstrip(".")


def ground_state_energy_and_vector(operator, num_qubits: int) -> tuple[float, np.ndarray]:
    sparse = get_sparse_operator(operator, n_qubits=num_qubits).tocsc()
    if sparse.shape[0] <= 4096:
        eigenvalues, eigenvectors = np.linalg.eigh(sparse.toarray())
        return float(np.real_if_close(eigenvalues[0])), eigenvectors[:, 0]
    eigenvalues, eigenvectors = eigsh(sparse, k=1, which="SA")
    return float(np.real_if_close(eigenvalues[0])), eigenvectors[:, 0]


def make_tencirchem_molecule(molecule_name: str, bond: float, basis: str | None = None):
    preset = MOLECULE_PRESETS[molecule_name]
    atom = [[symbol, *coords] for symbol, coords in preset.geometry_factory(bond)]
    return M(atom=atom, basis=basis or preset.basis, unit="Angstrom", charge=preset.charge, spin=0)


def bit_reverse_index(index: int, n_qubits: int) -> int:
    reversed_index = 0
    for qubit in range(n_qubits):
        if index & (1 << qubit):
            reversed_index |= 1 << (n_qubits - 1 - qubit)
    return reversed_index


def tencirchem_ci_to_openfermion_state(civector: np.ndarray, n_qubits: int, n_elec: int) -> np.ndarray:
    """Embed a TenCirChem CI vector into OpenFermion's sparse-operator basis."""

    ci_strings = np.asarray(get_ci_strings(n_qubits, n_elec, hcb=False), dtype=np.uint64)
    tc_state = np.zeros(2**n_qubits, dtype=np.complex128)
    tc_state[ci_strings] = np.asarray(civector, dtype=np.complex128).reshape(-1)

    openfermion_state = np.zeros_like(tc_state)
    for tc_index, amplitude in enumerate(tc_state):
        if amplitude == 0:
            continue
        openfermion_state[bit_reverse_index(tc_index, n_qubits)] = amplitude
    return openfermion_state


def compare_ground_state(
    molecule_name: str = "HF",
    bond: float = 1.0,
    basis: str | None = None,
    hamiltonian_pkl: Path | None = None,
    energy_atol: float = 1e-7,
    min_vector_overlap: float = 0.999,
) -> dict[str, float]:
    preset = MOLECULE_PRESETS[molecule_name]
    n_qubits = 2 * preset.active_space[1]
    if hamiltonian_pkl is None:
        hamiltonian_pkl = REPO_ROOT / "Pauli_Ham" / f"{molecule_name}_bond_{bond_token(bond)}_of.pkl"
    hamiltonian_pkl = hamiltonian_pkl.expanduser().resolve()
    if not hamiltonian_pkl.exists():
        raise FileNotFoundError(
            f"Generated Hamiltonian pickle not found: {hamiltonian_pkl}. "
            "Run June_main/generate_molecular_hamiltonians.py first."
        )
    with hamiltonian_pkl.open("rb") as file:
        qubit_hamiltonian = pickle.load(file)
    of_energy, of_vector = ground_state_energy_and_vector(qubit_hamiltonian, n_qubits)

    mol = make_tencirchem_molecule(molecule_name, bond, basis)
    ucc = UCCSD(
        mol,
        active_space=preset.active_space,
        run_fci=True,
        run_mp2=True,
        run_ccsd=False,
        init_method="mp2",
        engine="civector",
    )
    tc_energy = float(ucc.e_fci)
    tc_vector = tencirchem_ci_to_openfermion_state(ucc.civector_fci, ucc.n_qubits, ucc.n_elec)

    energy_abs_diff = abs(of_energy - tc_energy)
    vector_overlap = abs(np.vdot(of_vector, tc_vector))
    phase = np.vdot(of_vector, tc_vector)
    if abs(phase) > 0:
        phase_aligned_diff = np.linalg.norm(of_vector - tc_vector * phase / abs(phase))
    else:
        phase_aligned_diff = np.inf

    result = {
        "hamiltonian_pkl": str(hamiltonian_pkl),
        "openfermion_ground_energy": of_energy,
        "tencirchem_fci_energy": tc_energy,
        "energy_abs_diff": energy_abs_diff,
        "vector_overlap_abs": float(vector_overlap),
        "phase_aligned_vector_l2_diff": float(phase_aligned_diff),
    }

    if energy_abs_diff > energy_atol:
        raise AssertionError(f"Energy mismatch: {result}")
    if vector_overlap < min_vector_overlap:
        raise AssertionError(f"Ground-state vector mismatch: {result}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--molecule", choices=sorted(MOLECULE_PRESETS), default="HF")
    parser.add_argument("--bond", type=float, default=1.0)
    parser.add_argument("--basis", default=None)
    parser.add_argument(
        "--hamiltonian-pkl",
        type=Path,
        default=None,
        help="Generated *_of.pkl file. Defaults to Pauli_Ham/<molecule>_bond_<bond>_of.pkl.",
    )
    parser.add_argument("--energy-atol", type=float, default=1e-7)
    parser.add_argument("--min-vector-overlap", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_ground_state(
        molecule_name=args.molecule,
        bond=args.bond,
        basis=args.basis,
        hamiltonian_pkl=args.hamiltonian_pkl,
        energy_atol=args.energy_atol,
        min_vector_overlap=args.min_vector_overlap,
    )
    print("TenCirChem validation passed")
    for key, value in result.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.12g}")


if __name__ == "__main__":
    main()
