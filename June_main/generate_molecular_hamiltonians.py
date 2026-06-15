#!/usr/bin/env python3
"""Generate active-space molecular Hamiltonians in the repo Pauli format.

The molecule presets mirror the active-space choices used in ``UCCSD_Mole``.
By default this script generates the HF molecule over the bond grid used there
and saves numbered Pauli Hamiltonians under ``Pauli_Ham``:

    coefficient I/X/Y/Z-as-number-for-qubit-0 ...

with ``I=0, X=1, Y=2, Z=3``.  Qubits are exported in spin-block order:

    [alpha spatial orbitals..., beta spatial orbitals...]
"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from openfermion import QubitOperator
from openfermion.chem import MolecularData
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from scipy.sparse.linalg import eigsh

REPO_ROOT = Path(__file__).resolve().parents[1]


GeometryFactory = Callable[[float], list[tuple[str, tuple[float, float, float]]]]


@dataclass(frozen=True)
class MoleculePreset:
    name: str
    active_space: tuple[int, int]
    geometry_factory: GeometryFactory
    default_bonds: tuple[float, ...]
    basis: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1


def _diatomic(symbol_a: str, symbol_b: str) -> GeometryFactory:
    def geometry(bond: float) -> list[tuple[str, tuple[float, float, float]]]:
        return [
            (symbol_a, (0.0, 0.0, -bond / 2.0)),
            (symbol_b, (0.0, 0.0, bond / 2.0)),
        ]

    return geometry


def _grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return tuple(values)


MOLECULE_PRESETS: dict[str, MoleculePreset] = {
    "HF": MoleculePreset("HF", (6, 4), _diatomic("H", "F"), _grid(1.0, 2.2, 0.2)),
    "F2": MoleculePreset("F2", (10, 6), _diatomic("F", "F"), _grid(1.0, 3.0, 0.2)),
    "Cl2": MoleculePreset("Cl2", (14, 8), _diatomic("Cl", "Cl"), _grid(1.0, 3.0, 0.2)),
    "Br2": MoleculePreset("Br2", (14, 8), _diatomic("Br", "Br"), _grid(1.0, 3.0, 0.2)),
}


def bond_token(bond: float) -> str:
    return f"{bond:.10g}".rstrip("0").rstrip(".")


def spin_block_permutation(num_spatial_orbitals: int) -> list[int]:
    """Map OpenFermion interleaved spin orbitals to spin-block wire labels."""

    index_map = [0] * (2 * num_spatial_orbitals)
    for spatial in range(num_spatial_orbitals):
        index_map[2 * spatial] = spatial
        index_map[2 * spatial + 1] = num_spatial_orbitals + spatial
    return index_map


def relabel_qubit_operator(operator: QubitOperator, index_map: list[int]) -> QubitOperator:
    """Apply a qubit-index permutation while preserving each Pauli term."""

    relabelled = QubitOperator()
    for term, coeff in operator.terms.items():
        new_term = tuple(sorted((index_map[index], pauli) for index, pauli in term))
        relabelled += QubitOperator(new_term, coeff)
    return relabelled


def openfermion_to_numbered_paulis(operator: QubitOperator, num_qubits: int) -> list[list[float]]:
    """Convert a QubitOperator to [coefficient, pauli_0, ..., pauli_n]."""

    pauli_map = {"X": 1, "Y": 2, "Z": 3}
    rows: list[list[float]] = []
    for term, coeff in sorted(operator.terms.items(), key=lambda item: item[0]):
        coeff = complex(coeff)
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Cannot export complex coefficient with nonzero imaginary part: {coeff}")
        paulis = [0] * num_qubits
        for qubit_index, pauli_char in term:
            paulis[qubit_index] = pauli_map[pauli_char]
        rows.append([float(coeff.real), *paulis])
    return rows


def active_indices_from_tencirchem_choice(
    total_electrons: int,
    total_spatial_orbitals: int,
    active_space: tuple[int, int],
) -> tuple[list[int], list[int]]:
    """Derive frozen occupied and active orbital indices from TenCirChem's CAS tuple."""

    active_electrons, active_spatial_orbitals = active_space
    inactive_occ = total_electrons // 2 - active_electrons // 2
    if inactive_occ < 0:
        raise ValueError(f"Active electron count {active_electrons} exceeds total electrons {total_electrons}.")
    active_start = inactive_occ
    active_stop = active_start + active_spatial_orbitals
    if active_stop > total_spatial_orbitals:
        raise ValueError(
            "Active space does not fit available orbitals: "
            f"active_stop={active_stop}, total_spatial_orbitals={total_spatial_orbitals}."
        )
    return list(range(inactive_occ)), list(range(active_start, active_stop))


def build_molecular_hamiltonian(
    molecule_name: str,
    bond: float,
    basis: str | None = None,
) -> tuple[QubitOperator, dict[str, object]]:
    preset = MOLECULE_PRESETS[molecule_name]
    basis = basis or preset.basis
    molecule = MolecularData(
        geometry=preset.geometry_factory(bond),
        basis=basis,
        multiplicity=preset.multiplicity,
        charge=preset.charge,
        description=f"{preset.name}_{bond_token(bond)}A_active_space",
    )
    molecule = run_pyscf(
        molecule,
        run_scf=True,
        run_mp2=False,
        run_cisd=False,
        run_ccsd=False,
        run_fci=False,
        verbose=False,
    )

    occupied_indices, active_indices = active_indices_from_tencirchem_choice(
        total_electrons=int(molecule.n_electrons),
        total_spatial_orbitals=int(molecule.n_orbitals),
        active_space=preset.active_space,
    )
    active_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    interleaved_qubit_hamiltonian = jordan_wigner(get_fermion_operator(active_hamiltonian))
    index_map = spin_block_permutation(preset.active_space[1])
    qubit_hamiltonian = relabel_qubit_operator(interleaved_qubit_hamiltonian, index_map)

    metadata: dict[str, object] = {
        "molecule": preset.name,
        "bond_angstrom": float(bond),
        "basis": basis,
        "charge": preset.charge,
        "multiplicity": preset.multiplicity,
        "geometry": preset.geometry_factory(bond),
        "active_space": list(preset.active_space),
        "occupied_indices": occupied_indices,
        "active_indices": active_indices,
        "n_qubits": 2 * preset.active_space[1],
        "n_terms": len(qubit_hamiltonian.terms),
        "rhf_energy": float(molecule.hf_energy),
        "export_qubit_layout": "[alpha spatial orbitals..., beta spatial orbitals...]",
        "openfermion_spin_orb_to_export_qubit": index_map,
    }
    return qubit_hamiltonian, metadata


def ground_state_energy_and_vector(operator: QubitOperator, num_qubits: int) -> tuple[float, np.ndarray]:
    sparse = get_sparse_operator(operator, n_qubits=num_qubits).tocsc()
    if sparse.shape[0] <= 4096:
        eigenvalues, eigenvectors = np.linalg.eigh(sparse.toarray())
        return float(np.real_if_close(eigenvalues[0])), eigenvectors[:, 0]
    eigenvalues, eigenvectors = eigsh(sparse, k=1, which="SA")
    return float(np.real_if_close(eigenvalues[0])), eigenvectors[:, 0]


def save_hamiltonian(
    qubit_hamiltonian: QubitOperator,
    output_dir: Path,
    stem: str,
    metadata: dict[str, object],
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_qubits = int(metadata["n_qubits"])

    pkl_path = output_dir / f"{stem}_of.pkl"
    number_path = output_dir / f"{stem}.txt"
    pauli_path = output_dir / f"{stem}_pauli_convention.txt"
    metadata_path = output_dir / f"{stem}_metadata.json"

    with pkl_path.open("wb") as file:
        pickle.dump(qubit_hamiltonian, file)

    numbered_terms = openfermion_to_numbered_paulis(qubit_hamiltonian, num_qubits)
    with number_path.open("w", encoding="utf-8") as file:
        for term in numbered_terms:
            file.write(" ".join(map(str, term)) + "\n")

    num_to_char = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    with pauli_path.open("w", encoding="utf-8") as file:
        for term in numbered_terms:
            coeff = term[0]
            pauli = "".join(num_to_char[int(p)] for p in term[1:])
            file.write(f"{pauli}\n")
            file.write(f"{complex(coeff)}\n")

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
        file.write("\n")

    return pkl_path, number_path, pauli_path, metadata_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule",
        choices=sorted(MOLECULE_PRESETS),
        default="HF",
        help="Molecule preset to generate. Defaults to HF.",
    )
    parser.add_argument("--basis", default=None, help="Override preset basis. Defaults to STO-3G.")
    parser.add_argument(
        "--bonds",
        type=float,
        nargs="*",
        default=None,
        help="Explicit bond lengths in Angstrom. Defaults to the preset scan grid.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "Pauli_Ham",
        help="Directory for Hamiltonian files. Defaults to repo Pauli_Ham.",
    )
    parser.add_argument(
        "--skip-ground-state",
        action="store_true",
        help="Skip exact ground-state calculation in the printed summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = MOLECULE_PRESETS[args.molecule]
    bonds = tuple(args.bonds) if args.bonds else preset.default_bonds
    output_dir = args.output_dir.expanduser().resolve()

    print(f"Generating {preset.name} Hamiltonians")
    print(f"active_space={preset.active_space}, basis={args.basis or preset.basis}")
    print(f"bonds={list(bonds)}")
    print(f"output_dir={output_dir}")
    print()

    for bond in bonds:
        qubit_hamiltonian, metadata = build_molecular_hamiltonian(args.molecule, bond, args.basis)
        if not args.skip_ground_state:
            gs_energy, _ = ground_state_energy_and_vector(qubit_hamiltonian, int(metadata["n_qubits"]))
            metadata["exact_ground_state_energy"] = gs_energy

        stem = f"{preset.name}_bond_{bond_token(bond)}"
        paths = save_hamiltonian(qubit_hamiltonian, output_dir, stem, metadata)
        print(
            f"{stem}: n_qubits={metadata['n_qubits']}, n_terms={metadata['n_terms']}, "
            f"rhf={metadata['rhf_energy']:.12f}"
            + (
                f", gs={metadata['exact_ground_state_energy']:.12f}"
                if "exact_ground_state_energy" in metadata
                else ""
            )
        )
        print(f"  saved: {paths[1]}")


if __name__ == "__main__":
    main()
