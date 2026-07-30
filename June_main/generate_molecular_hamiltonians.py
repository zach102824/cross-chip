#!/usr/bin/env python3
"""Generate active-space molecular Hamiltonians in the repo Pauli format.

The molecule presets mirror the active-space choices used in ``UCCSD_Mole``.
By default this script generates the HF molecule over the bond grid used there
and saves one numbered Pauli Hamiltonian per bond under ``Pauli_Ham``:

    coefficient I/X/Y/Z-as-number-for-qubit-0 ...

with ``I=0, X=1, Y=2, Z=3``.  Qubits are exported in spin-block order:

    [alpha spatial orbitals..., beta spatial orbitals...]

For Cl2 and Br2 the active orbitals are symmetry-pinned (same method as
``UCCSD_Mole/Cl2.ipynb`` / ``Br2.ipynb``): the occupied MO columns are reordered
so each active-space slot holds the same physical orbital (fixed irrep, fixed
slot order) at every bond length, instead of whatever the canonical energy
ordering happens to put there.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from openfermion import QubitOperator
from openfermion.chem import MolecularData
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from pyscf import ao2mo, gto, scf, symm
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
    # Irrep labels (fixed slot order) pinning the ACTIVE OCCUPIED orbitals.
    # None -> plain canonical (energy-ordered) orbitals.  See symmetry_pinned_mo_coeff.
    pinned_occ_irreps: tuple[str, ...] | None = None
    # Optional explicit CAS orbital lists (OpenFermion / PySCF spatial indices).
    # When set, these override ``active_indices_from_tencirchem_choice``.
    occupied_indices: tuple[int, ...] | None = None
    active_indices: tuple[int, ...] | None = None


def _diatomic(symbol_a: str, symbol_b: str) -> GeometryFactory:
    def geometry(bond: float) -> list[tuple[str, tuple[float, float, float]]]:
        return [
            (symbol_a, (0.0, 0.0, -bond / 2.0)),
            (symbol_b, (0.0, 0.0, bond / 2.0)),
        ]

    return geometry


def _h_chain(n_h: int) -> GeometryFactory:
    """Linear H_n chain with equal H–H spacing ``bond`` (atoms at 0, d, …, (n-1)d)."""

    def geometry(bond: float) -> list[tuple[str, tuple[float, float, float]]]:
        return [("H", (0.0, 0.0, i * bond)) for i in range(n_h)]

    return geometry


def _grid(start: float, stop: float, step: float) -> tuple[float, ...]:
    values: list[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return tuple(values)


MOLECULE_PRESETS: dict[str, MoleculePreset] = {
    "HF": MoleculePreset("HF", (6, 4), _diatomic("H", "F"), _grid(0.4, 2.2, 0.2)),
    # Cl2/Br2 use symmetry-pinned active orbitals (matching UCCSD_Mole/Cl2.ipynb
    # and Br2.ipynb): the canonical MO energy ordering reshuffles with bond
    # length, so an energy-windowed active space silently swaps orbitals along
    # the scan.  Pinning by irrep keeps every active-space slot on the same
    # physical orbital at every bond length.  This also fixes the sigma/pi
    # near-degeneracy problem that previously forced skipping Cl2 at 2.0 A.
    "Cl2": MoleculePreset(
        "Cl2",
        (8, 5),
        _diatomic("Cl", "Cl"),
        _grid(1.0, 4.0, 0.2),
        pinned_occ_irreps=("A1u", "A1g", "E1gx", "E1gy"),
    ),
    "Br2": MoleculePreset(
        "Br2",
        (10, 6),
        _diatomic("Br", "Br"),
        _grid(1.9, 4.1, 0.2),
        pinned_occ_irreps=("A1g", "E1uy", "E1ux", "E1gx", "E1gy"),
    ),
    # Guo et al. Nat. Commun. / arXiv:2212.08006 (paper in papers/):
    # LiH STO-3G, freeze Li 1s, active MOs {1,2,5} -> 2e/3orb -> 6 qubits.
    "LiH": MoleculePreset(
        "LiH",
        (2, 3),
        _diatomic("Li", "H"),
        _grid(1.0, 3.0, 0.2),
        occupied_indices=(0,),
        active_indices=(1, 2, 5),
    ),
    # Same paper / UCCSD_Mole/F2.ipynb: freeze 1a1-4a1, CAS(10,6) -> 12 qubits.
    "F2": MoleculePreset(
        "F2",
        (10, 6),
        _diatomic("F", "F"),
        _grid(1.0, 3.0, 0.2),
    ),
    # Linear H4 STO-3G (UCCSD_Mole/H4.ipynb): full valence CAS(4,4) -> 8 qubits;
    # H–H spacing scan matches d_grid = np.arange(0.6, 2.0, 0.2).
    "H4": MoleculePreset("H4", (4, 4), _h_chain(4), _grid(0.6, 1.8, 0.2)),
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


def symmetry_pinned_mo_coeff(preset: MoleculePreset, bond: float, basis: str | None = None) -> np.ndarray:
    """RHF MO coefficients with the active occupied orbitals pinned by symmetry.

    Same method as ``symmetry_pinned_mo`` in UCCSD_Mole/Cl2.ipynb and Br2.ipynb:
    run RHF with ``symmetry=True``, label MOs by irrep, then reorder the occupied
    columns so the last ``len(preset.pinned_occ_irreps)`` occupied slots always
    hold the highest-energy occupied orbital of each pinned irrep, in the fixed
    order given by ``preset.pinned_occ_irreps``.  The returned coefficients are
    used on a symmetry-disabled molecule (identical AO basis).
    """
    if preset.pinned_occ_irreps is None:
        raise ValueError(f"Preset {preset.name} does not define pinned_occ_irreps.")
    if len(preset.pinned_occ_irreps) != preset.active_space[0] // 2:
        raise ValueError(
            f"Preset {preset.name}: {len(preset.pinned_occ_irreps)} pinned irreps "
            f"but active space has {preset.active_space[0] // 2} occupied orbitals."
        )

    atom = [[s, *coords] for s, coords in preset.geometry_factory(bond)]
    mol_s = gto.M(
        atom=atom,
        basis=basis or preset.basis,
        unit="Angstrom",
        charge=preset.charge,
        spin=preset.multiplicity - 1,
        symmetry=True,
    )
    mf = scf.RHF(mol_s)
    mf.verbose = 0
    mf.kernel()
    if not mf.converged:
        raise RuntimeError(f"RHF (symmetry=True) did not converge for {preset.name} at {bond} A.")
    labels = symm.label_orb_symm(mol_s, mol_s.irrep_name, mol_s.symm_orb, mf.mo_coeff)
    nocc = mol_s.nelectron // 2
    occ = list(range(nocc))

    def highest(lbl: str) -> int:
        candidates = [i for i in occ if labels[i] == lbl]
        if not candidates:
            raise ValueError(f"No occupied orbital with irrep {lbl} for {preset.name} at {bond} A.")
        return max(candidates, key=lambda i: mf.mo_energy[i])

    sel = [highest(lbl) for lbl in preset.pinned_occ_irreps]
    rest = [i for i in occ if i not in sel]
    order = rest + sel + list(range(nocc, mf.mo_coeff.shape[1]))
    return mf.mo_coeff[:, order]


def _integrals_from_mo_coeff(pyscf_mol, pyscf_scf, mo_coeff: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One- and two-body MO integrals in the OpenFermion convention for given orbitals."""

    n_orbitals = mo_coeff.shape[1]
    one_body = (mo_coeff.T @ pyscf_scf.get_hcore() @ mo_coeff).astype(float)
    eri = ao2mo.restore(1, ao2mo.kernel(pyscf_mol, mo_coeff), n_orbitals)
    # OpenFermion PQRS convention: h[p,q,r,s] = (ps|qr)
    two_body = np.asarray(eri.transpose(0, 2, 3, 1), order="C")
    return one_body, two_body


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

    if preset.pinned_occ_irreps is not None:
        # Replace the canonical-orbital integrals with symmetry-pinned ones so the
        # exported Hamiltonian matches the UCCSD_Mole notebooks at every bond length.
        pinned_mo = symmetry_pinned_mo_coeff(preset, bond, basis)
        one_body, two_body = _integrals_from_mo_coeff(
            molecule._pyscf_data["mol"], molecule._pyscf_data["scf"], pinned_mo
        )
        # PyscfMolecularData exposes these as read-only properties backed by
        # private caches; overwrite the caches directly.
        molecule._canonical_orbitals = pinned_mo
        molecule._one_body_integrals = one_body
        molecule._two_body_integrals = two_body

    if preset.occupied_indices is not None and preset.active_indices is not None:
        occupied_indices = list(preset.occupied_indices)
        active_indices = list(preset.active_indices)
        if len(active_indices) != preset.active_space[1]:
            raise ValueError(
                f"Preset {preset.name}: len(active_indices)={len(active_indices)} "
                f"!= active_space n_spatial={preset.active_space[1]}"
            )
    else:
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
        "orbital_selection": (
            "canonical"
            if preset.pinned_occ_irreps is None
            else f"symmetry_pinned:{','.join(preset.pinned_occ_irreps)}"
        ),
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
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    num_qubits = int(metadata["n_qubits"])

    number_path = output_dir / f"{stem}.txt"

    numbered_terms = openfermion_to_numbered_paulis(qubit_hamiltonian, num_qubits)
    with number_path.open("w", encoding="utf-8") as file:
        for term in numbered_terms:
            file.write(" ".join(map(str, term)) + "\n")

    return number_path


def write_bond_energy_summary(
    output_dir: Path,
    molecule_name: str,
    active_space: tuple[int, int],
    basis: str,
    rows: list[dict[str, float | None]],
) -> Path:
    """Write HF and ground-state energies for each bond length."""

    summary_path = output_dir / f"{molecule_name}_bond_scan_summary.txt"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write(f"# {molecule_name} active-space bond scan\n")
        file.write(f"# active_space: {active_space}\n")
        file.write(f"# basis: {basis}\n")
        file.write("# energies in Hartree; HF_minus_GS in milli-Hartree\n")
        file.write("bond_angstrom\tE_HF_Ha\tE_GS_Ha\tHF_minus_GS_mHa\n")
        for row in rows:
            bond = float(row["bond_angstrom"])
            hf_energy = float(row["rhf_energy"])
            gs_energy = row.get("exact_ground_state_energy")
            if gs_energy is None:
                file.write(f"{bond_token(bond)}\t{hf_energy:.12f}\t\t\n")
            else:
                gs_energy = float(gs_energy)
                hf_minus_gs_mha = (hf_energy - gs_energy) * 1000.0
                file.write(
                    f"{bond_token(bond)}\t{hf_energy:.12f}\t{gs_energy:.12f}\t{hf_minus_gs_mha:.6f}\n"
                )
    return summary_path


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

    summary_rows: list[dict[str, float | None]] = []
    basis = args.basis or preset.basis

    for bond in bonds:
        qubit_hamiltonian, metadata = build_molecular_hamiltonian(args.molecule, bond, basis)
        if not args.skip_ground_state:
            gs_energy, _ = ground_state_energy_and_vector(qubit_hamiltonian, int(metadata["n_qubits"]))
            metadata["exact_ground_state_energy"] = gs_energy

        stem = f"{preset.name}_bond_{bond_token(bond)}"
        saved_path = save_hamiltonian(qubit_hamiltonian, output_dir, stem, metadata)
        summary_rows.append(
            {
                "bond_angstrom": float(bond),
                "rhf_energy": float(metadata["rhf_energy"]),
                "exact_ground_state_energy": metadata.get("exact_ground_state_energy"),
            }
        )
        print(
            f"{stem}: n_qubits={metadata['n_qubits']}, n_terms={metadata['n_terms']}, "
            f"rhf={metadata['rhf_energy']:.12f}"
            + (
                f", gs={metadata['exact_ground_state_energy']:.12f}"
                if "exact_ground_state_energy" in metadata
                else ""
            )
        )
        print(f"  saved: {saved_path}")

    summary_path = write_bond_energy_summary(
        output_dir,
        preset.name,
        preset.active_space,
        basis,
        summary_rows,
    )
    print()
    print(f"Energy summary saved: {summary_path}")


if __name__ == "__main__":
    main()
