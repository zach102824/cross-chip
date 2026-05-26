#!/usr/bin/env python3
"""Scan LiH active-space choices for an 8-qubit Hamiltonian.

This script keeps one doubly occupied core orbital frozen by default and scans
all valid active-space selections with 4 spatial orbitals (8 qubits) to find
the lowest exact ground-state energy.

No Hamiltonian files are written; results are printed for inspection.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import pickle
import sys

import numpy as np
from openfermion import QubitOperator
from openfermion.chem import MolecularData
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import get_fermion_operator, jordan_wigner
from openfermionpyscf import run_pyscf
from scipy.sparse.linalg import eigsh

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from get_mole_ham_local_MO_pauli import openfermion_to_custom_format

TARGET_ACTIVE_INDICES = (1, 2, 4, 5)


def spin_block_permutation(num_spatial_orbitals: int) -> list[int]:
    """Map OF interleaved spin-orbital ordering to spin-block ordering."""
    perm: list[int] = [0] * (2 * num_spatial_orbitals)
    for s in range(num_spatial_orbitals):
        perm[2 * s] = s
        perm[2 * s + 1] = num_spatial_orbitals + s
    return perm


def relabel_qubit_operator(of_operator: QubitOperator, index_map: list[int]) -> QubitOperator:
    """Apply a permutation of Pauli-wire indices."""
    out = QubitOperator()
    for term, coeff in of_operator.terms.items():
        new_term = tuple(sorted((index_map[q], p) for q, p in term))
        out += QubitOperator(new_term, coeff)
    return out


def exact_ground_energy(qubit_hamiltonian: QubitOperator) -> float:
    sparse = get_sparse_operator(qubit_hamiltonian).tocsc()
    dim = sparse.shape[0]
    if dim <= 256:
        evals = np.linalg.eigvalsh(sparse.toarray())
        return float(np.real_if_close(evals[0]))
    eigvals, _ = eigsh(sparse, k=1, which="SA")
    return float(np.real_if_close(eigvals[0]))


@dataclass
class ScanResult:
    active_indices: tuple[int, ...]
    n_terms: int
    gs_energy: float


def run_scan(
    bond: float,
    basis: str,
    occupied_indices: list[int],
    active_spatial_orbitals: int,
) -> tuple[float, list[ScanResult], dict[tuple[int, ...], QubitOperator]]:
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond))]
    molecule = MolecularData(
        geometry=geometry,
        basis=basis,
        multiplicity=1,
        charge=0,
        description=f"LiH_{bond}A_scan_{active_spatial_orbitals}orb",
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

    all_spatial = list(range(int(molecule.n_orbitals)))
    candidate_pool = [i for i in all_spatial if i not in occupied_indices]
    if len(candidate_pool) < active_spatial_orbitals:
        raise ValueError(
            f"Not enough orbitals for active space: pool={candidate_pool}, "
            f"needed={active_spatial_orbitals}"
        )
    if len(TARGET_ACTIVE_INDICES) != active_spatial_orbitals:
        raise ValueError(
            "TARGET_ACTIVE_INDICES length must match --active-spatial. "
            f"Got {len(TARGET_ACTIVE_INDICES)} vs {active_spatial_orbitals}."
        )
    if any(i in occupied_indices for i in TARGET_ACTIVE_INDICES):
        raise ValueError(
            f"TARGET_ACTIVE_INDICES={TARGET_ACTIVE_INDICES} overlaps frozen occupied "
            f"orbitals={occupied_indices}."
        )
    if any(i not in candidate_pool for i in TARGET_ACTIVE_INDICES):
        raise ValueError(
            f"TARGET_ACTIVE_INDICES={TARGET_ACTIVE_INDICES} must come from "
            f"candidate pool={candidate_pool}."
        )

    index_map = spin_block_permutation(active_spatial_orbitals)
    results: list[ScanResult] = []
    hamiltonians: dict[tuple[int, ...], QubitOperator] = {}
    active_indices = TARGET_ACTIVE_INDICES
    active_h = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=list(active_indices),
    )
    of_qubit = jordan_wigner(get_fermion_operator(active_h))
    qubit_h = relabel_qubit_operator(of_qubit, index_map)
    hamiltonians[active_indices] = qubit_h
    gs = exact_ground_energy(qubit_h)
    results.append(
        ScanResult(
            active_indices=active_indices,
            n_terms=len(qubit_h.terms),
            gs_energy=gs,
        )
    )

    results.sort(key=lambda r: r.gs_energy)
    return float(molecule.hf_energy), results, hamiltonians


def bond_token(bond: float) -> str:
    return f"{bond}".rstrip("0").rstrip(".")


def save_hamiltonian(qubit_hamiltonian: QubitOperator, out_dir: Path, stem: str, num_qubits: int) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = out_dir / f"{stem}_of.pkl"
    number_path = out_dir / f"{stem}.txt"
    pauli_path = out_dir / f"{stem}_pauli_convention.txt"

    with pkl_path.open("wb") as f:
        pickle.dump(qubit_hamiltonian, f)

    terms = openfermion_to_custom_format(qubit_hamiltonian, num_qubits)
    num_to_char = {0: "I", 1: "X", 2: "Y", 3: "Z"}

    with number_path.open("w", encoding="utf-8") as f:
        for term in terms:
            f.write(" ".join(map(str, term)) + "\n")

    with pauli_path.open("w", encoding="utf-8") as f:
        for term in terms:
            coeff = term[0]
            pauli = "".join(num_to_char[int(p)] for p in term[1:])
            f.write(f"{pauli}\n")
            f.write(f"{complex(coeff)}\n")

    return pkl_path, number_path, pauli_path


def save_hf_summary(out_dir: Path, stem: str, hf_total_energy: float, exact_ground_energy: float) -> Path:
    hf_summary_path = out_dir / f"{stem}_hf_summary.txt"
    with hf_summary_path.open("w", encoding="utf-8") as f:
        f.write(f"HF total energy (Hartree): {hf_total_energy:.12f}\n")
        f.write(f"Exact ground energy (Hartree): {exact_ground_energy:.12f}\n")
        f.write("\n")
    return hf_summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=float, default=1.5, help="Li-H bond length in Angstrom.")
    parser.add_argument("--basis", type=str, default="sto-3g", help="Basis set for PySCF.")
    parser.add_argument(
        "--occupied",
        type=int,
        nargs="*",
        default=[0],
        help="Frozen occupied spatial orbitals (default: 0).",
    )
    parser.add_argument(
        "--active-spatial",
        type=int,
        default=4,
        help="Number of active spatial orbitals (4 => 8 qubits).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Show this many best active-space selections.",
    )
    parser.add_argument(
        "--save-best",
        action="store_true",
        help="Save Hamiltonian files for all tied-best active-space selections.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(REPO_ROOT / "Pauli_Ham"),
        help="Directory for saved Hamiltonian files (default: repo Pauli_Ham/).",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="LiH_active8_scan",
        help="Filename stem prefix for saved Hamiltonians.",
    )
    args = parser.parse_args()

    if args.active_spatial != 4:
        raise ValueError("This script is intended for 8-qubit runs, so --active-spatial must be 4.")

    hf_energy, results, hamiltonians = run_scan(
        bond=args.bond,
        basis=args.basis,
        occupied_indices=sorted(args.occupied),
        active_spatial_orbitals=args.active_spatial,
    )

    print("=== LiH active-space scan (8 qubits) ===")
    print(f"bond: {args.bond} Angstrom")
    print(f"basis: {args.basis}")
    print(f"frozen occupied orbitals: {sorted(args.occupied)}")
    print(f"active spatial orbitals: {args.active_spatial} (=> {2 * args.active_spatial} qubits)")
    print(f"fixed active_indices: {list(TARGET_ACTIVE_INDICES)}")
    print(f"PySCF RHF total energy: {hf_energy:.10f} Eh")
    print("combinations scanned: 1 (fixed active space)")
    print()

    best = results[0]
    print("--- Best active-space choice ---")
    print(f"active_indices: {list(best.active_indices)}")
    print(f"exact ground-state energy: {best.gs_energy:.10f} Eh")
    print(f"Pauli term count: {best.n_terms}")
    print()
    print("--- HF-style summary ---")
    print(f"HF total energy (Hartree): {hf_energy:.12f}")
    print(f"Exact ground energy (Hartree): {best.gs_energy:.12f}")
    print()

    print(f"--- Top {min(args.topk, len(results))} by lowest exact energy ---")
    for i, r in enumerate(results[: args.topk], start=1):
        print(
            f"{i:>2}. active_indices={list(r.active_indices)} | "
            f"E0={r.gs_energy:.10f} Eh | terms={r.n_terms}"
        )

    if args.save_best:
        tol = 1e-10
        best_energy = results[0].gs_energy
        tied_best = [r for r in results if abs(r.gs_energy - best_energy) <= tol]
        out_dir = args.output_dir.expanduser().resolve()
        print()
        print(f"--- Saving {len(tied_best)} tied-best Hamiltonian(s) to {out_dir} ---")
        for r in tied_best:
            idx_tag = "-".join(str(i) for i in r.active_indices)
            stem = (
                f"{args.basename}_bond_{bond_token(args.bond)}"
                f"_active_{idx_tag}_q{2 * args.active_spatial}"
            )
            pkl_path, number_path, pauli_path = save_hamiltonian(
                hamiltonians[r.active_indices],
                out_dir,
                stem,
                2 * args.active_spatial,
            )
            print(f"saved for active_indices={list(r.active_indices)}")
            print(f"  {pkl_path}")
            print(f"  {number_path}")
            print(f"  {pauli_path}")
            hf_summary_path = save_hf_summary(out_dir, stem, hf_energy, r.gs_energy)
            print(f"  {hf_summary_path}")


if __name__ == "__main__":
    main()
