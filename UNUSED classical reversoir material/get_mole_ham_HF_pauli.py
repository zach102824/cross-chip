"""Build Pauli Hamiltonians from canonical HF orbitals and save under Pauli_Ham with _HF suffix."""

from __future__ import annotations

from math import comb
from pathlib import Path
import os
import pickle

import numpy as np
import openfermion as of
import scipy.sparse.linalg
from pyscf import ao2mo, gto, scf
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper


def qiskit_to_openfermion(qiskit_op):
    """Transform a Qiskit SparsePauliOp into an OpenFermion QubitOperator."""
    of_op = of.QubitOperator()
    for pauli_str, coef in qiskit_op.to_list():
        term_str = []
        for qubit_idx, pauli_char in enumerate(reversed(pauli_str)):
            if pauli_char != "I":
                term_str.append(f"{pauli_char}{qubit_idx}")
        of_term_str = " ".join(term_str)
        of_op += of.QubitOperator(of_term_str, complex(coef))
    return of_op


def openfermion_to_custom_format(of_op, num_qubits: int) -> list[list[float]]:
    """Convert QubitOperator to [coeff, pauli_q0, ...] with I=0, X=1, Y=2, Z=3."""
    pauli_map = {"X": 1, "Y": 2, "Z": 3}
    custom_hamiltonian: list[list[float]] = []
    for term, coeff in sorted(of_op.terms.items(), key=lambda item: item[0]):
        pauli_string = [0] * num_qubits
        for qubit_idx, pauli_char in term:
            pauli_string[qubit_idx] = pauli_map[pauli_char]
        custom_hamiltonian.append([coeff.real] + pauli_string)
    return custom_hamiltonian


def build_hf_pauli_hamiltonian(mol, mf, *, chop_threshold: float = 1e-6):
    """Jordan–Wigner Pauli Hamiltonian from canonical HF MO integrals."""
    nmo = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    h2 = ao2mo.restore(1, eri, nmo)

    electronic_energy = ElectronicEnergy.from_raw_integrals(h1, h2)
    electronic_energy.nuclear_repulsion_energy = mol.energy_nuc()
    mapper = JordanWignerMapper()
    H_qiskit = mapper.map(electronic_energy.second_q_op())
    if chop_threshold > 0:
        H_qiskit = H_qiskit.chop(chop_threshold)
    return H_qiskit, qiskit_to_openfermion(H_qiskit)


def lowest_eigenvalues(of_op: of.QubitOperator, k: int = 5) -> np.ndarray:
    H_sparse = of.linalg.get_sparse_operator(of_op).tocsc()
    k_eff = min(k, H_sparse.shape[0] - 1)
    eigvals, _ = scipy.sparse.linalg.eigsh(H_sparse, k=k_eff, which="SA")
    return np.sort(eigvals.real)


def save_hamiltonians(
    output_dir: Path,
    *,
    h_atom: int,
    bond_length: float | int,
    H_qiskit,
    H_of: of.QubitOperator,
) -> None:
    stem = f"H{h_atom}_bond_{bond_length}_HF"
    output_dir.mkdir(parents=True, exist_ok=True)

    pkl_path = output_dir / f"{stem}.pkl"
    with pkl_path.open("wb") as f:
        pickle.dump(H_of, f)
    print(f"OpenFermion Hamiltonian saved to {pkl_path}")

    qiskit_path = output_dir / f"{stem}_qiskit.pkl"
    with qiskit_path.open("wb") as f:
        pickle.dump(H_qiskit, f)
    print(f"Qiskit Hamiltonian saved to {qiskit_path}")

    num_qubits = H_qiskit.num_qubits
    custom = openfermion_to_custom_format(H_of, num_qubits)

    numbered_path = output_dir / f"{stem}_number_convention.txt"
    with numbered_path.open("w", encoding="utf-8") as f:
        for term in custom:
            f.write(" ".join(map(str, term)) + "\n")
    print(f"Numbered convention saved to {numbered_path}")

    pauli_path = output_dir / f"{stem}_pauli_convention.txt"
    num_to_char = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    with pauli_path.open("w", encoding="utf-8") as f:
        for term in custom:
            coeff = term[0]
            pauli_str = "".join(num_to_char[p] for p in term[1:])
            f.write(f"{pauli_str}\n")
            f.write(f"{complex(coeff)}\n")
    print(f"IXYZ convention saved to {pauli_path}")


def main() -> None:
    # ==== Parameters you can adjust ====
    n_h = 4
    bond_length = 2
    chop_threshold = 1e-6
    # ====================================

    atom_lines = [
        f"H 0.000000 0.000000 {i * bond_length:.6f}" for i in range(n_h)
    ]

    mol = gto.Mole()
    mol.atom = "\n".join(atom_lines)
    mol.basis = "sto-6g"
    mol.unit = "Angstrom"
    mol.verbose = 1
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()
    mf.analyze()
    ss, mult = mf.spin_square()
    print(f"RHF <S^2> = {ss:.6f}, multiplicity = {mult}")

    n_ao = mol.nao_nr()
    n_orb = mf.mo_coeff.shape[1]
    ci_dim = comb(n_orb, mol.nelec[0]) * comb(n_orb, mol.nelec[1])
    print(f"AOs = {n_ao}, MOs = {n_orb}, CI dim = {ci_dim}")

    print("\n=== Building Pauli Hamiltonian (canonical HF orbitals) ===")
    H_qiskit, H_of = build_hf_pauli_hamiltonian(mol, mf, chop_threshold=chop_threshold)
    print(f"Pauli terms (chop={chop_threshold}): {len(H_qiskit)}")
    print(f"Qubits: {H_qiskit.num_qubits}")

    print("\n=== Lowest eigenvalues (electronic, no extra E_nuc add) ===")
    for i, energy in enumerate(lowest_eigenvalues(H_of, k=5), start=1):
        print(f"  State {i}: {energy:.12f}")

    output_dir = Path(__file__).resolve().parent / "Pauli_Ham"
    print(f"\n=== Saving to {output_dir} (_HF suffix) ===")
    save_hamiltonians(
        output_dir,
        h_atom=n_h,
        bond_length=bond_length,
        H_qiskit=H_qiskit,
        H_of=H_of,
    )


if __name__ == "__main__":
    main()
