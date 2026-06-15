from math import comb
from pathlib import Path
import multiprocessing
import os
import pickle

import numpy as np
import openfermion as of
import scipy.sparse.linalg
from pyscf import ao2mo, fci, gto, lo, scf
from pyscf.fci import spin_op
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper


def localize_mos(mol, mo_coeff, method="er"):
    """
    Localize the MOs in `mo_coeff` using the given method.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object.
    mo_coeff : ndarray (nAO, nMO)
        Canonical MO coefficients from SCF.
    method : str
        One of 'boys', 'pm' (Pipek-Mezey), or 'er' (Edmiston-Ruedenberg).
    """
    m = method.lower()
    if m == "boys":
        localizer = lo.Boys(mol, mo_coeff)
    elif m in ("pm", "pipek-mezey"):
        localizer = lo.PM(mol, mo_coeff)
    elif m in ("er", "edmiston-ruedenberg"):
        localizer = lo.ER(mol, mo_coeff)
    else:
        raise ValueError(f"Unknown localization method: {method!r}")
    return localizer.kernel()


def qiskit_to_openfermion(qiskit_op):
    """Transforms a Qiskit SparsePauliOp into an OpenFermion QubitOperator."""
    of_op = of.QubitOperator()
    for pauli_str, coef in qiskit_op.to_list():
        term_str = []
        for qubit_idx, pauli_char in enumerate(reversed(pauli_str)):
            if pauli_char != "I":
                term_str.append(f"{pauli_char}{qubit_idx}")
        of_term_str = " ".join(term_str)
        of_op += of.QubitOperator(of_term_str, complex(coef))
    return of_op


def openfermion_to_custom_format(of_op, num_qubits):
    """
    Converts an OpenFermion QubitOperator to:
    [coefficient, pauli_q0, pauli_q1, ..., pauli_qn] with I=0, X=1, Y=2, Z=3.
    """
    pauli_map = {"X": 1, "Y": 2, "Z": 3}
    custom_hamiltonian = []
    sorted_terms = sorted(of_op.terms.items(), key=lambda item: item[0])

    for term, coeff in sorted_terms:
        pauli_string = [0] * num_qubits
        for qubit_idx, pauli_char in term:
            pauli_string[qubit_idx] = pauli_map[pauli_char]
        term_list = [coeff.real] + pauli_string
        custom_hamiltonian.append(term_list)
    return custom_hamiltonian


def main():
    num_workers = max(1, multiprocessing.cpu_count())
    print(num_workers)

    # ==== Parameters you can adjust ====
    n_h = 4
    bond_length = 2
    r = bond_length
    # ====================================

    atom_lines = []
    for i in range(n_h):
        x = i * bond_length
        atom_lines.append(f"H 0.000000 0.000000  {x:.6f}")

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
    print(f"RHF <S^2> = {ss:.6f},  multiplicity = {mult}")

    n_ao = mol.nao_nr()
    n_orb = mf.mo_coeff.shape[1]
    dim = comb(n_orb, mol.nelec[0]) * comb(n_orb, mol.nelec[1])
    print(f"Number of AOs = {n_ao},  Number of MOs = {n_orb},  CI dim = {dim}")

    method = "er"
    loc_coeff = localize_mos(mol, mf.mo_coeff, method=method)

    dip = mol.intor_symmetric("int1e_r", comp=3)
    r2 = mol.intor_symmetric("int1e_r2")
    nmo = loc_coeff.shape[1]
    spreads = np.zeros(nmo)

    for i in range(nmo):
        c = loc_coeff[:, i]
        rx, ry, rz = (c @ dip[k] @ c for k in range(3))
        r2_mean = c @ r2 @ c
        spreads[i] = r2_mean - (rx * rx + ry * ry + rz * rz)

    print("Orbital spreads (<r^2> - <r>^2):")
    for i, s in enumerate(spreads, start=1):
        print(f"  Orbital {i:2d}: {s:.6f}")

    h1_loc = loc_coeff.T @ mf.get_hcore() @ loc_coeff
    eri_loc = ao2mo.kernel(mol, loc_coeff)
    h2_loc = ao2mo.restore(8, eri_loc, nmo)

    # if using normal HF integral
    h1_loc = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_canonical = ao2mo.kernel(mol, mf.mo_coeff)
    h2_loc = ao2mo.restore(8, eri_canonical, nmo)
    print("Canonical HF integrals generated and stored as 'h1_loc' and 'h2_loc'.")

    fci_solver = fci.FCI(mol, loc_coeff)
    fci_solver.nroots = 10
    E_loc, civec_loc = fci_solver.kernel(h1_loc, h2_loc, nmo, mol.nelec)
    print("\n=== FCI Results on Localized MOs ===")
    for i, Ei in enumerate(E_loc, start=1):
        ss_val, _ = spin_op.spin_square(civec_loc[i - 1], nmo, mol.nelec)
        print(f"State {i:2d}: Energy = {Ei:.12f}  <S^2> = {ss_val:.6f}")

    print("\n=== Building Qiskit Hamiltonian from Localized Integrals ===")
    h2_loc_4d = ao2mo.restore(1, h2_loc, nmo)
    electronic_energy = ElectronicEnergy.from_raw_integrals(h1_loc, h2_loc_4d)
    electronic_energy.nuclear_repulsion_energy = mol.energy_nuc()
    hamiltonian_fermionic = electronic_energy.second_q_op()
    mapper = JordanWignerMapper()
    H_qubit_qiskit = mapper.map(hamiltonian_fermionic)
    print(f"Qiskit Pauli Hamiltonian generated with {len(H_qubit_qiskit)} terms.")
    print("\n=== Double Checking Eigenvalues ===")

    mapper = JordanWignerMapper()
    print("=== Comparing Hamiltonian Term Counts ===\n")
    h1_localized = loc_coeff.T @ mf.get_hcore() @ loc_coeff
    eri_loc_raw = ao2mo.kernel(mol, loc_coeff)
    h2_localized = ao2mo.restore(1, eri_loc_raw, nmo)
    ee_loc = ElectronicEnergy.from_raw_integrals(h1_localized, h2_localized)
    ee_loc.nuclear_repulsion_energy = mol.energy_nuc()
    H_qubit_loc = mapper.map(ee_loc.second_q_op()).chop(1e-6)
    print(f"Number of Pauli terms (Localized Integrals, chopped): {len(H_qubit_loc)}")

    h1_can = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri_can_raw = ao2mo.kernel(mol, mf.mo_coeff)
    h2_can = ao2mo.restore(8, eri_can_raw, nmo)
    threshold = 1e-8
    h1_can[np.abs(h1_can) < threshold] = 0.0
    h2_can[np.abs(h2_can) < threshold] = 0.0
    h2_can_4d = ao2mo.restore(1, h2_can, nmo)
    ee_can = ElectronicEnergy.from_raw_integrals(h1_can, h2_can_4d)
    ee_can.nuclear_repulsion_energy = mol.energy_nuc()
    H_qubit_can = mapper.map(ee_can.second_q_op())
    print(f"Number of Pauli terms (Canonical Integrals with noise truncated): {len(H_qubit_can)}")

    threshold = 1e-6
    H_loc_chopped = H_qubit_loc.chop(threshold)
    H_can_chopped = H_qubit_can.chop(threshold)
    l1_norm_loc = np.sum(np.abs(H_loc_chopped.coeffs))
    l1_norm_can = np.sum(np.abs(H_can_chopped.coeffs))
    print(f"=== After applying {threshold} threshold ===")
    print(f"Localized Hamiltonian: {len(H_loc_chopped)} terms, L1 Norm = {l1_norm_loc:.6f}")
    print(f"Canonical Hamiltonian: {len(H_can_chopped)} terms, L1 Norm = {l1_norm_can:.6f}")

    H_qubit_of = qiskit_to_openfermion(H_qubit_qiskit)
    print("Successfully transformed to OpenFermion format.")
    print(f"Number of terms: {len(H_qubit_of.terms)}")
    print("\n=== Verifying Eigenvalues of the Transformed Object ===")
    H_sparse_of = of.linalg.get_sparse_operator(H_qubit_of).tocsc()
    num_states = min(5, H_sparse_of.shape[0] - 1)
    eigenvalues, _ = scipy.sparse.linalg.eigsh(H_sparse_of, k=num_states, which="SA")
    for i, energy in enumerate(eigenvalues):
        print(f"  State {i + 1}: {energy:.12f}")
    print(eigenvalues[0] + mol.energy_nuc())

    output_dir = Path(__file__).resolve().parent / "Pauli_Ham"
    os.makedirs(output_dir, exist_ok=True)

    print("\n=== Saving the OF Object ===")
    file_path = output_dir / f"H4_bond_{bond_length}.pkl"
    with open(file_path, "wb") as file:
        pickle.dump(H_qubit_of, file)
    print(f"Hamiltonian safely saved to {file_path}!")

    print("\n=== Saving the Qiskit Object ===")
    file_path_qiskit = output_dir / f"H4_bond_{bond_length}_qiskit.pkl"
    with open(file_path_qiskit, "wb") as file:
        pickle.dump(H_qubit_qiskit, file)
    print(f"Qiskit Hamiltonian safely saved to {file_path_qiskit}!")

    num_qubits = H_qubit_qiskit.num_qubits
    custom_hamiltonian = openfermion_to_custom_format(H_qubit_of, num_qubits)
    print(f"Converted to custom format. Number of terms: {len(custom_hamiltonian)}")
    print("First 5 terms:")
    for term in custom_hamiltonian[:5]:
        print(term)

    print("\n=== Saving the Numbered Convention Hamiltonian to TXT ===")
    file_path_numbered_txt = output_dir / f"H4_bond_{bond_length}_number_convention.txt"
    with open(file_path_numbered_txt, "w") as file:
        for term in custom_hamiltonian:
            line = " ".join(map(str, term))
            file.write(line + "\n")
    print(f"Numbered convention Hamiltonian safely saved to {file_path_numbered_txt}!")

    print("\n=== Saving the IXYZ Convention Hamiltonian to TXT ===")
    file_path_ixyz_txt = output_dir / f"H4_bond_{bond_length}_pauli_convention.txt"
    num_to_char = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    with open(file_path_ixyz_txt, "w") as file:
        for term in custom_hamiltonian:
            coeff = term[0]
            pauli_str = "".join([num_to_char[p] for p in term[1:]])
            file.write(f"{pauli_str}\n")
            file.write(f"{complex(coeff)}\n")
    print(f"IXYZ convention Hamiltonian safely saved to {file_path_ixyz_txt}!")

    print("\n=== Applying 2-Qubit Reduction ===")
    mapper_tapered = ParityMapper(num_particles=(1, 1))
    H_qubit_tapered = mapper_tapered.map(hamiltonian_fermionic)
    print(f"Original JW Hamiltonian size: {H_qubit_qiskit.num_qubits} qubits")
    print(f"Tapered Parity Hamiltonian size: {H_qubit_tapered.num_qubits} qubits")
    print(f"Number of Pauli terms: {len(H_qubit_tapered)}")
    print("\n=== Double Checking Tapered Eigenvalues ===")
    H_tapered_sparse = H_qubit_tapered.to_matrix(sparse=True).tocsc()
    num_states = min(1, H_tapered_sparse.shape[0] - 1)
    eigenvalues_tapered, _ = scipy.sparse.linalg.eigsh(H_tapered_sparse, k=num_states, which="SA")
    for i, energy in enumerate(eigenvalues_tapered):
        print(f"  State {i + 1}: {energy + electronic_energy.nuclear_repulsion_energy:.12f}")


if __name__ == "__main__":
    main()
