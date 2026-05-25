"""Build Pauli Hamiltonians from canonical HF orbitals and save under Pauli_Ham with _HF suffix."""

from __future__ import annotations

from math import comb
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import openfermion as of
import scipy.sparse.linalg
from pyscf import ao2mo, fci, gto, scf
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.mappers import JordanWignerMapper


def qiskit_to_openfermion(qiskit_op):
    """Transform a Qiskit SparsePauliOp into an OpenFermion QubitOperator."""
    of_op = of.QubitOperator()
    terms = qiskit_op.to_list() if hasattr(qiskit_op, "to_list") else qiskit_op.primitive.to_list()
    for pauli_str, coef in terms:
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
        H_qiskit = H_qiskit.reduce(atol=chop_threshold)
    return H_qiskit, qiskit_to_openfermion(H_qiskit)


def lowest_eigenvalues(of_op: of.QubitOperator, k: int = 5) -> np.ndarray:
    H_sparse = of.linalg.get_sparse_operator(of_op).tocsc()
    k_eff = min(k, H_sparse.shape[0] - 1)
    eigvals, _ = scipy.sparse.linalg.eigsh(H_sparse, k=k_eff, which="SA")
    return np.sort(eigvals.real)


def hf_overlap_and_ground_energy(mol, mf) -> tuple[float, float]:
    """Return HF-determinant overlap and exact electronic ground energy from FCI."""
    exact_ground_energy, ci_vec = fci_ground_energy_and_state(mol, mf)
    hf_overlap = float(abs(ci_vec[0, 0]))
    return hf_overlap, exact_ground_energy


def fci_ground_energy_and_state(mol, mf) -> tuple[float, np.ndarray]:
    """Return exact electronic ground energy and FCI vector in canonical HF MOs."""
    nmo = mf.mo_coeff.shape[1]
    h1 = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
    eri = ao2mo.kernel(mol, mf.mo_coeff)
    h2 = ao2mo.restore(1, eri, nmo)

    cisolver = fci.direct_spin1.FCI()
    exact_ground_energy, ci_vec = cisolver.kernel(h1, h2, nmo, mol.nelec)
    return float(exact_ground_energy), ci_vec


def make_h2_molecule(bond_length: float, basis: str, *, verbose: int = 0):
    """Create an H2 molecule aligned on z with the requested bond length."""
    mol = gto.Mole()
    mol.atom = f"H 0.000000 0.000000 0.000000\nH 0.000000 0.000000 {bond_length:.6f}"
    mol.basis = basis
    mol.unit = "Angstrom"
    mol.verbose = verbose
    mol.build()
    return mol


def plot_hf_overlap_and_energy_diff(
    output_path: Path,
    *,
    basis: str,
    equilibrium_bond: float = 0.7414,
    bond_min: float = 0.3,
    bond_max: float = 2.5,
    num_points: int = 30,
) -> None:
    """Plot HF overlap, adjacent-state overlap, and (Exact-HF electronic) vs bond length."""
    bond_lengths = np.linspace(bond_min, bond_max, num_points)
    hf_overlaps: list[float] = []
    state_overlaps: list[float] = []
    energy_diffs: list[float] = []
    prev_mol = None
    prev_mf = None
    prev_ci_vec = None

    print("\n=== Scanning bond lengths for HF overlap/energy-difference plot ===")
    for bond in bond_lengths:
        mol_scan = make_h2_molecule(float(bond), basis, verbose=0)
        mf_scan = scf.RHF(mol_scan)
        mf_scan.kernel()

        exact_elec, ci_vec = fci_ground_energy_and_state(mol_scan, mf_scan)
        hf_overlap = float(abs(ci_vec[0, 0]))
        hf_electronic = float(mf_scan.e_tot - mol_scan.energy_nuc())
        energy_diff = exact_elec - hf_electronic

        if prev_ci_vec is None:
            # No previous geometry for the first point.
            state_overlap = 1.0
        else:
            # Compare CI states in a consistent cross-geometry orbital metric.
            s_ao_cross = gto.intor_cross("int1e_ovlp", prev_mol, mol_scan)
            s_mo_cross = prev_mf.mo_coeff.T @ s_ao_cross @ mf_scan.mo_coeff
            state_overlap = float(
                abs(
                    fci.addons.overlap(
                        prev_ci_vec,
                        ci_vec,
                        mf_scan.mo_coeff.shape[1],
                        mol_scan.nelec,
                        s=s_mo_cross,
                    )
                )
            )

        hf_overlaps.append(hf_overlap)
        state_overlaps.append(state_overlap)
        energy_diffs.append(energy_diff)
        prev_mol = mol_scan
        prev_mf = mf_scan
        prev_ci_vec = ci_vec

        print(
            f"  R={bond:.4f} A | HF-overlap={hf_overlap:.8f} | "
            f"state-overlap={state_overlap:.8f} | DeltaE(exact-HF)={energy_diff:.8f} Ha"
        )

    eq_mol = make_h2_molecule(float(equilibrium_bond), basis, verbose=0)
    eq_mf = scf.RHF(eq_mol)
    eq_mf.kernel()
    eq_exact_elec, eq_ci_vec = fci_ground_energy_and_state(eq_mol, eq_mf)
    eq_hf_overlap = float(abs(eq_ci_vec[0, 0]))
    eq_energy_diff = eq_exact_elec - float(eq_mf.e_tot - eq_mol.energy_nuc())

    fig, ax1 = plt.subplots(figsize=(8, 5))
    l1 = ax1.plot(
        bond_lengths,
        hf_overlaps,
        color="tab:blue",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="HF overlap |<HF|GS>|",
    )
    l2 = ax1.plot(
        bond_lengths,
        state_overlaps,
        color="tab:cyan",
        marker="^",
        markersize=3,
        linewidth=1.3,
        label="State overlap |<GS(R-dR)|GS(R)>|",
    )
    ax1.plot(
        [equilibrium_bond],
        [eq_hf_overlap],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=14,
        linestyle="None",
        zorder=5,
    )
    ax1.set_xlabel("Bond length (Angstrom)")
    ax1.set_ylabel("Overlap", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_ylim(0.8, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    l3 = ax2.plot(
        bond_lengths,
        energy_diffs,
        color="tab:red",
        marker="s",
        markersize=3,
        linewidth=1.5,
        label="Energy difference (Exact - HF electronic)",
    )
    ax2.plot(
        [equilibrium_bond],
        [eq_energy_diff],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=14,
        linestyle="None",
        zorder=5,
    )
    ax2.set_ylabel("Energy difference [Exact - HF electronic] [Hartree]", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    ax1.set_title(f"H2 ({basis}) overlaps and correlation energy vs bond length")
    lines = l1 + l2 + l3
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def plot_exact_vs_hf_total_energy(
    output_path: Path,
    *,
    basis: str,
    equilibrium_bond: float = 0.7414,
    bond_min: float = 0.3,
    bond_max: float = 2.5,
    num_points: int = 40,
) -> None:
    """Plot exact total and HF total energies versus H2 bond length."""
    bond_lengths = np.linspace(bond_min, bond_max, num_points)
    hf_total_energies: list[float] = []
    exact_total_energies: list[float] = []

    for bond in bond_lengths:
        mol_scan = make_h2_molecule(float(bond), basis, verbose=0)
        mf_scan = scf.RHF(mol_scan)
        mf_scan.kernel()
        _, exact_elec = hf_overlap_and_ground_energy(mol_scan, mf_scan)
        e_nuc = float(mol_scan.energy_nuc())
        hf_total_energies.append(float(mf_scan.e_tot))
        exact_total_energies.append(exact_elec + e_nuc)

    eq_mol = make_h2_molecule(float(equilibrium_bond), basis, verbose=0)
    eq_mf = scf.RHF(eq_mol)
    eq_mf.kernel()
    _, eq_exact_elec = hf_overlap_and_ground_energy(eq_mol, eq_mf)
    eq_exact_total = eq_exact_elec + float(eq_mol.energy_nuc())
    eq_hf_total = float(eq_mf.e_tot)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        bond_lengths,
        exact_total_energies,
        color="tab:green",
        marker="o",
        markersize=3,
        linewidth=1.5,
        label="Exact total energy",
    )
    ax.plot(
        bond_lengths,
        hf_total_energies,
        color="tab:purple",
        marker="s",
        markersize=3,
        linewidth=1.5,
        label="HF total energy",
    )

    # Mark equilibrium energies on both curves.
    ax.plot(
        [equilibrium_bond],
        [eq_exact_total],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=14,
        linestyle="None",
        zorder=5,
    )
    ax.plot(
        [equilibrium_bond],
        [eq_hf_total],
        marker="*",
        color="gold",
        markeredgecolor="black",
        markersize=14,
        linestyle="None",
        zorder=5,
    )

    ax.set_xlabel("Bond length (Angstrom)")
    ax.set_ylabel("Total energy (Hartree)")
    ax.set_title(f"H2 ({basis}) exact and HF total energies vs bond length")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Figure saved to {output_path}")


def save_hamiltonians(
    output_dir: Path,
    *,
    h_atom: int,
    bond_length: float | int,
    basis: str,
    H_qiskit,
    H_of: of.QubitOperator,
    hf_total_energy: float,
    hf_electronic_energy: float,
    hf_overlap: float,
    exact_ground_energy: float,
) -> None:
    basis_tag = basis.lower().replace(" ", "").replace("*", "s")
    stem = f"H{h_atom}_bond_{bond_length}_basis_{basis_tag}_HF"
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

    hf_summary_path = output_dir / f"{stem}_hf_summary.txt"
    with hf_summary_path.open("w", encoding="utf-8") as f:
        f.write(f"HF total energy (Hartree): {hf_total_energy:.12f}\n")
        f.write(f"HF electronic energy (Hartree): {hf_electronic_energy:.12f}\n")
        f.write(f"HF overlap |<HF|GS>|: {hf_overlap:.12f}\n")
        f.write(f"Exact ground energy (Hartree): {exact_ground_energy:.12f}\n")
    print(f"HF summary saved to {hf_summary_path}")


def main() -> None:
    # ==== Parameters you can adjust ====
    n_h = 2
    bond_length = 0.7414  # H2 equilibrium bond length in Angstrom
    basis = "6-31g"
    chop_threshold = 1e-6
    # ====================================

    mol = make_h2_molecule(float(bond_length), basis, verbose=1)

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

    hf_total_energy = float(mf.e_tot)
    hf_electronic_energy = hf_total_energy - float(mol.energy_nuc())
    hf_overlap, exact_ground_energy = hf_overlap_and_ground_energy(mol, mf)
    print(f"HF total energy (Hartree): {hf_total_energy:.12f}")
    print(f"HF electronic energy (Hartree): {hf_electronic_energy:.12f}")
    print(f"HF overlap |<HF|GS>|: {hf_overlap:.12f}")
    print(f"Exact ground energy (Hartree): {exact_ground_energy:.12f}")

    print("\n=== Lowest eigenvalues (electronic, no extra E_nuc add) ===")
    for i, energy in enumerate(lowest_eigenvalues(H_of, k=5), start=1):
        print(f"  State {i}: {energy:.12f}")

    output_dir = Path(__file__).resolve().parent / "Pauli_Ham"
    print(f"\n=== Saving to {output_dir} (_HF suffix) ===")
    save_hamiltonians(
        output_dir,
        h_atom=n_h,
        bond_length=bond_length,
        basis=basis,
        H_qiskit=H_qiskit,
        H_of=H_of,
        hf_total_energy=hf_total_energy,
        hf_electronic_energy=hf_electronic_energy,
        hf_overlap=hf_overlap,
        exact_ground_energy=exact_ground_energy,
    )

    fig_energy_path = Path(__file__).resolve().parent / "h2_631g_exact_vs_hf_energy.png"
    plot_exact_vs_hf_total_energy(fig_energy_path, basis=basis, equilibrium_bond=float(bond_length))
    fig_overlap_path = Path(__file__).resolve().parent / "h2_631g_hf_overlap_energydiff_dual_axis.png"
    plot_hf_overlap_and_energy_diff(fig_overlap_path, basis=basis, equilibrium_bond=float(bond_length))


if __name__ == "__main__":
    main()
