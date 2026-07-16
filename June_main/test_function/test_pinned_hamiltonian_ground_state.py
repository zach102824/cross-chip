#!/usr/bin/env python3
"""Validate symmetry-pinned Cl2/Br2 Hamiltonians: ground-state eigenvector check.

For each case the qubit Hamiltonian is built with
``June_main/generate_molecular_hamiltonians.build_molecular_hamiltonian`` (which
for Cl2/Br2 uses symmetry-pinned active orbitals, matching
``UCCSD_Mole/Cl2.ipynb`` / ``Br2.ipynb``).  The exact ground state of that
Hamiltonian is compared against a PySCF CASCI reference computed with the SAME
orbitals (``symmetry_pinned_mo_coeff`` for Cl2/Br2, canonical RHF for HF):

  * ground-state energies must agree to ``ENERGY_ATOL``
  * ground-state eigenvectors must agree:
    ``|<gs_qubit_hamiltonian | gs_casci>| > MIN_VECTOR_OVERLAP``

The CASCI CI vector is mapped determinant-by-determinant into the exported
qubit basis (spin-block label order ``[alpha spatial..., beta spatial...]``,
OpenFermion big-endian sparse convention), including the fermionic reordering
sign between PySCF's (alpha block)(beta block) operator order and the
Jordan-Wigner label order.  With the correct mapping the overlap is 1 to
machine precision, so the threshold is tight.

HF (canonical orbitals, unchanged code path) is included as a regression case.

Run as a script for a summary table (also validates the exported ``Pauli_Ham``
txt files when they exist), or via pytest for the in-memory checks.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from pyscf import gto, mcscf, scf
from pyscf.fci import cistring
from scipy.sparse.linalg import eigsh

JUNE_MAIN = Path(__file__).resolve().parents[1]
REPO_ROOT = JUNE_MAIN.parent
if str(JUNE_MAIN) not in sys.path:
    sys.path.insert(0, str(JUNE_MAIN))

from generate_molecular_hamiltonians import (  # noqa: E402
    MOLECULE_PRESETS,
    bond_token,
    build_molecular_hamiltonian,
    symmetry_pinned_mo_coeff,
)

ENERGY_ATOL = 1e-8
MIN_VECTOR_OVERLAP = 1.0 - 1e-8

# (molecule, bond): compressed, equilibrium and stretched geometries.  The
# stretched Cl2/Br2 points are exactly where the old energy-ordered active
# space silently swapped orbitals.
CASES: list[tuple[str, float]] = [
    ("HF", 1.0),
    ("Cl2", 1.0),
    ("Cl2", 2.0),
    ("Cl2", 3.0),
    ("Br2", 2.2),
    ("Br2", 3.0),
    ("Br2", 3.8),
]


def ground_state_energy_and_vector(operator: QubitOperator, num_qubits: int) -> tuple[float, np.ndarray]:
    sparse = get_sparse_operator(operator, n_qubits=num_qubits).tocsc()
    if sparse.shape[0] <= 4096:
        eigenvalues, eigenvectors = np.linalg.eigh(sparse.toarray())
        return float(np.real_if_close(eigenvalues[0])), eigenvectors[:, 0]
    eigenvalues, eigenvectors = eigsh(sparse, k=1, which="SA")
    return float(np.real_if_close(eigenvalues[0])), eigenvectors[:, 0]


def casci_reference(molecule_name: str, bond: float) -> tuple[float, np.ndarray, int, int]:
    """CASCI energy and CI matrix using the exact same orbitals as the generator."""

    preset = MOLECULE_PRESETS[molecule_name]
    ncas = preset.active_space[1]
    nelecas = preset.active_space[0]

    atom = [[symbol, *coords] for symbol, coords in preset.geometry_factory(bond)]
    mol = gto.M(atom=atom, basis=preset.basis, unit="Angstrom", charge=preset.charge, spin=0)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    if preset.pinned_occ_irreps is not None:
        mo_coeff = symmetry_pinned_mo_coeff(preset, bond)
    else:
        mo_coeff = mf.mo_coeff

    mc = mcscf.CASCI(mf, ncas, nelecas)
    # Keep the supplied orbitals as-is; canonicalization would rotate the active
    # space and break the correspondence with the exported Hamiltonian.
    mc.canonicalization = False
    mc.verbose = 0
    mc.kernel(mo_coeff)
    return float(mc.e_tot), np.asarray(mc.ci), ncas, nelecas


def casci_vector_to_qubit_state(ci: np.ndarray, ncas: int, nelecas: int) -> np.ndarray:
    """Map a PySCF CASCI CI matrix to the exported qubit basis.

    Export layout: qubit ``i`` = alpha spatial orbital ``i``, qubit ``ncas+i`` =
    beta spatial orbital ``i``; OpenFermion's sparse operator treats qubit 0 as
    the most significant bit.  PySCF determinants are ordered (alpha block)
    (beta block) with ascending orbital index inside each block, while the
    Jordan-Wigner transform of the exported Hamiltonian assumes creation
    operators ordered by qubit label.  Moving each beta creation operator past
    the alpha ones with a HIGHER spatial index gives the parity factor below.
    """

    n_qubits = 2 * ncas
    neleca = nelecas // 2
    strings = cistring.make_strings(range(ncas), neleca)
    n_strings = len(strings)
    ci = ci.reshape(n_strings, n_strings)

    state = np.zeros(2**n_qubits)
    for ia, string_a in enumerate(strings):
        alpha_occ = [i for i in range(ncas) if string_a >> i & 1]
        for ib, string_b in enumerate(strings):
            beta_occ = [j for j in range(ncas) if string_b >> j & 1]
            swaps = sum(1 for j in beta_occ for i in alpha_occ if i > j)
            sign = -1.0 if swaps % 2 else 1.0
            index = 0
            for i in alpha_occ:
                index |= 1 << (n_qubits - 1 - i)
            for j in beta_occ:
                index |= 1 << (n_qubits - 1 - (ncas + j))
            state[index] = sign * ci[ia, ib]
    return state


def compare_ground_state(molecule_name: str, bond: float) -> dict[str, float | str]:
    preset = MOLECULE_PRESETS[molecule_name]
    n_qubits = 2 * preset.active_space[1]

    qubit_hamiltonian, metadata = build_molecular_hamiltonian(molecule_name, bond)
    of_energy, of_vector = ground_state_energy_and_vector(qubit_hamiltonian, n_qubits)

    casci_energy, ci, ncas, nelecas = casci_reference(molecule_name, bond)
    casci_state = casci_vector_to_qubit_state(ci, ncas, nelecas)

    energy_abs_diff = abs(of_energy - casci_energy)
    vector_overlap = float(abs(np.vdot(of_vector, casci_state)))

    result: dict[str, float | str] = {
        "molecule": molecule_name,
        "bond_angstrom": float(bond),
        "orbital_selection": str(metadata["orbital_selection"]),
        "qubit_hamiltonian_ground_energy": of_energy,
        "casci_energy": casci_energy,
        "energy_abs_diff": energy_abs_diff,
        "vector_overlap_abs": vector_overlap,
    }

    if energy_abs_diff > ENERGY_ATOL:
        raise AssertionError(f"Ground-state energy mismatch: {result}")
    if vector_overlap < MIN_VECTOR_OVERLAP:
        raise AssertionError(f"Ground-state eigenvector mismatch: {result}")
    return result


# ---------------------------------------------------------------------------
# pytest entry points
# ---------------------------------------------------------------------------

def test_hf_canonical_bond_1_0():
    compare_ground_state("HF", 1.0)


def test_cl2_pinned_bond_1_0():
    compare_ground_state("Cl2", 1.0)


def test_cl2_pinned_bond_2_0():
    # Previously skipped in the preset grid: the energy-ordered active space
    # split a sigma/pi near-degeneracy here.  Pinned orbitals fix it.
    compare_ground_state("Cl2", 2.0)


def test_cl2_pinned_bond_3_0():
    compare_ground_state("Cl2", 3.0)


def test_br2_pinned_bond_2_2():
    compare_ground_state("Br2", 2.2)


def test_br2_pinned_bond_3_0():
    compare_ground_state("Br2", 3.0)


def test_br2_pinned_bond_3_8():
    compare_ground_state("Br2", 3.8)


# ---------------------------------------------------------------------------
# exported-file round trip (script mode)
# ---------------------------------------------------------------------------

def load_pauli_txt(path: Path, num_qubits: int) -> QubitOperator:
    """Load a repo-format numbered Pauli Hamiltonian txt file."""

    pauli_chars = {1: "X", 2: "Y", 3: "Z"}
    operator = QubitOperator()
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.split()
            if not parts:
                continue
            coeff = float(parts[0])
            codes = [int(p) for p in parts[1:]]
            if len(codes) != num_qubits:
                raise ValueError(f"{path}: expected {num_qubits} Pauli codes, got {len(codes)}")
            term = tuple((q, pauli_chars[c]) for q, c in enumerate(codes) if c != 0)
            operator += QubitOperator(term, coeff)
    return operator


def check_exported_file(molecule_name: str, bond: float) -> dict[str, float | str] | None:
    """Verify the exported Pauli_Ham txt file matches the freshly built Hamiltonian."""

    preset = MOLECULE_PRESETS[molecule_name]
    n_qubits = 2 * preset.active_space[1]
    path = REPO_ROOT / "Pauli_Ham" / f"{molecule_name}_bond_{bond_token(bond)}.txt"
    if not path.exists():
        return None

    file_ham = load_pauli_txt(path, n_qubits)
    fresh_ham, _ = build_molecular_hamiltonian(molecule_name, bond)
    diff = file_ham - fresh_ham
    max_term_diff = max((abs(c) for c in diff.terms.values()), default=0.0)
    file_energy, _ = ground_state_energy_and_vector(file_ham, n_qubits)
    fresh_energy, _ = ground_state_energy_and_vector(fresh_ham, n_qubits)

    result: dict[str, float | str] = {
        "file": str(path),
        "max_term_coeff_diff": float(max_term_diff),
        "file_ground_energy": file_energy,
        "fresh_ground_energy": fresh_energy,
        "energy_abs_diff": abs(file_energy - fresh_energy),
    }
    if max_term_diff > 1e-8:
        raise AssertionError(f"Exported file differs from freshly built Hamiltonian: {result}")
    return result


def main() -> None:
    print(f"energy_atol={ENERGY_ATOL}, min_vector_overlap={MIN_VECTOR_OVERLAP}\n")
    for molecule_name, bond in CASES:
        result = compare_ground_state(molecule_name, bond)
        print(
            f"{molecule_name:4s} d={bond:4.2f} A  [{result['orbital_selection']}]  "
            f"E_gs={result['qubit_hamiltonian_ground_energy']:.10f}  "
            f"E_casci={result['casci_energy']:.10f}  "
            f"dE={result['energy_abs_diff']:.3e}  "
            f"|overlap|={result['vector_overlap_abs']:.12f}"
        )
        file_result = check_exported_file(molecule_name, bond)
        if file_result is None:
            print("     (no exported Pauli_Ham txt file for this bond; skipped file check)")
        else:
            print(
                f"     exported file OK: max term diff {file_result['max_term_coeff_diff']:.2e}, "
                f"GS energy diff {file_result['energy_abs_diff']:.2e}"
            )
    print("\nAll ground-state eigenvector checks passed")


if __name__ == "__main__":
    main()
