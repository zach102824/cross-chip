"""Pre-taper vs post-taper Hamiltonian checks (eigenvalues, symmetries, HF).

These formalize the ad-hoc checks used while building the pipeline:

1. the two spin-block parity generators commute with the full Hamiltonian;
2. the tapering involutions square to the identity;
3. the tapered ground-state energy EQUALS the full ground-state energy
   (tapering is exact, not an approximation);
4. the tapered spectrum is a sub-multiset of the full spectrum (the tapered
   Hamiltonian is the full Hamiltonian restricted to one symmetry sector);
5. the HF determinant energy is identical before and after tapering;
6. the saved numbered Pauli_Ham file reproduces the freshly tapered operator.
"""

from __future__ import annotations

import numpy as np
import pytest
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator

from conftest import MOLECULE, TEST_BONDS, STATE_TRANSFER, hf_state_vector

GS_TOL = 1e-9
SPECTRUM_TOL = 1e-8


def _dense(operator, n_qubits):
    return get_sparse_operator(operator, n_qubits=n_qubits).toarray()


def test_symmetry_generators_commute_with_hamiltonian(taper_lib, taper, hamiltonians):
    for bond, (full_op, _tapered_op) in hamiltonians.items():
        for gen_string in taper.symmetry_generators:
            g = taper_lib.pauli_string_to_qubit_operator(gen_string)
            commutator = g * full_op - full_op * g
            assert commutator.induced_norm() < 1e-8, (
                f"bond {bond}: generator {gen_string} does not commute with H"
            )


def test_tapering_involutions_square_to_identity(taper_lib, taper):
    identity = QubitOperator((), 1.0)
    for gen_string, q in zip(taper.symmetry_generators, taper.removed_qubits):
        g = taper_lib.pauli_string_to_qubit_operator(gen_string)
        sigma = QubitOperator(((q, "X"),), 1.0)
        u = (g + sigma) * (1.0 / np.sqrt(2.0))
        assert (u * u - identity).induced_norm() < 1e-9


@pytest.mark.parametrize("bond", TEST_BONDS)
def test_ground_state_energy_is_preserved(taper, hamiltonians, bond):
    full_op, tapered_op = hamiltonians[bond]
    e_full = np.linalg.eigvalsh(_dense(full_op, taper.n_qubits_full))[0].real
    e_tapered = np.linalg.eigvalsh(_dense(tapered_op, taper.n_qubits_tapered))[0].real
    assert abs(e_full - e_tapered) < GS_TOL, (
        f"bond {bond}: full GS {e_full:.12f} != tapered GS {e_tapered:.12f}"
    )


@pytest.mark.parametrize("bond", TEST_BONDS)
def test_tapered_spectrum_is_subset_of_full_spectrum(taper, hamiltonians, bond):
    """Every tapered eigenvalue must appear in the full spectrum (sector restriction)."""
    full_op, tapered_op = hamiltonians[bond]
    evals_full = np.sort(np.linalg.eigvalsh(_dense(full_op, taper.n_qubits_full)).real)
    evals_tap = np.sort(np.linalg.eigvalsh(_dense(tapered_op, taper.n_qubits_tapered)).real)

    # Greedy multiset-subset match on sorted arrays.
    i = 0
    for ev in evals_tap:
        while i < len(evals_full) and evals_full[i] < ev - SPECTRUM_TOL:
            i += 1
        assert i < len(evals_full) and abs(evals_full[i] - ev) < SPECTRUM_TOL, (
            f"bond {bond}: tapered eigenvalue {ev:.10f} missing from full spectrum"
        )
        i += 1


@pytest.mark.parametrize("bond", TEST_BONDS)
def test_hf_determinant_energy_is_preserved(taper, hamiltonians, bond):
    full_op, tapered_op = hamiltonians[bond]
    occ_full = [q for q, b in enumerate(taper.hf_bitstring_full) if b == "1"]
    occ_tap = [q for q, b in enumerate(taper.hf_bitstring_tapered) if b == "1"]

    psi_full = hf_state_vector(taper.n_qubits_full, occ_full)
    psi_tap = hf_state_vector(taper.n_qubits_tapered, occ_tap)
    e_full = np.real(psi_full.conj() @ _dense(full_op, taper.n_qubits_full) @ psi_full)
    e_tap = np.real(psi_tap.conj() @ _dense(tapered_op, taper.n_qubits_tapered) @ psi_tap)
    assert abs(e_full - e_tap) < GS_TOL


def test_saved_pauli_ham_file_matches_fresh_tapering(gm, taper, hamiltonians):
    """The numbered file written by generate_tapered_hamiltonians.py must
    reproduce the operator obtained by tapering from scratch."""
    bond = TEST_BONDS[0]
    _full_op, tapered_op = hamiltonians[bond]
    path = STATE_TRANSFER / "Pauli_Ham" / f"{MOLECULE}_tapered_bond_{gm.bond_token(bond)}.txt"
    if not path.is_file():
        pytest.skip(f"{path} not generated yet (run generate_tapered_hamiltonians.py)")

    idx_to_pauli = {1: "X", 2: "Y", 3: "Z"}
    loaded = QubitOperator()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split()
        coeff = float(parts[0])
        codes = [int(x) for x in parts[1:]]
        assert len(codes) == taper.n_qubits_tapered
        term = tuple((q, idx_to_pauli[c]) for q, c in enumerate(codes) if c != 0)
        loaded += QubitOperator(term, coeff)

    difference = loaded - tapered_op
    difference.compress(1e-10)
    assert not difference.terms, f"saved file differs from fresh tapering: {difference}"
