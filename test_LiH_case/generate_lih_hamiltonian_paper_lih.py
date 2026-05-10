#!/usr/bin/env python3
"""Generate LiH paper-style Hamiltonian and export repo-compatible artifacts.

Qubit layout (after export)
----------------------------
OpenFermion/PySCF use **interleaved** spin orbitals within the active space::

    [α(spatial 0), β(spatial 0), α(spatial 1), β(spatial 1), α(spatial 2), β(spatial 2)]

Before saving, Pauli strings are **relabelled** to a spin-block layout (lower-energy spatial
index first within each spin block)::

    [α(spa0), α(spa1), α(spa2), β(spa0), β(spa1), β(spa2)]

With ``cirq.LineQubit(i)`` read left-to-right in bitstrings as printed by this repo (qubit 0 is
the MSB of the 6-bit computational index), restricted Hartree–Fock is the computational state
``100100``: ``X`` on the lowest spatial's α and β lines (qubits 0 and 3).

Outputs (pickle + Pauli/number conventions) are written under **``Pauli_Ham/``** at the repository
root by default (override with ``--output-dir``).
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import cirq
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

# region agent log
_DEBUG_LOG = Path("/Users/zacharyhe/cross_chips_sim/.cursor/debug-1436a0.log")


def _dbg(hypothesis_id: str, location: str, message: str, data: dict, run_id: str = "pre-fix") -> None:
    try:
        payload = {
            "sessionId": "1436a0",
            "timestamp": int(time.time() * 1000),
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "runId": run_id,
        }
        with _DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except OSError:
        pass


# endregion

# Map OpenFermion spin-orbital index j -> exported qubit index (spin-block layout).
# OF order: [αs0, βs0, αs1, βs1, αs2, βs2]. Export order: [αs0, αs1, αs2, βs0, βs1, βs2].
_OPENFERMION_SPIN_ORB_TO_EXPORT_QUBIT = [0, 3, 1, 4, 2, 5]


def bond_token(bond: float) -> str:
    return f"{bond}".rstrip("0").rstrip(".")


def relabel_qubit_operator(of_operator: QubitOperator, of_index_to_export_index: list[int]) -> QubitOperator:
    """Apply a permutation of Pauli wire indices (same physics, different labelling)."""

    out = QubitOperator()
    for term, coeff in of_operator.terms.items():
        new_term = tuple(sorted((of_index_to_export_index[j], p) for j, p in term))
        out += QubitOperator(new_term, coeff)
    return out


def build_paper_lih_hamiltonian(bond: float) -> tuple[QubitOperator, dict[str, object]]:
    geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, bond))]
    molecule = MolecularData(
        geometry=geometry,
        basis="sto-3g",
        multiplicity=1,
        charge=0,
        description=f"LiH_{bond}A_active_space",
    )
    # Explicit PySCF driver flags (only RHF): keep MP2/CISD/CCSD/FCI off, no extra logging.
    molecule = run_pyscf(
        molecule,
        run_scf=True,
        run_mp2=False,
        run_cisd=False,
        run_ccsd=False,
        run_fci=False,
        verbose=False,
    )

    # Match the active-space construction used in the user's reference script.
    occupied_indices = [0]
    active_indices = [1, 2, 3]
    active_space_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    qubit_of = jordan_wigner(get_fermion_operator(active_space_hamiltonian))
    qubit_hamiltonian = relabel_qubit_operator(qubit_of, _OPENFERMION_SPIN_ORB_TO_EXPORT_QUBIT)

    meta = {
        "group": "openfermionpyscf",
        "irrep_labels": ["1a1(core)", "2a1(active)", "3a1(active)", "1b1/1b2(active pair)"],
        "mo_occupations": "frozen core=[0], active=[1,2,3]",
        "active_indices": active_indices,
        "occupied_indices": occupied_indices,
        "n_qubits": 2 * len(active_indices),
        "n_terms": len(qubit_hamiltonian.terms),
        "rhf_energy": float(molecule.hf_energy),
        "export_qubit_layout": "[α(spa0),α(spa1),α(spa2),β(spa0),β(spa1),β(spa2)]",
        "rhf_bitstring_spin_blocked": "100100",
        "openfermion_spin_orb_to_export_qubit": list(_OPENFERMION_SPIN_ORB_TO_EXPORT_QUBIT),
    }
    return qubit_hamiltonian, meta


def save_hamiltonian(qubit_hamiltonian: QubitOperator, out_dir: Path, stem: str, num_qubits: int) -> tuple[Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Names aligned with repo convention: <stem>_of.pkl, <stem>.txt (numbered Paulis), <stem>_pauli_convention.txt
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


def ground_state_energy_and_vector(qubit_hamiltonian: QubitOperator) -> tuple[float, np.ndarray]:
    sparse = get_sparse_operator(qubit_hamiltonian).tocsc()
    if sparse.shape[0] <= 256:
        dense = sparse.toarray()
        w, v = np.linalg.eigh(dense)
        return float(np.real_if_close(w[0])), v[:, 0]
    eigvals, eigvecs = eigsh(sparse, k=1, which="SA")
    return float(np.real_if_close(eigvals[0])), eigvecs[:, 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=float, default=2.2)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Directory for Hamiltonian files (default: repository Pauli_Ham → {REPO_ROOT / 'Pauli_Ham'})",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="LiH",
        help="Stem prefix (default: LiH → LiH_bond_<bond>_of.pkl, LiH_bond_<bond>.txt, …).",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (REPO_ROOT / "Pauli_Ham").resolve()
    )

    h_qubit, meta = build_paper_lih_hamiltonian(args.bond)
    stem = f"{args.basename}_bond_{bond_token(args.bond)}"
    pkl_path, number_path, pauli_path = save_hamiltonian(
        h_qubit,
        output_dir,
        stem,
        int(meta["n_qubits"]),
    )
    gs, gs_vec = ground_state_energy_and_vector(h_qubit)
    hf_energy = float(meta["rhf_energy"])
    qubits = cirq.LineQubit.range(int(meta["n_qubits"]))
    psi_rhf = cirq.Simulator().simulate(
        cirq.Circuit(cirq.X(qubits[0]), cirq.X(qubits[3])),
        qubit_order=qubits,
    ).final_state_vector
    hf_fidelity = float(abs(np.vdot(psi_rhf, gs_vec)) ** 2)
    sparse_h = get_sparse_operator(h_qubit).tocsc()
    psi_np = np.asarray(psi_rhf, dtype=np.complex128).reshape(-1)
    e_rhf = float(np.real(np.vdot(psi_np, sparse_h @ psi_np)))

    # region agent log
    _dbg(
        "H1",
        "generate_lih_hamiltonian_paper_lih.py:main",
        "export_spin_block_rhf_consistency",
        {
            "bond": float(args.bond),
            "stem": stem,
            "n_terms": int(meta["n_terms"]),
            "hf_energy_pyscf": hf_energy,
            "e_rhf_active_h": e_rhf,
            "exact_gs_active": float(gs),
            "hf_fidelity_wrt_gs": hf_fidelity,
            "rhf_bitstring": str(meta["rhf_bitstring_spin_blocked"]),
        },
    )
    # endregion

    print("=== LiH paper-style Hamiltonian generated ===")
    print(f"output directory: {output_dir}")
    print(f"bond: {args.bond} Angstrom")
    print(f"group: {meta['group']}")
    print(f"orbital irreps: {meta['irrep_labels']}")
    print(f"MO occupations: {meta['mo_occupations']}")
    print(f"frozen occupied orbitals: {meta['occupied_indices']}")
    print(f"active spatial orbitals: {meta['active_indices']}")
    print(f"num qubits: {meta['n_qubits']}")
    print(f"num pauli terms: {meta['n_terms']}")
    print(f"export layout: {meta['export_qubit_layout']}")
    print(f"RHF computational bitstring (spin-block): {meta['rhf_bitstring_spin_blocked']}")
    print()
    print("--- Energy Comparison ---")
    print(f"Hartree-Fock Energy:                              {hf_energy:.8f} Eh")
    print(f"⟨H⟩ on RHF ket (X on qubits 0 and 3):              {e_rhf:.8f} Eh")
    print(f"Exact Ground State Energy (6-qubit active space): {gs:.8f} Eh")
    print(f"Hartree-Fock State Fidelity:                      {hf_fidelity:.6f}")
    print()
    print("saved files:")
    print(f"  {pkl_path}")
    print(f"  {number_path}")
    print(f"  {pauli_path}")


if __name__ == "__main__":
    main()
