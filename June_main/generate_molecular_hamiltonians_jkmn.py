#!/usr/bin/env python3
"""Generate active-space molecular Hamiltonians with the COSMA JKMN mapping.

Same molecule presets / active-space construction as
``generate_molecular_hamiltonians.py``, but fermion → qubit uses the
product-preserving ternary-tree JKMN map from ``H4_circuits/cosma_uccsd``
instead of OpenFermion Jordan–Wigner.

Exports numbered Pauli files under ``Pauli_Ham_JKMN``:

    coefficient I/X/Y/Z-as-number-for-qubit-0 ...

with ``I=0, X=1, Y=2, Z=3``.  Qubit labels follow OpenFermion's interleaved
spin-orbital order (no spin-block relabel).

Use ``--compare`` to print JW vs JKMN term counts, Pauli-weight stats, and
ground-state energy agreement (H4 is the default molecule).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from openfermion import FermionOperator, QubitOperator
from openfermion.linalg import get_sparse_operator
from openfermion.transforms import jordan_wigner
from scipy.sparse.linalg import eigsh

JUNE_MAIN = Path(__file__).resolve().parent
REPO_ROOT = JUNE_MAIN.parent
H4_CIRCUITS = REPO_ROOT / "H4_circuits"
if str(JUNE_MAIN) not in sys.path:
    sys.path.insert(0, str(JUNE_MAIN))
if str(H4_CIRCUITS) not in sys.path:
    sys.path.insert(0, str(H4_CIRCUITS))

from cosma_uccsd.majorana import _accumulate, majorana_to_pauli  # noqa: E402
from cosma_uccsd.mapping import MappingBasis, build_mapping  # noqa: E402
from generate_molecular_hamiltonians import (  # noqa: E402
    MOLECULE_PRESETS,
    bond_token,
    build_active_fermion_operator,
    save_hamiltonian,
    write_bond_energy_summary,
)


def fermion_term_to_majorana(term: tuple, coeff: complex) -> dict[tuple[int, ...], complex]:
    """Expand one FermionOperator term in native ladder order into Majoranas.

    Convention (matches cosma_uccsd / COSMA):
      a_p     = (γ_{2p} + i γ_{2p+1}) / 2
      a_p^dag = (γ_{2p} - i γ_{2p+1}) / 2
    """

    factors: list[list[tuple[int, complex]]] = []
    for orbital, is_creation in term:
        if is_creation:
            factors.append([(2 * orbital, 0.5), (2 * orbital + 1, -0.5j)])
        else:
            factors.append([(2 * orbital, 0.5), (2 * orbital + 1, 0.5j)])

    acc: dict[tuple[int, ...], complex] = {(): complex(coeff)}
    for choices in factors:
        nxt: dict[tuple[int, ...], complex] = {}
        for modes, c0 in acc.items():
            for mode, cm in choices:
                _accumulate(nxt, list(modes) + [mode], c0 * cm)
        acc = nxt
    return {k: v for k, v in acc.items() if abs(v) > 1e-15}


def pauli_string_dict_to_qubit_operator(
    paulis: dict[str, complex],
    cutoff: float = 1e-12,
) -> QubitOperator:
    op = QubitOperator()
    for label, coeff in paulis.items():
        if abs(coeff) <= cutoff:
            continue
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"Non-Hermitian Pauli coeff after mapping: {label} {coeff}")
        term = tuple((i, ch) for i, ch in enumerate(label) if ch != "I")
        op += QubitOperator(term, float(coeff.real))
    return op


def fermion_operator_to_qubit(
    fermion_op: FermionOperator,
    basis: MappingBasis,
    cutoff: float = 1e-12,
) -> QubitOperator:
    """Map a FermionOperator through Majoranas and a COSMA PPTT basis."""

    merged: dict[str, complex] = {}
    for term, coeff in fermion_op.terms.items():
        maj = fermion_term_to_majorana(term, complex(coeff))
        for label, c in majorana_to_pauli(maj, basis, cutoff=cutoff).items():
            merged[label] = merged.get(label, 0.0) + c
    return pauli_string_dict_to_qubit_operator(merged, cutoff=cutoff)


def ground_state_energy(operator: QubitOperator, num_qubits: int) -> float:
    sparse = get_sparse_operator(operator, n_qubits=num_qubits).tocsc()
    if sparse.shape[0] <= 4096:
        eigenvalues = np.linalg.eigvalsh(sparse.toarray())
        return float(np.real_if_close(eigenvalues[0]))
    eigenvalues, _ = eigsh(sparse, k=1, which="SA")
    return float(np.real_if_close(eigenvalues[0]))


def pauli_weights(operator: QubitOperator) -> np.ndarray:
    weights = [len(term) for term in operator.terms]
    if not weights:
        return np.zeros(0, dtype=float)
    return np.asarray(weights, dtype=float)


def weight_stats(operator: QubitOperator) -> dict[str, float | int]:
    w = pauli_weights(operator)
    if w.size == 0:
        return {"n_terms": 0, "mean_weight": 0.0, "median_weight": 0.0, "max_weight": 0}
    return {
        "n_terms": int(w.size),
        "mean_weight": float(w.mean()),
        "median_weight": float(np.median(w)),
        "max_weight": int(w.max()),
    }


def build_molecular_hamiltonian_jkmn(
    molecule_name: str,
    bond: float,
    basis: str | None = None,
) -> tuple[QubitOperator, dict[str, object], FermionOperator]:
    fermion_op, metadata = build_active_fermion_operator(molecule_name, bond, basis)
    n_qubits = int(metadata["n_qubits"])
    basis_map = build_mapping("JKMN", n_qubits)
    qubit_hamiltonian = fermion_operator_to_qubit(fermion_op, basis_map)
    metadata = {
        **metadata,
        "n_terms": len(qubit_hamiltonian.terms),
        "export_qubit_layout": "openfermion_interleaved_spin_orbitals",
        "fermion_to_qubit_mapping": "JKMN_ternary_tree_cosma",
    }
    return qubit_hamiltonian, metadata, fermion_op


def compare_mappings(
    fermion_op: FermionOperator,
    jkmn_op: QubitOperator,
    n_qubits: int,
    *,
    skip_ground_state: bool = False,
) -> dict[str, object]:
    """Compare OpenFermion JW, Majorana-pipeline JW, and JKMN."""

    of_jw = jordan_wigner(fermion_op)
    pipeline_jw = fermion_operator_to_qubit(fermion_op, build_mapping("JW", n_qubits))
    jkmn_stats = weight_stats(jkmn_op)
    of_stats = weight_stats(of_jw)
    pipe_stats = weight_stats(pipeline_jw)

    result: dict[str, object] = {
        "openfermion_jw": of_stats,
        "pipeline_jw": pipe_stats,
        "jkmn": jkmn_stats,
    }

    if not skip_ground_state:
        e_of = ground_state_energy(of_jw, n_qubits)
        e_pipe = ground_state_energy(pipeline_jw, n_qubits)
        e_jkmn = ground_state_energy(jkmn_op, n_qubits)
        result["E_openfermion_jw"] = e_of
        result["E_pipeline_jw"] = e_pipe
        result["E_jkmn"] = e_jkmn
        result["abs_E_pipe_minus_of"] = abs(e_pipe - e_of)
        result["abs_E_jkmn_minus_of"] = abs(e_jkmn - e_of)
    return result


def _print_weight_histograms(
    fermion_op: FermionOperator,
    jkmn_op: QubitOperator,
    n_qubits: int,
) -> None:
    of_jw = jordan_wigner(fermion_op)
    print("  weight histogram (weight: JW_count / JKMN_count)")
    max_w = max(
        int(pauli_weights(of_jw).max() if of_jw.terms else 0),
        int(pauli_weights(jkmn_op).max() if jkmn_op.terms else 0),
    )
    jw_w = pauli_weights(of_jw)
    jk_w = pauli_weights(jkmn_op)
    for w in range(0, max_w + 1):
        jw_c = int(np.sum(jw_w == w)) if jw_w.size else 0
        jk_c = int(np.sum(jk_w == w)) if jk_w.size else 0
        if jw_c or jk_c:
            print(f"    w={w}: {jw_c} / {jk_c}")
    print()


def _print_compare(bond: float, cmp: dict[str, object]) -> None:
    of_s = cmp["openfermion_jw"]
    pipe_s = cmp["pipeline_jw"]
    jk_s = cmp["jkmn"]
    assert isinstance(of_s, dict) and isinstance(pipe_s, dict) and isinstance(jk_s, dict)
    print(f"--- compare bond={bond_token(bond)} ---")
    print(
        f"  OpenFermion JW : n_terms={of_s['n_terms']}, "
        f"mean_w={of_s['mean_weight']:.3f}, median_w={of_s['median_weight']:.1f}, "
        f"max_w={of_s['max_weight']}"
    )
    print(
        f"  Pipeline JW    : n_terms={pipe_s['n_terms']}, "
        f"mean_w={pipe_s['mean_weight']:.3f}, median_w={pipe_s['median_weight']:.1f}, "
        f"max_w={pipe_s['max_weight']}"
    )
    print(
        f"  JKMN           : n_terms={jk_s['n_terms']}, "
        f"mean_w={jk_s['mean_weight']:.3f}, median_w={jk_s['median_weight']:.1f}, "
        f"max_w={jk_s['max_weight']}"
    )
    if "E_openfermion_jw" in cmp:
        print(
            f"  E_OF_JW={cmp['E_openfermion_jw']:.12f}, "
            f"E_pipe_JW={cmp['E_pipeline_jw']:.12f}, "
            f"E_JKMN={cmp['E_jkmn']:.12f}"
        )
        print(
            f"  |E_pipe-E_OF|={cmp['abs_E_pipe_minus_of']:.3e}, "
            f"|E_JKMN-E_OF|={cmp['abs_E_jkmn_minus_of']:.3e}"
        )
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule",
        choices=sorted(MOLECULE_PRESETS),
        default="H4",
        help="Molecule preset. Defaults to H4.",
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
        default=REPO_ROOT / "Pauli_Ham_JKMN",
        help="Directory for JKMN Hamiltonian files. Defaults to repo Pauli_Ham_JKMN.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print JW vs JKMN term/weight/energy comparison for each bond.",
    )
    parser.add_argument(
        "--skip-ground-state",
        action="store_true",
        help="Skip exact ground-state calculation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = MOLECULE_PRESETS[args.molecule]
    bonds = tuple(args.bonds) if args.bonds else preset.default_bonds
    output_dir = args.output_dir.expanduser().resolve()
    basis = args.basis or preset.basis

    print(f"Generating {preset.name} JKMN Hamiltonians")
    print(f"active_space={preset.active_space}, basis={basis}")
    print(f"bonds={list(bonds)}")
    print(f"output_dir={output_dir}")
    print()

    summary_rows: list[dict[str, float | None]] = []

    for bond in bonds:
        jkmn_op, metadata, fermion_op = build_molecular_hamiltonian_jkmn(
            args.molecule, bond, basis
        )
        n_qubits = int(metadata["n_qubits"])

        if args.compare:
            cmp = compare_mappings(
                fermion_op,
                jkmn_op,
                n_qubits,
                skip_ground_state=args.skip_ground_state,
            )
            _print_compare(bond, cmp)
            _print_weight_histograms(fermion_op, jkmn_op, n_qubits)

        if not args.skip_ground_state:
            gs_energy = ground_state_energy(jkmn_op, n_qubits)
            metadata["exact_ground_state_energy"] = gs_energy

        stem = f"{preset.name}_bond_{bond_token(bond)}"
        saved_path = save_hamiltonian(jkmn_op, output_dir, stem, metadata)
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
