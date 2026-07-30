#!/usr/bin/env python3
"""Generate qubit-tapered active-space molecular Hamiltonians.

This is the tapered counterpart of
``June_main/generate_molecular_hamiltonians.py``.  It reuses that script's
molecule presets and full-register Hamiltonian builder, then applies the exact
two-parity Z2 tapering from :mod:`taper_lib` to remove 2 qubits (HF/H4 8 -> 6,
Cl2 10 -> 8, Br2 12 -> 10).

Defaults to HF; pass ``--molecule Cl2`` / ``Br2`` / ``LiH`` / ``F2`` / ``H4``
(or ``--bonds``) to generate the others.  LiH/F2 match Guo et al. arXiv:2212.08006
(6-qubit LiH active ``{1,2,5}``, 12-qubit F2 CAS(10,6)); H4 is linear CAS(4,4)
(8 -> 6 qubits, ``UCCSD_Mole/H4.ipynb``).  For each bond length it writes, under
``state_transfer/Pauli_Ham``:

    <mol>_tapered_bond_<b>.txt        numbered Pauli Hamiltonian (I=0,X=1,Y=2,Z=3)
    <mol>_tapered_bond_<b>_meta.json  tapering description (taper_lib.TaperData)

and a ``<mol>_tapered_bond_scan_summary.txt`` with per-bond HF / ground-state
energies.  A per-bond assertion checks that the tapered ground-state energy
equals the full-register ground-state energy (tapering is exact).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_JUNE_MAIN = _REPO_ROOT / "June_main"
for _p in (str(_THIS_DIR), str(_JUNE_MAIN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import generate_molecular_hamiltonians as gm  # noqa: E402  (June_main builder + presets)
import taper_lib  # noqa: E402

EXACTNESS_TOL = 1e-7


def build_tapered_hamiltonian(molecule_name: str, bond: float, basis: str | None = None):
    """Build the full-register Hamiltonian then taper it by 2 qubits.

    Returns ``(tapered_operator, taper_data, metadata)`` where ``metadata``
    extends the June_main metadata with tapering fields and (optionally) the
    full/tapered ground-state energies filled in by the caller.
    """
    preset = gm.MOLECULE_PRESETS[molecule_name]
    full_operator, metadata = gm.build_molecular_hamiltonian(molecule_name, bond, basis)

    n_spatial = int(preset.active_space[1])
    n_electrons = int(preset.active_space[0])
    taper = taper_lib.build_taper_data(n_spatial=n_spatial, n_electrons=n_electrons)

    tapered_operator = taper_lib.taper_qubit_operator(full_operator, taper)

    metadata = dict(metadata)
    metadata.update(
        {
            "tapered": True,
            "n_qubits_full": taper.n_qubits_full,
            "n_qubits": taper.n_qubits_tapered,
            "n_terms_full": len(full_operator.terms),
            "n_terms": len(tapered_operator.terms),
            "removed_qubits": list(taper.removed_qubits),
            "tapering_values": list(taper.tapering_values),
            "symmetry_generators": list(taper.symmetry_generators),
            "kept_qubits": list(taper.kept_qubits),
            "hf_bitstring_full": taper.hf_bitstring_full,
            "hf_bitstring_tapered": taper.hf_bitstring_tapered,
        }
    )
    return full_operator, tapered_operator, taper, metadata


def _ground_state_energy(operator, num_qubits: int) -> float:
    energy, _ = gm.ground_state_energy_and_vector(operator, num_qubits)
    return float(energy)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--molecule",
        choices=sorted(gm.MOLECULE_PRESETS),
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
        default=_THIS_DIR / "Pauli_Ham",
        help="Directory for tapered Hamiltonian files. Defaults to state_transfer/Pauli_Ham.",
    )
    parser.add_argument(
        "--skip-exactness-check",
        action="store_true",
        help="Skip the full-vs-tapered ground-state energy assertion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = gm.MOLECULE_PRESETS[args.molecule]
    bonds = tuple(args.bonds) if args.bonds else preset.default_bonds
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basis = args.basis or preset.basis

    print(f"Generating TAPERED {preset.name} Hamiltonians")
    print(f"active_space={preset.active_space}, basis={basis}")
    print(f"bonds={list(bonds)}")
    print(f"output_dir={output_dir}")
    print()

    summary_rows: list[dict[str, float | None]] = []

    for bond in bonds:
        full_op, tapered_op, taper, metadata = build_tapered_hamiltonian(
            args.molecule, bond, basis
        )
        n_qubits_tapered = taper.n_qubits_tapered

        e_gs_tapered = _ground_state_energy(tapered_op, n_qubits_tapered)
        metadata["exact_ground_state_energy"] = e_gs_tapered
        if not args.skip_exactness_check:
            e_gs_full = _ground_state_energy(full_op, taper.n_qubits_full)
            metadata["exact_ground_state_energy_full"] = e_gs_full
            if abs(e_gs_full - e_gs_tapered) > EXACTNESS_TOL:
                raise AssertionError(
                    f"{preset.name} bond {bond}: tapered GS {e_gs_tapered:.12f} != "
                    f"full GS {e_gs_full:.12f} (diff {abs(e_gs_full - e_gs_tapered):.3e})"
                )

        token = gm.bond_token(bond)
        stem = f"{preset.name}_tapered_bond_{token}"
        ham_path = output_dir / f"{stem}.txt"
        taper_lib.write_numbered_hamiltonian(tapered_op, n_qubits_tapered, ham_path)

        meta_path = output_dir / f"{stem}_meta.json"
        taper.save_json(meta_path)

        summary_rows.append(
            {
                "bond_angstrom": float(bond),
                "rhf_energy": float(metadata["rhf_energy"]),
                "exact_ground_state_energy": e_gs_tapered,
            }
        )
        print(
            f"{stem}: n_qubits {taper.n_qubits_full} -> {n_qubits_tapered}, "
            f"n_terms {metadata['n_terms_full']} -> {metadata['n_terms']}, "
            f"rhf={metadata['rhf_energy']:.10f}, gs={e_gs_tapered:.10f}"
        )
        print(f"  saved: {ham_path.name}  (+ {meta_path.name})")

    summary_path = gm.write_bond_energy_summary(
        output_dir,
        f"{preset.name}_tapered",
        preset.active_space,
        basis,
        summary_rows,
    )
    print()
    print(f"Energy summary saved: {summary_path}")


if __name__ == "__main__":
    main()
