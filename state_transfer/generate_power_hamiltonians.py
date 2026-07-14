#!/usr/bin/env python3
"""Generate powers of tapered molecular Hamiltonians (H^2, H^3) in Pauli format.

Tapered counterpart of ``June_main/generate_power_hamiltonians.py``.  Reads base
files written by ``generate_tapered_hamiltonians.py``:

    state_transfer/Pauli_Ham/<mol>_tapered_bond_<b>.txt

and writes:

    state_transfer/Pauli_Ham/<mol>_tapered_square_bond_<b>.txt   (= H @ H)
    state_transfer/Pauli_Ham/<mol>_tapered_triple_bond_<b>.txt   (= H @ H @ H)

Reuses the Pauli algebra and I/O helpers from the June_main script.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_JUNE_MAIN = _REPO_ROOT / "June_main"
if str(_JUNE_MAIN) not in sys.path:
    sys.path.insert(0, str(_JUNE_MAIN))

import generate_power_hamiltonians as gph  # noqa: E402

DEFAULT_PAULI_DIR = _THIS_DIR / "Pauli_Ham"
SUPPORTED_MOLECULES = ("HF", "Cl2", "Br2")
TAPERED_TAG = "tapered"


def base_tapered_bond_files(pauli_dir: Path, molecule: str) -> list[tuple[str, Path]]:
    """Find ``{molecule}_tapered_bond_{token}.txt`` base Hamiltonian files."""

    pattern = re.compile(
        rf"^{re.escape(molecule)}_{TAPERED_TAG}_bond_(?P<token>[^_]+)\.txt$"
    )
    found: list[tuple[str, Path]] = []
    for path in sorted(pauli_dir.glob(f"{molecule}_{TAPERED_TAG}_bond_*.txt")):
        if path.name.endswith("_scan_summary.txt") or path.name.endswith("_meta.json"):
            continue
        match = pattern.match(path.name)
        if match:
            found.append((match.group("token"), path))
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--molecule",
        choices=SUPPORTED_MOLECULES,
        default="HF",
        help="Molecule whose tapered base Hamiltonians to raise to powers. Defaults to HF.",
    )
    parser.add_argument(
        "--powers",
        type=int,
        nargs="*",
        default=sorted(gph.POWER_LABELS),
        help=f"Hamiltonian powers to generate. Defaults to {sorted(gph.POWER_LABELS)}.",
    )
    parser.add_argument(
        "--pauli-dir",
        type=Path,
        default=DEFAULT_PAULI_DIR,
        help="Directory holding tapered base Hamiltonians and receiving the powers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pauli_dir = args.pauli_dir.expanduser().resolve()

    for power in args.powers:
        if power not in gph.POWER_LABELS and power != 1:
            raise SystemExit(
                f"No file label defined for power {power}. Known labels: {gph.POWER_LABELS}."
            )

    bond_files = base_tapered_bond_files(pauli_dir, args.molecule)
    if not bond_files:
        raise SystemExit(
            f"No tapered base Hamiltonians found for {args.molecule} in {pauli_dir}. "
            "Run generate_tapered_hamiltonians.py first."
        )

    print(f"Generating tapered Hamiltonian powers for {args.molecule}")
    print(f"powers={list(args.powers)}, pauli_dir={pauli_dir}")
    print()

    for token, base_path in bond_files:
        operator, num_qubits = gph.read_pauli_file(base_path)
        print(f"{base_path.name}: n_qubits={num_qubits}, n_terms={len(operator)}")
        for power in args.powers:
            label = "linear" if power == 1 else gph.POWER_LABELS[power]
            powered = gph.operator_power(operator, power)
            rows = gph.to_real_pauli_rows(powered)
            out_path = pauli_dir / f"{args.molecule}_{TAPERED_TAG}_{label}_bond_{token}.txt"
            gph.write_pauli_file(out_path, rows)
            print(f"  H^{power} ({label}): n_terms={len(rows)} -> {out_path.name}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
