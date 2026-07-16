#!/usr/bin/env python3
"""Generate powers of a molecular Hamiltonian (H^2, H^3, ...) in the repo Pauli format.

The connected-moments method needs the Hamiltonian powers H^k.  This script reads
the base Pauli Hamiltonians produced by ``generate_molecular_hamiltonians.py`` (e.g.
``Pauli_Ham/HF_bond_1.2.txt``) and produces the squared and cubed operators:

    Pauli_Ham/HF_square_bond_1.2.txt   (= H @ H)
    Pauli_Ham/HF_triple_bond_1.2.txt   (= H @ H @ H)

Each output keeps the same line format as the base file:

    coefficient I/X/Y/Z-as-number-for-qubit-0 ...

with ``I=0, X=1, Y=2, Z=3``.

Pauli products are evaluated analytically with the single-qubit Pauli algebra and
identical Pauli strings are combined (their coefficients are summed), which is the
standard simplification for these operators.  Because H is Hermitian every power is
Hermitian too, so the combined coefficients are real (a small imaginary tolerance is
enforced as a safety check).

By default this runs for HF, but ``--molecule`` switches to Cl2 or Br2 once their
base Hamiltonians have been generated.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PAULI_DIR = REPO_ROOT / "Pauli_Ham"

SUPPORTED_MOLECULES = ("HF", "Cl2", "Br2")

# Human-readable labels used in the output file stems for each Hamiltonian power.
POWER_LABELS: dict[int, str] = {2: "square", 3: "triple"}

# Single-qubit Pauli multiplication table (left * right) with the global phase.
# Encoding: I=0, X=1, Y=2, Z=3.  Value is (result_pauli, phase) where phase is one
# of {1, 1j, -1, -1j}.  Example: X * Y = i Z  -> (1, 2) -> (3, 1j).
_PAULI_PRODUCT: dict[tuple[int, int], tuple[int, complex]] = {
    (0, 0): (0, 1), (0, 1): (1, 1), (0, 2): (2, 1), (0, 3): (3, 1),
    (1, 0): (1, 1), (1, 1): (0, 1), (1, 2): (3, 1j), (1, 3): (2, -1j),
    (2, 0): (2, 1), (2, 1): (3, -1j), (2, 2): (0, 1), (2, 3): (1, 1j),
    (3, 0): (3, 1), (3, 1): (2, 1j), (3, 2): (1, -1j), (3, 3): (0, 1),
}

# Term -> complex coefficient, where a term is a tuple of per-qubit Pauli ints.
PauliOperator = dict[tuple[int, ...], complex]

IMAG_TOLERANCE = 1e-8
COEFF_DROP_TOLERANCE = 1e-12


def read_pauli_file(path: Path) -> tuple[PauliOperator, int]:
    """Load a numbered-Pauli Hamiltonian file into a {term: coeff} operator."""

    operator: PauliOperator = {}
    num_qubits: int | None = None
    with path.open("r", encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            coeff = complex(float(tokens[0]))
            paulis = tuple(int(token) for token in tokens[1:])
            if num_qubits is None:
                num_qubits = len(paulis)
            elif len(paulis) != num_qubits:
                raise ValueError(
                    f"{path}:{line_number}: expected {num_qubits} qubit labels, got {len(paulis)}."
                )
            operator[paulis] = operator.get(paulis, 0j) + coeff
    if num_qubits is None:
        raise ValueError(f"{path}: no Pauli terms found.")
    return operator, num_qubits


def multiply_pauli_terms(
    left_term: tuple[int, ...], right_term: tuple[int, ...]
) -> tuple[tuple[int, ...], complex]:
    """Multiply two equal-length Pauli strings, returning (result_term, phase)."""

    result = [0] * len(left_term)
    phase: complex = 1 + 0j
    for index, (left, right) in enumerate(zip(left_term, right_term)):
        product_pauli, product_phase = _PAULI_PRODUCT[(left, right)]
        result[index] = product_pauli
        phase *= product_phase
    return tuple(result), phase


def multiply_operators(left: PauliOperator, right: PauliOperator) -> PauliOperator:
    """Multiply two operators, combining identical resulting Pauli strings."""

    product: PauliOperator = {}
    for left_term, left_coeff in left.items():
        for right_term, right_coeff in right.items():
            result_term, phase = multiply_pauli_terms(left_term, right_term)
            product[result_term] = (
                product.get(result_term, 0j) + left_coeff * right_coeff * phase
            )
    return product


def operator_power(operator: PauliOperator, power: int) -> PauliOperator:
    """Compute operator ** power for integer power >= 1."""

    if power < 1:
        raise ValueError("power must be >= 1")
    result = dict(operator)
    for _ in range(power - 1):
        result = multiply_operators(result, operator)
    return result


def to_real_pauli_rows(operator: PauliOperator) -> list[tuple[float, tuple[int, ...]]]:
    """Drop negligible terms, enforce real coefficients, and sort for stable output."""

    rows: list[tuple[float, tuple[int, ...]]] = []
    for term, coeff in operator.items():
        if abs(coeff.imag) > IMAG_TOLERANCE:
            raise ValueError(
                f"Term {term} has a non-negligible imaginary coefficient {coeff}; "
                "the operator is not Hermitian as expected."
            )
        real_coeff = coeff.real
        if abs(real_coeff) <= COEFF_DROP_TOLERANCE:
            continue
        rows.append((real_coeff, term))
    rows.sort(key=lambda row: row[1])
    return rows


def write_pauli_file(path: Path, rows: list[tuple[float, tuple[int, ...]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for coeff, term in rows:
            file.write(" ".join([repr(coeff), *map(str, term)]) + "\n")


def base_bond_files(pauli_dir: Path, molecule: str) -> list[tuple[str, Path]]:
    """Find base Hamiltonian files ``{molecule}_bond_{token}.txt`` and their tokens."""

    pattern = re.compile(rf"^{re.escape(molecule)}_bond_(?P<token>[^_]+)\.txt$")
    found: list[tuple[str, Path]] = []
    for path in sorted(pauli_dir.glob(f"{molecule}_bond_*.txt")):
        match = pattern.match(path.name)
        if match:
            found.append((match.group("token"), path))
    return found


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--molecule",
        choices=SUPPORTED_MOLECULES,
        default="HF",
        help="Molecule whose base Hamiltonians to raise to powers. Defaults to HF.",
    )
    parser.add_argument(
        "--powers",
        type=int,
        nargs="*",
        default=sorted(POWER_LABELS),
        help=f"Hamiltonian powers to generate. Defaults to {sorted(POWER_LABELS)}.",
    )
    parser.add_argument(
        "--pauli-dir",
        type=Path,
        default=DEFAULT_PAULI_DIR,
        help="Directory holding the base Hamiltonians and receiving the powers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pauli_dir = args.pauli_dir.expanduser().resolve()

    for power in args.powers:
        if power not in POWER_LABELS and power != 1:
            raise SystemExit(
                f"No file label defined for power {power}. Known labels: {POWER_LABELS}."
            )

    bond_files = base_bond_files(pauli_dir, args.molecule)
    if not bond_files:
        raise SystemExit(
            f"No base Hamiltonians found for {args.molecule} in {pauli_dir}. "
            "Run generate_molecular_hamiltonians.py first."
        )

    print(f"Generating Hamiltonian powers for {args.molecule}")
    print(f"powers={list(args.powers)}, pauli_dir={pauli_dir}")
    print()

    for token, base_path in bond_files:
        operator, num_qubits = read_pauli_file(base_path)
        print(f"{base_path.name}: n_qubits={num_qubits}, n_terms={len(operator)}")
        for power in args.powers:
            label = "linear" if power == 1 else POWER_LABELS[power]
            powered = operator_power(operator, power)
            rows = to_real_pauli_rows(powered)
            out_path = pauli_dir / f"{args.molecule}_{label}_bond_{token}.txt"
            write_pauli_file(out_path, rows)
            print(f"  H^{power} ({label}): n_terms={len(rows)} -> {out_path.name}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
