#!/usr/bin/env python3
"""Molecule-agnostic Z2 qubit tapering for the spin-block qubit layout.

This module implements *exact* qubit tapering using the two Z2 symmetries that
every closed-shell molecule has in the repo's spin-block layout
(``[alpha spatial orbitals..., beta spatial orbitals...]``):

* alpha-block spin parity   ``Z_0 Z_1 ... Z_{n_spatial-1}``
* beta-block  spin parity   ``Z_{n_spatial} ... Z_{2*n_spatial-1}``

Using exactly these two generators guarantees a 2-qubit reduction
(HF 8 -> 6, Cl2 10 -> 8, Br2 12 -> 10) instead of ``find_Z2_symmetries``,
which may discover more symmetries and over-taper.

Tapering is a Clifford conjugation ``U (.) U``, with the involution
``U_i = (g_i + X_{q_i}) / sqrt(2)`` per generator (``q_i`` = removed qubit).
The conjugation is evaluated *exactly* via the expansion

    U_i P U_i = 1/2 (g_i P g_i + g_i P X_{q_i} + X_{q_i} P g_i + X_{q_i} P X_{q_i})

which only ever produces clean ``0.5`` prefactors (no irrational ``1/sqrt(2)``),
so a symmetry-respecting single Pauli string maps to a single Pauli string
times a sign -- exactly what the UCCSD excitation generators need.

Nothing here is molecule-specific: the caller passes ``n_spatial`` and
``n_electrons`` and gets back a :class:`TaperData` describing the identical
transformation used by the Hamiltonian generator, the notebook and the circuit
generator.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from openfermion import QubitOperator

PAULI_TO_CODE = {"I": 0, "X": 1, "Y": 2, "Z": 3}
CODE_TO_PAULI = {v: k for k, v in PAULI_TO_CODE.items()}

_SIGN_TOL = 1e-9


# ----------------------------------------------------------------------
# Pauli-string helpers (strings are length-n over the alphabet I/X/Y/Z,
# index 0 == qubit 0, matching the numbered-Pauli export order)
# ----------------------------------------------------------------------
def pauli_string_to_qubit_operator(string: str) -> QubitOperator:
    term = tuple((q, p) for q, p in enumerate(string) if p != "I")
    return QubitOperator(term, 1.0)


def qubit_operator_term_to_string(term, num_qubits: int) -> str:
    letters = ["I"] * num_qubits
    for q, p in term:
        letters[q] = p
    return "".join(letters)


# ----------------------------------------------------------------------
# Tapering description (round-trips through JSON so every downstream file
# applies the identical transformation)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TaperData:
    """Everything needed to reproduce one tapering transformation."""

    n_qubits_full: int
    n_spatial: int
    n_electrons: int
    removed_qubits: list[int]          # e.g. [3, 7]
    tapering_values: list[int]         # sector eigenvalues (+-1), aligned to removed_qubits
    symmetry_generators: list[str]     # full-length Pauli strings, aligned to removed_qubits
    kept_qubits: list[int]             # e.g. [0, 1, 2, 4, 5, 6]
    hf_bitstring_full: str             # HF occupation on the full register
    hf_bitstring_tapered: str          # HF occupation on the kept qubits

    @property
    def n_qubits_tapered(self) -> int:
        return self.n_qubits_full - len(self.removed_qubits)

    @property
    def old_to_new(self) -> dict[int, int]:
        return {q: i for i, q in enumerate(self.kept_qubits)}

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TaperData":
        return cls(
            n_qubits_full=int(data["n_qubits_full"]),
            n_spatial=int(data["n_spatial"]),
            n_electrons=int(data["n_electrons"]),
            removed_qubits=[int(q) for q in data["removed_qubits"]],
            tapering_values=[int(v) for v in data["tapering_values"]],
            symmetry_generators=[str(s) for s in data["symmetry_generators"]],
            kept_qubits=[int(q) for q in data["kept_qubits"]],
            hf_bitstring_full=str(data["hf_bitstring_full"]),
            hf_bitstring_tapered=str(data["hf_bitstring_tapered"]),
        )

    def save_json(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path) -> "TaperData":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


# ----------------------------------------------------------------------
# Build the tapering description for a closed-shell active space
# ----------------------------------------------------------------------
def build_taper_data(n_spatial: int, n_electrons: int) -> TaperData:
    """Two spin-block parities; remove the last qubit of each block.

    The sector (``tapering_values``) is fixed by the closed-shell Hartree-Fock
    determinant: the lowest ``n_electrons // 2`` orbitals occupied in each spin
    block.  The removed qubits are the last orbital of each block, which are
    unoccupied for every active space used here.
    """
    if n_electrons % 2 != 0:
        raise ValueError("closed-shell tapering requires an even electron count")
    n_qubits = 2 * n_spatial
    alpha = list(range(n_spatial))
    beta = list(range(n_spatial, 2 * n_spatial))

    eta = n_electrons // 2
    occupied = set(range(eta)) | set(range(n_spatial, n_spatial + eta))

    blocks = [alpha, beta]
    removed_qubits = [alpha[-1], beta[-1]]
    tapering_values = [
        int((-1) ** sum(1 for q in block if q in occupied)) for block in blocks
    ]
    symmetry_generators = [
        qubit_operator_term_to_string(tuple((q, "Z") for q in block), n_qubits)
        for block in blocks
    ]
    kept_qubits = [q for q in range(n_qubits) if q not in removed_qubits]

    hf_bits_full = ["0"] * n_qubits
    for q in occupied:
        hf_bits_full[q] = "1"
    hf_bitstring_full = "".join(hf_bits_full)
    hf_bitstring_tapered = "".join(hf_bits_full[q] for q in kept_qubits)

    return TaperData(
        n_qubits_full=n_qubits,
        n_spatial=n_spatial,
        n_electrons=n_electrons,
        removed_qubits=removed_qubits,
        tapering_values=tapering_values,
        symmetry_generators=symmetry_generators,
        kept_qubits=kept_qubits,
        hf_bitstring_full=hf_bitstring_full,
        hf_bitstring_tapered=hf_bitstring_tapered,
    )


# ----------------------------------------------------------------------
# Exact Clifford conjugation and qubit removal
# ----------------------------------------------------------------------
def _conjugate_all(operator: QubitOperator, taper: TaperData) -> QubitOperator:
    """Apply ``U (.) U`` for every (generator, removed-qubit) involution."""
    result = operator
    for gen_string, q in zip(taper.symmetry_generators, taper.removed_qubits):
        g = pauli_string_to_qubit_operator(gen_string)
        s = QubitOperator(((q, "X"),), 1.0)
        result = (g * result * g + g * result * s + s * result * g + s * result * s) * 0.5
    return result


def _drop_removed_qubits(term, coeff, taper: TaperData):
    """Substitute X on removed qubits by the sector value, then relabel."""
    letters = dict(term)
    new_coeff = coeff
    for q, value in zip(taper.removed_qubits, taper.tapering_values):
        letter = letters.get(q, "I")
        if letter == "X":
            new_coeff *= value
        elif letter != "I":
            raise ValueError(
                f"removed qubit {q} carries non-I/X Pauli {letter!r}; the operator "
                "does not respect the tapering symmetries."
            )
        letters.pop(q, None)
    old_to_new = taper.old_to_new
    new_term = tuple(sorted((old_to_new[q], p) for q, p in letters.items()))
    return new_term, new_coeff


def taper_qubit_operator(operator: QubitOperator, taper: TaperData) -> QubitOperator:
    """Taper a full-register :class:`QubitOperator` to the kept qubits."""
    conjugated = _conjugate_all(operator, taper)
    tapered = QubitOperator()
    for term, coeff in conjugated.terms.items():
        new_term, new_coeff = _drop_removed_qubits(term, coeff, taper)
        tapered += QubitOperator(new_term, new_coeff)
    tapered.compress(1e-12)
    return tapered


def taper_pauli_string(string: str, taper: TaperData) -> tuple[str, int]:
    """Taper a single symmetry-respecting Pauli string.

    Returns ``(tapered_string, sign)`` where ``sign in {+1, -1}`` and
    ``tapered_string`` has length ``taper.n_qubits_tapered``.  Raises if the
    input does not commute with both symmetries (would not map to one Pauli).
    """
    conjugated = _conjugate_all(pauli_string_to_qubit_operator(string), taper)
    conjugated.compress(_SIGN_TOL)
    if len(conjugated.terms) != 1:
        raise ValueError(
            f"Pauli string {string!r} does not respect the tapering symmetries "
            f"(maps to {len(conjugated.terms)} terms)."
        )
    (term, coeff), = conjugated.terms.items()
    new_term, new_coeff = _drop_removed_qubits(term, coeff, taper)
    if abs(new_coeff.imag) > _SIGN_TOL or abs(abs(new_coeff.real) - 1.0) > _SIGN_TOL:
        raise ValueError(f"tapered coefficient {new_coeff} is not a real sign (+-1).")
    sign = 1 if new_coeff.real > 0 else -1
    tapered_string = qubit_operator_term_to_string(new_term, taper.n_qubits_tapered)
    return tapered_string, sign


# ----------------------------------------------------------------------
# Numbered-Pauli export helpers (same format as the repo Pauli_Ham files)
# ----------------------------------------------------------------------
def qubit_operator_to_numbered_rows(operator: QubitOperator, num_qubits: int) -> list[list[float]]:
    """Convert to ``[coefficient, pauli_0, ..., pauli_{n-1}]`` (I=0,X=1,Y=2,Z=3)."""
    rows: list[list[float]] = []
    for term, coeff in sorted(operator.terms.items(), key=lambda item: item[0]):
        coeff = complex(coeff)
        if abs(coeff.imag) > 1e-10:
            raise ValueError(f"complex coefficient with nonzero imaginary part: {coeff}")
        codes = [0] * num_qubits
        for qubit_index, pauli_char in term:
            codes[qubit_index] = PAULI_TO_CODE[pauli_char]
        rows.append([float(coeff.real), *codes])
    return rows


def write_numbered_hamiltonian(operator: QubitOperator, num_qubits: int, path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = qubit_operator_to_numbered_rows(operator, num_qubits)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(" ".join(map(str, row)) + "\n")
    return path
