"""Mapping-dependent Z₂ tapering for COSMA-mapped UCCSD generators."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_ST = _ROOT / "state_transfer"
for p in (_ROOT, _ST):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from taper_lib import (  # type: ignore
    TaperData,
    build_taper_data,
    pauli_string_to_qubit_operator,
    qubit_operator_term_to_string,
    taper_pauli_string,
    taper_qubit_operator,
)

from .mapping import MappingBasis, multiply_pauli, pauli_to_string
from .workload import N_ELECTRONS, N_SPATIAL, tencirchem_to_export_wire


def mapped_spin_parity(basis: MappingBasis, spin: str) -> dict[str, complex]:
    """Map fermion parity ∏_{p in block} (-i γ_{2p} γ_{2p+1})."""
    n_spatial = basis.num_modes // 2
    if spin == "beta":
        modes = list(range(n_spatial))
    elif spin == "alpha":
        modes = list(range(n_spatial, 2 * n_spatial))
    else:
        raise ValueError(spin)

    x = np.zeros(basis.num_modes, dtype=np.uint8)
    z = np.zeros(basis.num_modes, dtype=np.uint8)
    phase = 1.0 + 0.0j
    for m in modes:
        phase *= -1j
        for maj in (2 * m, 2 * m + 1):
            mx, mz, mc = basis.majorana_pauli[maj]
            x, z, phase = multiply_pauli(x, z, phase, mx, mz, mc)
    return {pauli_to_string(x, z): phase}


def _normalize_parity_generator(op: dict[str, complex]) -> str:
    if len(op) != 1:
        raise ValueError(f"parity map is multi-term ({len(op)})")
    (s, c), = op.items()
    if abs(c) < 1e-12:
        raise ValueError("zero parity")
    # Global phase does not change the stabilizer group element up to i-phase;
    # require P^2 = I which single Paulis satisfy.
    return s


def build_mapping_taper_data(basis: MappingBasis) -> TaperData | None:
    """Build TaperData from mapped α/β parity generators when they are single Paulis."""
    try:
        gen_a = _normalize_parity_generator(mapped_spin_parity(basis, "alpha"))
        gen_b = _normalize_parity_generator(mapped_spin_parity(basis, "beta"))
    except ValueError:
        return None

    def remove_qubit(gen: str) -> int:
        zs = [i for i, c in enumerate(gen) if c == "Z"]
        if zs:
            return zs[-1]
        xs = [i for i, c in enumerate(gen) if c != "I"]
        if not xs:
            raise ValueError("empty generator")
        return xs[-1]

    try:
        removed = [remove_qubit(gen_a), remove_qubit(gen_b)]
    except ValueError:
        return None
    if len(set(removed)) != 2:
        return None

    kept = [q for q in range(basis.num_modes) if q not in removed]
    eta = N_ELECTRONS // 2
    occupied = set(range(eta)) | set(range(N_SPATIAL, N_SPATIAL + eta))
    blocks = [
        list(range(N_SPATIAL, 2 * N_SPATIAL)),  # alpha (TenCirChem)
        list(range(N_SPATIAL)),  # beta
    ]
    taper_vals = [int((-1) ** sum(1 for q in block if q in occupied)) for block in blocks]
    hf_full = ["0"] * basis.num_modes
    for q in occupied:
        hf_full[q] = "1"
    return TaperData(
        n_qubits_full=basis.num_modes,
        n_spatial=N_SPATIAL,
        n_electrons=N_ELECTRONS,
        removed_qubits=removed,
        tapering_values=taper_vals,
        symmetry_generators=[gen_a, gen_b],
        kept_qubits=kept,
        hf_bitstring_full="".join(hf_full),
        hf_bitstring_tapered="".join(hf_full[q] for q in kept),
    )


def taper_pauli_operator(op: dict[str, complex], taper: TaperData) -> dict[str, complex]:
    """Taper a general multi-Pauli operator."""
    qop = None
    for p, c in op.items():
        term = complex(c) * pauli_string_to_qubit_operator(p)
        qop = term if qop is None else qop + term
    if qop is None:
        return {}
    tapered = taper_qubit_operator(qop, taper)
    out: dict[str, complex] = {}
    for term, coeff in tapered.terms.items():
        s = qubit_operator_term_to_string(term, taper.n_qubits_tapered)
        out[s] = out.get(s, 0.0) + complex(coeff)
    return {k: v for k, v in out.items() if abs(v) > 1e-12}


def jw_export_taper() -> TaperData:
    return build_taper_data(N_SPATIAL, N_ELECTRONS)


def relabel_op_tencirchem_to_export(op: dict[str, complex]) -> dict[str, complex]:
    out: dict[str, complex] = {}
    for p, c in op.items():
        letters = ["I"] * len(p)
        for q, ch in enumerate(p):
            if ch != "I":
                letters[tencirchem_to_export_wire(q)] = ch
        s = "".join(letters)
        out[s] = out.get(s, 0.0) + c
    return out


def taper_string_export(pauli_tc: str, taper: TaperData) -> tuple[str, int]:
    """Taper a TenCirChem-layout Pauli via export relabel + standard taper."""
    letters = ["I"] * len(pauli_tc)
    for q, ch in enumerate(pauli_tc):
        if ch != "I":
            letters[tencirchem_to_export_wire(q)] = ch
    return taper_pauli_string("".join(letters), taper)
