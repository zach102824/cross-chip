"""Fermion excitation → sparse Majorana → Pauli (COSMA-style).

Majorana convention (matches COSMA / common chemistry codes):
  a_p     = (γ_{2p} + i γ_{2p+1}) / 2
  a_p^dag = (γ_{2p} - i γ_{2p+1}) / 2
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .mapping import MappingBasis, multiply_pauli, pauli_to_string


@dataclass(frozen=True)
class MajoranaTerm:
    modes: tuple[int, ...]  # sorted ascending; empty = identity
    coeff: complex


def _sort_and_parity(modes: list[int]) -> tuple[tuple[int, ...], int]:
    """Bubble-sort modes; return (sorted_tuple, parity) with parity = (-1)^swaps."""
    arr = list(modes)
    swaps = 0
    n = len(arr)
    for i in range(n):
        for j in range(n - 1, i, -1):
            if arr[j] < arr[j - 1]:
                arr[j], arr[j - 1] = arr[j - 1], arr[j]
                swaps += 1
    # Cancel identical adjacent modes: γ_k^2 = 1
    stack: list[int] = []
    for m in arr:
        if stack and stack[-1] == m:
            stack.pop()
        else:
            stack.append(m)
    return tuple(stack), (1 if swaps % 2 == 0 else -1)


def _accumulate(acc: dict[tuple[int, ...], complex], modes: list[int], coeff: complex) -> None:
    key, parity = _sort_and_parity(modes)
    acc[key] = acc.get(key, 0.0) + parity * coeff


def fermion_op_to_majorana(creations: list[int], annihilations: list[int]) -> dict[tuple[int, ...], complex]:
    """Map ∏ a†_c ∏ a_a to Majorana monomials (unordered accumulator)."""
    # Each a†_p contributes (γ_{2p} - i γ_{2p+1})/2
    # Each a_p  contributes (γ_{2p} + i γ_{2p+1})/2
    factors: list[list[tuple[int, complex]]] = []
    for p in creations:
        factors.append([(2 * p, 0.5), (2 * p + 1, -0.5j)])
    for p in annihilations:
        factors.append([(2 * p, 0.5), (2 * p + 1, 0.5j)])

    acc: dict[tuple[int, ...], complex] = {(): 1.0 + 0.0j}
    for choices in factors:
        nxt: dict[tuple[int, ...], complex] = {}
        for modes, c0 in acc.items():
            for m, cm in choices:
                _accumulate(nxt, list(modes) + [m], c0 * cm)
        acc = nxt
    return {k: v for k, v in acc.items() if abs(v) > 1e-15}


def excitation_generator_majorana(ex_op: tuple[int, ...]) -> dict[tuple[int, ...], complex]:
    """Majorana expansion of G = T - T† for T = a_k† a_l† a_i a_j (len 4)."""
    if len(ex_op) != 4:
        raise ValueError(f"expected 4-tuple double excitation, got {ex_op}")
    k, l, i, j = ex_op
    t = fermion_op_to_majorana([k, l], [i, j])
    td = fermion_op_to_majorana([i, j], [k, l])  # T† = a_i† a_j† a_k a_l for distinct indices
    # Actually T† = (a_k† a_l† a_i a_j)† = a_j† a_i† a_l a_k
    td = fermion_op_to_majorana([j, i], [l, k])
    acc: dict[tuple[int, ...], complex] = {}
    for key, val in t.items():
        acc[key] = acc.get(key, 0.0) + val
    for key, val in td.items():
        acc[key] = acc.get(key, 0.0) - val
    return {k: v for k, v in acc.items() if abs(v) > 1e-14}


def majorana_to_pauli(
    maj: dict[tuple[int, ...], complex],
    basis: MappingBasis,
    cutoff: float = 1e-12,
) -> dict[str, complex]:
    """Map sparse Majorana operator through a COSMA PPTT basis."""
    n = basis.num_modes
    out: dict[str, complex] = {}
    for modes, coeff in maj.items():
        # Start from identity
        x = np.zeros(n, dtype=np.uint8)
        z = np.zeros(n, dtype=np.uint8)
        phase = complex(coeff)
        for m in modes:
            mx, mz, mc = basis.majorana_pauli[m]
            x, z, phase = multiply_pauli(x, z, phase, mx, mz, mc)
        if abs(phase) <= cutoff:
            continue
        s = pauli_to_string(x, z)
        out[s] = out.get(s, 0.0) + phase
    return {k: v for k, v in out.items() if abs(v) > cutoff}


def jw_odd_y_string(ex_op: tuple[int, int, int, int], n_qubits: int) -> str:
    """Reference odd-Y JW representative used by the existing H4 pipeline."""
    k, l, i, j = ex_op
    sup = sorted({k, l, i, j})
    if len(sup) != 4:
        raise ValueError("double must have four distinct indices")
    s = ["I"] * n_qubits
    for m, q in enumerate(sup):
        s[q] = "Y" if m == 0 else "X"
    for q in range(sup[0] + 1, sup[1]):
        s[q] = "Z"
    for q in range(sup[2] + 1, sup[3]):
        s[q] = "Z"
    return "".join(s)
