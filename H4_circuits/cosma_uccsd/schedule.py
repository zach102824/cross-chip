"""Commutation-legal Pauli scheduling (COSMA Gray / Hamming within legal moves)."""

from __future__ import annotations

from dataclasses import dataclass

from .mapping import paulis_commute


@dataclass
class PauliFactor:
    """One rotation factor exp(-i * angle_sign * θ/2 * P) in the UCC product."""

    label: str
    pid: int
    ex_op_index: int
    pauli: str
    angle_sign: float  # multiplies θ in exp(-i * angle_sign * θ / 2 * P)
    theta_name: str
    coeff: complex  # original mapped coefficient (for bookkeeping)


def support_mask(pauli: str) -> int:
    m = 0
    for i, c in enumerate(pauli):
        if c != "I":
            m |= 1 << i
    return m


def support_delta(order: list[PauliFactor]) -> int:
    if len(order) < 2:
        return 0
    cost = 0
    for a, b in zip(order, order[1:]):
        cost += bin(support_mask(a.pauli) ^ support_mask(b.pauli)).count("1")
    return cost


def gray_key(pauli: str) -> int:
    m = support_mask(pauli)
    return m ^ (m >> 1)


def build_precedence(factors: list[PauliFactor]) -> list[set[int]]:
    """pred[j] = indices that must precede j (noncommuting earlier factors)."""
    n = len(factors)
    pred = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if not paulis_commute(factors[i].pauli, factors[j].pauli):
                pred[j].add(i)
    return pred


def topological_gray_schedule(factors: list[PauliFactor]) -> list[PauliFactor]:
    """Reorder factors with COSMA Gray preference subject to precedence DAG."""
    n = len(factors)
    pred = build_precedence(factors)
    remaining = set(range(n))
    indeg = {i: len(pred[i]) for i in range(n)}
    # also track successors for indeg updates
    succ = [set() for _ in range(n)]
    for j in range(n):
        for i in pred[j]:
            succ[i].add(j)

    order_idx: list[int] = []
    prev_mask = 0
    while remaining:
        ready = [i for i in remaining if indeg[i] == 0]
        if not ready:
            raise RuntimeError("cycle in commutation precedence — impossible for a total order")
        # Gray / Hamming: minimize Hamming distance of support to previous
        def key(i: int):
            m = support_mask(factors[i].pauli)
            ham = bin(m ^ prev_mask).count("1") if order_idx else 0
            return (ham, gray_key(factors[i].pauli), factors[i].label)

        pick = min(ready, key=key)
        order_idx.append(pick)
        remaining.remove(pick)
        prev_mask = support_mask(factors[pick].pauli)
        for j in succ[pick]:
            indeg[j] -= 1

    return [factors[i] for i in order_idx]


def expand_operator_to_factors(
    label: str,
    pid: int,
    ex_op_index: int,
    theta_name: str,
    pauli_op: dict[str, complex],
) -> list[PauliFactor]:
    """Turn mapped G = Σ c_P P into rotation factors.

    For anti-Hermitian G, coefficients are pure imaginary on Hermitian Paulis.
    We emit exp(θ G) ≈ ∏ exp(θ c_P P). When all Paulis in the sum commute this
    is exact; otherwise this is a first-order Trotter step (flagged by caller).
    Convention: exp(θ * c * P) = exp(-i * angle_sign * θ/2 * P) with
    angle_sign = -2 * Im(c) when c = i * Im(c) and P Hermitian... 

    If c = α + iβ and P²=I, exp(θ c P) = cos(|z|θ) I + (c/|z|)sin(|z|θ) P ...
    For UCC, G†=-G so eigenvalues are imaginary; c should be pure imaginary for
    each Hermitian Pauli term. Then exp(θ * (iβ) P) = exp(i β θ P) =
    exp(-i * (-2β) * θ/2 * P), so angle_sign = -2*β = -2*Im(c).
    """
    factors: list[PauliFactor] = []
    # Drop identity
    terms = [(p, c) for p, c in pauli_op.items() if p != "I" * len(p) and abs(c) > 1e-12]
    # Sort for determinism
    terms.sort(key=lambda pc: pc[0])
    for k, (p, c) in enumerate(terms):
        # Prefer pure-imaginary anti-Hermitian generators
        beta = float(c.imag)
        # Also allow real skew pieces that appear from phase conventions
        if abs(c.real) > 1e-8 and abs(c.imag) < 1e-8:
            # c real → exp(θ c P) is not unitary unless we interpret as i*( -i c)
            # treat as angle from real via c = -i * (i c)
            angle_sign = float(-2.0 * (-c.real))  # if c was stored as real standing for i*(-c)?
            # Better: rewrite c_eff = -1j * c so Im(c_eff)= -Re(c); angle_sign=-2*Im=2*Re(c)
            angle_sign = float(2.0 * c.real)
        else:
            angle_sign = float(-2.0 * beta)
        if abs(angle_sign) < 1e-14:
            continue
        factors.append(
            PauliFactor(
                label=f"{label}#{k}",
                pid=pid,
                ex_op_index=ex_op_index,
                pauli=p,
                angle_sign=angle_sign,
                theta_name=theta_name,
                coeff=c,
            )
        )
    return factors


def factors_all_commute(factors: list[PauliFactor]) -> bool:
    for i in range(len(factors)):
        for j in range(i + 1, len(factors)):
            if not paulis_commute(factors[i].pauli, factors[j].pauli):
                return False
    return True
