"""Error-budget proxy matching error_reduction_methods.md §0.

fidelity ≈ ∏(1-p) with p = 0.0005 (1q) / 0.01 (on-chip 2q) / 0.1 (cross-chip 2q).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

P_1Q = 5e-4
P_ONCHIP = 1e-2
P_CROSS = 1e-1


@dataclass(frozen=True)
class Budget:
    n1: int
    n_onchip: int
    n_cross: int
    depth: int | None = None
    pairs: tuple[tuple[tuple[int, int], int], ...] = ()

    @property
    def fidelity(self) -> float:
        return (
            (1.0 - P_1Q) ** self.n1
            * (1.0 - P_ONCHIP) ** self.n_onchip
            * (1.0 - P_CROSS) ** self.n_cross
        )

    @property
    def error(self) -> float:
        return 1.0 - self.fidelity

    def as_dict(self) -> dict:
        return {
            "n1": self.n1,
            "n_onchip": self.n_onchip,
            "n_cross": self.n_cross,
            "depth": self.depth,
            "fidelity": self.fidelity,
            "error": self.error,
            "pairs": {f"{a}-{b}": c for (a, b), c in self.pairs},
        }


def score_gates(
    gates: Iterable[dict],
    cross_pairs: set[frozenset[int]],
    depth: int | None = None,
) -> Budget:
    n1 = n_on = n_x = 0
    pairs: Counter = Counter()
    for g in gates:
        qs = list(g["qubits"])
        if len(qs) == 1:
            n1 += 1
        elif len(qs) == 2:
            fr = frozenset(qs)
            pairs[tuple(sorted(qs))] += 1
            if fr in cross_pairs:
                n_x += 1
            else:
                n_on += 1
    return Budget(
        n1=n1,
        n_onchip=n_on,
        n_cross=n_x,
        depth=depth,
        pairs=tuple(sorted(pairs.items())),
    )


def spin_split_cross_pairs(n_qubits: int) -> set[frozenset[int]]:
    """Every alpha–beta pair is treated as a cross-chip link."""
    half = n_qubits // 2
    out: set[frozenset[int]] = set()
    for a in range(half):
        for b in range(half, n_qubits):
            out.add(frozenset((a, b)))
    return out


def cross_from_list(pairs: Iterable[tuple[int, int]]) -> set[frozenset[int]]:
    return {frozenset(p) for p in pairs}
