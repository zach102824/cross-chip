"""2q-scaffold then 1q-adjust ansatz for approximating UCCSD doubles.

Layout
------
HF 6q (tapered spin-block):
  α: 0 1 2
  β: 3 4 5
  vertical: (0,3),(1,4),(2,5)

Cl2 10q:
  α: 0 1 2 3 4
  β: 5 6 7 8 9
  vertical: (0,5),(1,6),(2,7),(3,8),(4,9)
  within-spin NN + chords (0,3)/(5,8)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter


# ----------------------------------------------------------------------
# Pair libraries
# ----------------------------------------------------------------------
PAIRS_HF = {
    "vert": [(0, 3), (1, 4), (2, 5)],
    "nn": [(0, 1), (1, 2), (3, 4), (4, 5)],
    "nn+vert": [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)],
}

PAIRS_CL2 = {
    "vert": [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)],
    "vert3": [(0, 5), (2, 7), (3, 8)],  # winner-like disjoint
    "nn": [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8), (8, 9),
    ],
    "nn+chord": [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 3),
        (5, 6), (6, 7), (7, 8), (8, 9), (5, 8),
    ],
    "nn+vert3": [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (5, 6), (6, 7), (7, 8), (8, 9),
        (0, 5), (2, 7), (3, 8),
    ],
    "nn+chord+vert3": [
        (0, 1), (1, 2), (2, 3), (3, 4), (0, 3),
        (5, 6), (6, 7), (7, 8), (8, 9), (5, 8),
        (0, 5), (2, 7), (3, 8),
    ],
}


@dataclass
class AnsatzSpec:
    name: str
    n_qubits: int
    pairs_2q: list[tuple[int, int]]
    twoq_kind: str = "rzx"  # "rzx" | "cz" | "rzx_on_vert_else_cz"
    prepend_1q: bool = True
    adjust_reps: int = 1  # layers of U3 after 2q
    occupied: Sequence[int] | None = None  # HF X prep


def _u3_layer(qc: QuantumCircuit, prefix: str, n: int) -> list[Parameter]:
    params = []
    for q in range(n):
        a = Parameter(f"{prefix}_a{q}")
        b = Parameter(f"{prefix}_b{q}")
        c = Parameter(f"{prefix}_c{q}")
        qc.rz(a, q)
        qc.ry(b, q)
        qc.rz(c, q)
        params.extend([a, b, c])
    return params


def _is_vertical(a: int, b: int, n: int) -> bool:
    half = n // 2
    lo, hi = (a, b) if a < b else (b, a)
    return hi == lo + half


def build_circuit(spec: AnsatzSpec):
    """Return (qc, all_params, params_2q, params_1q).

    Structure: HF-prep → [U3] → 2q scaffold → [U3]×reps
    """
    n = spec.n_qubits
    qc = QuantumCircuit(n)
    if spec.occupied:
        for q in spec.occupied:
            qc.x(q)

    p1: list[Parameter] = []
    p2: list[Parameter] = []

    if spec.prepend_1q:
        p1.extend(_u3_layer(qc, "pre", n))

    for i, (a, b) in enumerate(spec.pairs_2q):
        kind = spec.twoq_kind
        if kind == "rzx_on_vert_else_cz":
            kind = "rzx" if _is_vertical(a, b, n) else "cz"
        if kind == "rzx":
            th = Parameter(f"w{i}")
            qc.rzx(th, a, b)
            p2.append(th)
        elif kind == "cz":
            qc.cz(a, b)
        else:
            raise ValueError(kind)

    for r in range(spec.adjust_reps):
        p1.extend(_u3_layer(qc, f"adj{r}", n))

    all_p = list(qc.parameters)
    return qc, all_p, p2, p1


def default_specs(case: str, occupied: Sequence[int]) -> list[AnsatzSpec]:
    """Sparse → denser scaffolds to probe how much 2q is needed."""
    if case == "HF_6q":
        n = 6
        lib = PAIRS_HF
        specs = [
            AnsatzSpec("vert|rzx|pre+u3", n, lib["vert"], "rzx", True, 1, occupied),
            AnsatzSpec("vert|rzx|u3", n, lib["vert"], "rzx", False, 1, occupied),
            AnsatzSpec("vert|cz|pre+u3", n, lib["vert"], "cz", True, 1, occupied),
            AnsatzSpec("nn+vert|rzx|pre+u3", n, lib["nn+vert"], "rzx", True, 1, occupied),
            AnsatzSpec("nn+vert|mix|pre+u3", n, lib["nn+vert"], "rzx_on_vert_else_cz", True, 1, occupied),
        ]
    elif case == "Cl2_10q":
        n = 10
        lib = PAIRS_CL2
        specs = [
            AnsatzSpec("vert3|rzx|pre+u3", n, lib["vert3"], "rzx", True, 1, occupied),
            AnsatzSpec("vert|rzx|pre+u3", n, lib["vert"], "rzx", True, 1, occupied),
            AnsatzSpec("vert3|rzx|pre+u3x2", n, lib["vert3"], "rzx", True, 2, occupied),
            AnsatzSpec("nn+vert3|mix|pre+u3", n, lib["nn+vert3"], "rzx_on_vert_else_cz", True, 1, occupied),
            AnsatzSpec("nn+chord+vert3|mix|pre+u3", n, lib["nn+chord+vert3"], "rzx_on_vert_else_cz", True, 1, occupied),
            AnsatzSpec("nn+chord+vert3|rzx|pre+u3", n, lib["nn+chord+vert3"], "rzx", True, 1, occupied),
            # 2q only first, no prepend — tests pure "then 1q"
            AnsatzSpec("nn+chord+vert3|mix|u3", n, lib["nn+chord+vert3"], "rzx_on_vert_else_cz", False, 1, occupied),
            AnsatzSpec("nn+chord+vert3|mix|u3x2", n, lib["nn+chord+vert3"], "rzx_on_vert_else_cz", False, 2, occupied),
        ]
    else:
        raise ValueError(case)
    return specs
