"""Parity-tree Pauli-gadget emission + A–B machine scoring."""

from __future__ import annotations

from dataclasses import dataclass

from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.transpiler import CouplingMap

from .schedule import PauliFactor, support_mask


# Physical embeddings from qubit_connectivity / H4_circuits README
def ladder_edges_6q() -> list[tuple[int, int]]:
    """Logical 6q A–B ladder: A={0,1,2}, B={3,4,5}."""
    return [(0, 1), (1, 2), (3, 4), (4, 5), (0, 3), (1, 4), (2, 5)]


def cube_edges_8q() -> list[tuple[int, int]]:
    """Logical 8q A–B cube (Case-0 rungs).

    chip A: 0-1-2-3 cycle; chip B: 4-5-6-7 cycle; rungs 0-4,1-5,2-6,3-7.
    """
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    return edges


def cross_rungs(n_qubits: int) -> set[frozenset[int]]:
    if n_qubits == 6:
        return {frozenset(e) for e in [(0, 3), (1, 4), (2, 5)]}
    if n_qubits == 8:
        return {frozenset(e) for e in [(0, 4), (1, 5), (2, 6), (3, 7)]}
    raise ValueError(n_qubits)


def coupling_map(n_qubits: int) -> CouplingMap:
    edges = ladder_edges_6q() if n_qubits == 6 else cube_edges_8q()
    undirected = edges + [(b, a) for a, b in edges]
    return CouplingMap(couplinglist=undirected)


def _support(pauli: str) -> list[int]:
    return [i for i, c in enumerate(pauli) if c != "I"]


def add_pauli_gadget(qc: QuantumCircuit, pauli: str, angle_param, angle_sign: float) -> None:
    """Append exp(-i * angle_sign * angle_param / 2 * P)."""
    sup = _support(pauli)
    if not sup:
        return
    # Basis change
    for q in sup:
        if pauli[q] == "X":
            qc.h(q)
        elif pauli[q] == "Y":
            qc.sdg(q)
            qc.h(q)
    # Parity tree: chain CX toward last support qubit as root (local-first: sort)
    # Prefer roots that reduce cross-chip: put same-chip qubits together.
    ordered = _parity_order(sup, qc.num_qubits)
    for a, b in zip(ordered[:-1], ordered[1:]):
        qc.cx(a, b)
    qc.rz(angle_sign * angle_param, ordered[-1])
    for a, b in reversed(list(zip(ordered[:-1], ordered[1:]))):
        qc.cx(a, b)
    for q in sup:
        if pauli[q] == "X":
            qc.h(q)
        elif pauli[q] == "Y":
            qc.h(q)
            qc.s(q)


def _parity_order(support: list[int], n_qubits: int) -> list[int]:
    """Order support so same-core qubits are contiguous, then join cores."""
    if n_qubits == 6:
        cores = [{0, 1, 2}, {3, 4, 5}]
    else:
        cores = [{0, 1, 2, 3}, {4, 5, 6, 7}]
    groups = []
    for core in cores:
        g = sorted(q for q in support if q in core)
        if g:
            groups.append(g)
    # concatenate groups; chain will create one cross edge between group roots
    out: list[int] = []
    for g in groups:
        out.extend(g)
    return out if out else list(support)


@dataclass
class CircuitScore:
    cross_chip_2q: int
    total_2q: int
    depth: int
    n_qubits: int
    n_factors: int
    support_delta: int

    def key(self) -> tuple[int, int, int]:
        return (self.cross_chip_2q, self.total_2q, self.depth)

    def to_dict(self) -> dict:
        return {
            "cross_chip_2q": self.cross_chip_2q,
            "total_2q": self.total_2q,
            "depth": self.depth,
            "n_qubits": self.n_qubits,
            "n_factors": self.n_factors,
            "support_delta": self.support_delta,
        }


def build_logical_circuit(factors: list[PauliFactor], n_qubits: int) -> tuple[QuantumCircuit, dict[str, Parameter]]:
    params: dict[str, Parameter] = {}
    qc = QuantumCircuit(n_qubits)
    for fac in factors:
        if fac.theta_name not in params:
            params[fac.theta_name] = Parameter(fac.theta_name)
        add_pauli_gadget(qc, fac.pauli, params[fac.theta_name], fac.angle_sign)
    return qc, params


def score_circuit(
    factors: list[PauliFactor],
    n_qubits: int,
    seed: int = 0,
    optimization_level: int = 3,
) -> tuple[CircuitScore, QuantumCircuit, QuantumCircuit]:
    from .schedule import support_delta

    logical, _ = build_logical_circuit(factors, n_qubits)
    cmap = coupling_map(n_qubits)
    rungs = cross_rungs(n_qubits)
    routed = transpile(
        logical,
        coupling_map=cmap,
        basis_gates=["rz", "sx", "x", "cz", "cx", "h", "s", "sdg"],
        optimization_level=optimization_level,
        seed_transpiler=seed,
        initial_layout=list(range(n_qubits)),
    )
    n2q = 0
    ncross = 0
    for inst in routed.data:
        if inst.operation.num_qubits != 2:
            continue
        n2q += 1
        qs = frozenset(routed.find_bit(q).index for q in inst.qubits)
        if qs in rungs:
            ncross += 1
    score = CircuitScore(
        cross_chip_2q=ncross,
        total_2q=n2q,
        depth=routed.depth(),
        n_qubits=n_qubits,
        n_factors=len(factors),
        support_delta=support_delta(factors),
    )
    return score, logical, routed


def best_score_over_seeds(
    factors: list[PauliFactor],
    n_qubits: int,
    seeds: range | list[int] | None = None,
) -> tuple[CircuitScore, QuantumCircuit, QuantumCircuit, int]:
    if seeds is None:
        seeds = range(8)
    best = None
    for seed in seeds:
        score, logical, routed = score_circuit(factors, n_qubits, seed=int(seed))
        if best is None or score.key() < best[0].key():
            best = (score, logical, routed, int(seed))
    assert best is not None
    return best
