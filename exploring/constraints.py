"""Global compilation rules for flexible α–β UCCSD circuits.

Rules (Cl2 and general):
1. Every qubit index must appear in at least one gate.
2. Parameterized RZX pairs are vertex-disjoint; any α ↔ any β is allowed for
   RZX only (no hub pile-up on one index).
3. Within a spin sector, CZ is nearest-neighbour only, plus the chord
   (0,3) on alpha and the mirrored (half, half+3) on beta when that index
   exists.  Cross-spin CZ is forbidden — α–β coupling is RZX-only.
"""
from __future__ import annotations


def spin_sectors(n_qubits: int) -> tuple[list[int], list[int]]:
    half = n_qubits // 2
    return list(range(half)), list(range(half, n_qubits))


def allowed_cz_edges(n_qubits: int) -> set[frozenset[int]]:
    """Within-spin CZ graph: NN path + chord (0,3) / (half, half+3)."""
    half = n_qubits // 2
    edges: set[frozenset[int]] = set()
    # alpha NN
    for i in range(half - 1):
        edges.add(frozenset((i, i + 1)))
    # beta NN
    for i in range(half, n_qubits - 1):
        edges.add(frozenset((i, i + 1)))
    # special chords
    if half >= 4:
        edges.add(frozenset((0, 3)))
        edges.add(frozenset((half, half + 3)))
    return edges


def is_alpha_beta_pair(a: int, b: int, n_qubits: int) -> bool:
    half = n_qubits // 2
    return (a < half) != (b < half)


def qubits_used(gates) -> set[int]:
    used: set[int] = set()
    for g in gates:
        used.update(g["qubits"])
    return used


def all_qubits_used(gates, n_qubits: int) -> bool:
    return qubits_used(gates) == set(range(n_qubits))


def rzx_pairs(gates) -> list[tuple[int, int]]:
    pairs = []
    for g in gates:
        if g["op"].lower() == "rzx" and len(g["qubits"]) == 2:
            a, b = g["qubits"]
            pairs.append(tuple(sorted((a, b))))
    return pairs


def cz_pairs(gates) -> list[tuple[int, int]]:
    pairs = []
    for g in gates:
        if g["op"].lower() == "cz" and len(g["qubits"]) == 2:
            a, b = g["qubits"]
            pairs.append(tuple(sorted((a, b))))
    return pairs


def rzx_pairs_disjoint(gates) -> bool:
    seen: set[int] = set()
    for a, b in rzx_pairs(gates):
        if a in seen or b in seen:
            return False
        seen.add(a)
        seen.add(b)
    return True


def bridges_disjoint(schedule) -> bool:
    seen: set[int] = set()
    for pair in schedule:
        if pair is None or pair[0] is None:
            continue
        a, b = pair
        if a in seen or b in seen:
            return False
        seen.add(a)
        seen.add(b)
    return True


def cz_edges_legal(gates, n_qubits: int) -> tuple[bool, str]:
    allowed = allowed_cz_edges(n_qubits)
    for a, b in cz_pairs(gates):
        if is_alpha_beta_pair(a, b, n_qubits):
            return False, f"cross_spin_cz={(a, b)}"
        if frozenset((a, b)) not in allowed:
            return False, f"long_range_cz={(a, b)}"
    return True, "ok"


def rzx_edges_are_alpha_beta(gates, n_qubits: int) -> tuple[bool, str]:
    for a, b in rzx_pairs(gates):
        if not is_alpha_beta_pair(a, b, n_qubits):
            return False, f"rzx_same_spin={(a, b)}"
    return True, "ok"


def satisfies_rules(gates, n_qubits: int) -> tuple[bool, str]:
    if not rzx_pairs_disjoint(gates):
        return False, "rzx_overlap"
    ok, why = rzx_edges_are_alpha_beta(gates, n_qubits)
    if not ok:
        return False, why
    ok, why = cz_edges_legal(gates, n_qubits)
    if not ok:
        return False, why
    if not all_qubits_used(gates, n_qubits):
        missing = sorted(set(range(n_qubits)) - qubits_used(gates))
        return False, f"unused_qubits={missing}"
    return True, "ok"
