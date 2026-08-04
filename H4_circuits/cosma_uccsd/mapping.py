"""Product-preserving ternary-tree (PPTT) fermion-to-qubit mappings (COSMA port)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

import numpy as np


class Edge(IntEnum):
    X = 0
    Y = 1
    Z = 2


_EDGE_CHAR = {Edge.X: "X", Edge.Y: "Y", Edge.Z: "Z"}


@dataclass
class MappingTree:
    n: int
    parent: list[int]
    incoming: list[int]  # Edge or -1
    children: list[list[int]]  # [X,Y,Z] child or -1

    @staticmethod
    def empty(n: int) -> "MappingTree":
        return MappingTree(
            n=n,
            parent=[-1] * n,
            incoming=[-1] * n,
            children=[[-1, -1, -1] for _ in range(n)],
        )

    def connect(self, parent: int, edge: Edge, child: int) -> None:
        if self.children[parent][int(edge)] != -1:
            raise ValueError("child slot occupied")
        if self.parent[child] != -1:
            raise ValueError("child already has parent")
        self.children[parent][int(edge)] = child
        self.parent[child] = parent
        self.incoming[child] = int(edge)

    def roots(self) -> list[int]:
        return [i for i, p in enumerate(self.parent) if p == -1]


def build_jw_tree(n: int) -> MappingTree:
    t = MappingTree.empty(n)
    for i in range(n - 1):
        t.connect(i, Edge.Z, i + 1)
    return t


def build_parity_tree(n: int) -> MappingTree:
    t = MappingTree.empty(n)
    for i in range(n - 1):
        t.connect(i, Edge.X, i + 1)
    return t


def build_jkmn_tree(n: int) -> MappingTree:
    t = MappingTree.empty(n)
    parent = 0
    for child in range(1, n):
        edge = Edge((child - 1) % 3)
        t.connect(parent, edge, child)
        if edge == Edge.Z:
            parent += 1
    return t


def _find_leg(tree: MappingTree, node: int, start: Edge) -> tuple[int, Edge]:
    current = node
    edge = start
    # Guard against malformed trees (e.g. self-loops from bad degree decode).
    for _ in range(tree.n + 1):
        child = tree.children[current][int(edge)]
        if child < 0:
            return current, edge
        current = child
        edge = Edge.Z
    raise ValueError("cycle detected in mapping tree")


def _pauli_from_leg(tree: MappingTree, node: int, edge: Edge) -> tuple[np.ndarray, np.ndarray, complex]:
    n = tree.n
    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)
    current = node
    current_edge = edge
    while current >= 0:
        ch = _EDGE_CHAR[Edge(current_edge) if not isinstance(current_edge, Edge) else current_edge]
        if isinstance(current_edge, int):
            ch = _EDGE_CHAR[Edge(current_edge)]
        if ch in ("X", "Y"):
            x[current] = 1
        if ch in ("Z", "Y"):
            z[current] = 1
        parent = tree.parent[current]
        current_edge = tree.incoming[current]
        current = parent
    return x, z, 1.0 + 0.0j


def tree_to_mapping(
    tree: MappingTree,
    mode_order: Optional[list[int]] = None,
) -> list[tuple[np.ndarray, np.ndarray, complex]]:
    """Return Majorana→Pauli basis: length 2n, entries (x_bits, z_bits, coeff)."""
    n = tree.n
    if mode_order is None:
        mode_order = list(range(n))
    if len(mode_order) != n:
        raise ValueError("mode_order size mismatch")
    mapping: list[tuple[np.ndarray, np.ndarray, complex]] = []
    for m in range(n):
        u = mode_order[m]
        lx, le = _find_leg(tree, u, Edge.X)
        ly, ley = _find_leg(tree, u, Edge.Y)
        mapping.append(_pauli_from_leg(tree, lx, le))
        mapping.append(_pauli_from_leg(tree, ly, ley))
    return mapping


def jordan_wigner_mapping(n: int) -> list[tuple[np.ndarray, np.ndarray, complex]]:
    """Direct JW: γ_{2q}→Z…ZX, γ_{2q+1}→Z…ZY (COSMA JordanWignerMapping)."""
    out = []
    for i in range(2 * n):
        q = i // 2
        x = np.zeros(n, dtype=np.uint8)
        z = np.zeros(n, dtype=np.uint8)
        for j in range(q):
            z[j] = 1
        if i % 2 == 0:
            x[q] = 1
        else:
            x[q] = 1
            z[q] = 1
        out.append((x, z, 1.0 + 0.0j))
    return out


@dataclass
class MappingBasis:
    name: str
    num_modes: int
    majorana_pauli: list[tuple[np.ndarray, np.ndarray, complex]]
    tree: Optional[MappingTree] = None
    mode_order: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "num_modes": self.num_modes,
            "mode_order": list(self.mode_order),
            "has_tree": self.tree is not None,
            "parent_of": None if self.tree is None else list(self.tree.parent),
            "child_of": None if self.tree is None else [list(c) for c in self.tree.children],
        }


def build_mapping(name: str, num_modes: int, mode_order: Optional[list[int]] = None) -> MappingBasis:
    name_u = name.strip().upper()
    if mode_order is None:
        mode_order = list(range(num_modes))

    if name_u == "JW":
        # Prefer explicit JW (matches COSMA JordanWignerMapping)
        return MappingBasis("JW", num_modes, jordan_wigner_mapping(num_modes), build_jw_tree(num_modes), mode_order)

    if name_u in ("PE", "PARITY"):
        tree = build_parity_tree(num_modes)
        return MappingBasis("PE", num_modes, tree_to_mapping(tree, mode_order), tree, mode_order)

    if name_u == "JKMN":
        tree = build_jkmn_tree(num_modes)
        return MappingBasis("JKMN", num_modes, tree_to_mapping(tree, mode_order), tree, mode_order)

    if name_u in ("PE_OPT", "JKMN_OPT"):
        base = "PE" if name_u.startswith("PE") else "JKMN"
        return build_mapping(base, num_modes, mode_order)

    raise ValueError(f"unsupported mapping {name}")


# ---------------------------------------------------------------------------
# Pauli algebra helpers
# ---------------------------------------------------------------------------

def multiply_pauli(
    x1: np.ndarray,
    z1: np.ndarray,
    c1: complex,
    x2: np.ndarray,
    z2: np.ndarray,
    c2: complex,
) -> tuple[np.ndarray, np.ndarray, complex]:
    """Return (x, z, coeff) for Pauli product ``P(x1,z1) * P(x2,z2)``.

    Bit encoding: X=(1,0), Z=(0,1), Y=(1,1).  The i-power uses the standard
    symplectic composition so that XY=+iZ, XZ=-iY, ZX=+iY, etc.
    """
    x = np.bitwise_xor(x1, x2)
    z = np.bitwise_xor(z1, z2)
    # Per qubit: i^(x1 z1 + x2 z2 - x z + 2 z1 x2)
    exp = (
        int(np.bitwise_and(x1, z1).sum())
        + int(np.bitwise_and(x2, z2).sum())
        - int(np.bitwise_and(x, z).sum())
        + 2 * int(np.bitwise_and(z1, x2).sum())
    ) & 3
    phase = (1 + 0j, 1j, -1 + 0j, -1j)[exp]
    return x, z, c1 * c2 * phase


def pauli_to_string(x: np.ndarray, z: np.ndarray) -> str:
    letters = []
    for xi, zi in zip(x, z):
        if xi and zi:
            letters.append("Y")
        elif xi:
            letters.append("X")
        elif zi:
            letters.append("Z")
        else:
            letters.append("I")
    return "".join(letters)


def string_to_xz(s: str) -> tuple[np.ndarray, np.ndarray]:
    n = len(s)
    x = np.zeros(n, dtype=np.uint8)
    z = np.zeros(n, dtype=np.uint8)
    for i, c in enumerate(s):
        if c == "X":
            x[i] = 1
        elif c == "Z":
            z[i] = 1
        elif c == "Y":
            x[i] = 1
            z[i] = 1
        elif c != "I":
            raise ValueError(c)
    return x, z


def paulis_commute(a: str, b: str) -> bool:
    xa, za = string_to_xz(a)
    xb, zb = string_to_xz(b)
    # symplectic product
    return int((np.bitwise_and(xa, zb).sum() + np.bitwise_and(za, xb).sum()) % 2) == 0


# ---------------------------------------------------------------------------
# Lightweight TREE_GA (shape + mode labels), COSMA-inspired
# ---------------------------------------------------------------------------

@dataclass
class TreeIndividual:
    # Prufer-like ternary: for nodes 1..n-1, parent and edge (encoded compactly)
    # degree sequence of preorder out-degrees (0..3) length n, sum = n-1
    degrees: list[int]
    labels: list[int]  # mode_order: labels[mode] = node

    def clone(self) -> "TreeIndividual":
        return TreeIndividual(list(self.degrees), list(self.labels))


def _degrees_to_tree(degrees: list[int]) -> MappingTree:
    """Decode out-degree sequence into a tree by left-to-right child filling."""
    n = len(degrees)
    if sum(degrees) != n - 1:
        raise ValueError("invalid degree sequence")
    t = MappingTree.empty(n)
    # nodes in order 0..n-1; assign next free child indices
    next_node = 1
    for parent, deg in enumerate(degrees):
        if deg > 3:
            raise ValueError("ternary degree > 3")
        if deg > 0 and next_node <= parent:
            raise ValueError("invalid preorder degree sequence")
        for e in range(deg):
            if next_node >= n:
                raise ValueError("degree decode overflow")
            if next_node <= parent:
                raise ValueError("child index must exceed parent")
            t.connect(parent, Edge(e), next_node)
            next_node += 1
    if next_node != n:
        raise ValueError("degree decode underfill")
    return t


def individual_to_basis(ind: TreeIndividual) -> MappingBasis:
    tree = _degrees_to_tree(ind.degrees)
    basis = tree_to_mapping(tree, ind.labels)
    return MappingBasis("TREE_GA", tree.n, basis, tree, list(ind.labels))


def _tree_to_degrees(tree: MappingTree) -> list[int]:
    return [sum(1 for c in tree.children[i] if c >= 0) for i in range(tree.n)]


def _valid_label_permutation(labels: list[int], n: int) -> list[int]:
    if len(labels) != n or len(set(labels)) != n or any(x < 0 or x >= n for x in labels):
        return list(range(n))
    return list(labels)


def _random_degrees(n: int, rng: np.random.Generator, max_attempts: int = 128) -> list[int]:
    # Sample ternary trees via random valid degree sequences
    for _ in range(max_attempts):
        deg = [0] * n
        remaining = n - 1
        for i in range(n - 1):
            max_d = min(3, remaining)
            d = int(rng.integers(0, max_d + 1))
            deg[i] = d
            remaining -= d
        deg[n - 1] = 0
        if remaining == 0 and sum(deg) == n - 1:
            try:
                _degrees_to_tree(deg)
                return deg
            except ValueError:
                continue
    # fallback: JW path tree (all Z-chain) as degrees [1,1,...,1,0]
    return [1] * (n - 1) + [0]


def _warm_start_population(n: int) -> list[TreeIndividual]:
    pops: list[TreeIndividual] = []
    fallback = [1] * (n - 1) + [0]
    for builder in (build_jw_tree, build_parity_tree, build_jkmn_tree):
        tree = builder(n)
        deg = _tree_to_degrees(tree)
        try:
            _degrees_to_tree(deg)
        except ValueError:
            deg = fallback
        pops.append(TreeIndividual(deg, list(range(n))))
        pops.append(TreeIndividual(list(deg), list(reversed(range(n)))))
    return pops


def run_tree_ga(
    n: int,
    fitness_fn,
    population: int = 24,
    generations: int = 12,
    seed: int = 42,
) -> tuple[MappingBasis, float]:
    """Maximize fitness_fn(MappingBasis). Returns best basis and fitness."""
    rng = np.random.default_rng(seed)
    pop: list[TreeIndividual] = _warm_start_population(n)
    while len(pop) < population:
        pop.append(TreeIndividual(_random_degrees(n, rng), rng.permutation(n).tolist()))

    def score(ind: TreeIndividual) -> float:
        try:
            return float(fitness_fn(individual_to_basis(ind)))
        except Exception:
            return -1e300

    scored = [(score(ind), ind) for ind in pop]
    scored.sort(key=lambda x: x[0], reverse=True)

    for _ in range(generations):
        elite = [ind.clone() for _, ind in scored[: max(2, population // 8)]]
        children: list[TreeIndividual] = list(elite)
        while len(children) < population:
            # tournament
            i1 = scored[int(rng.integers(0, min(8, len(scored))))][1]
            i2 = scored[int(rng.integers(0, min(8, len(scored))))][1]
            # label OX-ish crossover
            cut = int(rng.integers(1, n))
            labels = _valid_label_permutation(
                i1.labels[:cut] + [x for x in i2.labels if x not in i1.labels[:cut]],
                n,
            )
            degrees = list(i1.degrees if rng.random() < 0.5 else i2.degrees)
            try:
                _degrees_to_tree(degrees)
            except ValueError:
                degrees = _random_degrees(n, rng)
            if rng.random() < 0.3:
                degrees = _random_degrees(n, rng)
            if rng.random() < 0.4:
                a, b = rng.integers(0, n, size=2)
                labels[a], labels[b] = labels[b], labels[a]
            labels = _valid_label_permutation(labels, n)
            children.append(TreeIndividual(degrees, labels))
        scored = [(score(ind), ind) for ind in children]
        scored.sort(key=lambda x: x[0], reverse=True)

    best_fit, best_ind = scored[0]
    return individual_to_basis(best_ind), best_fit
