"""Flexible α–β hub compilation for UCCSD Pauli strings.

Generalises ``_compile_tworow`` so the inter-row link may join ANY alpha
support qubit to ANY beta support qubit (emitted as CZ then fused to RZX).

Within each spin row, fan-in uses only the allowed CZ graph (nearest-neighbour
plus the (0,3)/(half,half+3) chords) — see ``constraints.allowed_cz_edges``.
"""
from __future__ import annotations

import importlib.util
import itertools
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter

from constraints import allowed_cz_edges

_ROOT = Path(__file__).resolve().parents[1]
_GEN_PATH = _ROOT / "UCCSD circuit" / "improved create UCCSD circuit .py"


def _load_gen():
    spec = importlib.util.spec_from_file_location("uccsd_gen", str(_GEN_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load_gen()


def _adj(n: int, *, chords: bool = True) -> dict[int, list[int]]:
    g: dict[int, list[int]] = defaultdict(list)
    for e in allowed_cz_edges(n, chords=chords):
        a, b = tuple(e)
        g[a].append(b)
        g[b].append(a)
    for k in g:
        g[k].sort()
    return g


def _shortest_path(adj, src: int, dst: int, allowed_nodes: set[int]):
    """BFS path src→dst using only nodes in allowed_nodes (and endpoints)."""
    if src == dst:
        return [src]
    q = deque([src])
    prev = {src: None}
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v in prev:
                continue
            if v != dst and v not in allowed_nodes:
                continue
            prev[v] = u
            if v == dst:
                path = [dst]
                while path[-1] != src:
                    path.append(prev[path[-1]])
                path.reverse()
                return path
            q.append(v)
    return None


def _fanin_on_graph(terminals: list[int], hub: int, n: int,
                    steiner_ok: set[int], *, chords: bool = True) -> list[tuple]:
    """CZ/H fan-in of terminals into hub along the allowed within-spin graph.

    Builds a shortest-path tree into ``hub``, then emits gates in leaf→hub
    order (same convention as ``gen._row_chain``).  Intermediate nodes must
    lie in ``steiner_ok``.  Raises ValueError if no legal path exists.
    """
    if hub not in terminals:
        raise ValueError("hub must be a terminal")
    terms = set(terminals)
    allowed_nodes = set(steiner_ok) | terms
    adj = _adj(n, chords=chords)
    parent: dict[int, int | None] = {hub: None}
    for t in sorted(terms):
        if t == hub:
            continue
        path = _shortest_path(adj, t, hub, allowed_nodes)
        if path is None:
            raise ValueError(f"no path {t}->{hub} on allowed CZ graph")
        for i in range(len(path) - 1):
            # path runs leaf → hub; parent[farther] = closer
            parent[path[i]] = path[i + 1]

    def _depth(q: int) -> int:
        d = 0
        while q != hub:
            q = parent[q]
            d += 1
        return d

    nodes = sorted((q for q in parent if q != hub), key=lambda q: -_depth(q))
    gates = []
    h_done: set[int] = set()
    for q in nodes:
        p = parent[q]
        gates.append(("CZ", q, p))
        if p != hub and p not in h_done:
            gates.append(("H", p))
            h_done.add(p)
    return gates


def compile_flexible(string: str, n: int, hub_a: int | None = None,
                     hub_b: int | None = None, hub_hint: int | None = None,
                     use_cz_graph: bool = True, *, chords: bool = True):
    """Compile exp(-i t/2 P) with a freely chosen α–β bridge (hub_a, hub_b).

    The α–β link is a single CZ(hub_a, hub_b) (later fused to RZX).  Within each
    spin row, fan-in respects ``allowed_cz_edges`` when use_cz_graph=True.
    ``chords=False`` restricts fan-in to nearest-neighbour edges only.
    """
    half = n // 2
    sup = [q for q in range(n) if string[q] != "I"]
    alpha = [q for q in sup if q < half]
    beta = [q for q in sup if q >= half]
    steiner_ok = set(sup)  # only existing support letters may be path couriers

    if not alpha or not beta:
        row = alpha or beta
        pivot = hub_hint if hub_hint in row else (
            hub_a if hub_a in row else row[len(row) // 2]
        )
        if use_cz_graph:
            ladder = _fanin_on_graph(row, pivot, n, steiner_ok, chords=chords)
        else:
            ladder = gen._row_chain(row, pivot)
        hub_a, hub_b = pivot, None
    else:
        if hub_a is None or hub_a not in alpha:
            hub_a = hub_hint if hub_hint in alpha else alpha[len(alpha) // 2]
        if hub_b is None or hub_b not in beta:
            want = hub_a + half
            hub_b = want if want in beta else min(beta, key=lambda q: abs(q - want))
        pivot = hub_a
        if use_cz_graph:
            ladder = (
                _fanin_on_graph(alpha, hub_a, n, steiner_ok, chords=chords)
                + _fanin_on_graph(beta, hub_b, n, steiner_ok, chords=chords)
            )
        else:
            ladder = gen._row_chain(alpha, hub_a) + gen._row_chain(beta, hub_b)
        ladder.append(("H", hub_b))
        ladder.append(("CZ", hub_a, hub_b))  # α–β bridge → fused to RZX

    _, lad = gen._frame(ladder, pivot, n)
    basis = []
    for q in sup:
        basis += gen._basis_for(lad[q], string[q], q)
    prefix = basis + ladder
    ph, letters = gen._frame(prefix, pivot, n)
    assert letters == list(string), ("".join(letters), string)
    assert ph in (1, -1)
    return prefix, pivot, ph, (hub_a, hub_b)


def candidate_bridges(string: str, n: int):
    half = n // 2
    alpha = [q for q in range(n) if string[q] != "I" and q < half]
    beta = [q for q in range(n) if string[q] != "I" and q >= half]
    if not alpha or not beta:
        return [(None, None)]
    return list(itertools.product(alpha, beta))


def prog_to_gates(prog):
    gates = []
    for g in prog:
        if g[0] == "H":
            gates.append({"op": "h", "qubits": [g[1]]})
        elif g[0] == "RX":
            gates.append({"op": "rx", "qubits": [g[1]], "value": float(g[2])})
        elif g[0] == "CZ":
            gates.append({"op": "cz", "qubits": [g[1], g[2]]})
        elif g[0] == "ROT":
            _, pivot, d, a_sgn = g
            gates.append({
                "op": "rx",
                "qubits": [pivot],
                "param": f"t{d}",
                "coeff": float(a_sgn),
            })
        else:
            raise ValueError(g)
    return gates


def compile_strings(strings, signs=None, order="auto", hub_schedule=None,
                    fuse=True, optimize=True, *, chords: bool = True):
    """Compile a list of Pauli strings with optional per-block (hub_a, hub_b).

    hub_schedule: list of (hub_a, hub_b) aligned to *ordered* strings, or None
    for the legacy vertical preference.
    ``chords=False`` uses NN-only within-spin CZ (no (0,3)/(half,half+3)).
    """
    n = len(strings[0])
    if signs is None:
        signs = [1] * len(strings)
    idx = gen._auto_order(strings) if order == "auto" else list(range(len(strings)))

    prog, expected = [], []
    hub_hint = None
    bridges_used = []
    for pos, d in enumerate(idx):
        s = strings[d]
        if hub_schedule is None:
            ha = hb = None
        else:
            ha, hb = hub_schedule[pos]
        prefix, pivot, ph, bridge = compile_flexible(
            s, n, hub_a=ha, hub_b=hb, hub_hint=hub_hint, chords=chords
        )
        hub_hint = pivot
        bridges_used.append(bridge)
        # frame sign ph: ROT angle multiplier keeps exp(-i * signs[d] * t / 2 * P)
        # when signs[d] absorbs the representative sign; here ph is from frame.
        prog += prefix
        prog.append(("ROT", pivot, d, signs[d] * ph))
        prog += gen._invert(prefix)
        expected.append((s, ph))
    if optimize:
        prog = gen._peephole(prog)
    gen._verify_program(prog, n, expected)
    gates = prog_to_gates(prog)
    if fuse:
        gates = gen.fuse_cz_rot_cz_to_rzx(gates)
    return {
        "gates": gates,
        "order": idx,
        "bridges": bridges_used,
        "n_qubits": n,
        "strings": [strings[i] for i in idx],
        "prog": prog,
    }


def gates_to_qc(gates, n_qubits, theta_values=None):
    qc = QuantumCircuit(n_qubits)
    params = {}
    for g in gates:
        op = g["op"].lower()
        qs = g["qubits"]
        if op == "h":
            qc.h(qs[0])
        elif op == "cz":
            qc.cz(qs[0], qs[1])
        elif op == "rx":
            if "param" in g:
                name = g["param"]
                p = params.setdefault(name, Parameter(name))
                qc.rx(float(g.get("coeff", 1.0)) * p, qs[0])
            else:
                qc.rx(float(g["value"]), qs[0])
        elif op == "rzx":
            name = g["param"]
            p = params.setdefault(name, Parameter(name))
            qc.rzx(float(g.get("coeff", 1.0)) * p, qs[0], qs[1])
        else:
            raise ValueError(op)
    if theta_values is not None:
        bind = {}
        for name, p in params.items():
            # t0 -> index 0
            k = int(name.lstrip("t"))
            bind[p] = float(theta_values[k])
        qc = qc.assign_parameters(bind)
    return qc


def statevector_overlap(qc_a, qc_b, n_qubits, init_bits=None):
    """|<b|U_a† U_b|init>|^2 with computational init (default |0...0>)."""
    from qiskit.quantum_info import Statevector

    init = QuantumCircuit(n_qubits)
    if init_bits:
        for q, bit in enumerate(init_bits):
            if bit == "1":
                init.x(q)
    sv_a = Statevector.from_instruction(init.compose(qc_a))
    sv_b = Statevector.from_instruction(init.compose(qc_b))
    return float(np.abs(np.vdot(sv_a.data, sv_b.data)) ** 2)
