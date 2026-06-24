"""Exact minimum cross-chip embedding search.

Given the logical 2-qubit interaction graph produced by the Cl2 circuit
(CZ pairs with multiplicity), find a bijection logical-orbital -> physical-qubit
such that *every* logical edge lands on a physical chip edge (so no SWAP routing
is ever needed, total 2q count unchanged) while minimising the total number of
2q gates that fall on the two high-error cross-chip edges.

Optionally also sweeps the 6! alpha/beta orbital reorderings, since the orbital
labelling changes both the circuit and its interaction graph.
"""
import importlib.util, itertools
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "gen_cl2", str(HERE / "improved create UCCSD circuit_Cl2.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
from qiskit.circuit import Parameter

NUM = 12
BASE = [(5, 11, 8, 2), (5, 11, 7, 1), (5, 11, 9, 3),
        (5, 11, 10, 4), (5, 11, 6, 0)]

# Physical chip graph (labels as in the drawing): three 2x2 squares A-B-C in a
# line, joined by two cross-chip bridges.
CHIP_EDGES = {tuple(sorted(e)) for e in [
    (8, 9), (7, 8), (6, 7), (6, 9),       # square A
    (10, 11), (4, 10), (5, 11), (4, 5),   # square B
    (0, 3), (2, 3), (0, 1), (1, 2),       # square C
    (9, 10), (3, 4),                      # cross-chip bridges (high error)
]}
BRIDGES = {tuple(sorted(e)) for e in [(9, 10), (3, 4)]}

CHIP_ADJ = {q: set() for q in range(NUM)}
for a, b in CHIP_EDGES:
    CHIP_ADJ[a].add(b)
    CHIP_ADJ[b].add(a)


def cz_weights(doubles):
    thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
    qc, *_ = gen.create_uccsd_circuit(
        NUM, doubles, thetas=thetas, optimize=True, order="auto",
        pair=False, init_state=None)
    pairs = []
    for inst in qc.data:
        if inst.operation.name == "cz":
            a, b = (qc.find_bit(q).index for q in inst.qubits)
            pairs.append(tuple(sorted((a, b))))
    return Counter(pairs), len(pairs)


def min_cross_embedding(weights):
    """Exact backtracking: bijection logical->physical with every logical edge
    on a chip edge, minimising weight on the two bridges. Returns (cost, map)
    or None if the logical graph does not embed into the chip at all."""
    logical_adj = {}
    for (a, b), w in weights.items():
        logical_adj.setdefault(a, {})[b] = w
        logical_adj.setdefault(b, {})[a] = w
    nodes = sorted(logical_adj, key=lambda n: -len(logical_adj[n]))

    best = [None, None]  # cost, mapping

    def bridge_cost(la, lb, pa, pb):
        e = tuple(sorted((pa, pb)))
        return logical_adj[la][lb] if e in BRIDGES else 0

    def bt(i, used, place, cost):
        if best[0] is not None and cost >= best[0]:
            return
        if i == len(nodes):
            best[0], best[1] = cost, dict(place)
            return
        ln = nodes[i]
        placed_neighbors = [(m, place[m]) for m in logical_adj[ln] if m in place]
        if placed_neighbors:
            _, p0 = placed_neighbors[0]
            cands = CHIP_ADJ[p0]
        else:
            cands = set(range(NUM))
        for pn in cands:
            if pn in used:
                continue
            ok = True
            add = 0
            for (m, pm) in placed_neighbors:
                if pn not in CHIP_ADJ[pm]:
                    ok = False
                    break
                add += bridge_cost(ln, m, pn, pm)
            if not ok:
                continue
            place[ln] = pn
            used.add(pn)
            bt(i + 1, used, place, cost + add)
            used.discard(pn)
            del place[ln]

    bt(0, set(), {}, 0)
    return (best[0], best[1]) if best[0] is not None else None


def relabel(doubles, sigma):
    def m(o):
        return sigma[o] if o < 6 else sigma[o - 6] + 6
    return [tuple(m(o) for o in d) for d in doubles]


if __name__ == "__main__":
    w, tot = cz_weights(BASE)
    print(f"BASE ordering: total CZ = {tot}")
    res = min_cross_embedding(w)
    print(f"  exact min cross-chip = {res[0]}")
    print(f"  placement logical->physical: {dict(sorted(res[1].items()))}")

    print("\nSweeping 6! orbital reorderings ...")
    best = None
    for sigma in itertools.permutations(range(6)):
        dbl = relabel(BASE, sigma)
        try:
            w, tot = cz_weights(dbl)
        except Exception:
            continue
        res = min_cross_embedding(w)
        if res is None:
            continue
        cross = res[0]
        key = (cross, tot)
        if best is None or key < best[0]:
            best = (key, sigma, res[1])
    print(f"BEST over reorderings: cross-chip={best[0][0]}, total CZ={best[0][1]}")
    print(f"  sigma (alpha/beta orbital order) = {best[1]}")
    print(f"  placement logical->physical: {dict(sorted(best[2].items()))}")
