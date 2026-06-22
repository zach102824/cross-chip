import importlib.util, itertools
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "gen_cl2", str(HERE / "improved create UCCSD circuit_Cl2.py"))
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
from qiskit.circuit import Parameter

NUM = 12
BASE_DOUBLES = [(5, 11, 8, 2), (5, 11, 7, 1), (5, 11, 9, 3),
                (5, 11, 10, 4), (5, 11, 6, 0)]

CHIP_EDGES = {tuple(sorted(e)) for e in [
    (8, 9), (7, 8), (6, 7), (6, 9),
    (10, 11), (4, 10), (5, 11), (4, 5),
    (0, 3), (2, 3), (0, 1), (1, 2),
    (9, 10), (3, 4),
]}
BAD = {tuple(sorted(e)) for e in [(9, 10), (3, 4)]}
# bridge corners: A<->B via (9,10); B<->C via (4,3)
BLOCKS = {"A": {6, 7, 8, 9}, "B": {4, 5, 10, 11}, "C": {0, 1, 2, 3}}


def cz_edges(doubles):
    thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
    qc, *_ = gen.create_uccsd_circuit(
        NUM, doubles, thetas=thetas, optimize=True, order="auto",
        pair=False, init_state=None)
    pairs = []
    for inst in qc.data:
        if inst.operation.name == "cz":
            a, b = (qc.find_bit(q).index for q in inst.qubits)
            pairs.append(tuple(sorted((a, b))))
    return pairs


def square_embeds(block_nodes, edges_in, corner_constraints, square_cycle):
    """Can the 4 block_nodes embed in a 4-cycle (square_cycle gives the 4
    physical corners in cyclic order) so that every interaction edge in
    edges_in maps to a square edge and corner_constraints {node:phys} hold?"""
    for perm in itertools.permutations(block_nodes):
        place = dict(zip(square_cycle, perm))   # phys->logical
        inv = {v: k for k, v in place.items()}  # logical->phys
        if any(inv[n] != p for n, p in corner_constraints.items()):
            continue
        ok = True
        for (a, b) in edges_in:
            if tuple(sorted((inv[a], inv[b]))) not in CHIP_EDGES:
                ok = False
                break
        if ok:
            return inv
    return None


def best_embedding(pairs, verbose=False):
    w = Counter(pairs)
    edges = list(w)
    nodes = sorted({q for e in pairs for q in e})
    cyc = {"A": [8, 9, 6, 7], "B": [10, 11, 5, 4], "C": [3, 0, 1, 2]}
    best = None
    for A in itertools.combinations(nodes, 4):
        sA = set(A)
        rest = [n for n in nodes if n not in sA]
        for B in itertools.combinations(rest, 4):
            sB = set(B)
            sC = set(rest) - sB
            grp = {n: "A" for n in sA}
            grp.update({n: "B" for n in sB})
            grp.update({n: "C" for n in sC})
            inter = [(a, b) for (a, b) in edges if grp[a] != grp[b]]
            if len(inter) != 2:
                continue
            tags = {frozenset((grp[a], grp[b])) for (a, b) in inter}
            # need a line: A-B and B-C (B in the middle)
            if tags != {frozenset(("A", "B")), frozenset(("B", "C"))}:
                continue
            cross = sum(w[e] for e in inter)
            if best is not None and cross >= best[0]:
                continue
            # connectors in B (to A and to C)
            connB = {}
            connA = connC = None
            for (a, b) in inter:
                ga, gb = grp[a], grp[b]
                if {ga, gb} == {"A", "B"}:
                    bnode = a if ga == "B" else b
                    anode = b if ga == "B" else a
                    connB["A"] = bnode
                    connA = anode
                else:
                    bnode = a if ga == "B" else b
                    cnode = b if ga == "B" else a
                    connB["C"] = bnode
                    connC = cnode
            # embed each block; bridge corners: A:9, B:(10 to A,4 to C), C:3
            embA = square_embeds(sA, [e for e in edges if grp[e[0]] == "A" and grp[e[1]] == "A"],
                                  {connA: 9}, cyc["A"])
            embC = square_embeds(sC, [e for e in edges if grp[e[0]] == "C" and grp[e[1]] == "C"],
                                  {connC: 3}, cyc["C"])
            embB = square_embeds(sB, [e for e in edges if grp[e[0]] == "B" and grp[e[1]] == "B"],
                                  {connB["A"]: 10, connB["C"]: 4}, cyc["B"])
            if embA and embB and embC:
                place = {**embA, **embB, **embC}
                best = (cross, place)
    return best


pairs = cz_edges(BASE_DOUBLES)
print("base total CZ:", len(pairs))
idw = sum(c for e, c in Counter(pairs).items() if e in BAD)
print("identity-placement cross-chip:", idw)
b = best_embedding(pairs)
if b:
    print("best embedding cross-chip:", b[0])
    print("placement logical->physical:", dict(sorted(b[1].items())))
else:
    print("no valid embedding found")
