"""Extend the exact min-cross-chip search to also sweep the DOUBLE ordering.

The order in which the doubles are emitted changes the interface cancellations
in the peephole pass, hence the per-edge CZ multiplicity, hence how much load
lands on the two cross-chip bridges. We sweep:
    - 5! orderings of the doubles            (order='given')
    - optionally 6! alpha/beta orbital relabelings
and for each we run the exact placement embedding (every gate on a chip edge,
no SWAPs) minimising the high-error count.
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

import _embed_search as E   # reuse cz_weights pieces + chip graph + embedder

NUM = 12
BASE = [(5, 11, 8, 2), (5, 11, 7, 1), (5, 11, 9, 3),
        (5, 11, 10, 4), (5, 11, 6, 0)]


def cz_weights_ordered(doubles, order):
    """order: 'auto' or 'given'."""
    thetas = [Parameter(f"t{d}") for d in range(len(doubles))]
    qc, *_ = gen.create_uccsd_circuit(
        NUM, doubles, thetas=thetas, optimize=True, order=order,
        pair=False, init_state=None)
    pairs = []
    for inst in qc.data:
        if inst.operation.name == "cz":
            a, b = (qc.find_bit(q).index for q in inst.qubits)
            pairs.append(tuple(sorted((a, b))))
    return Counter(pairs), len(pairs)


def relabel(doubles, sigma):
    def m(o):
        return sigma[o] if o < 6 else sigma[o - 6] + 6
    return [tuple(m(o) for o in d) for d in doubles]


def best_for(doubles, order):
    w, tot = cz_weights_ordered(doubles, order)
    res = E.min_cross_embedding(w)
    if res is None:
        return None
    return res[0], tot, res[1]


if __name__ == "__main__":
    # 1) sweep only the double ordering (base labels)
    print("== sweep double ordering only (base orbital labels) ==")
    best = None
    for perm in itertools.permutations(range(5)):
        dbl = [BASE[i] for i in perm]
        r = best_for(dbl, "given")
        if r is None:
            continue
        cross, tot, place = r
        key = (cross, tot)
        if best is None or key < best[0]:
            best = (key, perm, place)
    print(f"  best: cross-chip={best[0][0]}, total CZ={best[0][1]}, "
          f"double order={best[1]}")
    print(f"  placement logical->physical: {dict(sorted(best[2].items()))}")

    # 2) full sweep: double ordering x orbital relabeling
    print("\n== full sweep: double order x orbital relabel ==")
    best = None
    for sigma in itertools.permutations(range(6)):
        rel = relabel(BASE, sigma)
        for perm in itertools.permutations(range(5)):
            dbl = [rel[i] for i in perm]
            try:
                r = best_for(dbl, "given")
            except Exception:
                continue
            if r is None:
                continue
            cross, tot, place = r
            key = (cross, tot)
            if best is None or key < best[0]:
                best = (key, sigma, perm, place)
    print(f"  best: cross-chip={best[0][0]}, total CZ={best[0][1]}")
    print(f"  sigma={best[1]}  double order={best[2]}")
    print(f"  placement logical->physical: {dict(sorted(best[3].items()))}")
