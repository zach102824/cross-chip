import importlib.util, itertools
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
import _search_cl2 as S   # reuse cz_edges + best_embedding + chip graph

S6 = list(itertools.permutations(range(6)))
BASE = [(5, 11, 8, 2), (5, 11, 7, 1), (5, 11, 9, 3),
        (5, 11, 10, 4), (5, 11, 6, 0)]


def relabel(doubles, sigma):
    def m(o):
        return sigma[o] if o < 6 else sigma[o - 6] + 6
    return [tuple(m(o) for o in d) for d in doubles]


# Phase 1: total CZ for every reordering (cheap).
totals = []
for sigma in S6:
    dbl = relabel(BASE, sigma)
    try:
        total = len(S.cz_edges(dbl))
    except Exception:
        continue
    totals.append((total, sigma))
totals.sort()
print("min total CZ over reorderings:", totals[0][0])
print("how many reach the minimum total:",
      sum(1 for t, _ in totals if t == totals[0][0]))

# Phase 2: embedding cross-chip only for reorderings with total <= 58.
best = None
checked = 0
for total, sigma in totals:
    if total > 58:
        break
    checked += 1
    pairs = S.cz_edges(relabel(BASE, sigma))
    emb = S.best_embedding(pairs)
    if emb is None:
        continue
    cross = emb[0]
    if best is None or (cross, total) < (best[0], best[1]):
        best = (cross, total, sigma, emb[1])
print(f"checked {checked} reorderings with total<=58")
print("best:", best[:3] if best else None)
if best:
    print("placement logical->physical:", dict(sorted(best[3].items())))
