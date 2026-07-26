"""JW Z-pair freeze / selective re-insert (Method 1).

For spin-paired doubles, Z_q Z_{q+half} = +1 on the reachable subspace, so
those letters may be deleted or re-inserted for free.
"""
from __future__ import annotations


def freeze_z_pairs(string: str, keep_pairs: set[int] | None = None) -> str:
    """Delete Z spin-pairs except those whose alpha index is in keep_pairs.

    keep_pairs: set of alpha-row indices q < half whose (q, q+half) Z-pair
    should be retained as parity couriers.
    """
    n = len(string)
    half = n // 2
    keep_pairs = keep_pairs or set()
    out = list(string)
    for q in range(half):
        if out[q] == "Z" and out[q + half] == "Z" and q not in keep_pairs:
            out[q] = "I"
            out[q + half] = "I"
    return "".join(out)


def fully_freeze(strings: list[str]) -> list[str]:
    return [freeze_z_pairs(s, keep_pairs=set()) for s in strings]


def all_keep_masks(string: str):
    """Yield keep_pairs subsets for Z-pairs present in string."""
    n = len(string)
    half = n // 2
    zpairs = [q for q in range(half) if string[q] == "Z" and string[q + half] == "Z"]
    m = len(zpairs)
    for mask in range(1 << m):
        keep = {zpairs[i] for i in range(m) if mask & (1 << i)}
        yield keep, freeze_z_pairs(string, keep)
