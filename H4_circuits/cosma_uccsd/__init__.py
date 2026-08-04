"""COSMA-faithful full-linked H4 UCCSD compilation (Python port).

Native COSMA C++ could not be built in this environment (missing C++ SDK
headers), so the PPTT mappings / Gray scheduling / parity-tree allocation
ideas are reimplemented here against the cloned COSMA sources.
"""

from .workload import FIXED_DOUBLE_PIDS, H4_LINKED_EX_OPS, load_h4_workload

__all__ = [
    "FIXED_DOUBLE_PIDS",
    "H4_LINKED_EX_OPS",
    "load_h4_workload",
]
