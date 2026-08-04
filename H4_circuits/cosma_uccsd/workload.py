"""H4 fixed-pid UCCSD workload with all linked fermionic excitations.

Wire convention (TenCirChem for (4e,4o)):
  spin-orbitals 0..3 = beta block (spatial 0..3)
  spin-orbitals 4..7 = alpha block (spatial 0..3)
COSMA num_modes = 8 (one mode per spin-orbital).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Fixed ansatz from UCCSD_Mole/H4.ipynb (stable across geometries).
FIXED_DOUBLE_PIDS: list[int] = [12, 5, 9, 14, 7, 4, 10, 13]

# All linked ex_ops per pid at d=1.0 A (notebook fixed_meta_df).
# Order within each pid matches TenCirChem probe.ex_ops order.
H4_LINKED_EX_OPS: dict[int, list[tuple[int, int, int, int]]] = {
    12: [(2, 6, 5, 1)],
    5: [(2, 6, 4, 0)],
    9: [(2, 7, 5, 0), (6, 3, 1, 4)],
    14: [(3, 7, 5, 1)],
    7: [(3, 7, 4, 0)],
    4: [(6, 7, 5, 4), (2, 3, 1, 0)],
    10: [(3, 6, 5, 0), (7, 2, 1, 4)],
    13: [(2, 7, 5, 1), (6, 3, 1, 5)],
}

N_SPATIAL = 4
N_MODES = 8  # spin-orbitals
N_ELECTRONS = 4


@dataclass(frozen=True)
class LinkedExcitation:
    pid: int
    ex_op_index: int
    ex_op: tuple[int, int, int, int]
    theta_name: str


@dataclass
class H4Workload:
    """Ordered product of linked excitations with shared θ per pid."""

    pids: list[int] = field(default_factory=lambda: list(FIXED_DOUBLE_PIDS))
    n_modes: int = N_MODES
    n_spatial: int = N_SPATIAL
    n_electrons: int = N_ELECTRONS
    bond_length_A: float = 1.0

    def linked_excitations(self) -> list[LinkedExcitation]:
        out: list[LinkedExcitation] = []
        for pid in self.pids:
            ops = H4_LINKED_EX_OPS[pid]
            for i, op in enumerate(ops):
                out.append(
                    LinkedExcitation(
                        pid=pid,
                        ex_op_index=i,
                        ex_op=op,
                        theta_name=f"t{pid}",
                    )
                )
        return out

    def product_order_labels(self) -> list[str]:
        """Labels for each factor in the UCC product (one per linked op)."""
        return [f"pid{e.pid}_op{e.ex_op_index}" for e in self.linked_excitations()]


def load_h4_workload(bond_length_A: float = 1.0) -> H4Workload:
    return H4Workload(bond_length_A=float(bond_length_A))


def tencirchem_to_export_wire(q: int, n_spatial: int = N_SPATIAL) -> int:
    """TenCirChem (beta|alpha) index → export/taper (alpha|beta) index."""
    if q < n_spatial:
        return q + n_spatial  # beta → upper block
    return q - n_spatial  # alpha → lower block


def export_to_tencirchem_wire(q: int, n_spatial: int = N_SPATIAL) -> int:
    """Inverse of :func:`tencirchem_to_export_wire`."""
    if q < n_spatial:
        return q + n_spatial
    return q - n_spatial


def relabel_pauli_string(pauli: str, remap: dict[int, int]) -> str:
    n = len(pauli)
    out = ["I"] * n
    for q, letter in enumerate(pauli):
        if letter == "I":
            continue
        out[remap[q]] = letter
    return "".join(out)
