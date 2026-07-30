#!/usr/bin/env python3
"""Cl2 2-doubles: same Z-pair freeze as exploring winner + fixed RZX hub.

Isolates the disjoint-RZX gain vs
``exploring/Cl2_10q_2doubles_disjoint_rzx`` by holding the freeze mask fixed
and compiling both doubles onto June_main's legacy bridge ``(3, 8)``.

Outputs:
    circuits2read/Cl2_10q_2doubles_freeze_fixed_rzx.json
    circuits2read/Cl2_10q_2doubles_freeze_fixed_rzx_circuit.png
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO = _THIS_DIR.parent
sys.path[:0] = [
    str(_REPO / "exploring"),
    str(_REPO / "UCCSD circuit"),
]

from error_budget import score_gates, spin_split_cross_pairs  # noqa: E402
from flexible_compile import (  # noqa: E402
    compile_strings,
    gates_to_qc,
    gen,
    statevector_overlap,
)
from freeze import freeze_z_pairs  # noqa: E402
import uccsd_circuit_io as cio  # noqa: E402

TAG = "Cl2_10q_2doubles_freeze_fixed_rzx"
BRIDGE = (3, 8)
N = 10
DOUBLES = [(4, 9, 6, 1), (4, 9, 5, 0)]
# Same keep masks as exploring/Cl2_10q_2doubles_disjoint_rzx.json
KEEP_PAIRS = [{2, 3}, {3}]
HF_BITS = "".join("1" if q in (0, 1, 2, 3, 5, 6, 7, 8) else "0" for q in range(N))


def run(out_dir: Path | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")

    if out_dir is None:
        out_dir = _THIS_DIR / "circuits2read"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signs = [1, 1]
    original = ["".join(gen.jw_string_for_double(N, d)) for d in DOUBLES]
    frozen = [freeze_z_pairs(s, keep) for s, keep in zip(original, KEEP_PAIRS)]

    # Sanity: match the exploring 2-doubles freeze choice.
    ref = json.loads(
        (_REPO / "exploring" / "Cl2_10q_2doubles_disjoint_rzx.json").read_text(
            encoding="utf-8"
        )
    )
    assert original == ref["original_strings"], (original, ref["original_strings"])
    assert frozen == ref["frozen_strings"], (frozen, ref["frozen_strings"])
    assert [sorted(k) for k in KEEP_PAIRS] == ref["keep_pairs"]

    out = compile_strings(
        frozen,
        signs=signs,
        order="given",
        hub_schedule=[BRIDGE, BRIDGE],
        fuse=True,
    )
    gates = out["gates"]
    rzx_pairs = [list(g["qubits"]) for g in gates if g["op"] == "rzx"]
    assert all(tuple(sorted(p)) == tuple(sorted(BRIDGE)) for p in rzx_pairs), rzx_pairs
    assert all(b == BRIDGE for b in out["bridges"]), out["bridges"]

    # Equivalence on the HF spin-paired subspace vs unfrozen fixed-hub compile.
    unfrozen = compile_strings(
        original,
        signs=signs,
        order="given",
        hub_schedule=[BRIDGE, BRIDGE],
        fuse=True,
    )
    rng = np.random.default_rng(0)
    ovs_unf = []
    ovs_disj = []
    for _ in range(8):
        th = rng.uniform(-np.pi, np.pi, len(signs))
        qa = gates_to_qc(gates, N, th)
        ovs_unf.append(
            statevector_overlap(qa, gates_to_qc(unfrozen["gates"], N, th), N, HF_BITS)
        )
        ovs_disj.append(
            statevector_overlap(qa, gates_to_qc(ref["gates"], N, th), N, HF_BITS)
        )
    if min(ovs_unf) < 1 - 1e-9:
        raise AssertionError(f"HF overlap vs unfrozen fixed failed: min={min(ovs_unf)}")
    if min(ovs_disj) < 1 - 1e-9:
        raise AssertionError(f"HF overlap vs disjoint failed: min={min(ovs_disj)}")

    bud = score_gates(gates, spin_split_cross_pairs(N))
    counts = Counter(g["op"] for g in gates)
    n_cz = counts.get("cz", 0)
    n_cz_disj = sum(1 for g in ref["gates"] if g["op"] == "cz")
    n_cz_legacy = sum(
        1
        for g in json.loads(
            (out_dir / "Cl2_10q_2doubles_rzx.json").read_text(encoding="utf-8")
        )["gates"]
        if g["op"] == "cz"
    )

    json_path = out_dir / f"{TAG}.json"
    cio.save_circuit_json(
        json_path,
        molecule="Cl2",
        bond_length=1.0,
        num_qubits=N,
        n_spatial=5,
        n_electrons=8,
        doubles=DOUBLES,
        signs=signs,
        theta_idx=list(range(len(signs))),
        logical_gates=gates,
        init_state=None,
        beta=None,
        extra={
            "method": "freeze_fixed_rzx",
            "freeze": "selective_z_pair",
            "keep_pairs": [sorted(k) for k in KEEP_PAIRS],
            "original_strings": original,
            "frozen_strings": frozen,
            "bridge_pair": list(BRIDGE),
            "hub_schedule": [list(BRIDGE), list(BRIDGE)],
            "rzx_pairs": rzx_pairs,
            "budget": bud.as_dict(),
            "depth": gates_to_qc(gates, N).depth(),
            "cz_count": n_cz,
            "cz_count_disjoint_same_freeze": n_cz_disj,
            "cz_count_legacy_unfrozen_fixed": n_cz_legacy,
            "hf_bitstring": HF_BITS,
            "hf_overlap_min_vs_unfrozen_fixed": float(min(ovs_unf)),
            "hf_overlap_min_vs_disjoint_same_freeze": float(min(ovs_disj)),
            "compare_to": {
                "disjoint_same_freeze": "exploring/Cl2_10q_2doubles_disjoint_rzx",
                "legacy_unfrozen_fixed": "June_main/circuits2read/Cl2_10q_2doubles_rzx",
            },
        },
    )
    gen.save_circuit_diagram(gates, N, out_dir / f"{TAG}_circuit.png", title=TAG)

    print(f"[{TAG}]")
    print(f"  original : {original}")
    print(f"  frozen   : {frozen}  keep_pairs={KEEP_PAIRS}")
    print(f"  gates    : {dict(counts)}")
    print(f"  RZX      : {rzx_pairs}  (fixed bridge {BRIDGE})")
    print(
        f"  CZ       : {n_cz}  "
        f"(disjoint same freeze: {n_cz_disj}, legacy unfrozen: {n_cz_legacy})"
    )
    print(f"  depth    : {gates_to_qc(gates, N).depth()}")
    print(f"  HF ov    : vs unfrozen={min(ovs_unf):.12f}, vs disjoint={min(ovs_disj):.12f}")
    print(f"  wrote {json_path.name} (+ {TAG}_circuit.png)")
    return {"json": json_path, "cz": n_cz, "cz_disj": n_cz_disj}


if __name__ == "__main__":
    run()
