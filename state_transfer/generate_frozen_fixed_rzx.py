#!/usr/bin/env python3
"""HF tapered doubles: selective Z-pair freeze + fixed RZX bridge.

Companion to ``generate_tapered_circuits.py`` (no freeze, fixed bridge) and
``exploring/HF_6q_disjoint_rzx`` (full freeze, three disjoint bridges).

Full freeze of the tapered strings yields weight-2 Paulis on *different*
α–β pairs, so a single shared RZX bridge is impossible.  Keeping the
spatial-2 courier pair ``Z2 Z5`` restores a common bridge ``(2, 5)`` while
still deleting the other spin-paired Z's:

    YZZXZZ → YIZXIZ
    IYZIXZ → IYZIXZ   (already needs Z2 Z5)
    IIYIIX → IIYIIX

Compile with unconstrained row fan-in onto hub ``(2, 5)``, fuse CZ·RX·CZ →
RZX.  Equivalent to the unfrozen tapered circuit on the HF spin-paired
subspace.

Outputs:
    circuits2read/HF_tapered_6q_3doubles_freeze_fixed_rzx.json
    circuits2read/HF_tapered_6q_3doubles_freeze_fixed_rzx_circuit.png
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
    str(_THIS_DIR),
]

from flexible_compile import (  # noqa: E402
    compile_flexible,
    gates_to_qc,
    gen,
    prog_to_gates,
    statevector_overlap,
)
from freeze import freeze_z_pairs  # noqa: E402
import uccsd_circuit_io as cio  # noqa: E402

TAG = "HF_tapered_6q_3doubles_freeze_fixed_rzx"
BRIDGE = (2, 5)
# Per-string keep_pairs (alpha indices of Z_q Z_{q+half} couriers to retain).
KEEP_PAIRS = [{2}, {2}, set()]


def _compile_fixed_hub(strings, signs, hub=BRIDGE, n=6):
    ha, hb = hub
    prog, expected, bridges = [], [], []
    for d, s in enumerate(strings):
        prefix, pivot, ph, bridge = compile_flexible(
            s, n, hub_a=ha, hub_b=hb, hub_hint=ha, use_cz_graph=False
        )
        bridges.append(bridge)
        prog += prefix
        prog.append(("ROT", pivot, d, signs[d] * ph))
        prog += gen._invert(prefix)
        expected.append((s, ph))
    prog = gen._peephole(prog)
    gen._verify_program(prog, n, expected)
    gates = gen.fuse_cz_rot_cz_to_rzx(prog_to_gates(prog))
    return gates, bridges


def run(out_dir: Path | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")

    if out_dir is None:
        out_dir = _THIS_DIR / "circuits2read"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = json.loads(
        (out_dir / "HF_tapered_6q_3doubles_rzx.json").read_text(encoding="utf-8")
    )
    strings = list(base["tapered_strings"])
    signs = [int(s) for s in base["signs"]]
    n = int(base["num_qubits"])
    frozen = [freeze_z_pairs(s, keep) for s, keep in zip(strings, KEEP_PAIRS)]

    gates, bridges = _compile_fixed_hub(frozen, signs, hub=BRIDGE, n=n)
    rzx_pairs = [list(g["qubits"]) for g in gates if g["op"] == "rzx"]
    assert all(tuple(sorted(p)) == tuple(sorted(BRIDGE)) for p in rzx_pairs), rzx_pairs
    assert all(b == BRIDGE for b in bridges), bridges

    # HF-subspace check vs the unfrozen tapered circuit.
    hf_bits = base["hf_bitstring_tapered"]
    rng = np.random.default_rng(0)
    ovs = []
    for _ in range(8):
        th = rng.uniform(-np.pi, np.pi, len(strings))
        ovs.append(
            statevector_overlap(
                gates_to_qc(base["gates"], n, th),
                gates_to_qc(gates, n, th),
                n,
                init_bits=hf_bits,
            )
        )
    if min(ovs) < 1 - 1e-9:
        raise AssertionError(f"HF overlap failed: min={min(ovs)}")

    n_cz = sum(1 for g in gates if g["op"] == "cz")
    n_cz_base = sum(1 for g in base["gates"] if g["op"] == "cz")
    counts = Counter(g["op"] for g in gates)

    json_path = out_dir / f"{TAG}.json"
    cio.save_circuit_json(
        json_path,
        molecule=base["molecule"],
        bond_length=base["bond_length"],
        num_qubits=n,
        n_spatial=base["n_spatial"],
        n_electrons=base["n_electrons"],
        doubles=base["doubles"],
        signs=signs,
        theta_idx=list(range(len(strings))),
        logical_gates=gates,
        init_state=None,
        beta=None,
        extra={
            "tapered": True,
            "freeze": "selective_z_pair",
            "keep_pairs": [sorted(k) for k in KEEP_PAIRS],
            "tapered_strings": strings,
            "frozen_strings": frozen,
            "bridge_pair": list(BRIDGE),
            "n_qubits_full": base.get("n_qubits_full"),
            "n_electrons_full": base.get("n_electrons_full"),
            "removed_qubits": base.get("removed_qubits"),
            "tapering_values": base.get("tapering_values"),
            "kept_qubits": base.get("kept_qubits"),
            "symmetry_generators": base.get("symmetry_generators"),
            "hf_bitstring_tapered": hf_bits,
            "hf_occupied_qubits": base.get("hf_occupied_qubits"),
            "cz_count": n_cz,
            "cz_count_unfrozen_fixed": n_cz_base,
            "cz_saved_vs_unfrozen_fixed": n_cz_base - n_cz,
            "hf_overlap_min_vs_unfrozen": float(min(ovs)),
        },
    )
    gen.save_circuit_diagram(gates, n, out_dir / f"{TAG}_circuit.png", title=TAG)

    print(f"[{TAG}]")
    print(f"  tapered  : {strings}")
    print(f"  frozen   : {frozen}  keep_pairs={KEEP_PAIRS}")
    print(f"  gates    : {dict(counts)}")
    print(f"  RZX      : {rzx_pairs}  (fixed bridge {BRIDGE})")
    print(f"  CZ       : {n_cz}  (unfrozen fixed: {n_cz_base}, saved {n_cz_base - n_cz})")
    print(f"  HF ov    : min={min(ovs):.12f}")
    print(f"  wrote {json_path.name} (+ {TAG}_circuit.png)")
    return {"json": json_path, "cz": n_cz, "cz_base": n_cz_base}


if __name__ == "__main__":
    run()
