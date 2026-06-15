"""
simplify decomposed circuit.py
==============================
Input  : a circuit JSON saved by uccsd_circuit_io.save_circuit_json
         (e.g. June_main/circuits2read/HF_bond_1.4.json).
Output : a "real-device mapped + simplified" JSON next to the input
         (<name>_decomposed_simplified.json) plus a printed gate-count report.

What it does
------------
1.  Decompose every CROSS-CHIP CZ into the cross-resonance native form
        CZ = (Rz_c(-pi/2) (x) H . Rx_t(-pi/2)) . RZX(pi/2) . (I (x) H_t)
    (on-chip CZs are kept as native CZ).
2.  Because CZ is symmetric, for each cross-chip CZ it tries BOTH
    control/target orientations and keeps whichever yields more gate
    cancellation against its neighbours.
3.  Runs a peephole simplification that maximises cancellation of BOTH
    kinds of gate:
      - H . H = I,
      - consecutive same-axis single-qubit rotations (Rz/Rx) are merged,
        and dropped when they fold to the identity,
      - X . X = I.
    Two-qubit gates (RZX / CZ / CX) block single-qubit moves on their qubits;
    this is the same commuting-peephole idea as the generator's ``_peephole``.

Usage:  python "simplify decomposed circuit.py" [path/to/circuit.json]
        (defaults to June_main/circuits2read/HF_bond_1.4.json)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import uccsd_circuit_io as cio

_TWO_PI = 2.0 * np.pi


# ----------------------------------------------------------------------
# Gate helpers
# ----------------------------------------------------------------------
def _is_1q(g: dict) -> bool:
    return len(g["qubits"]) == 1


def _is_numeric_rot(g: dict) -> bool:
    return g["op"] in ("rz", "rx") and "param" not in g and "value" in g


def _wrap(angle: float) -> float:
    """Wrap to (-pi, pi]."""
    a = (float(angle) + np.pi) % _TWO_PI - np.pi
    return float(a)


def _cancel_or_merge(g1: dict, g2: dict):
    """Return None (no rule), [] (annihilate), or [merged] for two 1q gates
    on the same qubit that are adjacent on that qubit."""
    if g1["op"] == g2["op"] == "h":
        return []
    if g1["op"] == g2["op"] == "x":
        return []
    if g1["op"] == g2["op"] and _is_numeric_rot(g1) and _is_numeric_rot(g2):
        merged_val = _wrap(g1["value"] + g2["value"])
        if abs(merged_val) < 1e-9:
            return []
        out = {"op": g1["op"], "qubits": list(g1["qubits"]), "value": merged_val}
        if g1.get("cross_chip") or g2.get("cross_chip"):
            out["cross_chip"] = True
        return [out]
    return None


def peephole(gates: list[dict]) -> list[dict]:
    """Cancel/merge adjacent single-qubit gates until no change."""
    gates = [dict(g) for g in gates]
    changed = True
    while changed:
        changed = False
        for i in range(len(gates)):
            gi = gates[i]
            if not _is_1q(gi):
                continue
            q = gi["qubits"][0]
            j = None
            for k in range(i + 1, len(gates)):
                if q in gates[k]["qubits"]:
                    j = k
                    break
            if j is None:
                continue
            gj = gates[j]
            if not (_is_1q(gj) and gj["qubits"][0] == q):
                continue
            res = _cancel_or_merge(gi, gj)
            if res is None:
                continue
            del gates[j]
            del gates[i]
            for off, g in enumerate(res):
                gates.insert(i + off, g)
            changed = True
            break
    return gates


# ----------------------------------------------------------------------
# Cross-chip decomposition with orientation choice
# ----------------------------------------------------------------------
def _adjacent_h_count(gates: list[dict], idx: int, q: int) -> int:
    """Number of single-qubit H gates immediately neighbouring position idx on q."""
    cnt = 0
    for k in range(idx - 1, -1, -1):
        if q in gates[k]["qubits"]:
            if gates[k]["op"] == "h" and _is_1q(gates[k]):
                cnt += 1
            break
    for k in range(idx + 1, len(gates)):
        if q in gates[k]["qubits"]:
            if gates[k]["op"] == "h" and _is_1q(gates[k]):
                cnt += 1
            break
    return cnt


def _choose_target(gates: list[dict], idx: int) -> int:
    """Pick the CZ qubit whose neighbouring H gates will cancel the
    decomposition's boundary H (the decomposition puts H on the target)."""
    a, b = gates[idx]["qubits"]
    return a if _adjacent_h_count(gates, idx, a) >= _adjacent_h_count(gates, idx, b) else b


def decompose_cross_chip(gates: list[dict], choose_orientation: bool = True) -> list[dict]:
    out: list[dict] = []
    for idx, g in enumerate(gates):
        if g["op"] == "cz" and g.get("cross_chip"):
            a, b = g["qubits"]
            if choose_orientation:
                target = _choose_target(gates, idx)
                control = b if target == a else a
            else:
                control, target = a, b
            out.extend(cio.decompose_cross_chip_cz(control, target))
        else:
            out.append(dict(g))
    return out


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def gate_counts(gates: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for g in gates:
        counts[g["op"]] = counts.get(g["op"], 0) + 1
    counts["__total__"] = len(gates)
    return counts


def _fmt(counts: dict) -> str:
    keys = [k for k in ("h", "rx", "rz", "ry", "x", "cx", "cz", "rzx") if k in counts]
    body = "  ".join(f"{k}={counts[k]}" for k in keys)
    return f"{body}  | total={counts['__total__']}"


def simplify_file(in_path: Path) -> Path:
    data = cio.load_circuit_json(in_path)
    logical = data["gates"]

    print(f"Input: {in_path}")
    print(f"  logical               : {_fmt(gate_counts(logical))}")

    # Naive orientation (control = first listed qubit) for comparison.
    naive = peephole(decompose_cross_chip(logical, choose_orientation=False))
    chosen = peephole(decompose_cross_chip(logical, choose_orientation=True))

    print(f"  decomposed (naive)    : {_fmt(gate_counts(naive))}")
    print(f"  decomposed (oriented) : {_fmt(gate_counts(chosen))}")
    saved = gate_counts(naive)["__total__"] - gate_counts(chosen)["__total__"]
    print(f"  orientation choice removed {saved} extra gate(s)")

    out = dict(data)
    out["gates"] = chosen
    out["decomposition"] = "cross_chip_cz_to_rzx"
    out["simplified"] = True
    out_path = in_path.with_name(in_path.stem + "_decomposed_simplified.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return out_path


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target = Path(sys.argv[1])
    else:
        target = (
            Path(__file__).resolve().parent.parent
            / "June_main"
            / "circuits2read"
            / "HF_bond_1.4.json"
        )
    simplify_file(target)
