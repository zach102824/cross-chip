#!/usr/bin/env python3
"""Rewrite HF tapered long-range RZX via QST ping-pong with ancilla q6.

Reads ``circuits2read/HF_tapered_6q_3doubles_rzx.json`` (bridge pair (2, 5))
and replaces every ``rzx`` on ``[5, 2]`` with:

    QST(q2 -> q6)  .  RZX(t_i) on [5, 6]  .  QST(q6 -> q2)

so the entangler is local on chip B (q5 next to ancilla q6).  QST is modeled
as a labeled SWAP (exact on the receiver-in-|0> subspace), matching
``qst_formulas.txt`` and ``first_attempt/make_state_transfer_circuits.py``.

Outputs under ``circuits2read/``:
    HF_tapered_6q_3doubles_rzx_qst.json
    HF_tapered_6q_3doubles_rzx_qst_circuit.png

Run:  python state_transfer/rewrite_hf_tapered_with_qst.py
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

_THIS_DIR = Path(__file__).resolve().parent
_CIRCUITS = _THIS_DIR / "circuits2read"
_INPUT_JSON = _CIRCUITS / "HF_tapered_6q_3doubles_rzx.json"
_TAG = "HF_tapered_6q_3doubles_rzx_qst"

SENDER, TARGET, ANCILLA = 2, 5, 6
BRIDGE = (SENDER, TARGET)
LOCAL_RZX = (TARGET, ANCILLA)
QST_PAIR = (SENDER, ANCILLA)

_SWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]],
    dtype=complex,
)

# Maroon QST boxes like the reference ping-pong figure; IQP palette elsewhere.
_DRAW_STYLE = {
    "name": "iqp",
    "displaycolor": {
        "h": ["#FA4D56", "#000000"],
        "x": ["#002D9C", "#FFFFFF"],
        "rx": ["#9F1853", "#FFFFFF"],
        "ry": ["#9F1853", "#FFFFFF"],
        "rz": ["#33B1FF", "#000000"],
        "rzx": ["#9F1853", "#FFFFFF"],
        "cz": ["#33B1FF", "#000000"],
        "cx": ["#002D9C", "#000000"],
        "unitary": ["#9F1853", "#FFFFFF"],
        "QST": ["#9F1853", "#FFFFFF"],
    },
}


def qst_gate() -> UnitaryGate:
    """Cable transfer drawn as a 'QST' box; simulated as SWAP."""
    return UnitaryGate(_SWAP, label="QST")


def rewrite_gates(gates: list[dict]) -> list[dict]:
    """Replace each long-range RZX(5, 2) with QST / local RZX / QST back."""
    out: list[dict] = []
    for g in gates:
        if g["op"] == "rzx":
            qs = list(g["qubits"])
            if sorted(qs) != sorted(BRIDGE):
                raise ValueError(f"unexpected RZX pair {qs}; expected {list(BRIDGE)}")
            # Preserve Z-on-TARGET, X-on-travelling (now ancilla).
            out.append({"op": "qst", "qubits": [SENDER, ANCILLA]})
            rzx = {"op": "rzx", "qubits": [TARGET, ANCILLA]}
            if "param" in g:
                rzx["param"] = g["param"]
                rzx["coeff"] = float(g.get("coeff", 1.0))
            else:
                rzx["angle"] = float(g.get("value", g.get("angle", 0.0)))
            out.append(rzx)
            out.append({"op": "qst", "qubits": [ANCILLA, SENDER]})
        else:
            out.append(dict(g))
    return out


def _params_from_gates(gates: list[dict]) -> dict[str, Parameter]:
    names = sorted({g["param"] for g in gates if g["op"] == "rzx" and "param" in g})
    return {n: Parameter(n) for n in names}


def qc_from_gates(gates: list[dict], num_qubits: int, params: dict[str, Parameter]) -> QuantumCircuit:
    q = QuantumRegister(num_qubits, "q")
    qc = QuantumCircuit(q)
    for g in gates:
        op, qs = g["op"], g["qubits"]
        if op == "h":
            qc.h(q[qs[0]])
        elif op == "rx":
            if "param" in g:
                qc.rx(float(g.get("coeff", 1.0)) * params[g["param"]], q[qs[0]])
            else:
                qc.rx(float(g["value"]), q[qs[0]])
        elif op == "cz":
            qc.cz(q[qs[0]], q[qs[1]])
        elif op == "rzx":
            theta = float(g.get("coeff", 1.0)) * params[g["param"]] if "param" in g else float(
                g.get("value", g.get("angle", 0.0))
            )
            qc.rzx(theta, q[qs[0]], q[qs[1]])
        elif op == "qst":
            qc.append(qst_gate(), [q[qs[0]], q[qs[1]]])
        else:
            raise ValueError(f"unsupported op {op!r}")
    return qc


def pad_original_to_7q(gates: list[dict], params: dict[str, Parameter]) -> QuantumCircuit:
    """Original 6q gates on q0..q5 with idle ancilla q6."""
    return qc_from_gates(gates, num_qubits=7, params=params)


def assert_equiv(qc_a: QuantumCircuit, qc_b: QuantumCircuit, name: str, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    values = {p.name: float(rng.uniform(-np.pi, np.pi)) for p in qc_a.parameters}
    bound_a = qc_a.assign_parameters({p: values[p.name] for p in qc_a.parameters})
    bound_b = qc_b.assign_parameters({p: values[p.name] for p in qc_b.parameters})
    if not Operator(bound_a).equiv(Operator(bound_b)):
        raise AssertionError(f"{name}: NOT equivalent to padded target!")
    print(f"    [ok] {name} == padded target (up to global phase)")


def save_png(qc: QuantumCircuit, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = qc.draw(output="mpl", style=_DRAW_STYLE, fold=-1)
    fig.suptitle(title, fontsize=14)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"    wrote {path.name}")


def save_json(data: dict, rewritten: list[dict], path: Path) -> None:
    payload = deepcopy(data)
    payload["num_qubits"] = 7
    payload["gates"] = rewritten
    payload["ancilla"] = ANCILLA
    payload["qst_pair"] = list(QST_PAIR)
    payload["local_rzx_pair"] = list(LOCAL_RZX)
    payload["qst_mode"] = "pingpong"
    # bridge_pair stays as the logical (2, 5) long-range target.
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"    wrote {path.name}")


def main() -> None:
    data = json.loads(_INPUT_JSON.read_text(encoding="utf-8"))
    original_gates = data["gates"]
    rewritten = rewrite_gates(original_gates)

    params = _params_from_gates(rewritten)
    qc_qst = qc_from_gates(rewritten, num_qubits=7, params=params)
    qc_pad = pad_original_to_7q(original_gates, params)

    print(f"[{_TAG}]")
    n_rzx = sum(1 for g in rewritten if g["op"] == "rzx")
    n_qst = sum(1 for g in rewritten if g["op"] == "qst")
    print(f"    qubits 6 -> 7 (ancilla q{ANCILLA}), RZX={n_rzx}, QST={n_qst}")
    print(f"    rewrite: RZX{list(BRIDGE)} -> QST{list(QST_PAIR)} · RZX{list(LOCAL_RZX)} · QST back")

    assert_equiv(qc_qst, qc_pad, "HF tapered QST ping-pong (7q)")

    json_path = _CIRCUITS / f"{_TAG}.json"
    png_path = _CIRCUITS / f"{_TAG}_circuit.png"
    save_json(data, rewritten, json_path)
    save_png(qc_qst, png_path, _TAG)
    print("done.")


if __name__ == "__main__":
    main()
