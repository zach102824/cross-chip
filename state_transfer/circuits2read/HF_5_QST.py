#!/usr/bin/env python3
"""Omit the final return QST from the HF tapered ping-pong circuit.

Reads ``HF_tapered_6q_3doubles_rzx_qst.json`` (6 QST swaps) and drops the last
``QST(q6 -> q2)`` after the final local RZX.  The travelling qubit then stays
on the ancilla, so the trailing single-qubit gates that acted on q2 (H, RX)
are rewritten onto q6.  Result: 5 QST swaps, same logical state up to a
SWAP(q2, q6) at the end.

Outputs under this folder:
    HF_tapered_6q_3doubles_rzx_5qst.json
    HF_tapered_6q_3doubles_rzx_5qst_circuit.png

Run from the repo root::

    .venv_py311/bin/python state_transfer/circuits2read/HF_5_QST.py
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.quantum_info import Operator, Statevector

_THIS_DIR = Path(__file__).resolve().parent
_STATE_TRANSFER = _THIS_DIR.parent
if str(_STATE_TRANSFER) not in sys.path:
    sys.path.insert(0, str(_STATE_TRANSFER))

import rewrite_hf_tapered_with_qst as qst_mod  # noqa: E402

_INPUT_JSON = _THIS_DIR / "HF_tapered_6q_3doubles_rzx_qst.json"
_TAG = "HF_tapered_6q_3doubles_rzx_5qst"

SENDER, ANCILLA = qst_mod.SENDER, qst_mod.ANCILLA  # 2, 6


def omit_final_return_qst(gates: list[dict]) -> list[dict]:
    """Drop last QST(ancilla -> sender); remap later gates on sender -> ancilla."""
    last_rzx = max(i for i, g in enumerate(gates) if g["op"] == "rzx")
    out: list[dict] = []
    skipped_return = False
    for i, g in enumerate(gates):
        if i <= last_rzx:
            out.append(dict(g))
            continue
        if (
            not skipped_return
            and g["op"] == "qst"
            and list(g["qubits"]) == [ANCILLA, SENDER]
        ):
            skipped_return = True
            continue
        new_g = dict(g)
        new_g["qubits"] = [ANCILLA if q == SENDER else q for q in g["qubits"]]
        out.append(new_g)
    if not skipped_return:
        raise ValueError(
            f"expected a final QST({ANCILLA}->{SENDER}) after the last RZX; none found"
        )
    return out


def assert_equiv_up_to_final_swap(
    qc_5: QuantumCircuit, qc_6: QuantumCircuit, name: str, seed: int = 7
) -> None:
    """``SWAP(sender, ancilla) · U_5 == U_6`` (global phase ok)."""
    rng = np.random.default_rng(seed)
    values = {p.name: float(rng.uniform(-np.pi, np.pi)) for p in qc_6.parameters}
    bound_5 = qc_5.assign_parameters({p: values[p.name] for p in qc_5.parameters})
    bound_6 = qc_6.assign_parameters({p: values[p.name] for p in qc_6.parameters})

    qc_swapped = bound_5.copy()
    qc_swapped.append(SwapGate(), [SENDER, ANCILLA])
    if not Operator(qc_swapped).equiv(Operator(bound_6)):
        # Fall back to statevector (same check, clearer residual).
        v5 = Statevector(qc_swapped).data
        v6 = Statevector(bound_6).data
        idx = int(np.argmax(np.abs(v6)))
        if abs(v6[idx]) > 1e-14 and abs(v5[idx]) > 1e-14:
            v5 = v5 * (v6[idx] / v5[idx])
        max_err = float(np.max(np.abs(v5 - v6)))
        raise AssertionError(f"{name}: NOT equivalent up to SWAP(q{SENDER},q{ANCILLA}) (max|Δ|={max_err})")
    print(f"    [ok] {name} == 6-QST after SWAP(q{SENDER},q{ANCILLA})")


def save_json(data: dict, rewritten: list[dict], path: Path) -> None:
    payload = deepcopy(data)
    payload["gates"] = rewritten
    payload["qst_mode"] = "pingpong_omit_final_return"
    payload["final_data_on_ancilla"] = True
    payload["logical_sender_ends_on"] = ANCILLA
    n_qst = sum(1 for g in rewritten if g["op"] == "qst")
    payload["n_qst"] = n_qst
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"    wrote {path.name}")


def main() -> None:
    data = json.loads(_INPUT_JSON.read_text(encoding="utf-8"))
    gates_6 = data["gates"]
    gates_5 = omit_final_return_qst(gates_6)

    params = qst_mod._params_from_gates(gates_5)
    qc_5 = qst_mod.qc_from_gates(gates_5, num_qubits=7, params=params)
    qc_6 = qst_mod.qc_from_gates(gates_6, num_qubits=7, params=params)

    n_rzx = sum(1 for g in gates_5 if g["op"] == "rzx")
    n_qst_5 = sum(1 for g in gates_5 if g["op"] == "qst")
    n_qst_6 = sum(1 for g in gates_6 if g["op"] == "qst")
    print(f"[{_TAG}]")
    print(f"    input  {_INPUT_JSON.name}: RZX={n_rzx}, QST={n_qst_6}")
    print(f"    omit final QST(q{ANCILLA}->q{SENDER}); remap trailing q{SENDER} -> q{ANCILLA}")
    print(f"    output: RZX={n_rzx}, QST={n_qst_5}")

    assert n_qst_5 == n_qst_6 - 1
    assert_equiv_up_to_final_swap(qc_5, qc_6, "HF 5-QST (omit final return)")

    json_path = _THIS_DIR / f"{_TAG}.json"
    png_path = _THIS_DIR / f"{_TAG}_circuit.png"
    save_json(data, gates_5, json_path)
    qst_mod.save_png(qc_5, png_path, _TAG)
    print("done.")


if __name__ == "__main__":
    main()
