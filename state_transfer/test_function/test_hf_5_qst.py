"""Statevector equivalence: 5-QST (omit final return) vs full 6-QST ping-pong.

After the last local RZX, the 5-QST circuit leaves the travelling qubit on
ancilla q6 and applies the trailing H / RX(-π/2) there.  That matches the
6-QST circuit up to a final SWAP(q2, q6).

Run from the repo root with a Qiskit venv::

    .venv_py311/bin/python -m pytest state_transfer/test_function/test_hf_5_qst.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.library import SwapGate
from qiskit.quantum_info import Statevector

from conftest import STATE_TRANSFER

import rewrite_hf_tapered_with_qst as qst_mod  # noqa: E402

# HF_5_QST lives under circuits2read; load via path so pytest need not cd.
import importlib.util

_HF5_PATH = STATE_TRANSFER / "circuits2read" / "HF_5_QST.py"
_spec = importlib.util.spec_from_file_location("HF_5_QST", _HF5_PATH)
hf5_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(hf5_mod)

QST6_JSON = STATE_TRANSFER / "circuits2read" / "HF_tapered_6q_3doubles_rzx_qst.json"
QST5_JSON = STATE_TRANSFER / "circuits2read" / "HF_tapered_6q_3doubles_rzx_5qst.json"

SENDER, ANCILLA = 2, 6
N_QUBITS = 7
N_TRIALS = 5
ATOL = 1e-10


@pytest.fixture(scope="module")
def qst6_data():
    if not QST6_JSON.is_file():
        pytest.skip(f"{QST6_JSON.name} missing (run rewrite_hf_tapered_with_qst.py)")
    return json.loads(QST6_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def qst5_data(qst6_data):
    if QST5_JSON.is_file():
        return json.loads(QST5_JSON.read_text(encoding="utf-8"))
    # Generate on the fly if the artifact has not been written yet.
    return {
        **qst6_data,
        "gates": hf5_mod.omit_final_return_qst(qst6_data["gates"]),
        "qst_mode": "pingpong_omit_final_return",
    }


def _bind_random(qc, rng: np.random.Generator) -> dict:
    return {p: float(rng.uniform(-np.pi, np.pi)) for p in qc.parameters}


def _global_phase_align(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(ref)))
    if abs(ref[idx]) < 1e-14 or abs(v[idx]) < 1e-14:
        return v
    return v * (ref[idx] / v[idx])


def test_5qst_json_metadata(qst5_data, qst6_data):
    n_doubles = sum(1 for g in qst6_data["gates"] if g["op"] == "rzx")
    assert sum(1 for g in qst6_data["gates"] if g["op"] == "qst") == 2 * n_doubles
    assert sum(1 for g in qst5_data["gates"] if g["op"] == "qst") == 2 * n_doubles - 1
    assert sum(1 for g in qst5_data["gates"] if g["op"] == "rzx") == n_doubles
    assert qst5_data.get("qst_mode") == "pingpong_omit_final_return"
    # No return QST after the last RZX.
    last_rzx = max(i for i, g in enumerate(qst5_data["gates"]) if g["op"] == "rzx")
    assert not any(
        g["op"] == "qst" and list(g["qubits"]) == [ANCILLA, SENDER]
        for g in qst5_data["gates"][last_rzx + 1 :]
    )
    # Trailing single-qubit cleanup lives on the ancilla, not q2.
    trailing = qst5_data["gates"][last_rzx + 1 :]
    assert any(g["op"] == "h" and g["qubits"] == [ANCILLA] for g in trailing)
    assert any(
        g["op"] == "rx" and g["qubits"] == [ANCILLA] and abs(g["value"] + np.pi / 2) < 1e-12
        for g in trailing
    )
    assert not any(SENDER in g["qubits"] for g in trailing)


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_5qst_matches_6qst_up_to_swap(qst5_data, qst6_data, seed):
    """SWAP(q2,q6) · |ψ_5⟩ == |ψ_6⟩ for random RZX angles."""
    params = qst_mod._params_from_gates(qst6_data["gates"])
    qc5 = qst_mod.qc_from_gates(qst5_data["gates"], num_qubits=N_QUBITS, params=params)
    qc6 = qst_mod.qc_from_gates(qst6_data["gates"], num_qubits=N_QUBITS, params=params)

    rng = np.random.default_rng(seed)
    binding = _bind_random(qc6, rng)
    by_name = {p.name: val for p, val in binding.items()}
    bound5 = qc5.assign_parameters({p: by_name[p.name] for p in qc5.parameters})
    bound6 = qc6.assign_parameters({p: by_name[p.name] for p in qc6.parameters})

    swapped = bound5.copy()
    swapped.append(SwapGate(), [SENDER, ANCILLA])

    v5 = Statevector(swapped).data
    v6 = Statevector(bound6).data
    v5 = _global_phase_align(v5, v6)
    assert np.allclose(v5, v6, atol=ATOL), (
        f"seed={seed}: max |Δ|={np.max(np.abs(v5 - v6))}"
    )


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_5qst_matches_6qst_from_hf_prep(qst5_data, qst6_data, seed):
    """Same check with tapered HF determinant on the data register."""
    params = qst_mod._params_from_gates(qst6_data["gates"])
    qc5 = qst_mod.qc_from_gates(qst5_data["gates"], num_qubits=N_QUBITS, params=params)
    qc6 = qst_mod.qc_from_gates(qst6_data["gates"], num_qubits=N_QUBITS, params=params)

    occupied = qst6_data["hf_occupied_qubits"]
    prep = QuantumCircuit(N_QUBITS)
    for q in occupied:
        prep.x(q)

    rng = np.random.default_rng(2000 + seed)
    binding = _bind_random(qc6, rng)
    by_name = {p.name: val for p, val in binding.items()}
    full5 = prep.compose(
        qc5.assign_parameters({p: by_name[p.name] for p in qc5.parameters})
    )
    full6 = prep.compose(
        qc6.assign_parameters({p: by_name[p.name] for p in qc6.parameters})
    )
    full5.append(SwapGate(), [SENDER, ANCILLA])

    v5 = _global_phase_align(Statevector(full5).data, Statevector(full6).data)
    v6 = Statevector(full6).data
    assert np.allclose(v5, v6, atol=ATOL), (
        f"seed={seed}: max |Δ|={np.max(np.abs(v5 - v6))}"
    )


def test_omit_helper_matches_saved_json(qst6_data):
    """Saved 5-QST JSON (if present) matches ``omit_final_return_qst``."""
    if not QST5_JSON.is_file():
        pytest.skip(f"{QST5_JSON.name} not written yet (run HF_5_QST.py)")
    saved = json.loads(QST5_JSON.read_text(encoding="utf-8"))
    expected = hf5_mod.omit_final_return_qst(qst6_data["gates"])
    assert saved["gates"] == expected
