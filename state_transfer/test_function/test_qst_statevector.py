"""Statevector equivalence: QST ping-pong circuit vs original (no ancilla).

With random RZX angles, the 7-qubit rewritten circuit
(``HF_tapered_6q_3doubles_rzx_qst.json``) must reproduce the same data-qubit
statevector as the original 6-qubit circuit
(``HF_tapered_6q_3doubles_rzx.json``), with ancilla q6 starting (and ending)
in |0>.

Run from the repo root with a Qiskit venv:

    .venv_py311/bin/python -m pytest state_transfer/test_function/test_qst_statevector.py -v
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from conftest import STATE_TRANSFER

# rewrite_hf_tapered_with_qst lives next to circuits2read; conftest already
# puts STATE_TRANSFER on sys.path.
import rewrite_hf_tapered_with_qst as qst_mod  # noqa: E402

ORIG_JSON = STATE_TRANSFER / "circuits2read" / "HF_tapered_6q_3doubles_rzx.json"
QST_JSON = STATE_TRANSFER / "circuits2read" / "HF_tapered_6q_3doubles_rzx_qst.json"

ANCILLA = 6
N_DATA = 6
N_TRIALS = 5
ATOL = 1e-10


@pytest.fixture(scope="module")
def orig_data():
    if not ORIG_JSON.is_file():
        pytest.skip(f"{ORIG_JSON.name} missing (run generate_tapered_circuits.py)")
    return json.loads(ORIG_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def qst_data():
    if not QST_JSON.is_file():
        pytest.skip(f"{QST_JSON.name} missing (run rewrite_hf_tapered_with_qst.py)")
    return json.loads(QST_JSON.read_text(encoding="utf-8"))


def _bind_random(qc, rng: np.random.Generator) -> dict:
    """Return {Parameter: float} with independent uniform angles in (-pi, pi)."""
    return {p: float(rng.uniform(-np.pi, np.pi)) for p in qc.parameters}


def _global_phase_align(v: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Multiply v by a global phase so the largest-|ref| entry matches."""
    idx = int(np.argmax(np.abs(ref)))
    if abs(ref[idx]) < 1e-14 or abs(v[idx]) < 1e-14:
        return v
    return v * (ref[idx] / v[idx])


def _data_amplitudes_ancilla_zero(v7: np.ndarray) -> np.ndarray:
    """Extract 6q amplitudes with ancilla q6 = |0> (Qiskit little-endian).

    Qubit 6 is the MSB, so ancilla-|0> lives in indices 0 .. 2^6 - 1.
    """
    return v7[: 2**N_DATA]


def test_qst_json_metadata(qst_data, orig_data):
    assert qst_data["num_qubits"] == N_DATA + 1
    assert qst_data["ancilla"] == ANCILLA
    assert qst_data["qst_mode"] == "pingpong"
    assert qst_data["local_rzx_pair"] == [5, 6]
    assert qst_data["qst_pair"] == [2, 6]
    assert qst_data["bridge_pair"] == orig_data["bridge_pair"]
    n_doubles = len(orig_data["doubles"])
    assert sum(1 for g in qst_data["gates"] if g["op"] == "rzx") == n_doubles
    assert sum(1 for g in qst_data["gates"] if g["op"] == "qst") == 2 * n_doubles


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_qst_matches_original_statevector_random_angles(orig_data, qst_data, seed):
    """Same random angles → identical data statevectors (ancilla ends in |0>)."""
    params = qst_mod._params_from_gates(qst_data["gates"])
    qc_orig = qst_mod.qc_from_gates(orig_data["gates"], num_qubits=N_DATA, params=params)
    qc_qst = qst_mod.qc_from_gates(qst_data["gates"], num_qubits=N_DATA + 1, params=params)

    rng = np.random.default_rng(seed)
    binding = _bind_random(qc_orig, rng)
    # Map by parameter name so both circuits share identical angle values.
    by_name = {p.name: val for p, val in binding.items()}
    bound_orig = qc_orig.assign_parameters({p: by_name[p.name] for p in qc_orig.parameters})
    bound_qst = qc_qst.assign_parameters({p: by_name[p.name] for p in qc_qst.parameters})

    # Default initial state |0...0> (data) and |0> on ancilla.
    v_orig = Statevector(bound_orig).data
    v_qst = Statevector(bound_qst).data

    # Ancilla must return to |0>.
    leaked = np.linalg.norm(v_qst[2**N_DATA :])
    assert leaked < ATOL, f"ancilla not back in |0> (leak norm={leaked})"

    v_data = _data_amplitudes_ancilla_zero(v_qst)
    v_data = _global_phase_align(v_data, v_orig)
    assert np.allclose(v_data, v_orig, atol=ATOL), (
        f"seed={seed}: max |Δ|={np.max(np.abs(v_data - v_orig))}"
    )


@pytest.mark.parametrize("seed", range(N_TRIALS))
def test_qst_matches_original_from_hf_prep(orig_data, qst_data, seed):
    """Same check starting from the tapered HF determinant ⊗ |0>_ancilla."""
    from qiskit import QuantumCircuit

    params = qst_mod._params_from_gates(qst_data["gates"])
    qc_orig = qst_mod.qc_from_gates(orig_data["gates"], num_qubits=N_DATA, params=params)
    qc_qst = qst_mod.qc_from_gates(qst_data["gates"], num_qubits=N_DATA + 1, params=params)

    occupied = orig_data["hf_occupied_qubits"]
    prep6 = QuantumCircuit(N_DATA)
    for q in occupied:
        prep6.x(q)
    prep7 = QuantumCircuit(N_DATA + 1)
    for q in occupied:
        prep7.x(q)

    rng = np.random.default_rng(1000 + seed)
    binding = _bind_random(qc_orig, rng)
    by_name = {p.name: val for p, val in binding.items()}
    full_orig = prep6.compose(
        qc_orig.assign_parameters({p: by_name[p.name] for p in qc_orig.parameters})
    )
    full_qst = prep7.compose(
        qc_qst.assign_parameters({p: by_name[p.name] for p in qc_qst.parameters})
    )

    v_orig = Statevector(full_orig).data
    v_qst = Statevector(full_qst).data

    leaked = np.linalg.norm(v_qst[2**N_DATA :])
    assert leaked < ATOL, f"ancilla not back in |0> (leak norm={leaked})"

    v_data = _global_phase_align(_data_amplitudes_ancilla_zero(v_qst), v_orig)
    assert np.allclose(v_data, v_orig, atol=ATOL), (
        f"seed={seed}: max |Δ|={np.max(np.abs(v_data - v_orig))}"
    )
