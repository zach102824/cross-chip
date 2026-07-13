"""Tests for the saved tapered RZX circuit (state_transfer/circuits2read).

Checks that the compiled circuit is doing exactly what it is expected to:

1. STRUCTURE: one parameterized RZX per double, ALL on the same qubit pair;
   every other gate is a local h / cz / rx (same schema as
   June_main/circuits2read/HF_8q_3doubles_rzx.json, on 6 qubits).
2. UNITARY: the ansatz equals the product of Pauli exponentials
   prod_k exp(-i sign_k * t_k / 2 * P~_k) at random angles (up to global phase).
3. ENERGY: with the tapered-HF X-layer prep,
   - theta = 0 reproduces the RHF determinant energy of the tapered Hamiltonian,
   - the optimized energy matches the pre-taper (full-register) single-string
     ansatz optimum (tapering costs no accuracy), and
   - the optimized energy is within chemical accuracy of the tapered FCI energy.
4. REPRODUCIBILITY: regenerating with generate_tapered_circuits.run() yields
   the same gate list as the saved JSON (also re-runs the internal fusion check).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from openfermion.linalg import get_sparse_operator
from scipy.linalg import expm
from scipy.optimize import minimize

from conftest import STATE_TRANSFER, hf_state_vector

CIRCUIT_JSON = STATE_TRANSFER / "circuits2read" / "HF_tapered_6q_3doubles_rzx.json"

HF_DOUBLES = [(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)]
CHEMICAL_ACCURACY_HA = 1.6e-3
OPTIMUM_TOL = 1e-6


@pytest.fixture(scope="module")
def circuit_data():
    if not CIRCUIT_JSON.is_file():
        pytest.skip(f"{CIRCUIT_JSON} not generated yet (run generate_tapered_circuits.py)")
    return json.loads(CIRCUIT_JSON.read_text(encoding="utf-8"))


def _pauli_matrix_openfermion_order(string, n_qubits, taper_lib):
    """Dense matrix of a Pauli string in openfermion ordering (qubit 0 = MSB)."""
    return get_sparse_operator(
        taper_lib.pauli_string_to_qubit_operator(string), n_qubits=n_qubits
    ).toarray()


def _qiskit_statevector_in_openfermion_order(qc):
    """Statevector of a Qiskit circuit re-indexed to openfermion bit order.

    Qiskit is little-endian (qubit 0 = least significant bit); openfermion's
    get_sparse_operator treats qubit 0 as the MOST significant bit, so the
    amplitude indices are bit-reversed relative to each other.
    """
    from qiskit.quantum_info import Statevector

    data = Statevector(qc).data
    n = qc.num_qubits
    out = np.empty_like(data)
    for index in range(len(data)):
        reversed_index = int(format(index, f"0{n}b")[::-1], 2)
        out[reversed_index] = data[index]
    return out


def test_structure_fixed_pair_rzx_rest_local(circuit_data):
    gates = circuit_data["gates"]
    n_doubles = len(circuit_data["doubles"])

    rzx_gates = [g for g in gates if g["op"] == "rzx"]
    assert len(rzx_gates) == n_doubles, "expected exactly one RZX per double"

    pairs = {tuple(sorted(g["qubits"])) for g in rzx_gates}
    assert len(pairs) == 1, f"RZX gates are not all on the same pair: {pairs}"
    assert pairs == {tuple(sorted(circuit_data["bridge_pair"]))}

    # Each RZX carries its own symbolic parameter (t0, t1, ...). Note the JSON's
    # top-level "param_names" is empty by design (save_circuit_json only scans
    # rx/rz gates; consumers recover names from gate "param" refs), so compare
    # against the expected t<i> set instead.
    assert sorted(g["param"] for g in rzx_gates) == [f"t{i}" for i in range(n_doubles)]

    # Everything else is a local single-qubit gate or a plain CZ.
    for g in gates:
        if g["op"] == "rzx":
            continue
        assert g["op"] in ("h", "cz", "rx"), f"unexpected op {g['op']!r}"
        assert len(g["qubits"]) in (1, 2)

    assert circuit_data["num_qubits"] == circuit_data["n_qubits_full"] - 2


def test_circuit_equals_product_of_pauli_exponentials(circuit_data, taper_lib, gen):
    """Bind random angles; the ansatz statevector (from |HF~>) must equal the
    product of exp(-i sign*t/2 P~) applied to |HF~>, up to a global phase."""
    from qiskit import QuantumCircuit

    n_qubits = circuit_data["num_qubits"]
    strings = circuit_data["tapered_strings"]
    signs = circuit_data["signs"]
    theta_order = circuit_data["theta_idx"]  # rotation order in the circuit
    occupied = circuit_data["hf_occupied_qubits"]

    rng = np.random.default_rng(7)
    thetas = rng.uniform(-np.pi, np.pi, size=len(strings))

    ansatz = gen.qc_from_logical_gates(circuit_data["gates"], n_qubits)
    binding = {p: float(thetas[int(p.name[1:])]) for p in ansatz.parameters}
    prep = QuantumCircuit(n_qubits)
    for q in occupied:
        prep.x(q)
    full = prep.compose(ansatz.assign_parameters(binding))
    v_circuit = _qiskit_statevector_in_openfermion_order(full)

    v_expected = hf_state_vector(n_qubits, occupied)
    for d in theta_order:
        pauli = _pauli_matrix_openfermion_order(strings[d], n_qubits, taper_lib)
        v_expected = expm(-0.5j * float(thetas[d]) * float(signs[d]) * pauli) @ v_expected

    overlap = abs(np.vdot(v_expected, v_circuit))
    assert overlap > 1 - 1e-9, f"|<expected|circuit>| = {overlap} (should be 1)"


def test_circuit_energy_vs_pre_taper_and_fci(circuit_data, taper_lib, taper, gen, hamiltonians):
    """theta=0 gives the RHF energy; the optimized circuit energy equals the
    pre-taper full-register optimum and is within chemical accuracy of FCI."""
    from qiskit import QuantumCircuit

    bond = float(circuit_data["bond_length"])
    assert bond in hamiltonians, f"add bond {bond} to conftest.TEST_BONDS"
    full_op, tapered_op = hamiltonians[bond]

    n_qubits = circuit_data["num_qubits"]
    occupied = circuit_data["hf_occupied_qubits"]
    h_tap = get_sparse_operator(tapered_op, n_qubits=n_qubits).toarray()

    ansatz = gen.qc_from_logical_gates(circuit_data["gates"], n_qubits)
    params = sorted(ansatz.parameters, key=lambda p: p.name)
    prep = QuantumCircuit(n_qubits)
    for q in occupied:
        prep.x(q)

    def energy(x):
        bound = prep.compose(
            ansatz.assign_parameters({p: float(v) for p, v in zip(params, x)})
        )
        v = _qiskit_statevector_in_openfermion_order(bound)
        return float(np.real(v.conj() @ h_tap @ v))

    # theta = 0 -> RHF determinant energy of the tapered Hamiltonian.
    psi_hf = hf_state_vector(n_qubits, occupied)
    e_rhf = float(np.real(psi_hf.conj() @ h_tap @ psi_hf))
    assert abs(energy(np.zeros(len(params))) - e_rhf) < 1e-9

    # Optimized circuit energy.
    res = minimize(energy, np.zeros(len(params)), method="BFGS", options={"gtol": 1e-9})
    e_circuit = float(res.fun)

    # Pre-taper reference: optimize the SAME doubles on the FULL 8-qubit register.
    n_full = taper.n_qubits_full
    h_full = get_sparse_operator(full_op, n_qubits=n_full).toarray()
    occ_full = [q for q, b in enumerate(taper.hf_bitstring_full) if b == "1"]
    psi_full = hf_state_vector(n_full, occ_full)
    full_strings = ["".join(gen.jw_string_for_double(n_full, d)) for d in HF_DOUBLES]
    paulis_full = [
        _pauli_matrix_openfermion_order(s, n_full, taper_lib) for s in full_strings
    ]

    def energy_full(x):
        v = psi_full.copy()
        for theta, pauli in zip(x, paulis_full):
            v = expm(-0.5j * float(theta) * pauli) @ v
        return float(np.real(v.conj() @ h_full @ v))

    res_full = minimize(
        energy_full, np.zeros(len(HF_DOUBLES)), method="BFGS", options={"gtol": 1e-9}
    )
    e_pre_taper = float(res_full.fun)

    assert abs(e_circuit - e_pre_taper) < OPTIMUM_TOL, (
        f"tapered circuit optimum {e_circuit:.10f} != pre-taper optimum {e_pre_taper:.10f}"
    )

    e_fci = float(np.linalg.eigvalsh(h_tap)[0].real)
    assert e_circuit - e_fci < CHEMICAL_ACCURACY_HA, (
        f"circuit optimum is {(e_circuit - e_fci) * 1000:.3f} mHa above FCI"
    )


def test_regeneration_reproduces_saved_circuit(circuit_data, tmp_path):
    """generate_tapered_circuits.run('HF') into a temp dir must reproduce the
    committed JSON gate list (and re-runs its internal RZX fusion verification)."""
    import generate_tapered_circuits as gtc

    result = gtc.run("HF", out_dir=tmp_path)
    regenerated = json.loads(result["json"].read_text(encoding="utf-8"))
    assert regenerated["gates"] == circuit_data["gates"]
    assert regenerated["tapered_strings"] == circuit_data["tapered_strings"]
    assert regenerated["bridge_pair"] == circuit_data["bridge_pair"]
