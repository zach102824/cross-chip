"""
test_cl2_against_formula.py
===========================
Validate the Cl2 device-mapped UCCSD-doubles circuit produced by
"improved create UCCSD circuit_Cl2.py" against the exact mathematical object it
is supposed to implement: the ordered product of single-Pauli rotations

        U  =  prod_k  exp(-i * signs[k] * theta[theta_idx[k]] / 2 * P_k)

where P_k = strings[k] is the Jordan-Wigner Pauli string of double k (the
generator's documented contract).  We bind random angles, evolve a random
statevector through both the circuit and the formula, and require equality up to
a global phase.  We test BOTH circuit forms:

    * the abstract fused circuit (continuous-theta RZX), and
    * the device-mapped circuit (nearest-neighbour on M1+M2+M3, bridge CZ -> RZX),

and we additionally assert the device circuit is genuinely nearest-neighbour and
uses 58 two-qubit gates under the rule that same-chip RZX is emitted as
CZ.RX(theta).CZ.

Run directly (python test_cl2_against_formula.py) or under pytest.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector, random_statevector

# ----------------------------------------------------------------------
# Locate and import the (space-named) Cl2 generator + its IO helper
# ----------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_GEN_DIR = _THIS.parent.parent                       # final_version_generate_circuit/
_PKG_DIR = _GEN_DIR.parent                           # UCCSD circuit/
_GEN_PATH = _GEN_DIR / "improved create UCCSD circuit_Cl2.py"

if str(_PKG_DIR) not in sys.path:
    sys.path.insert(0, str(_PKG_DIR))
import uccsd_circuit_io as cio                        # noqa: E402


def _load_generator():
    spec = importlib.util.spec_from_file_location("cl2_generator", str(_GEN_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GEN = _load_generator()

# Cl2: 12 qubits, top-5 doubles (UCCSD_Mole/Cl2.ipynb), matching the generator.
DOUBLES = [(5, 11, 8, 2), (5, 11, 7, 1), (5, 11, 9, 3),
           (5, 11, 10, 4), (5, 11, 6, 0)]
N = 12


# ----------------------------------------------------------------------
# Build the circuits exactly as the generator does
# ----------------------------------------------------------------------
def _build():
    thetas = [Parameter(f"t{d}") for d in range(len(DOUBLES))]
    qc, strings, signs, theta_idx = GEN.create_uccsd_circuit(
        N, DOUBLES, thetas=thetas, optimize=True, order="auto", init_state=None)
    bare = cio.circuit_to_logical_gates(qc, N)
    fused = GEN.fuse_cz_rot_cz_to_rzx(bare)
    phi = GEN.embed_to_device(fused, N, GEN.DEVICE_12Q)
    dgates, n_cz, n_rzx = GEN.device_gates(fused, phi, GEN.DEVICE_12Q)
    return dict(fused=fused, dgates=dgates, strings=strings, signs=signs,
                theta_idx=theta_idx, phi=phi, n_cz=n_cz, n_rzx=n_rzx)


# ----------------------------------------------------------------------
# Reference: ordered product of exp(-i sign*theta/2 P) on a statevector
# ----------------------------------------------------------------------
def _apply_pauli_exp(psi, pauli_label, t):
    """exp(-i t P) |psi>  =  cos(t)|psi> - i sin(t) P|psi>  (exact; P^2 = I)."""
    P = SparsePauliOp.from_list([(pauli_label, 1.0)]).to_matrix(sparse=True)
    v = psi.data
    return Statevector(np.cos(t) * v - 1j * np.sin(t) * (P @ v))


def _formula_state(psi0, strings, signs, theta_idx, values):
    """Apply the rotation blocks in circuit order (block 0 first)."""
    psi = psi0
    for k, s in enumerate(strings):
        # qiskit Pauli labels are written qubit (N-1) .. 0; strings are q0..qN-1.
        label = s[::-1]
        t = signs[k] * values[theta_idx[k]] / 2.0
        psi = _apply_pauli_exp(psi, label, t)
    return psi


def _equal_up_to_phase(v1, v2, atol=1e-8):
    idx = int(np.argmax(np.abs(v1)))
    if abs(v2[idx]) < 1e-12:
        return False
    phase = v1[idx] / v2[idx]
    return bool(np.allclose(v1, phase * v2, atol=atol))


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_device_circuit_is_nearest_neighbour():
    b = _build()
    edges = {frozenset(e) for e in GEN.DEVICE_12Q["cz"]} | \
            {frozenset(e) for e in GEN.DEVICE_12Q["rzx"]}
    phi = b["phi"]
    for g in b["dgates"]:
        if g["op"] in ("cz", "rzx") and len(g["qubits"]) == 2:
            a, c = g["qubits"]
            assert frozenset((phi[a], phi[c])) in edges, \
                f"{g['op']} on logical {(a, c)} is not a device edge"


def test_device_two_qubit_count():
    b = _build()
    assert b["n_cz"] + b["n_rzx"] == 58, (b["n_cz"], b["n_rzx"])


def test_fused_matches_formula():
    b = _build()
    qc = GEN.qc_from_logical_gates(b["fused"], N)
    _check_against_formula(qc, b, seeds=range(4))


def test_device_matches_formula():
    b = _build()
    qc = GEN.qc_from_logical_gates(b["dgates"], N)
    _check_against_formula(qc, b, seeds=range(4))


def _check_against_formula(qc, b, seeds):
    for seed in seeds:
        rng = np.random.default_rng(seed)
        values = [float(rng.uniform(-np.pi, np.pi)) for _ in range(len(DOUBLES))]
        bound = qc.assign_parameters(
            {p: values[int(p.name[1:])] for p in qc.parameters})
        psi0 = random_statevector(2 ** N, seed=seed + 100)
        v_circ = psi0.evolve(bound).data
        v_form = _formula_state(psi0, b["strings"], b["signs"],
                                b["theta_idx"], values).data
        assert _equal_up_to_phase(v_form, v_circ), f"mismatch at seed {seed}"


if __name__ == "__main__":
    info = _build()
    print(f"Cl2 device circuit: 2q = {info['n_cz'] + info['n_rzx']} "
          f"(CZ={info['n_cz']}, RZX={info['n_rzx']})")
    print("placement q_i -> Q:",
          ["Q%02d" % (info["phi"][i] + 1) for i in range(N)])
    test_device_circuit_is_nearest_neighbour()
    print("[ok] device circuit is nearest-neighbour on M1+M2+M3")
    test_device_two_qubit_count()
    print("[ok] device circuit uses 58 two-qubit gates")
    test_fused_matches_formula()
    print("[ok] abstract fused circuit == product of Pauli exponentials")
    test_device_matches_formula()
    print("[ok] device-mapped circuit == product of Pauli exponentials")
    print("ALL CHECKS PASSED")
