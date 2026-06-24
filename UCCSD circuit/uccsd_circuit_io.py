"""
uccsd_circuit_io.py
===================
Bridge between the Qiskit UCCSD generator ("improved create UCCSD circuit .py")
and the Cirq noisy-simulation notebook (June_main/main.ipynb).

Responsibilities
----------------
1.  Build a UCCSD circuit with the generator (Qiskit), for a chosen molecule.
2.  Classify every CZ as on-chip or *cross-chip* using the 2-spatial-orbital
    chip layout (see ``chip_of`` / ``is_cross_chip``).
3.  Serialise the circuit to a small JSON gate-list (``save_circuit_json``).
    The JSON keeps the LOGICAL gates (h / rx / cz / ry / cx / x) with a
    ``cross_chip`` flag on every CZ, plus the metadata the notebook needs to
    map its sympy symbols onto the RX angles (``doubles``, ``signs``,
    ``theta_idx``, ``param_names``).
4.  Provide the cross-chip CZ -> RZX decomposition used by
    "simplify decomposed circuit.py" for real-device mapping.

Qubit layout (paper / generator convention): spin-up orbitals on 0..N/2-1,
spin-down orbitals on N/2..N-1.  Chips hold 2 spatial orbitals each (4 qubits:
2 up + 2 down):  ``chip(q) = (q % (N//2)) // 2``.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

# Tag used (in the Cirq notebook) for cross-chip CZs that get the higher
# depolarising probability, and stored in the JSON gate-list.
CZ_CROSS_CHIP_TAG = "cz_cross_chip"

_THIS_DIR = Path(__file__).resolve().parent
_GENERATOR_PATH = _THIS_DIR / "improved create UCCSD circuit .py"


# ----------------------------------------------------------------------
# Load the space-named generator module
# ----------------------------------------------------------------------
def load_generator():
    """Import "improved create UCCSD circuit .py" (a filename with spaces)."""
    spec = importlib.util.spec_from_file_location(
        "improved_create_uccsd_circuit", str(_GENERATOR_PATH)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generator at {_GENERATOR_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------
# Chip layout / cross-chip classification
# ----------------------------------------------------------------------
def chip_of(q: int, num_qubits: int) -> int:
    """Chip index of qubit ``q``: chips hold 2 spatial orbitals (4 qubits)."""
    half = num_qubits // 2
    return (q % half) // 2


def is_cross_chip(a: int, b: int, num_qubits: int) -> bool:
    """True when a CZ on qubits (a, b) spans two different chips."""
    return chip_of(a, num_qubits) != chip_of(b, num_qubits)


# ----------------------------------------------------------------------
# Build the UCCSD circuit via the generator
# ----------------------------------------------------------------------
def build_uccsd_circuit(
    num_qubits: int,
    doubles,
    *,
    thetas=None,
    optimize: bool = True,
    order: str = "auto",
    pair: bool = False,
    init_state=None,
    n_electrons=None,
    occupied=None,
    beta=None,
):
    """Thin wrapper around generator.create_uccsd_circuit.

    Returns (qiskit QuantumCircuit, strings, signs, theta_idx).
    """
    gen = load_generator()
    return gen.create_uccsd_circuit(
        num_qubits,
        list(doubles),
        thetas=thetas,
        optimize=optimize,
        order=order,
        pair=pair,
        init_state=init_state,
        n_electrons=n_electrons,
        occupied=occupied,
        beta=beta,
    )


# ----------------------------------------------------------------------
# Qiskit circuit -> logical gate list
# ----------------------------------------------------------------------
def _angle_param_info(angle):
    """Return (param_name, coeff) for a (possibly symbolic) Qiskit angle.

    Numeric angle -> (None, float(angle)).
    Linear ParameterExpression  coeff * p  -> (p.name, coeff).
    """
    try:
        return None, float(angle)
    except (TypeError, RuntimeError):
        pass
    params = list(getattr(angle, "parameters", []))
    if not params:
        return None, float(angle)
    if len(params) != 1:
        raise ValueError(f"Unsupported multi-parameter angle: {angle!r}")
    p = params[0]
    v1 = float(angle.bind({p: 1.0}))
    v0 = float(angle.bind({p: 0.0}))
    if abs(v0) > 1e-12:
        raise ValueError(f"Non-linear / offset angle not supported: {angle!r}")
    return p.name, v1 - v0


def circuit_to_logical_gates(qc, num_qubits: int) -> list[dict]:
    """Flatten a Qiskit UCCSD circuit into a JSON-friendly gate list.

    Each entry: {"op", "qubits", ...}.  CZ entries carry "cross_chip": bool.
    RX entries carry either {"param": name, "coeff": c} or {"value": float}.
    RY entries carry {"value": float}.  Barriers are dropped (visual only).
    """
    gates: list[dict] = []
    for inst in qc.data:
        op = inst.operation
        name = op.name
        qs = [qc.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "cz":
            a, b = qs
            gates.append(
                {
                    "op": "cz",
                    "qubits": [a, b],
                    "cross_chip": bool(is_cross_chip(a, b, num_qubits)),
                }
            )
        elif name == "h":
            gates.append({"op": "h", "qubits": qs})
        elif name == "x":
            gates.append({"op": "x", "qubits": qs})
        elif name == "cx":
            gates.append({"op": "cx", "qubits": qs})
        elif name == "rx":
            pname, coeff = _angle_param_info(op.params[0])
            entry = {"op": "rx", "qubits": qs}
            if pname is None:
                entry["value"] = float(coeff)
            else:
                entry["param"] = pname
                entry["coeff"] = float(coeff)
            gates.append(entry)
        elif name == "ry":
            gates.append({"op": "ry", "qubits": qs, "value": float(op.params[0])})
        elif name == "rz":
            pname, coeff = _angle_param_info(op.params[0])
            entry = {"op": "rz", "qubits": qs}
            if pname is None:
                entry["value"] = float(coeff)
            else:
                entry["param"] = pname
                entry["coeff"] = float(coeff)
            gates.append(entry)
        else:
            raise ValueError(f"Unhandled Qiskit gate in UCCSD circuit: {name!r}")
    return gates


# ----------------------------------------------------------------------
# Cross-chip CZ -> RZX decomposition (real-device native form)
# ----------------------------------------------------------------------
def decompose_cross_chip_cz(control: int, target: int) -> list[dict]:
    """CZ(control, target) in the cross-resonance native form

        CZ = (Rz_c(-pi/2) (x) H . Rx_t(-pi/2)) . RZX(pi/2) . (I (x) H_t)

    Returned as a time-ordered gate list (first gate applied first).  All
    entries carry ``cross_chip=True``; the RZX is the (noisy) two-qubit gate.
    """
    return [
        {"op": "h", "qubits": [target], "cross_chip": True},
        {"op": "rzx", "qubits": [control, target], "angle": float(np.pi / 2), "cross_chip": True},
        {"op": "rz", "qubits": [control], "value": float(-np.pi / 2), "cross_chip": True},
        {"op": "rx", "qubits": [target], "value": float(-np.pi / 2), "cross_chip": True},
        {"op": "h", "qubits": [target], "cross_chip": True},
    ]


# ----------------------------------------------------------------------
# Serialisation
# ----------------------------------------------------------------------
def save_circuit_json(
    path,
    *,
    molecule: str,
    bond_length: float,
    num_qubits: int,
    n_spatial: int,
    n_electrons: int,
    doubles,
    signs,
    theta_idx,
    logical_gates: list[dict],
    init_state=None,
    beta=None,
    extra: dict | None = None,
) -> Path:
    """Write the canonical logical-circuit JSON consumed by main.ipynb."""
    param_names = sorted(
        {g["param"] for g in logical_gates if g["op"] in ("rx", "rz") and "param" in g},
        key=lambda s: (len(s), s),
    )
    payload = {
        "molecule": molecule,
        "bond_length": float(bond_length),
        "num_qubits": int(num_qubits),
        "n_spatial": int(n_spatial),
        "n_electrons": int(n_electrons),
        "init_state": init_state,
        "beta": None if beta is None else float(beta),
        "doubles": [list(map(int, d)) for d in doubles],
        "signs": [float(s) for s in signs],
        "theta_idx": [int(i) for i in theta_idx],
        "param_names": param_names,
        "gates": logical_gates,
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_circuit_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# High-level convenience used by the molecule notebooks
# ----------------------------------------------------------------------
def build_and_save(
    *,
    molecule: str,
    bond_length: float,
    num_qubits: int,
    doubles,
    n_electrons: int,
    out_dir,
    init_state="multiref",
    beta=None,
    order: str = "auto",
    pair: bool = False,
    filename: str | None = None,
) -> Path:
    """Build the UCCSD circuit and write circuits2read/<molecule>_bond_<d>.json."""
    n_spatial = num_qubits // 2
    qc, strings, signs, theta_idx = build_uccsd_circuit(
        num_qubits,
        doubles,
        thetas=None,
        optimize=True,
        order=order,
        pair=pair,
        init_state=init_state,
        n_electrons=n_electrons,
        beta=beta,
    )
    logical_gates = circuit_to_logical_gates(qc, num_qubits)
    if filename is None:
        filename = f"{molecule}_bond_{bond_length:.1f}.json"
    out_path = Path(out_dir) / filename
    save_circuit_json(
        out_path,
        molecule=molecule,
        bond_length=bond_length,
        num_qubits=num_qubits,
        n_spatial=n_spatial,
        n_electrons=n_electrons,
        doubles=doubles,
        signs=signs,
        theta_idx=theta_idx,
        logical_gates=logical_gates,
        init_state=init_state,
        beta=beta,
    )
    n_cz = sum(1 for g in logical_gates if g["op"] == "cz")
    n_cross = sum(1 for g in logical_gates if g["op"] == "cz" and g["cross_chip"])
    print(
        f"Saved {out_path}  (qubits={num_qubits}, CZ={n_cz}, cross-chip CZ={n_cross}, "
        f"params={len(set(theta_idx))})"
    )
    return out_path


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import cirq

    # 1) Verify the cross-chip CZ decomposition equals CZ up to global phase.
    c, t = cirq.LineQubit.range(2)
    ZX = np.kron(np.array([[1, 0], [0, -1]]), np.array([[0, 1], [1, 0]]))
    rzx_u = lambda ang: cirq.MatrixGate(
        __import__("scipy.linalg", fromlist=["expm"]).expm(-0.5j * ang * ZX),
        name="RZX",
    )
    name_to_op = {
        "h": lambda qs, g: cirq.H(c if qs[0] == 0 else t),
        "rz": lambda qs, g: cirq.rz(g["value"]).on(c if qs[0] == 0 else t),
        "rx": lambda qs, g: cirq.rx(g["value"]).on(c if qs[0] == 0 else t),
        "rzx": lambda qs, g: rzx_u(g["angle"]).on(c, t),
    }
    dec = decompose_cross_chip_cz(0, 1)
    circ = cirq.Circuit(name_to_op[g["op"]](g["qubits"], g) for g in dec)
    u = circ.unitary(qubit_order=[c, t])
    cz = cirq.unitary(cirq.CZ)
    ratio = u / cz
    diag = np.diagonal(ratio)
    ok = np.allclose(u, cz * diag[0], atol=1e-9) and np.allclose(diag, diag[0], atol=1e-9)
    print(f"RZX decomposition == CZ up to global phase: {ok}  (phase={diag[0]:.3f})")
    assert ok, "cross-chip CZ decomposition is NOT equivalent to CZ"

    # 2) Build + save the default HF circuit (8 qubits, three doubles).
    repo = _THIS_DIR.parent
    out = build_and_save(
        molecule="HF",
        bond_length=1.4,
        num_qubits=8,
        doubles=[(7, 3, 4, 0), (7, 3, 5, 1), (7, 3, 6, 2)],
        n_electrons=6,
        out_dir=repo / "June_main" / "circuits2read",
        init_state="multiref",
        beta=0.1,
    )
    data = load_circuit_json(out)
    print(f"param_names={data['param_names']}  signs={data['signs']}  theta_idx={data['theta_idx']}")
