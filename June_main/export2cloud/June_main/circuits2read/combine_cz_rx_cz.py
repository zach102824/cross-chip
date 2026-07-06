"""
combine_cz_rx_cz.py
===================
Read a bare CZ-form UCCSD-doubles ansatz JSON and fuse every
CZ(a,b) . RX(theta)@q . CZ(a,b)  sandwich into a single continuous-angle
native RZX gate.  Each double contributes one such sandwich on the
alpha<->beta vertical link, so every CZ pair collapses into an RZX gate
while every angle stays symbolic.

Originally written for HF (``HF_8q_3doubles.json``, 8 qubits, 3 doubles); now
generic over any input JSON in this folder.  Output names/titles are derived
from the input file stem, e.g.
    Br2_12q_4doubles.json -> Br2_12q_4doubles_rzx.json (+ _rzx_circuit.png)
    Cl2_10q_3doubles.json -> Cl2_10q_3doubles_rzx.json (+ _rzx_circuit.png)

Why this is exact.  CZ is symmetric and Z-diagonal, so
    CZ(a,b) . exp(-i th/2 X_q) . CZ(a,b)  ==  exp(-i th/2  X_q (x) Z_other)
which is precisely Qiskit's RZX(th) with the Z on ``other`` and the X on ``q``,
i.e. ``rzx(th, other, q)``.

Outputs (same folder, same format/style as the UCCSD generator pipeline):
    <stem>_rzx.json          -- updated logical gate-list JSON
    <stem>_rzx_circuit.png   -- folded-out Qiskit mpl diagram
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO = _THIS_DIR.parent.parent
_UCCSD_DIR = _REPO / "UCCSD circuit"
_GENERATOR_PATH = _UCCSD_DIR / "improved create UCCSD circuit .py"


def _load_module(name: str, path: Path):
    """Import a module from an explicit path (handles filenames with spaces)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module at {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_helpers():
    """Bring in the generator (fusion + drawing) and the I/O helpers."""
    if str(_UCCSD_DIR) not in sys.path:
        sys.path.insert(0, str(_UCCSD_DIR))
    gen = _load_module("improved_create_uccsd_circuit", _GENERATOR_PATH)
    cio = _load_module("uccsd_circuit_io", _UCCSD_DIR / "uccsd_circuit_io.py")
    return gen, cio


def _verify_fusion(gen, bare_gates, fused_gates, num_qubits, seed=0):
    """Confirm the fused circuit equals the bare one up to a global phase
    (random angle binding + random-state evolution)."""
    from qiskit.quantum_info import random_statevector

    qc_bare = gen.qc_from_logical_gates(bare_gates, num_qubits)
    qc_fused = gen.qc_from_logical_gates(fused_gates, num_qubits)
    rng = np.random.default_rng(seed)
    by_name = {p.name: float(rng.uniform(-np.pi, np.pi)) for p in qc_bare.parameters}
    qc_bare = qc_bare.assign_parameters({p: by_name[p.name] for p in qc_bare.parameters})
    qc_fused = qc_fused.assign_parameters({p: by_name[p.name] for p in qc_fused.parameters})
    sv0 = random_statevector(2 ** num_qubits, seed=seed + 1)
    v1 = sv0.evolve(qc_bare).data
    v2 = sv0.evolve(qc_fused).data
    idx = int(np.argmax(np.abs(v1)))
    phase = v1[idx] / v2[idx]
    if not np.allclose(v1, phase * v2, atol=1e-8):
        raise AssertionError("RZX fusion changed the unitary!")


def main(in_json: Path | None = None):
    gen, cio = _load_helpers()

    in_json = Path(in_json) if in_json is not None else _THIS_DIR / "HF_8q_3doubles.json"
    stem = in_json.stem
    data = cio.load_circuit_json(in_json)
    num_qubits = int(data["num_qubits"])
    bare_gates = data["gates"]

    fused_gates = gen.fuse_cz_rot_cz_to_rzx(bare_gates)

    n_cz_in = sum(1 for g in bare_gates if g["op"] == "cz")
    n_cz_out = sum(1 for g in fused_gates if g["op"] == "cz")
    n_rzx = sum(1 for g in fused_gates if g["op"] == "rzx")

    _verify_fusion(gen, bare_gates, fused_gates, num_qubits)

    out_json = _THIS_DIR / f"{stem}_rzx.json"
    cio.save_circuit_json(
        out_json,
        molecule=data["molecule"],
        bond_length=data["bond_length"],
        num_qubits=num_qubits,
        n_spatial=int(data["n_spatial"]),
        n_electrons=int(data["n_electrons"]),
        doubles=data["doubles"],
        signs=data["signs"],
        theta_idx=data["theta_idx"],
        logical_gates=fused_gates,
        init_state=data.get("init_state"),
        beta=data.get("beta"),
    )

    out_png = _THIS_DIR / f"{stem}_rzx_circuit.png"
    gen.save_circuit_diagram(
        fused_gates, num_qubits, out_png, title=f"{stem} (CZ-RX-CZ -> RZX)"
    )

    print(
        f"[{stem}] fused CZ-RX-CZ -> RZX\n"
        f"    in  : {len(bare_gates):3d} gates (CZ={n_cz_in})\n"
        f"    out : {len(fused_gates):3d} gates (CZ={n_cz_out}, RZX={n_rzx})\n"
        f"    wrote {out_json.name} (+ {out_png.name})"
    )
    return out_json, out_png


if __name__ == "__main__":
    if len(sys.argv) > 1:
        inputs = [Path(a) for a in sys.argv[1:]]
    else:
        inputs = [
            _THIS_DIR / "HF_8q_3doubles.json",
            _THIS_DIR / "Cl2_10q_3doubles.json",
            _THIS_DIR / "Br2_12q_4doubles.json",
        ]
    for p in inputs:
        if not p.is_absolute():
            p = _THIS_DIR / p
        if p.exists():
            main(p)
        else:
            print(f"[skip] not found: {p}")
