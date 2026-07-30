"""Reference UCCSD doubles circuits from exploring/ (exact on the HF subspace)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "exploring"))
sys.path.insert(0, str(REPO / "UCCSD circuit"))
sys.path.insert(0, str(REPO / "state_transfer"))

from flexible_compile import (  # noqa: E402
    compile_flexible,
    compile_strings,
    gates_to_qc,
    gen,
    prog_to_gates,
)
from freeze import fully_freeze  # noqa: E402


def compile_fully_frozen(strings, signs=None, fuse_rzx: bool = True):
    """Compile fully Z-frozen weight-4 strings.

    Exploring's graph-aware fan-in forbids Steiner nodes outside the Pauli
    support, so fully-frozen Cl2 strings need the unconstrained row chain.
    """
    n = len(strings[0])
    if signs is None:
        signs = [1] * len(strings)
    prog, expected = [], []
    hub_hint = None
    bridges = []
    for d, s in enumerate(strings):
        prefix, pivot, ph, bridge = compile_flexible(
            s, n, hub_hint=hub_hint, use_cz_graph=False
        )
        hub_hint = pivot
        bridges.append(bridge)
        prog += prefix
        prog.append(("ROT", pivot, d, signs[d] * ph))
        prog += gen._invert(prefix)
        expected.append((s, ph))
    prog = gen._peephole(prog)
    gen._verify_program(prog, n, expected)
    gates = prog_to_gates(prog)
    if fuse_rzx:
        gates = gen.fuse_cz_rot_cz_to_rzx(gates)
    return {"gates": gates, "bridges": bridges, "n_qubits": n}


def _hf_bits_cl2(n: int = 10) -> str:
    # occupied spatial 0..3 both spins on 10q JW layout
    return "".join("1" if q in (0, 1, 2, 3, 5, 6, 7, 8) else "0" for q in range(n))


def prep_hf(n_qubits: int, hf_bits: str) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits)
    for q, b in enumerate(hf_bits):
        if b == "1":
            qc.x(q)
    return qc


def load_hf6q():
    """Tapered HF 6q, 3 doubles — exploring winner is already exact (3 RZX)."""
    import taper_lib

    taper = taper_lib.build_taper_data(n_spatial=4, n_electrons=6)
    doubles = [(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)]
    strings, signs = [], []
    for d in doubles:
        jw = "".join(gen.jw_string_for_double(8, d))
        ts, sg = taper_lib.taper_pauli_string(jw, taper)
        strings.append(ts)
        signs.append(int(sg))
    frozen = fully_freeze(strings)
    ref_full = compile_strings(strings, signs=signs, order="given", fuse=True)
    ref_frozen = compile_fully_frozen(frozen, signs=signs)
    winner = json.loads((REPO / "exploring/HF_6q_disjoint_rzx.json").read_text())
    return {
        "name": "HF_6q",
        "n_qubits": 6,
        "n_params": 3,
        "hf_bits": taper.hf_bitstring_tapered,
        "doubles": doubles,
        "strings": strings,
        "frozen_strings": frozen,
        "signs": signs,
        "gates_full": ref_full["gates"],
        "gates_frozen": ref_frozen["gates"],
        "gates_winner": winner["gates"],
        "winner_rzx": winner["rzx_pairs"],
    }


def load_cl2_10q():
    """Cl2 10q, 2 spin-paired doubles (exploring winner: RZX (2,7)+(0,5))."""
    doubles = [(4, 9, 6, 1), (4, 9, 5, 0)]
    strings = ["".join(gen.jw_string_for_double(10, d)) for d in doubles]
    signs = [1, 1]
    frozen = fully_freeze(strings)
    ref_full = compile_strings(strings, signs=signs, order="given", fuse=True)
    ref_frozen = compile_fully_frozen(frozen, signs=signs)
    winner = json.loads(
        (REPO / "exploring/Cl2_10q_2doubles_disjoint_rzx.json").read_text()
    )
    return {
        "name": "Cl2_10q",
        "n_qubits": 10,
        "n_params": 2,
        "hf_bits": _hf_bits_cl2(10),
        "doubles": doubles,
        "strings": strings,
        "frozen_strings": frozen,
        "signs": signs,
        "gates_full": ref_full["gates"],
        "gates_frozen": ref_frozen["gates"],
        "gates_winner": winner["gates"],
        "winner_rzx": winner["rzx_pairs"],
        "winner_frozen": winner["frozen_strings"],
    }


def target_statevector(gates, n_qubits: int, theta, hf_bits: str):
    from qiskit.quantum_info import Statevector

    qc = prep_hf(n_qubits, hf_bits).compose(gates_to_qc(gates, n_qubits, theta))
    return Statevector.from_instruction(qc)


def sample_thetas(n_params: int, n_samples: int, seed: int = 0, scale: float = 0.3):
    rng = np.random.default_rng(seed)
    return rng.uniform(-scale, scale, size=(n_samples, n_params))
