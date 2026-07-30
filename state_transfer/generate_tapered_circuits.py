#!/usr/bin/env python3
"""Generate tapered UCCSD-doubles circuits with fixed-pair long-range RZX.

Tapered counterpart of ``UCCSD circuit/improved create UCCSD circuit .py``.
Each double's Jordan-Wigner representative string is tapered to the reduced
register via :mod:`taper_lib`, then compiled with the SAME two-row hub layout
used by the full-register generator.  Because the two spin-block parities remove
one qubit from each block, the reduced register keeps its two-row structure
(alpha on ``0..half-1``, beta on ``half..n-1``), so every tapered double still
sandwiches a single CZ on one shared vertical bridge pair.  Fusing
``CZ . RX(theta) . CZ`` into a native ``RZX`` then yields the target structure:

    one long-range RZX per double, all on the SAME qubit pair, everything else
    local single-/two-qubit gates -- exactly like
    ``June_main/circuits2read/HF_8q_3doubles_rzx.json`` but on fewer qubits.

Defaults to HF (8 -> 6 qubits, 3 doubles).  Cl2 / Br2 / H4 entries live in
``CASES``.  Outputs go to
``state_transfer/circuits2read/<mol>_tapered_<n>q_<k>doubles_rzx.json`` (+ PNG).
"""

from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
_UCCSD_DIR = _REPO_ROOT / "UCCSD circuit"
_GENERATOR_PATH = _UCCSD_DIR / "improved create UCCSD circuit .py"

for _p in (str(_THIS_DIR), str(_UCCSD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import taper_lib  # noqa: E402


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "improved_create_uccsd_circuit", str(_GENERATOR_PATH)
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generator at {_GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gen = _load_generator()
import uccsd_circuit_io as cio  # noqa: E402  (lives in UCCSD circuit/)

from qiskit import QuantumCircuit  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402


# ----------------------------------------------------------------------
# Per-molecule configuration (default HF active; Cl2/Br2 stubbed for later)
# ----------------------------------------------------------------------
CASES = {
    "HF": dict(
        molecule="HF",
        bond_length=1.0,
        n_qubits_full=8,
        n_electrons=6,
        # UCCSD_Mole/HF.ipynb, active_space=(6, 4) -> 8 qubits, top-3 doubles;
        # the shared-creation set (N/2-1, N-1, k+N/2, k) = (3, 7, k+4, k).
        doubles=[(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)],
        pair=False,
        order="given",
        hub0=None,  # auto-detected shared pivot (== 2 for HF)
    ),
    "Cl2": dict(
        molecule="Cl2",
        # Matches June_main/export2cloud/main_Cl2.py (CL2_BOND_LENGTH default).
        bond_length=2.2,
        n_qubits_full=10,
        n_electrons=8,
        # UCCSD_Mole/Cl2.ipynb, active_space=(8, 5) -> 10 qubits, top-3 doubles.
        doubles=[(4, 9, 6, 1), (4, 9, 5, 0), (4, 9, 7, 2)],
        pair=False,
        order="given",
        hub0=None,
    ),
    "Br2": dict(
        molecule="Br2",
        # Matches June_main/export2cloud/main_Br2.py (BR2_BOND_LENGTH default).
        bond_length=2.2,
        n_qubits_full=12,
        n_electrons=10,
        # UCCSD_Mole/Br2.ipynb, active_space=(10, 6) -> 12 qubits, top-4 doubles.
        doubles=[(5, 11, 6, 0), (5, 11, 10, 4), (5, 11, 9, 3), (5, 11, 8, 2)],
        pair=False,
        order="given",
        hub0=None,
    ),
    "H4": dict(
        molecule="H4",
        # UCCSD_Mole/H4.ipynb fixed ansatz at d=1.0 A: param_ids
        # [12, 5, 9, 14, 7, 4, 10, 13] (first linked ex_op per param).
        bond_length=1.0,
        n_qubits_full=8,
        n_electrons=4,
        doubles=[
            (2, 6, 5, 1),
            (2, 6, 4, 0),
            (2, 7, 5, 0),
            (3, 7, 5, 1),
            (3, 7, 4, 0),
            (6, 7, 5, 4),
            (3, 6, 5, 0),
            (2, 7, 5, 1),
        ],
        param_ids=[12, 5, 9, 14, 7, 4, 10, 13],
        pair=False,
        order="given",
        # No shared vertical bridge across all 8 tapered strings; pin hub to q2.
        hub0=2,
    ),
}


# ----------------------------------------------------------------------
# Taper the doubles' JW strings
# ----------------------------------------------------------------------
def taper_doubles(doubles, taper: taper_lib.TaperData):
    """Return (tapered_strings, signs) for each double's JW representative."""
    tapered_strings, signs = [], []
    for double in doubles:
        jw = "".join(gen.jw_string_for_double(taper.n_qubits_full, double))
        tapered_string, sign = taper_lib.taper_pauli_string(jw, taper)
        tapered_strings.append(tapered_string)
        signs.append(int(sign))
    return tapered_strings, signs


def _support(string: str) -> set[int]:
    return {q for q, p in enumerate(string) if p != "I"}


def find_shared_pivot(tapered_strings, n_qubits: int) -> int:
    """Smallest alpha-row qubit that is in every string's support AND whose
    vertical partner (q + n/2) is also in every string's support.

    That guarantees the two-row compiler pins the RX pivot to ``q`` and the
    single vertical bridge to ``(q, q + n/2)`` for all doubles.
    """
    half = n_qubits // 2
    common = set(range(n_qubits))
    for string in tapered_strings:
        common &= _support(string)
    candidates = [q for q in sorted(common) if q < half and (q + half) in common]
    if not candidates:
        raise ValueError(
            "No shared vertical bridge across the tapered strings; set 'hub0' "
            f"explicitly in CASES.  Common support = {sorted(common)}."
        )
    return candidates[0]


# ----------------------------------------------------------------------
# Compile the tapered strings into a two-row circuit (same core as the
# full-register generator, but over explicit Pauli strings + signs)
# ----------------------------------------------------------------------
def create_tapered_circuit(n_qubits, tapered_strings, signs, thetas, hub0,
                           order="given", optimize=True):
    """Return (QuantumCircuit, out_signs, theta_idx).

    Rotation ``k`` implements ``exp(-i out_signs[k] * thetas[theta_idx[k]] / 2 *
    tapered_strings[k])`` (mirrors the full-register generator's contract).
    """
    if order == "auto":
        idx = gen._auto_order(tapered_strings)
    elif order == "given":
        idx = list(range(len(tapered_strings)))
    else:
        raise ValueError(f"unknown order {order!r}")

    prog, expected, theta_idx = [], [], []
    out_signs = [0] * len(tapered_strings)
    hub = hub0
    for d in idx:
        string = tapered_strings[d]
        prefix, pivot, ph = gen._compile_tworow(string, n_qubits, hub_hint=hub)
        hub = pivot  # hub continuity pins the pivot across blocks
        prog += prefix
        # block = exp(-i (sign*ph) * theta/2 * ph*P) = exp(-i sign*theta/2 P)
        prog.append(("ROT", pivot, d, signs[d] * ph))
        prog += gen._invert(prefix)
        expected.append((string, ph))
        theta_idx.append(d)
        out_signs[d] = int(signs[d])

    if optimize:
        prog = gen._peephole(prog)
    gen._verify_program(prog, n_qubits, expected)

    qc = QuantumCircuit(n_qubits)
    for g in prog:
        if g[0] == "H":
            qc.h(g[1])
        elif g[0] == "RX":
            qc.rx(g[2], g[1])
        elif g[0] == "CZ":
            qc.cz(g[1], g[2])
        else:
            _, pivot, d, a_sgn = g
            qc.rx(a_sgn * thetas[d], pivot)
    return qc, out_signs, theta_idx


def _verify_fusion(bare_gates, fused_gates, num_qubits, seed=0):
    """Bind random angles and confirm fused == bare up to a global phase."""
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
    ratio = v1[int(np.argmax(np.abs(v1)))] / v2[int(np.argmax(np.abs(v1)))]
    if not np.allclose(v1, ratio * v2, atol=1e-8):
        raise AssertionError("RZX fusion changed the unitary!")


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------
def run(name: str, out_dir: Path | None = None) -> dict:
    import matplotlib

    matplotlib.use("Agg")

    cfg = CASES[name]
    if out_dir is None:
        out_dir = _THIS_DIR / "circuits2read"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_full = int(cfg["n_qubits_full"])
    n_spatial = n_full // 2
    taper = taper_lib.build_taper_data(n_spatial=n_spatial, n_electrons=int(cfg["n_electrons"]))
    n_qubits = taper.n_qubits_tapered

    tapered_strings, tsigns = taper_doubles(cfg["doubles"], taper)
    hub0 = cfg.get("hub0")
    if hub0 is None:
        hub0 = find_shared_pivot(tapered_strings, n_qubits)
    bridge_pair = sorted((hub0, hub0 + n_qubits // 2))

    thetas = [Parameter(f"t{d}") for d in range(len(cfg["doubles"]))]
    qc, out_signs, theta_idx = create_tapered_circuit(
        n_qubits, tapered_strings, tsigns, thetas,
        hub0=hub0, order=cfg.get("order", "given"),
    )

    bare_gates = cio.circuit_to_logical_gates(qc, n_qubits)
    bare_gates = [{k: v for k, v in g.items() if k != "cross_chip"} for g in bare_gates]
    fused_gates = gen.fuse_cz_rot_cz_to_rzx(bare_gates)
    _verify_fusion(bare_gates, fused_gates, n_qubits)

    rzx_pairs = [tuple(g["qubits"]) for g in fused_gates if g["op"] == "rzx"]
    hf_occupied = [q for q, b in enumerate(taper.hf_bitstring_tapered) if b == "1"]
    n_electrons_tapered = len(hf_occupied)

    tag = f"{cfg['molecule']}_tapered_{n_qubits}q_{len(cfg['doubles'])}doubles_rzx"
    json_path = out_dir / f"{tag}.json"
    cio.save_circuit_json(
        json_path,
        molecule=cfg["molecule"],
        bond_length=cfg["bond_length"],
        num_qubits=n_qubits,
        n_spatial=n_qubits // 2,
        n_electrons=n_electrons_tapered,
        doubles=cfg["doubles"],
        signs=out_signs,
        theta_idx=theta_idx,
        logical_gates=fused_gates,
        init_state=None,
        beta=None,
        extra={
            "tapered": True,
            "n_qubits_full": taper.n_qubits_full,
            "n_electrons_full": int(cfg["n_electrons"]),
            "removed_qubits": list(taper.removed_qubits),
            "tapering_values": list(taper.tapering_values),
            "kept_qubits": list(taper.kept_qubits),
            "symmetry_generators": list(taper.symmetry_generators),
            "hf_bitstring_tapered": taper.hf_bitstring_tapered,
            "hf_occupied_qubits": hf_occupied,
            "tapered_strings": tapered_strings,
            "bridge_pair": bridge_pair,
            **({"param_ids": list(cfg["param_ids"])} if "param_ids" in cfg else {}),
        },
    )
    gen.save_circuit_diagram(fused_gates, n_qubits, out_dir / f"{tag}_circuit.png", title=tag)

    counts = Counter(g["op"] for g in fused_gates)
    print(
        f"[{tag}] qubits {taper.n_qubits_full} -> {n_qubits}, doubles={len(cfg['doubles'])}, "
        f"params={len(set(theta_idx))}"
    )
    print(f"    tapered strings : {tapered_strings}  signs={out_signs}")
    print(f"    fused gates     : {dict(counts)}")
    print(f"    RZX pairs       : {rzx_pairs}  (bridge pair {tuple(bridge_pair)})")
    print(f"    wrote {json_path.name} (+ {tag}_circuit.png)")
    return {"json": json_path, "rzx_pairs": rzx_pairs}


def run_all(out_dir: Path | None = None) -> dict:
    return {name: run(name, out_dir=out_dir) for name in CASES}


if __name__ == "__main__":
    run_all()
