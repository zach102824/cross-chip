"""
method1_HF_frozen_circuit_png.py
================================
Method 1 from error_reduction_methods.md, applied to the HF case, in a
HARDWARE-ROUTABLE form that respects qubit_connectivity.png.

Step 1 (freeze):  delete the JW Z-parity tails from every selected double.
This is EXACT on the reachable subspace because the doubles are spin-paired:
Z_q Z_{q+N/2} = +1 on every branch of the state.

Step 2 (routing fix):  a fully frozen string like  Y0 X3 | X4 X7  compiles to
a STAR (CZ(0,3), CZ(1,3), CZ(2,3) all into hub q3) -- but the chip graph is
two 2x2 squares joined by ONE bridge, so the hub cannot have 4 neighbours and
long-range CZs are not physical edges.  The same identity used for freezing
also works in reverse: any Z spin-pair may be RE-INSERTED for free.  So we
re-insert exactly the Z-pairs needed to turn each string's support into a
connected PATH of physical edges (the Z qubits act as parity couriers again),
and choose the logical -> physical placement so both hubs sit on the bridge:

    physical chip 1 = {1,2,3,4},  chip 2 = {5,6,7,8},  bridge = (4,5)
    logical -> physical:  0->2  1->1  2->3  3->4  |  4->7  5->6  6->8  7->5

    t2 = IIYXIIXX   (fully frozen: CZ(2,3), CZ(6,7) are physical edges)
    t1 = IYIXIXIX   (fully frozen: CZ(1,3), CZ(5,7) are physical edges)
    t0 = YZIXXZIX   (Z1/Z5 re-inserted: chains 0-1-3 and 4-5-7; Z2/Z6 stay frozen)

Every CZ then maps to a chip edge and the three parameterized RZX all sit on
the physical bridge (logical (3,7) -> physical (4,5)).  NOTE for main_HF.py:
CROSS_CHIP_QUBIT_PAIRS becomes {(3, 7)} with this circuit.

Outputs (repo root):
    HF_8q_3doubles_frozen_rzx.png   -- frozen, hardware-routable, RZX-fused

Run from the repo root:  python method1_HF_frozen_circuit_png.py
"""
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def _load(name, rel_path):
    spec = importlib.util.spec_from_file_location(name, str(ROOT / rel_path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


gen = _load("gen", "UCCSD circuit/improved create UCCSD circuit .py")
cio = _load("cio", "UCCSD circuit/uccsd_circuit_io.py")

# ----------------------------------------------------------------------
# HF case (same registry values as MoleculeCircuitRunner / main_HF.py)
# ----------------------------------------------------------------------
NUM_QUBITS = 8
N_ELECTRONS = 6
DOUBLES = [(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)]
BASELINE_JSON = ROOT / "June_main" / "circuits2read" / "HF_8q_3doubles_rzx.json"
OUT_PNG = ROOT / "HF_8q_3doubles_frozen_rzx.png"

# Hardware graph (first two chips of qubit_connectivity.png, 1-indexed)
PHYSICAL_EDGES = {
    frozenset(e)
    for e in [(1, 2), (2, 3), (3, 4), (1, 4),          # chip 1 square
              (5, 6), (6, 7), (7, 8), (5, 8),          # chip 2 square
              (4, 5)]                                   # cross-chip bridge
}
# Logical wire -> physical qubit.  Hubs (3, 7) land on the bridge (4, 5).
LOGICAL_TO_PHYSICAL = {0: 2, 1: 1, 2: 3, 3: 4, 4: 7, 5: 6, 6: 8, 7: 5}

# Selectively frozen strings (see module docstring).  Compared to the raw
# jw_string_for_double output, all Z spin-pairs are deleted EXCEPT Z1/Z5 in
# t0, which are re-inserted as parity couriers so t0 routes 0-1-3 / 4-5-7.
FROZEN_STRINGS = [
    "YZIXXZIX",   # t0  (was YZZXXZZX)
    "IYIXIXIX",   # t1  (was IYZXIXZX)
    "IIYXIIXX",   # t2  (unchanged, no tail)
]


def build_circuit(strings, num_qubits, thetas, hub0=None):
    """Compile explicit Pauli strings with the generator's pipeline
    (two-row hub ladders + hub continuity + peephole + symbolic verify)."""
    from qiskit import QuantumCircuit

    idx = gen._auto_order(strings)
    prog, expected, hub = [], [], hub0
    for i in idx:
        s = strings[i]
        prefix, pivot, ph = gen._compile_tworow(s, num_qubits, hub_hint=hub)
        hub = pivot
        prog += prefix + [("ROT", pivot, i, ph)] + gen._invert(prefix)
        expected.append((s, ph))
    prog = gen._peephole(prog)
    gen._verify_program(prog, num_qubits, expected)

    qc = QuantumCircuit(num_qubits)
    for g in prog:
        if g[0] == "H":
            qc.h(g[1])
        elif g[0] == "RX":
            qc.rx(g[2], g[1])
        elif g[0] == "CZ":
            qc.cz(g[1], g[2])
        else:  # ("ROT", pivot, theta_index, sign)
            qc.rx(g[3] * thetas[g[2]], g[1])
    return qc


def check_routability(gates):
    """Assert every 2-qubit gate maps onto a physical chip edge."""
    bad = []
    for g in gates:
        if len(g["qubits"]) == 2:
            a, b = (LOGICAL_TO_PHYSICAL[q] for q in g["qubits"])
            if frozenset((a, b)) not in PHYSICAL_EDGES:
                bad.append((g["op"], tuple(g["qubits"]), (a, b)))
    if bad:
        raise AssertionError(f"non-physical 2q gates: {bad}")


def main():
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector

    thetas = [Parameter(f"t{d}") for d in range(len(DOUBLES))]

    orig_strings = ["".join(gen.jw_string_for_double(NUM_QUBITS, d)) for d in DOUBLES]
    print("Method 1 selective freeze (HF, hardware-routable):")
    for s0, s1 in zip(orig_strings, FROZEN_STRINGS):
        print(f"    {s0}  ->  {s1}")

    # ---- exactness check on the HF determinant with random angles ----
    rng = np.random.default_rng(7)
    vals = {t: float(a) for t, a in zip(thetas, rng.uniform(-0.3, 0.3, len(thetas)))}
    init = QuantumCircuit(NUM_QUBITS)
    eta = N_ELECTRONS // 2
    for q in list(range(eta)) + list(range(NUM_QUBITS // 2, NUM_QUBITS // 2 + eta)):
        init.x(q)
    qc_orig = build_circuit(orig_strings, NUM_QUBITS, thetas).assign_parameters(vals)
    qc_froz_bound = build_circuit(FROZEN_STRINGS, NUM_QUBITS, thetas,
                                  hub0=3).assign_parameters(vals)
    v0 = Statevector(init.compose(qc_orig)).data
    v1 = Statevector(init.compose(qc_froz_bound)).data
    overlap = abs(np.vdot(v0, v1))
    print(f"|<orig|frozen>| on HF reference (random thetas) = {overlap:.12f}")
    assert overlap > 1 - 1e-10, "selective freeze is NOT exact here!"

    # ---- frozen symbolic circuit -> logical gates -> RZX fusion ----
    qc_frozen = build_circuit(FROZEN_STRINGS, NUM_QUBITS, thetas, hub0=3)
    bare = cio.circuit_to_logical_gates(qc_frozen, NUM_QUBITS)
    fused = gen.fuse_cz_rot_cz_to_rzx(bare)
    gen.MoleculeCircuitRunner._verify_fusion(None, bare, fused, NUM_QUBITS)

    # ---- connectivity check against qubit_connectivity.png ----
    check_routability(fused)
    print("connectivity: every 2q gate maps to a physical edge under")
    print(f"    logical->physical {LOGICAL_TO_PHYSICAL}  (bridge = physical (4,5))")

    # ---- gate-count comparison against the saved baseline ----
    def counts(gates):
        out = {}
        for g in gates:
            out[g["op"]] = out.get(g["op"], 0) + 1
        return out

    baseline = json.loads(BASELINE_JSON.read_text())["gates"]
    qc_base = gen.qc_from_logical_gates(baseline, NUM_QUBITS)
    qc_fused = gen.qc_from_logical_gates(fused, NUM_QUBITS)
    print(f"baseline ({BASELINE_JSON.name}): {counts(baseline)}  depth={qc_base.depth()}")
    print(f"frozen + routed (Method 1)     : {counts(fused)}  depth={qc_fused.depth()}")
    rzx_links = sorted({tuple(sorted(g["qubits"])) for g in fused if g["op"] == "rzx"})
    print(f"parameterized RZX link(s): logical {rzx_links} -> physical bridge (4,5)")

    # ---- diagram ----
    gen.save_circuit_diagram(
        fused,
        NUM_QUBITS,
        OUT_PNG,
        title="HF 8q 3 doubles -- Method 1 frozen, hardware-routable (hubs on bridge)",
    )
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
