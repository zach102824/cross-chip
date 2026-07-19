"""
method1_routed_circuits_pngs.py
===============================
Method 1 (Z-tail freeze, see error_reduction_methods.md) for ALL THREE
molecules, in hardware-routable form on the real chip graph
(qubit_connectivity/qubit_connectivity.png: four 2x2 chips, bridges
4-5 top, 7-9 bottom, 11-13 top; 1-indexed physical qubits).

Recipe per molecule:
  1. freeze every JW Z spin-pair (exact: Z_q Z_{q+N/2} = +1 on the reachable
     subspace);
  2. re-insert only the Z-pairs needed as parity couriers so each string's
     support is a PATH of physical edges (re-insertion is equally exact);
  3. choose the logical -> physical placement so the two hubs (the shared
     LUMO pair, which carries the fused RZX(theta)) sit on a physical bridge.

Every 2-qubit gate is asserted to land on a physical edge, every circuit is
verified by statevector against the ORIGINAL (unfrozen) strings, and the
CZ.RX.CZ -> RZX fusion is verified by statevector too.

Outputs (repo root):
    HF_8q_3doubles_frozen_rzx.png
    Cl2_10q_3doubles_frozen_rzx.png
    Br2_12q_4doubles_frozen_rzx.png
    method1_placement_maps.png      -- where each logical qubit sits on-chip

Run from the repo root:  python method1_routed_circuits_pngs.py
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
# Hardware graph (1-indexed physical qubits, from qubit_connectivity.png)
# ----------------------------------------------------------------------
CHIPS = [(1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16)]
PHYSICAL_EDGES = {frozenset(e) for e in [
    (2, 3), (3, 4), (4, 1), (1, 2),          # chip 1 square
    (5, 6), (6, 7), (7, 8), (8, 5),          # chip 2 square
    (10, 9), (9, 12), (12, 11), (11, 10),    # chip 3 square
    (13, 14), (14, 15), (15, 16), (16, 13),  # chip 4 square
    (4, 5), (7, 9), (11, 13),                # cross-chip bridges
]}
BRIDGES = {frozenset((4, 5)), frozenset((7, 9)), frozenset((11, 13))}

# Node positions for the placement-map figure (match the png layout).
POS = {3: (0, 1), 4: (1, 1), 2: (0, 0), 1: (1, 0),
       5: (2, 1), 8: (3, 1), 6: (2, 0), 7: (3, 0),
       10: (4, 1), 11: (5, 1), 9: (4, 0), 12: (5, 0),
       13: (6, 1), 14: (7, 1), 16: (6, 0), 15: (7, 0)}

# ----------------------------------------------------------------------
# Molecule cases: selectively frozen strings + logical->physical placement.
#
# String notation: original = raw jw_string_for_double output; frozen = after
# deleting all Z spin-pairs EXCEPT the couriers listed (re-inserted so the
# support is a physical path).  hub = the shared LUMO pair, always on a
# bridge, carrying the fused RZX(theta).
# ----------------------------------------------------------------------
CASES = {
    "HF": dict(
        num_qubits=8, n_electrons=6,
        doubles=[(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)],
        # hub (3,7) on bridge (4,5); t0 keeps Z1/Z5 as couriers (chains
        # 0-1-3 and 4-5-7); t1, t2 fully frozen.
        frozen=["YZIXXZIX", "IYIXIXIX", "IIYXIIXX"],
        hub0=3,
        placement={0: 2, 1: 1, 2: 3, 3: 4, 4: 7, 5: 6, 6: 8, 7: 5},
        baseline="HF_8q_3doubles_rzx.json",
    ),
    "Cl2": dict(
        num_qubits=10, n_electrons=8,
        doubles=[(4, 9, 6, 1), (4, 9, 5, 0), (4, 9, 7, 2)],
        # hub (4,9) on bridge (7,9).  t2 fully frozen (direct edges via the
        # square diagonal qubits 8 / 12); t0 keeps Z3/Z8 (chains 1-3-4,
        # 6-8-9); t1 keeps Z1,Z3/Z6,Z8 (chains 0-1-3-4, 5-6-8-9).
        frozen=["IYIZXIXIZX", "YZIZXXZIZX", "IIYIXIIXIX"],
        hub0=4,
        placement={0: 4, 1: 5, 2: 8, 3: 6, 4: 7,
                   5: 13, 6: 11, 7: 12, 8: 10, 9: 9},
        baseline="Cl2_10q_3doubles_rzx.json",
    ),
    "Br2": dict(
        num_qubits=12, n_electrons=10,
        # 5 doubles: the 4 MP2-selected ones PLUS (5,11,7,1), the next paired
        # double, so that spatial orbital 1 (logical q1/q7) participates and
        # ALL 12 qubits carry gates.  theta4 = 0 recovers the 4-double ansatz,
        # so the variational floor can only improve.
        doubles=[(5, 11, 6, 0), (5, 11, 10, 4), (5, 11, 9, 3), (5, 11, 8, 2),
                 (5, 11, 7, 1)],
        # hub (5,11) on bridge (7,9).  t0 (the 3-chip string!) becomes FULLY
        # frozen -- Y0 X5 | X6 X11 attaches directly to the hubs via the
        # square qubits 8 / 12.  t1 has no tail; t2 keeps Z4/Z10; t3 keeps
        # Z3,Z4/Z9,Z10 (chains along 4-5-6-7 and 13-11-10-9).  t4 (the new
        # double) keeps its full tail: its couriers ARE the chain qubits
        # 1-2-3-4-5 / 7-8-9-10-11, so nothing to freeze.
        frozen=["YIIIIXXIIIIX",      # t0: fully frozen
                "IIIIYXIIIIXX",      # t1: no tail anyway
                "IIIYZXIIIXZX",      # t2: Z4/Z10 kept as couriers
                "IIYZZXIIXZZX",      # t3: Z3,Z4/Z9,Z10 kept as couriers
                "IYZZZXIXZZZX"],     # t4: new double, full tail = couriers
        hub0=5,
        placement={0: 8, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7,
                   6: 12, 7: 14, 8: 13, 9: 11, 10: 10, 11: 9},
        baseline="Br2_12q_4doubles_rzx.json",
    ),
}


def build_circuit(strings, num_qubits, thetas, hub0=None):
    """Compile explicit Pauli strings with the generator's pipeline."""
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
        else:
            qc.rx(g[3] * thetas[g[2]], g[1])
    return qc


def run_case(name, cfg):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Statevector

    n = cfg["num_qubits"]
    doubles = cfg["doubles"]
    place = cfg["placement"]
    thetas = [Parameter(f"t{d}") for d in range(len(doubles))]

    orig = ["".join(gen.jw_string_for_double(n, d)) for d in doubles]
    print(f"\n=== {name} ===")
    for s0, s1 in zip(orig, cfg["frozen"]):
        tag = "(unchanged)" if s0 == s1 else ""
        print(f"    {s0} -> {s1} {tag}")

    # ---- exactness: frozen circuit == original circuit on the HF state ----
    rng = np.random.default_rng(11)
    vals = {t: float(a) for t, a in zip(thetas, rng.uniform(-0.3, 0.3, len(thetas)))}
    eta = cfg["n_electrons"] // 2
    init = QuantumCircuit(n)
    for q in list(range(eta)) + list(range(n // 2, n // 2 + eta)):
        init.x(q)
    v0 = Statevector(init.compose(
        build_circuit(orig, n, thetas).assign_parameters(vals)))
    v1 = Statevector(init.compose(
        build_circuit(cfg["frozen"], n, thetas, hub0=cfg["hub0"]).assign_parameters(vals)))
    ov = abs(np.vdot(v0.data, v1.data))
    print(f"    exactness |<orig|frozen>| = {ov:.12f}")
    assert ov > 1 - 1e-10

    # ---- compile symbolically, fuse RZX, check routability ----
    qc = build_circuit(cfg["frozen"], n, thetas, hub0=cfg["hub0"])
    bare = cio.circuit_to_logical_gates(qc, n)
    fused = gen.fuse_cz_rot_cz_to_rzx(bare)
    gen.MoleculeCircuitRunner._verify_fusion(None, bare, fused, n)

    n_cross = 0
    for g in fused:
        if len(g["qubits"]) == 2:
            pq = frozenset(place[q] for q in g["qubits"])
            assert pq in PHYSICAL_EDGES, \
                f"{name}: {g['op']} on logical {g['qubits']} -> physical {sorted(pq)} is NOT an edge"
            if pq in BRIDGES:
                n_cross += 1
    print(f"    routable: yes (all 2q gates on physical edges)")

    # ---- counts vs baseline ----
    base = json.loads((ROOT / "June_main" / "circuits2read" / cfg["baseline"]).read_text())["gates"]

    def stats(gates):
        c = {}
        for g in gates:
            c[g["op"]] = c.get(g["op"], 0) + 1
        depth = gen.qc_from_logical_gates(gates, n).depth()
        return c, depth

    cb, db = stats(base)
    cf, df = stats(fused)
    print(f"    baseline: {cb}  depth={db}")
    print(f"    method 1: {cf}  depth={df}  cross-chip 2q={n_cross}")

    rzx_link = sorted({tuple(sorted(g["qubits"])) for g in fused if g["op"] == "rzx"})
    out_png = ROOT / f"{name}_{n}q_{len(doubles)}doubles_frozen_rzx.png"
    gen.save_circuit_diagram(
        fused, n, out_png,
        title=(f"{name} {n}q {len(doubles)} doubles -- Method 1 frozen, "
               f"routable (RZX on logical {rzx_link[0]})"),
    )
    print(f"    wrote {out_png.name}")
    return fused


def placement_map_figure():
    """One panel per molecule: the physical chip graph with logical labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(3, 1, figsize=(11, 10.5))
    for ax, (name, cfg) in zip(axes, CASES.items()):
        n = cfg["num_qubits"]
        place = cfg["placement"]
        inv = {p: l for l, p in place.items()}
        half = n // 2

        # roles for coloring
        excited = set()
        couriers = set()
        for s in cfg["frozen"]:
            for q, c in enumerate(s):
                if c in "XY":
                    excited.add(q)
                elif c == "Z":
                    couriers.add(q)
        hubs = {cfg["hub0"], cfg["hub0"] + half}
        couriers -= excited

        for e in PHYSICAL_EDGES:
            a, b = sorted(e)
            (x1, y1), (x2, y2) = POS[a], POS[b]
            is_bridge = e in BRIDGES
            used = a in inv and b in inv
            ax.plot([x1, x2], [y1, y2],
                    color=("#d62728" if is_bridge and used else
                           "#999999" if used else "#dddddd"),
                    lw=(3 if is_bridge and used else 1.5),
                    zorder=1)
        for p, (x, y) in POS.items():
            if p in inv:
                l = inv[p]
                if l in hubs:
                    fc = "#ff7f0e"           # hub (RZX endpoint)
                elif l in excited - hubs:
                    fc = "#2ca02c"           # excited (X/Y support)
                elif l in couriers:
                    fc = "#aec7e8"           # Z courier
                else:
                    fc = "#f0f0f0"           # idle spectator
                label = f"q{l}"
            else:
                fc, label = "white", ""
            ax.scatter([x], [y], s=1300, c=fc, edgecolors="black", zorder=2)
            ax.text(x, y, label, ha="center", va="center", fontsize=11,
                    fontweight="bold", zorder=3)
            ax.text(x, y - 0.28, f"p{p}", ha="center", va="center",
                    fontsize=7, color="#666666", zorder=3)
        ax.set_title(f"{name}: logical qubit placement "
                     f"(hub pair q{cfg['hub0']}/q{cfg['hub0'] + half} on a bridge)",
                     fontsize=11)
        ax.set_xlim(-0.6, 7.6)
        ax.set_ylim(-0.55, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    legend = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#ff7f0e",
               markeredgecolor="k", markersize=12, label="hub (RZX endpoint)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="k", markersize=12, label="excited orbital (X/Y)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#aec7e8",
               markeredgecolor="k", markersize=12, label="Z parity courier"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#f0f0f0",
               markeredgecolor="k", markersize=12, label="idle spectator"),
        Line2D([0], [0], color="#d62728", lw=3, label="cross-chip bridge used"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=9,
               frameon=False)
    fig.suptitle("Method 1 frozen circuits: logical -> physical placement "
                 "(pN = physical qubit in qubit_connectivity.png)", fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out = ROOT / "method1_placement_maps.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {out.name}")


if __name__ == "__main__":
    for name, cfg in CASES.items():
        run_case(name, cfg)
    placement_map_figure()
