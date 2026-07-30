#!/usr/bin/env python3
"""From-scratch compilation of the H4 fixed-ansatz unitary onto the 4-chip machine.

Target unitary (tapered 6-qubit register, alpha = q0,q1,q2 / beta = q3,q4,q5):

    U = prod_k exp(-i theta_k / 2 * P_k)

over the 8 fixed-ansatz doubles of UCCSD_Mole/H4.ipynb (param ids
[12, 5, 9, 14, 7, 4, 10, 13] at d = 1.00 A), with tapered JW strings

    P12 = IYXIXX   P5  = YZXXZX   P9  = YZXIXZ   P14 = IYZIXZ
    P7  = YZZXZZ   P10 = YZZIXX   P4  = IIIYXX   P13 = IYXIXZ

KEY ALGEBRA (found numerically, verified in this script):
  * {P12, P5, P9, P14, P7, P10} pairwise commute; P4 and P13 do not
    (P13 anticommutes with all others, P4 with P9/P14/P7/P13).
  * The commuting family has GF(2) rank 4:
        P7  = +P12 * P5 * P14
        P10 = -P12 * P9 * P14
  * Therefore ONE Clifford frame V maps
        P12 -> Z-string, P5 -> Z-string, P9 -> Z-string, P14 -> Z-string
    and automatically P7, P10 -> Z-strings too.  Six of the eight
    rotations become DIAGONAL (RZ / small CZ-parity ladders) inside a
    single V ... V^dag sandwich; only the conjugated images of P4 and
    P13 remain generic Pauli rotations.

The script searches over
  * which 4 of the 6 commuting strings form the independent set,
  * a random "gauge" Clifford G that preserves the diagonal images
    (so V' = V.G is an equally valid frame but may synthesize smaller
    and may shrink the images of P4 / P13),
  * rotation ordering within the free (commuting) block,
then transpiles the best candidates at optimization_level 3 and verifies
the final circuit against the exact matrix product of exponentials.

Finally the logical 6q circuit is routed onto the flexible 4-chip
machine (qubit_connectivity/all_connection_cases.txt).  We only need
chips A+B; choosing the cross links (0,4),(1,5),(2,6),(3,7) (the A-B
half of Case 0) makes the 8 physical qubits a CUBE graph:

        chip A            chip B
        1 -- 2            5 -- 6
        |    |    rungs   |    |
        0 -- 3            4 -- 7      rungs: 0-4, 1-5, 2-6, 3-7

Cross-chip 2q gates (on rungs) can be realised three ways (see
state_transfer/qst_formulas.txt):
  1. QST state transfer through the cable using a spare qubit as
     ancilla (couplers on together, tau ~ 66 ns),
  2. QST/2 Bell pair (|10> + |01>)/sqrt2 + local gates + feed-forward,
  3. a native cross-chip RZX pulse.

Outputs (this directory):
  H4_8doubles_diagframe.json          logical circuit + all metadata
  H4_8doubles_diagframe_logical.png   logical 6q circuit
  H4_8doubles_diagframe_routed.png    routed 8q (cube) circuit
  README.md                           derivation + costs (written by hand)

Run:  .venv_py311/bin/python H4_circuits/build_h4_circuit.py
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import scipy.linalg as sla
import stim
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import Parameter
from qiskit.quantum_info import Clifford, Operator
from qiskit.synthesis import synth_clifford_full, synth_clifford_greedy
from qiskit.transpiler import CouplingMap

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------
# Problem definition
# ----------------------------------------------------------------------
STRINGS = {
    12: "IYXIXX", 5: "YZXXZX", 9: "YZXIXZ", 14: "IYZIXZ",
    7: "YZZXZZ", 10: "YZZIXX", 4: "IIIYXX", 13: "IYXIXZ",
}
COMMUTING = [12, 5, 9, 14, 7, 10]   # mutually commuting sextet (rank 4)
EXTRAS = [4, 13]                    # anticommute with part of the rest
N = 6

BASIS = ["rz", "sx", "x", "cz"]


def to_stim(s: str) -> stim.PauliString:
    return stim.PauliString(s.replace("I", "_"))


def pauli_mat(s: str) -> np.ndarray:
    P = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]),
         "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1.0, -1.0])}
    m = np.eye(1)
    for c in reversed(s):          # qiskit little-endian
        m = np.kron(m, P[c])
    return m


def gf2_rank(rows: np.ndarray) -> int:
    M = rows.copy() % 2
    r = 0
    for c in range(M.shape[1]):
        piv = next((i for i in range(r, M.shape[0]) if M[i, c]), None)
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        for i in range(M.shape[0]):
            if i != r and M[i, c]:
                M[i] ^= M[r]
        r += 1
    return r


def xz_row(s: str) -> np.ndarray:
    x = [c in "XY" for c in s]
    z = [c in "ZY" for c in s]
    return np.array(x + z, dtype=int)


# ----------------------------------------------------------------------
# stim tableau -> qiskit circuit
# ----------------------------------------------------------------------
def stim_tableau_to_qiskit(t: stim.Tableau) -> QuantumCircuit:
    qc = QuantumCircuit(N)
    for inst in t.to_circuit(method="elimination"):
        nm, ts = inst.name, [tt.value for tt in inst.targets_copy()]
        if nm == "CX":
            for i in range(0, len(ts), 2):
                qc.cx(ts[i], ts[i + 1])
        elif nm == "CZ":
            for i in range(0, len(ts), 2):
                qc.cz(ts[i], ts[i + 1])
        elif nm == "SWAP":
            for i in range(0, len(ts), 2):
                qc.swap(ts[i], ts[i + 1])
        elif nm == "H":
            for q in ts:
                qc.h(q)
        elif nm == "S":
            for q in ts:
                qc.s(q)
        elif nm == "S_DAG":
            for q in ts:
                qc.sdg(q)
        elif nm == "X":
            for q in ts:
                qc.x(q)
        elif nm == "Y":
            for q in ts:
                qc.y(q)
        elif nm == "Z":
            for q in ts:
                qc.z(q)
        elif nm in ("I", "TICK"):
            pass
        else:
            raise ValueError(f"unhandled stim gate {nm}")
    return qc


# ----------------------------------------------------------------------
# Gauge sampling: Cliffords G with G^{-1}(Z_k) still Z-type for k < 4
# ----------------------------------------------------------------------
GAUGE_GATES = (
    [("CX", i, j) for i in range(N) for j in range(N) if i != j]
    + [("CZ", i, j) for i in range(N) for j in range(i + 1, N)]
    + [("S", i) for i in range(N)]
    + [("H", 4), ("H", 5)]
    + [("SWAP", i, j) for i in range(N) for j in range(i + 1, N)]
)


def random_gauge(rng: np.random.Generator, max_len: int = 10) -> stim.Tableau:
    ln = int(rng.integers(0, max_len + 1))
    c = stim.Circuit()
    c.append("I", [N - 1])          # pin the tableau to N qubits
    for _ in range(ln):
        g = GAUGE_GATES[int(rng.integers(len(GAUGE_GATES)))]
        if len(g) == 3:
            c.append(g[0], [g[1], g[2]])
        else:
            c.append(g[0], [g[1]])
    return stim.Tableau.from_circuit(c)


def frame_is_valid(t_inv: stim.Tableau) -> bool:
    """All six commuting strings must map to Z-type (diagonal) Paulis."""
    for pid in COMMUTING:
        img = t_inv(to_stim(STRINGS[pid]))
        if any(c in "XY" for c in str(img)):
            return False
    return True


# ----------------------------------------------------------------------
# Rotation-block construction in the diagonal frame
# ----------------------------------------------------------------------
def support(body: str) -> list[int]:
    return [i for i, c in enumerate(body) if c not in "_I"]


def add_diag_rotation(qc: QuantumCircuit, zs: list[int], angle) -> None:
    """exp(-i angle/2 * Z...Z) on qubits zs via a CX parity ladder."""
    if len(zs) == 1:
        qc.rz(angle, zs[0])
        return
    for a, b in zip(zs[:-1], zs[1:]):
        qc.cx(a, b)
    qc.rz(angle, zs[-1])
    for a, b in reversed(list(zip(zs[:-1], zs[1:]))):
        qc.cx(a, b)


def add_pauli_rotation(qc: QuantumCircuit, body: str, angle) -> None:
    """exp(-i angle/2 * body) for a generic Pauli body."""
    sup = support(body)
    for q in sup:
        if body[q] == "X":
            qc.h(q)
        elif body[q] == "Y":
            qc.sdg(q)
            qc.h(q)
    for a, b in zip(sup[:-1], sup[1:]):
        qc.cx(a, b)
    qc.rz(angle, sup[-1])
    for a, b in reversed(list(zip(sup[:-1], sup[1:]))):
        qc.cx(a, b)
    for q in sup:
        if body[q] == "X":
            qc.h(q)
        elif body[q] == "Y":
            qc.h(q)
            qc.s(q)


def parse_pauli(ps: stim.PauliString) -> tuple[int, str]:
    st = str(ps)
    sign = -1 if st.startswith("-") else 1
    return sign, st.lstrip("+-").replace("_", "I")


def _count_2q(qc: QuantumCircuit) -> int:
    return sum(1 for inst in qc.data if inst.operation.num_qubits == 2)


def synthesize_v(t_frame: stim.Tableau) -> QuantumCircuit:
    """Clifford V with C Z_k C† = P_k. Keep h/s/cx — do NOT transpile to rz/sx
    (that silently breaks Clifford.inverse() composition via RZ global phases)."""
    raw = stim_tableau_to_qiskit(t_frame)
    cliff = Clifford(raw)
    cands = [synth_clifford_full(cliff), synth_clifford_greedy(cliff)]
    v_best = min(cands, key=_count_2q)
    assert Clifford(v_best) == cliff
    return v_best


def build_candidate(t_frame: stim.Tableau, rot_order: list[int], thetas: dict):
    """Return the full logical circuit for frame t_frame (U_T) and rotation order.

    Qiskit apply-order: compose(V†); compose(rot); compose(V)  implements
    U = V · rot · V† on states, and with V Z V† = P this equals exp(-i θ/2 P).
    """
    t_inv = t_frame.inverse()
    conj = {pid: parse_pauli(t_inv(to_stim(STRINGS[pid]))) for pid in STRINGS}
    v_best = synthesize_v(t_frame)

    rot = QuantumCircuit(N)
    for pid in rot_order:
        sign, body = conj[pid]
        ang = sign * thetas[pid]
        if all(c in "IZ" for c in body):
            add_diag_rotation(rot, support(body), ang)
        else:
            add_pauli_rotation(rot, body, ang)

    full = QuantumCircuit(N)
    full.compose(v_best.inverse(), inplace=True)
    full.compose(rot, inplace=True)
    full.compose(v_best, inplace=True)
    return full, conj


def verify_unitary(circ: QuantumCircuit, thetas: dict, rot_order: list[int],
                   seed: int = 11) -> float:
    rng = np.random.default_rng(seed)
    th = {pid: float(rng.uniform(-0.8, 0.8)) for pid in STRINGS}
    U = np.eye(2 ** N, dtype=complex)
    for pid in rot_order:
        U = sla.expm(-0.5j * th[pid] * pauli_mat(STRINGS[pid])) @ U
    bound = circ.assign_parameters({thetas[p]: th[p] for p in STRINGS})
    # Undo routing permutation so Operator is in logical qubit order.
    if bound.layout is not None and bound.layout.final_layout is not None:
        from qiskit.circuit.library import PermutationGate
        fil = bound.layout.final_index_layout(filter_ancillas=True)
        pattern = [0] * N
        for logical, physical in enumerate(fil):
            pattern[physical] = logical
        undo = QuantumCircuit(N)
        undo.compose(bound, inplace=True)
        undo.append(PermutationGate(pattern), range(N))
        bound = undo
    Uc = Operator(bound).data
    if Uc.shape != U.shape:
        return float("inf")
    k = int(np.argmax(np.abs(U)))
    phase = U.flat[k] / Uc.flat[k]
    return float(np.max(np.abs(U - phase * Uc)))


def proxy_cost(t_frame: stim.Tableau) -> tuple[int, dict]:
    """Cheap estimate: 2*CX(V via greedy synth) + parity-ladder CX."""
    t_inv = t_frame.inverse()
    conj = {pid: parse_pauli(t_inv(to_stim(STRINGS[pid]))) for pid in STRINGS}
    v_qc = stim_tableau_to_qiskit(t_frame)
    syn = synth_clifford_greedy(Clifford(v_qc))
    n2 = sum(1 for inst in syn.data if inst.operation.num_qubits == 2)
    ladders = 0
    for pid in STRINGS:
        w = len(support(conj[pid][1]))
        ladders += 2 * (w - 1) if w > 1 else 0
    return 2 * n2 + ladders, conj


# ----------------------------------------------------------------------
# Search
# ----------------------------------------------------------------------
def search_frames(n_samples: int = 4000, seed: int = 7):
    rng = np.random.default_rng(seed)
    rows = {pid: xz_row(STRINGS[pid]) for pid in COMMUTING}

    bases = []
    for combo in itertools.combinations(COMMUTING, 4):
        if gf2_rank(np.array([rows[p] for p in combo])) != 4:
            continue
        try:
            t = stim.Tableau.from_stabilizers(
                [to_stim(STRINGS[p]) for p in combo],
                allow_underconstrained=True,
            )
        except ValueError:
            continue
        bases.append((combo, t))
    print(f"independent 4-subsets usable as bases: {len(bases)}")

    scored = []
    for combo, t_base in bases:
        cost, _ = proxy_cost(t_base)
        scored.append((cost, combo, t_base))

    for _ in range(n_samples):
        combo, t_base = bases[int(rng.integers(len(bases)))]
        g = random_gauge(rng)
        t_new = g.then(t_base)          # t_new(P) = t_base(g(P))
        if not frame_is_valid(t_new.inverse()):
            continue
        cost, _ = proxy_cost(t_new)
        scored.append((cost, combo, t_new))

    scored.sort(key=lambda x: x[0])
    return scored


def main():
    thetas = {pid: Parameter(f"t{pid}") for pid in STRINGS}
    # ansatz order: commuting block first (internal order free), extras last
    rot_order = [12, 5, 9, 14, 7, 10, 4, 13]

    print("searching frames ...")
    scored = search_frames(n_samples=2000)
    print("best proxy costs:", [s[0] for s in scored[:8]])

    best = None
    for cost, combo, t_frame in scored[:30]:
        full, conj = build_candidate(t_frame, rot_order, thetas)
        err0 = verify_unitary(full, thetas, rot_order)
        if err0 > 1e-9:
            continue
        for seed in (0, 1, 2, 3):
            # Keep h/s/cx in the basis so the Clifford sandwich stays exact.
            t_full = transpile(
                full,
                basis_gates=["rz", "sx", "x", "cz", "cx", "h", "s", "sdg"],
                optimization_level=3,
                seed_transpiler=seed,
            )
            err = verify_unitary(t_full, thetas, rot_order)
            if err > 1e-9:
                continue
            n2q = _count_2q(t_full)
            dep = t_full.depth()
            if best is None or (n2q, dep) < (best[0], best[1]):
                best = (n2q, dep, t_full, conj, combo, t_frame, err)
    if best is None:
        raise RuntimeError("no verified candidate found")
    n2q, dep, circ, conj, combo, t_frame, err = best
    print(f"\nBEST logical circuit: 2q={n2q}, depth={dep}, base={combo}, err={err:.2e}")
    for pid in rot_order:
        s, b = conj[pid]
        print(f"  P{pid:2d} {STRINGS[pid]} -> {'-' if s < 0 else '+'}{b}")

    # Also report CZ-only transpile (may inflate count via CX->H-CZ-H)
    circ_cz = transpile(circ, basis_gates=BASIS, optimization_level=3)
    err_cz = verify_unitary(circ_cz, thetas, rot_order)
    ncz_only = circ_cz.count_ops().get("cz", 0)
    print(
        f"CZ-basis transpile: cz={ncz_only}, "
        f"depth={circ_cz.depth()}, err={err_cz:.2e}"
    )
    if err_cz < 1e-9 and (ncz_only, circ_cz.depth()) < (n2q, dep):
        circ = circ_cz
        n2q = ncz_only
        dep = circ.depth()
        print(f"  -> using CZ-basis circuit (2q={n2q}, depth={dep})")
    else:
        print(f"  -> keeping mixed-basis circuit (2q={n2q}, depth={dep})")

    # ------------------------------------------------------------------
    # route onto chips A+B subgraph (ladder embedding of the 6q register)
    #
    #   logical 0,1,2  →  chip A qubits 0,1,2
    #   logical 3,4,5  →  chip B qubits 4,5,6
    #   cross rungs    →  (0,4),(1,5),(2,6)  == logical (0,3),(1,4),(2,5)
    # ------------------------------------------------------------------
    edges = [
        (0, 1), (1, 2),          # chip A path
        (3, 4), (4, 5),          # chip B path
        (0, 3), (1, 4), (2, 5),  # cross-chip rungs
    ]
    cmap = CouplingMap(couplinglist=edges + [(b, a) for a, b in edges])
    rungs = {frozenset(e) for e in [(0, 3), (1, 4), (2, 5)]}

    routed_best = None
    for seed in range(32):
        r = transpile(
            circ, coupling_map=cmap,
            basis_gates=["rz", "sx", "x", "cz", "cx", "h", "s", "sdg"],
            optimization_level=3, seed_transpiler=seed,
            initial_layout=list(range(N)),
        )
        if r.num_qubits != N:
            continue
        if verify_unitary(r, thetas, rot_order) > 1e-9:
            continue
        n2q_r = _count_2q(r)
        ncross = sum(
            1 for inst in r.data
            if inst.operation.num_qubits == 2
            and frozenset(r.find_bit(q).index for q in inst.qubits) in rungs
        )
        key = (ncross, n2q_r, r.depth())
        if routed_best is None or key < routed_best[0]:
            routed_best = (key, r, seed)
    if routed_best is None:
        raise RuntimeError("no verified routed circuit")
    (ncross, n2q_r, dep_r), routed, seed_r = routed_best
    print(f"routed on ladder: 2q={n2q_r} (cross-chip={ncross}), depth={dep_r}, seed={seed_r}")

    # ------------------------------------------------------------------
    # artifacts
    # ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")

    fig = circ.draw(output="mpl", fold=-1, idle_wires=True)
    fig.savefig(HERE / "H4_8doubles_diagframe_logical.png", dpi=160, bbox_inches="tight")
    fig2 = routed.draw(output="mpl", fold=-1, idle_wires=True)
    fig2.savefig(HERE / "H4_8doubles_diagframe_routed.png", dpi=160, bbox_inches="tight")

    def circ_to_gates(c: QuantumCircuit) -> list[dict]:
        out = []
        for inst in c.data:
            qs = [c.find_bit(q).index for q in inst.qubits]
            op = inst.operation
            if op.name == "rz" and op.params and hasattr(op.params[0], "parameters") \
                    and op.params[0].parameters:
                out.append({"op": "rz", "qubits": qs, "param": str(op.params[0])})
            else:
                out.append({"op": op.name, "qubits": qs,
                            "value": [float(p) for p in op.params] if op.params else []})
        return out

    payload = {
        "molecule": "H4",
        "bond_length_A": 1.0,
        "num_qubits_logical": N,
        "param_ids": rot_order,
        "strings": {str(p): STRINGS[p] for p in rot_order},
        "convention": "U = prod_k exp(-i t_k/2 P_k), k applied in listed order",
        "algebra": {
            "commuting_set": COMMUTING,
            "rank": 4,
            "dependencies": ["P7 = +P12*P5*P14", "P10 = -P12*P9*P14"],
            "independent_base": list(combo),
            "conjugated_images": {
                str(p): ("-" if conj[p][0] < 0 else "+") + conj[p][1] for p in rot_order
            },
        },
        "logical_budget": {"two_qubit_gates": int(n2q), "depth": int(dep),
                            "ops": {k: int(v) for k, v in circ.count_ops().items()}},
        "machine": {
            "source": "qubit_connectivity/all_connection_cases.txt",
            "chips_used": ["A", "B"],
            "embedding": {
                "logical_to_physical": {
                    "0": "A.0", "1": "A.1", "2": "A.2",
                    "3": "B.4", "4": "B.5", "5": "B.6",
                },
                "intrachip_edges_logical": [[0, 1], [1, 2], [3, 4], [4, 5]],
                "cross_rungs_logical": [[0, 3], [1, 4], [2, 5]],
            },
            "cross_link_case_family": "Case 0 (A-B rungs 0-4,1-5,2-6)",
            "cross_chip_realisations": [
                "QST full state transfer via cable + spare ancilla "
                "(state_transfer/qst_formulas.txt, tau~66ns, cable must end in |0>)",
                "QST/2 Bell pair (|10>+|01>)/sqrt2 then gate teleportation",
                "native cross-chip RZX pulse",
            ],
        },
        "routed_budget": {"two_qubit_gates": int(n2q_r), "cross_chip_2q": int(ncross),
                           "depth": int(dep_r),
                           "ops": {k: int(v) for k, v in routed.count_ops().items()}},
        "verification_max_err": float(err),
        "gates_logical": circ_to_gates(circ),
        "gates_routed": circ_to_gates(routed),
    }
    (HERE / "H4_8doubles_diagframe.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("wrote H4_8doubles_diagframe.json / _logical.png / _routed.png")


if __name__ == "__main__":
    main()
