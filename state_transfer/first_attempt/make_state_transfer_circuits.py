#!/usr/bin/env python3
"""
make_state_transfer_circuits.py
===============================
Companion generator for ``state_transfer_migration.tex``.

Builds and draws the circuits that explain how cross-chip two-qubit gates
can be implemented with *coherent state transfer only* (QST, no mid-circuit
measurement, no feedforward), and numerically verifies that every
implementation is exactly unitarily equivalent to its nonlocal target.

The QST cable transfer
    (a|0> + b|1>)_sender |0>_receiver  ->  |0>_sender (a|0> + b|1>)_receiver
is modeled as a SWAP (exact on the required receiver-in-|0> subspace);
it is drawn as an opaque box labeled "QST".

Toy model (4 qubits):
    chip A: A_0, A_1 (data; A_1 is the seam qubit that travels)
    chip B: B_0 (data), e (communication ancilla, starts in |0>)

Outputs (written next to this script):
    fig1_toy_target.png              nonlocal target: 3x RZZ(A_1,B_0) + local layers
    fig2_toy_pingpong.png            ping-pong implementation, 6 cable traversals
    fig3_toy_migration.png           migration implementation, 2 cable traversals
    fig4_toy_blocked.png             counterexample: CZ(A_0,A_1) blocks migration
    fig5_HF_pingpong.png             HF_8q_3doubles_rzx + 1 ancilla, ping-pong (exact, runnable)
    fig6_HF_migration_template.png   desired (redesigned-ansatz) HF structure, 2 traversals

Run:  python3 make_state_transfer_circuits.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Gate, Parameter
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator

HERE = Path(__file__).resolve().parent
HF_JSON = HERE.parent / "June_main" / "circuits2read" / "HF_8q_3doubles_rzx.json"

_SWAP = np.array(
    [[1, 0, 0, 0],
     [0, 0, 1, 0],
     [0, 1, 0, 0],
     [0, 0, 0, 1]], dtype=complex)


def qst_gate() -> UnitaryGate:
    """Cable transfer, drawn as a 'QST' box, simulated as a SWAP."""
    return UnitaryGate(_SWAP, label="QST")


def save_png(qc: QuantumCircuit, path: Path, title: str, fold: int = -1) -> None:
    fig = qc.draw("mpl", fold=fold)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"    wrote {path.name}")


# --------------------------------------------------------------------------
# Toy model
# --------------------------------------------------------------------------
T0, T1, T2 = (Parameter("t0"), Parameter("t1"), Parameter("t2"))
PHI = Parameter("phi")


def toy_registers():
    A = QuantumRegister(2, "A")   # chip A: A_0, A_1 (A_1 = seam/travelling qubit)
    B = QuantumRegister(1, "B")   # chip B data
    E = QuantumRegister(1, "e")   # chip B communication ancilla, starts |0>
    return A, B, E


def toy_target() -> QuantumCircuit:
    """U = R3 . V2 . R2 . V1 . R1 with nonlocal Rk = RZZ(tk) on (A_1, B_0)."""
    A, B, E = toy_registers()
    qc = QuantumCircuit(A, B, E)
    qc.rzz(T0, A[1], B[0])                    # R1  (cross-chip)
    qc.barrier()
    qc.h(A[0]); qc.rz(PHI, A[1])              # V1: local, does not couple A_1 to chip A
    qc.barrier()
    qc.rzz(T1, A[1], B[0])                    # R2  (cross-chip)
    qc.barrier()
    qc.h(A[0]); qc.z(B[0])                    # V2
    qc.barrier()
    qc.rzz(T2, A[1], B[0])                    # R3  (cross-chip)
    return qc


def toy_pingpong() -> QuantumCircuit:
    """Each nonlocal RZZ -> [QST out, local RZZ(e,B_0), QST back]: 6 traversals."""
    A, B, E = toy_registers()
    qc = QuantumCircuit(A, B, E)

    def crossing(theta):
        qc.append(qst_gate(), [A[1], E[0]])   # ship A_1's state to e
        qc.rzz(theta, E[0], B[0])             # gate is local on chip B
        qc.append(qst_gate(), [E[0], A[1]])   # ship it home

    crossing(T0)
    qc.barrier()
    qc.h(A[0]); qc.rz(PHI, A[1])
    qc.barrier()
    crossing(T1)
    qc.barrier()
    qc.h(A[0]); qc.z(B[0])
    qc.barrier()
    crossing(T2)
    return qc


def toy_migration() -> QuantumCircuit:
    """One QST out, all three RZZ local on chip B, one QST home: 2 traversals.
    Single-qubit gates on the travelling qubit are applied to e instead."""
    A, B, E = toy_registers()
    qc = QuantumCircuit(A, B, E)
    qc.append(qst_gate(), [A[1], E[0]])       # migrate A_1 -> e (traversal 1)
    qc.barrier()
    qc.rzz(T0, E[0], B[0])                    # R1, now local
    qc.barrier()
    qc.h(A[0]); qc.rz(PHI, E[0])              # V1: the RZ travels with the qubit
    qc.barrier()
    qc.rzz(T1, E[0], B[0])                    # R2, local
    qc.barrier()
    qc.h(A[0]); qc.z(B[0])                    # V2
    qc.barrier()
    qc.rzz(T2, E[0], B[0])                    # R3, local
    qc.barrier()
    qc.append(qst_gate(), [E[0], A[1]])       # migrate home (traversal 2)
    return qc


def toy_blocked() -> QuantumCircuit:
    """Migration attempt when V1 = CZ(A_0, A_1): after migrating, the commuted
    gate is CZ(A_0, e) -- itself cross-chip. The telescoping dies here."""
    A, B, E = toy_registers()
    qc = QuantumCircuit(A, B, E)
    qc.append(qst_gate(), [A[1], E[0]])
    qc.barrier()
    qc.rzz(T0, E[0], B[0])
    qc.barrier()
    qc.cz(A[0], E[0])                         # <-- BLOCKED: spans the chip cut
    qc.barrier()
    qc.rzz(T1, E[0], B[0])
    qc.barrier()
    qc.append(qst_gate(), [E[0], A[1]])
    return qc


# --------------------------------------------------------------------------
# HF_8q_3doubles_rzx: ping-pong version (exact) + migration template (schematic)
# --------------------------------------------------------------------------
def _hf_params(data: dict) -> dict:
    names = sorted({g["param"] for g in data["gates"] if g["op"] == "rzx"})
    return {n: Parameter(n) for n in names}


def hf_target_9q(data: dict, params: dict) -> QuantumCircuit:
    """The original 8-qubit circuit on 9 wires (ancilla e idle)."""
    q = QuantumRegister(8, "q")
    e = QuantumRegister(1, "e")
    qc = QuantumCircuit(q, e)
    for g in data["gates"]:
        op, qs = g["op"], g["qubits"]
        if op == "h":
            qc.h(qs[0])
        elif op == "rx":
            qc.rx(g["value"], qs[0])
        elif op == "cz":
            qc.cz(qs[0], qs[1])
        elif op == "rzx":
            qc.rzx(g["coeff"] * params[g["param"]], qs[0], qs[1])
        else:
            raise ValueError(f"unknown op {op}")
    return qc


def hf_pingpong_9q(data: dict, params: dict) -> QuantumCircuit:
    """Every cross-chip rzx(6,2) -> QST(2->e), local rzx(6,e), QST(e->2).
    Chip A = q0..q3, chip B = q4..q7 + e. Exact, runnable today: 6 traversals."""
    q = QuantumRegister(8, "q")
    e = QuantumRegister(1, "e")
    qc = QuantumCircuit(q, e)
    for g in data["gates"]:
        op, qs = g["op"], g["qubits"]
        if op == "h":
            qc.h(qs[0])
        elif op == "rx":
            qc.rx(g["value"], qs[0])
        elif op == "cz":
            qc.cz(qs[0], qs[1])
        elif op == "rzx":
            theta = g["coeff"] * params[g["param"]]
            qc.append(qst_gate(), [q[2], e[0]])       # data q2 -> chip B ancilla
            qc.rzx(theta, q[6], e[0])                 # gate local on chip B
            qc.append(qst_gate(), [e[0], q[2]])       # and home again
        else:
            raise ValueError(f"unknown op {op}")
    return qc


def hf_migration_template() -> QuantumCircuit:
    """Schematic of the DESIRED circuit structure for a redesigned ansatz:
    q2 migrates once, all three RZX are local on chip B, q2 returns once.
    V_A blocks act on chip A WITHOUT q2 (it is empty while abroad);
    V_B blocks are chip-B local dressing (q4..q7). Opaque boxes: structure only."""
    q = QuantumRegister(8, "q")
    e = QuantumRegister(1, "e")
    qc = QuantumCircuit(q, e)
    t = [Parameter("t0"), Parameter("t1"), Parameter("t2")]

    qc.append(Gate("V_A-init", 4, []), [q[0], q[1], q[2], q[3]])
    qc.append(Gate("V_B-init", 4, []), [q[4], q[5], q[6], q[7]])
    qc.append(qst_gate(), [q[2], e[0]])               # traversal 1: q2 -> e
    qc.barrier()
    for k in range(3):
        qc.rzx(t[k], q[6], e[0])                      # crossing k, local on chip B
        if k < 2:
            qc.barrier()
            # allowed between crossings: chip A gates NOT touching q2 ...
            qc.append(Gate(f"V_A{k+1}", 3, []), [q[0], q[1], q[3]])
            # ... and chip B gates (may touch q6 and even e)
            qc.append(Gate(f"V_B{k+1}", 4, []), [q[4], q[5], q[6], q[7]])
            qc.barrier()
    qc.barrier()
    qc.append(qst_gate(), [e[0], q[2]])               # traversal 2: home
    qc.append(Gate("V_A-end", 4, []), [q[0], q[1], q[2], q[3]])
    qc.append(Gate("V_B-end", 4, []), [q[4], q[5], q[6], q[7]])
    return qc


# --------------------------------------------------------------------------
# Verification: every implementation == its target as a full unitary
# --------------------------------------------------------------------------
def assert_equiv(qc_a: QuantumCircuit, qc_b: QuantumCircuit, name: str, seed: int = 7):
    rng = np.random.default_rng(seed)
    values = {p.name: float(rng.uniform(-np.pi, np.pi)) for p in qc_a.parameters}
    bound_a = qc_a.assign_parameters({p: values[p.name] for p in qc_a.parameters})
    bound_b = qc_b.assign_parameters({p: values[p.name] for p in qc_b.parameters})
    if not Operator(bound_a).equiv(Operator(bound_b)):
        raise AssertionError(f"{name}: NOT equivalent to target!")
    print(f"    [ok] {name} == target (up to global phase)")


def main():
    print("[toy model]")
    tgt = toy_target()
    pp = toy_pingpong()
    mig = toy_migration()
    blk = toy_blocked()
    assert_equiv(pp, tgt, "ping-pong (6 traversals)")
    assert_equiv(mig, tgt, "migration (2 traversals)")
    save_png(tgt, HERE / "fig1_toy_target.png",
             "Target: 3 cross-chip RZZ(A1,B0) with local layers V1, V2")
    save_png(pp, HERE / "fig2_toy_pingpong.png",
             "Ping-pong: QST out / local RZZ(e,B0) / QST back -- 6 cable traversals")
    save_png(mig, HERE / "fig3_toy_migration.png",
             "Migration: 1 QST out, all gates local on chip B, 1 QST home -- 2 traversals")
    save_png(blk, HERE / "fig4_toy_blocked.png",
             "Blocked: V1=CZ(A0,A1) commutes into CZ(A0,e), itself cross-chip")

    print("[HF_8q_3doubles_rzx]")
    data = json.loads(HF_JSON.read_text())
    params = _hf_params(data)
    hf_tgt = hf_target_9q(data, params)
    hf_pp = hf_pingpong_9q(data, params)
    assert_equiv(hf_pp, hf_tgt, "HF ping-pong (9 qubits, 6 traversals)")
    save_png(hf_pp, HERE / "fig5_HF_pingpong.png",
             "HF_8q_3doubles_rzx, ping-pong QST version (exact; chip A=q0-q3, chip B=q4-q7,e)")
    save_png(hf_migration_template(), HERE / "fig6_HF_migration_template.png",
             "DESIRED structure (redesigned ansatz): q2 abroad for all 3 crossings -- 2 traversals")

    print("done.")


if __name__ == "__main__":
    main()
