from __future__ import annotations

from typing import Iterable

import cirq
import numpy as np


def decompose_fsim_theta_only(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> list[cirq.Operation]:
    """Analytical CZ + 1Q decomposition for FSim(theta, 0), up to global phase."""
    pi = np.pi
    # Gate count summary (parameter-dependent gates treated as non-Clifford):
    # total=20, clifford=6, non_clifford=14, T_or_Tdg=8
    return [
        cirq.rz(0.25 * pi).on(q0),   # non-Clifford; T gate (Rz(+pi/4))
        cirq.rx(0.5 * pi).on(q0),    # Clifford; sqrt(X) (Rx(pi/2))
        cirq.rz(-0.25 * pi).on(q0),  # non-Clifford; Tdg gate (Rz(-pi/4))
        cirq.rz(0.75 * pi).on(q1),   # non-Clifford; Z^(3/4), not pure T
        cirq.rx(0.5 * pi).on(q1),    # Clifford; sqrt(X) (Rx(pi/2))
        cirq.rz(-0.75 * pi).on(q1),  # non-Clifford; Z^(-3/4), not pure T
        cirq.CZ(q0, q1),             # Clifford; 2Q entangling gate
        cirq.rz(0.25 * pi).on(q0),   # non-Clifford; T gate (Rz(+pi/4))
        cirq.rx(theta).on(q0),       # parameter-dependent -> counted as non-Clifford
        cirq.rz(-0.25 * pi).on(q0),  # non-Clifford; Tdg gate (Rz(-pi/4))
        cirq.rz(-0.25 * pi).on(q1),  # non-Clifford; Tdg gate (Rz(-pi/4))
        cirq.rx(theta).on(q1),       # parameter-dependent -> counted as non-Clifford
        cirq.rz(0.25 * pi).on(q1),   # non-Clifford; T gate (Rz(+pi/4))
        cirq.CZ(q0, q1),             # Clifford; 2Q entangling gate
        cirq.rz(-0.75 * pi).on(q0),  # non-Clifford; Z^(-3/4), not pure T
        cirq.rx(0.5 * pi).on(q0),    # Clifford; sqrt(X) (Rx(pi/2))
        cirq.rz(0.75 * pi).on(q0),   # non-Clifford; Z^(3/4), not pure T
        cirq.rz(-0.25 * pi).on(q1),  # non-Clifford; Tdg gate (Rz(-pi/4))
        cirq.rx(0.5 * pi).on(q1),    # Clifford; sqrt(X) (Rx(pi/2))
        cirq.rz(0.25 * pi).on(q1),   # non-Clifford; T gate (Rz(+pi/4))
    ]


def decompose_fsim_phi_only(phi: float, q0: cirq.Qid, q1: cirq.Qid) -> list[cirq.Operation]:
    """Analytical CZ + 1Q decomposition for FSim(0, phi), up to global phase.

    Uses:
        FSim(0, phi) = CZPow(exponent=-phi/pi)
    and the identity:
        CZPow(t) ~ Z(q0)^(t/2) Z(q1)^(t/2) CNOT Z(q1)^(-t/2) CNOT
    where CNOT is implemented as H-CZ-H so the final gate set stays CZ + 1Q.
    """
    # Gate count summary (parameter-dependent gates treated as non-Clifford):
    # total=9, clifford=6, non_clifford=3, T_or_Tdg=0
    return [
        # ZPowGate(exponent=e) ~ Rz(e*pi)
        cirq.rz(-phi / 2).on(q0),  # parameter-dependent -> counted as non-Clifford
        cirq.rz(-phi / 2).on(q1),  # parameter-dependent -> counted as non-Clifford
        cirq.H(q1),                # Clifford; Hadamard
        cirq.CZ(q0, q1),           # Clifford; 2Q entangling gate
        cirq.H(q1),                # Clifford; Hadamard
        cirq.rz(phi / 2).on(q1),   # parameter-dependent -> counted as non-Clifford
        cirq.H(q1),                # Clifford; Hadamard
        cirq.CZ(q0, q1),           # Clifford; 2Q entangling gate
        cirq.H(q1),                # Clifford; Hadamard
    ]


def build_decomposed_circuit(case: str, angle: float, q0: cirq.Qid, q1: cirq.Qid) -> cirq.Circuit:
    """Build a circuit for a selected restricted FSim decomposition case."""
    case_normalized = case.strip().lower()
    if case_normalized in {"theta", "theta_only", "fsim_theta"}:
        ops = decompose_fsim_theta_only(angle, q0, q1)
    elif case_normalized in {"phi", "phi_only", "fsim_phi"}:
        ops = decompose_fsim_phi_only(angle, q0, q1)
    else:
        raise ValueError(f"Unsupported case '{case}'. Use 'theta' or 'phi'.")
    return cirq.Circuit(ops)


def is_cz_plus_single_qubit(operations: Iterable[cirq.Operation]) -> bool:
    """Return True iff all operations are either CZ or 1-qubit gates."""
    for op in operations:
        if len(op.qubits) == 1:
            continue
        if len(op.qubits) == 2 and isinstance(op.gate, cirq.CZPowGate):
            # Restrict to full CZ only, not fractional CZPow.
            if np.isclose(float(op.gate.exponent), 1.0):
                continue
        return False
    return True
