from __future__ import annotations

from typing import Iterable

import cirq
import numpy as np


def decompose_fsim_theta_only(theta: float, q0: cirq.Qid, q1: cirq.Qid) -> list[cirq.Operation]:
    """Analytical CZ + 1Q decomposition for FSim(theta, 0), up to global phase."""
    theta_exponent = theta / np.pi
    return [
        cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=0.5).on(q0),
        cirq.PhasedXPowGate(phase_exponent=-0.75, exponent=0.5).on(q1),
        cirq.CZ(q0, q1),
        cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=theta_exponent).on(q0),
        cirq.PhasedXPowGate(phase_exponent=0.25, exponent=theta_exponent).on(q1),
        cirq.CZ(q0, q1),
        cirq.PhasedXPowGate(phase_exponent=0.75, exponent=0.5).on(q0),
        cirq.PhasedXPowGate(phase_exponent=0.25, exponent=0.5).on(q1),
    ]


def decompose_fsim_phi_only(phi: float, q0: cirq.Qid, q1: cirq.Qid) -> list[cirq.Operation]:
    """Analytical CZ + 1Q decomposition for FSim(0, phi), up to global phase.

    Uses:
        FSim(0, phi) = CZPow(exponent=-phi/pi)
    and the identity:
        CZPow(t) ~ Z(q0)^(t/2) Z(q1)^(t/2) CNOT Z(q1)^(-t/2) CNOT
    where CNOT is implemented as H-CZ-H so the final gate set stays CZ + 1Q.
    """
    t = -phi / np.pi
    return [
        cirq.ZPowGate(exponent=t / 2).on(q0),
        cirq.ZPowGate(exponent=t / 2).on(q1),
        cirq.H(q1),
        cirq.CZ(q0, q1),
        cirq.H(q1),
        cirq.ZPowGate(exponent=-t / 2).on(q1),
        cirq.H(q1),
        cirq.CZ(q0, q1),
        cirq.H(q1),
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
