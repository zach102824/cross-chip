"""Sanity checks for the vnCDR (folding + Pauli-twirling) machinery.

Run with:  python June_main/test_vncdr.py
These mirror the checks in section 7/8 of the implementation plan.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cirq
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import shot_measurement_vnCDR as sm
from main_cursor_vnCDR import CZ_CROSS_CHIP_TAG, GateArityDepolarizingNoise


# Spec Table A: CZ compensators  (R = CZ·P·CZ)
SPEC_TABLE_A = {
    "II": "II", "IX": "ZX", "IY": "ZY", "IZ": "IZ",
    "XI": "XZ", "XX": "YY", "XY": "YX", "XZ": "XI",
    "YI": "YZ", "YX": "XY", "YY": "XX", "YZ": "YI",
    "ZI": "ZI", "ZX": "IX", "ZY": "IY", "ZZ": "ZZ",
}

# Spec Table B: RZX(pi/2) compensators (generator Z1 X2)
SPEC_TABLE_B = {
    "II": "II", "IX": "IX", "IY": "ZZ", "IZ": "ZY",
    "XI": "YX", "XX": "YI", "XY": "XY", "XZ": "XZ",
    "YI": "XX", "YX": "XI", "YY": "YY", "YZ": "YZ",
    "ZI": "ZI", "ZX": "ZX", "ZY": "IZ", "ZZ": "IY",
}


def test_twirl_tables() -> None:
    cz_u = cirq.unitary(cirq.CZ)
    cz_table = sm.compensator_table_for_unitary(cz_u)
    assert cz_table == SPEC_TABLE_A, f"CZ table mismatch: {cz_table}"

    rzx_u = sm._rzx_unitary(np.pi / 2)
    rzx_table = sm.compensator_table_for_unitary(rzx_u)
    assert rzx_table == SPEC_TABLE_B, f"RZX table mismatch: {rzx_table}"
    print("[ok] twirl tables match spec Tables A and B")


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray, atol: float = 1e-8) -> bool:
    overlap = np.vdot(a.ravel(), b.ravel())
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return bool(np.isclose(abs(overlap) / norm, 1.0, atol=atol))


def test_compensator_validity() -> None:
    for u in (cirq.unitary(cirq.CZ), sm._rzx_unitary(np.pi / 2)):
        table = sm.compensator_table_for_unitary(u)
        for p_label, r_label in table.items():
            p = sm._PAULI_2Q_MATRIX[p_label]
            r = sm._PAULI_2Q_MATRIX[r_label]
            # circuit order: P then G then R  =>  R @ G @ P should equal G
            assert _equal_up_to_phase(r @ u @ p, u), f"compensator fails for {p_label}"
    print("[ok] compensator validity R*G*P == G (up to phase)")


def test_fold_identity() -> None:
    q = cirq.LineQubit.range(2)
    gates = {
        "CZ": cirq.CZ(q[0], q[1]),
        "RZX": cirq.MatrixGate(sm._rzx_unitary(np.pi / 2), name="RZX").on(q[0], q[1]),
    }
    for name, op in gates.items():
        base_u = cirq.Circuit([op]).unitary(qubit_order=q)
        for c in (1, 3, 5):
            folded = sm.fold_two_qubit_gates(cirq.Circuit([op]), c)
            fu = folded.unitary(qubit_order=q)
            assert _equal_up_to_phase(fu, base_u), f"fold identity fails {name} c={c}"
    print("[ok] fold identity: folded block == single gate for c in {1,3,5}")


def _build_demo_circuit():
    import sympy

    q = cirq.LineQubit.range(3)
    th = [sympy.Symbol(f"th_{i}") for i in range(3)]
    c = cirq.Circuit()
    c.append([cirq.H(q[0]), cirq.H(q[1]), cirq.H(q[2])])
    c.append(cirq.rz(th[0]).on(q[0]))
    c.append(cirq.CZ(q[0], q[1]))
    c.append(cirq.MatrixGate(sm._rzx_unitary(np.pi / 2), name="RZX").on(q[1], q[2]).with_tags(CZ_CROSS_CHIP_TAG))
    c.append(cirq.rx(th[1]).on(q[1]))
    c.append(cirq.rz(th[2]).on(q[2]))
    c.append(cirq.CZ(q[1], q[2]))
    return c, q, th


def test_noise_off_twirl_invariance() -> None:
    circuit, q, th = _build_demo_circuit()
    resolver = {th[0]: 0.37, th[1]: 1.1, th[2]: -0.8}
    rng = np.random.default_rng(7)

    resolved = cirq.resolve_parameters(circuit, resolver)
    psi_ref = cirq.Simulator().simulate(resolved, qubit_order=q).final_state_vector

    folded = sm.fold_two_qubit_gates(circuit, 3)
    twirled = sm.twirl_two_qubit_gates(folded, rng)
    resolved_t = cirq.resolve_parameters(twirled, resolver)
    fused = sm.fuse_single_qubit_gates(resolved_t)
    psi_tw = cirq.Simulator().simulate(fused, qubit_order=q).final_state_vector

    assert _equal_up_to_phase(psi_ref, psi_tw), "fold+twirl+fuse changed the ideal state"
    print("[ok] noise-off + fold + twirl + fuse leaves the ideal state unchanged")


def test_end_to_end_vncdr() -> None:
    circuit, q, th = _build_demo_circuit()
    observable = (
        cirq.Z(q[0]) * cirq.Z(q[1])
        + 0.5 * cirq.X(q[0])
        + 0.25 * cirq.Z(q[2])
    )
    target_resolver = {th[0]: 0.3, th[1]: 0.7, th[2]: -0.4}

    base_noise_cfg = {
        "two_qubit_depol_prob": 0.01,
        "one_qubit_depol_prob": 0.005,
        "cross_chip_two_qubit_depol_prob": 0.05,
        "coherent_2q_overrotation": 0.15,  # coherent error -> twirling matters
        "coherent_1q_overrotation": 0.0,
    }
    shot_cfg = {
        "num_shots": 4000,
        "measurement_scheme": "direct_pauli",
        "apply_readout_noise": False,
        "sampling_seed": 11,
    }
    cdr_cfg = {
        "num_circuits": 6,
        "t_max": 0,
        "seed": 3,
        "cdr_training": "random_clifford",
        "noise_levels": (1, 3, 5),
        "twirl_samples": 4,
    }

    out = sm.run_mitigation(
        "cdr",
        ansatz_circuit=circuit,
        observable_h=observable,
        qubits=list(q),
        target_resolver=target_resolver,
        target_params=target_resolver,
        symbols=list(th),
        base_noise_cfg=base_noise_cfg,
        shot_cfg=shot_cfg,
        readout_cal={},
        cdr_cfg=cdr_cfg,
        simulator_seed=5,
    )

    models = out["cdr_models"]
    coeffs = np.asarray(models["coeffs_rem_to_exact_per_term"], dtype=float)
    assert coeffs.shape[1] == 3, f"expected length-3 coeff vectors, got {coeffs.shape}"
    for key in ("unmit_target", "rem_target", "cdr_unmit_corrected", "cdr_rem_corrected"):
        assert np.isfinite(out[key]), f"{key} not finite"

    # Exact reference energy.
    resolved = cirq.resolve_parameters(circuit, target_resolver)
    psi = cirq.Simulator().simulate(resolved, qubit_order=list(q)).final_state_vector
    h_mat = observable.matrix(qubits=list(q))
    e_exact = float(np.vdot(psi, h_mat @ psi).real)

    print("[ok] end-to-end vnCDR run")
    print(f"      exact            = {e_exact:+.6f}")
    print(f"      raw (unmit/REM)  = {out['unmit_target']:+.6f} / {out['rem_target']:+.6f}")
    print(f"      vnCDR (unmit/REM)= {out['cdr_unmit_corrected']:+.6f} / {out['cdr_rem_corrected']:+.6f}")
    print(f"      noise levels     = {out['vncdr_noise_levels']}, twirl_samples={out['vncdr_twirl_samples']}")


if __name__ == "__main__":
    test_twirl_tables()
    test_compensator_validity()
    test_fold_identity()
    test_noise_off_twirl_invariance()
    test_end_to_end_vncdr()
    print("\nAll vnCDR sanity checks passed.")
