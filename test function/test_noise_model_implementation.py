from __future__ import annotations

from pathlib import Path
import sys

import cirq
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_cursor_lib import CZ_HIGH_TAG, LocationAwareDecomposedNoise


def _ops_from_model(model: LocationAwareDecomposedNoise, op: cirq.Operation) -> list[cirq.Operation]:
    return list(model.noisy_operation(op))


def test_noise_model_leaves_measurement_gate_unmodified() -> None:
    q = cirq.LineQubit(0)
    model = LocationAwareDecomposedNoise()
    meas = cirq.measure(q, key="m")
    out = _ops_from_model(model, meas)
    assert len(out) == 1
    assert out[0] == meas


def test_czpow_branch_applies_channel_noise_per_qubit() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    model = LocationAwareDecomposedNoise(
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.03,
        depol_prob=0.04,
        high_cz_multiplier=2.0,
        leakage_approx_prob=0.05,
    )
    out = _ops_from_model(model, cirq.CZ(q0, q1))

    # CZ itself + (AD, PD, depol) on each qubit.
    assert len(out) == 7
    assert isinstance(out[0].gate, cirq.CZPowGate)
    one_q_noise = out[1:]
    assert sum(isinstance(op.gate, cirq.AmplitudeDampingChannel) for op in one_q_noise) == 2
    assert sum(isinstance(op.gate, cirq.PhaseDampingChannel) for op in one_q_noise) == 2
    assert sum(isinstance(op.gate, cirq.DepolarizingChannel) for op in one_q_noise) == 2


def test_single_qubit_branch_uses_tenth_scaled_noise() -> None:
    q = cirq.LineQubit(0)
    model = LocationAwareDecomposedNoise(
        amp_damp_gamma=0.2,
        phase_damp_gamma=0.3,
        depol_prob=0.4,
    )
    out = _ops_from_model(model, cirq.X(q))
    assert len(out) == 4
    ad = next(op.gate for op in out if isinstance(op.gate, cirq.AmplitudeDampingChannel))
    pd = next(op.gate for op in out if isinstance(op.gate, cirq.PhaseDampingChannel))
    dp = next(op.gate for op in out if isinstance(op.gate, cirq.DepolarizingChannel))
    assert np.isclose(ad.gamma, 0.02)
    assert np.isclose(pd.gamma, 0.03)
    assert np.isclose(dp.p, 0.04)


def test_cz_high_tag_increases_effective_noise_vs_normal_cz() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    model = LocationAwareDecomposedNoise(
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.03,
        depol_prob=0.04,
        high_cz_multiplier=5.0,
        leakage_approx_prob=0.10,
    )
    normal_out = _ops_from_model(model, cirq.CZ(q0, q1))
    high_out = _ops_from_model(model, cirq.CZ(q0, q1).with_tags(CZ_HIGH_TAG))

    normal_depol = [op.gate.p for op in normal_out if isinstance(op.gate, cirq.DepolarizingChannel)]
    high_depol = [op.gate.p for op in high_out if isinstance(op.gate, cirq.DepolarizingChannel)]
    assert len(normal_depol) == len(high_depol) == 2
    assert all(h > n for h, n in zip(high_depol, normal_depol))


def test_noise_nonzero_when_noise_enabled() -> None:
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([cirq.H(q0), cirq.CZ(q0, q1)])
    ideal = cirq.DensityMatrixSimulator(seed=1).simulate(circuit, qubit_order=[q0, q1]).final_density_matrix

    noisy_circuit = circuit.with_noise(
        LocationAwareDecomposedNoise(
            amp_damp_gamma=0.05,
            phase_damp_gamma=0.05,
            depol_prob=0.05,
            high_cz_multiplier=2.0,
            leakage_approx_prob=0.02,
        )
    )
    noisy = cirq.DensityMatrixSimulator(seed=1).simulate(noisy_circuit, qubit_order=[q0, q1]).final_density_matrix
    # Use Frobenius norm as a simple state-distance proxy.
    assert np.linalg.norm(noisy - ideal) > 1e-6
