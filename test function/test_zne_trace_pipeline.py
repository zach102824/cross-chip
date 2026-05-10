from __future__ import annotations

from pathlib import Path
import sys

import cirq
import numpy as np
import pytest
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_cursor_lib import run_trace_zne, trace_energy_at_noise_scale


def _simple_problem() -> tuple[cirq.Circuit, cirq.ParamResolver, list[cirq.Qid], np.ndarray, float]:
    q0, q1 = cirq.LineQubit.range(2)
    th = sympy.Symbol("th")
    circuit = cirq.Circuit(
        [
            cirq.ry(th).on(q0),
            cirq.rx(0.31).on(q1),
            cirq.CZ(q0, q1),
            cirq.rz(-0.22).on(q0),
        ]
    )
    resolver = cirq.ParamResolver({"th": 0.41})
    qubits = [q0, q1]
    observable = 0.7 * cirq.Z(q0) + 0.2 * cirq.Z(q1) - 0.3 * cirq.X(q0) * cirq.X(q1)
    hmat = observable.matrix(qubits=qubits)

    state = cirq.Simulator(seed=123).simulate(
        cirq.resolve_parameters(circuit, resolver), qubit_order=qubits
    ).final_state_vector
    exact = float(np.vdot(state, hmat @ state).real)
    return circuit, resolver, qubits, hmat, exact


def test_trace_zne_returns_expected_keys_and_shapes() -> None:
    circuit, resolver, qubits, hmat, _ = _simple_problem()
    out = run_trace_zne(
        circuit,
        resolver,
        qubits,
        hmat,
        noise_scales=[1.0, 2.0, 3.0],
        fit_order=1,
        simulator_seed=11,
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
    )
    assert set(out.keys()) == {
        "noise_scales",
        "trace_energies",
        "energy_zne",
        "fit_order",
        "fit_coefficients",
        "baseline_noisy_energy",
    }
    assert len(out["noise_scales"]) == 3
    assert len(out["trace_energies"]) == 3
    assert out["fit_order"] == 1
    assert len(out["fit_coefficients"]) == 2


def test_trace_zne_moves_toward_noiseless_energy() -> None:
    circuit, resolver, qubits, hmat, exact = _simple_problem()
    out = run_trace_zne(
        circuit,
        resolver,
        qubits,
        hmat,
        noise_scales=[1.0, 2.0, 3.0],
        fit_order=1,
        simulator_seed=17,
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
    )
    baseline = float(out["baseline_noisy_energy"])
    zne = float(out["energy_zne"])
    assert abs(zne - exact) <= abs(baseline - exact) + 1e-10


def test_trace_zne_is_reproducible_with_fixed_seed() -> None:
    circuit, resolver, qubits, hmat, _ = _simple_problem()
    kwargs = dict(
        noise_scales=[1.0, 2.0, 3.0],
        fit_order=1,
        simulator_seed=19,
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
    )
    out1 = run_trace_zne(circuit, resolver, qubits, hmat, **kwargs)
    out2 = run_trace_zne(circuit, resolver, qubits, hmat, **kwargs)
    np.testing.assert_allclose(out1["trace_energies"], out2["trace_energies"], atol=1e-6, rtol=0.0)
    assert abs(float(out1["energy_zne"]) - float(out2["energy_zne"])) <= 1e-6


def test_invalid_zne_scales_raise() -> None:
    circuit, resolver, qubits, hmat, _ = _simple_problem()
    with pytest.raises(ValueError, match="noise_scale must be > 0"):
        run_trace_zne(
            circuit,
            resolver,
            qubits,
            hmat,
            noise_scales=[1.0, 0.0, 2.0],
            fit_order=1,
        )

    with pytest.raises(ValueError, match="at least two scale points"):
        run_trace_zne(
            circuit,
            resolver,
            qubits,
            hmat,
            noise_scales=[1.0],
            fit_order=1,
        )


def test_fit_order_validation_raises() -> None:
    circuit, resolver, qubits, hmat, _ = _simple_problem()
    with pytest.raises(ValueError, match="fit_order must be < number of points"):
        run_trace_zne(
            circuit,
            resolver,
            qubits,
            hmat,
            noise_scales=[1.0, 2.0, 3.0],
            fit_order=3,
        )


def test_trace_energy_at_noise_scale_rejects_nonpositive_scale() -> None:
    circuit, resolver, qubits, hmat, _ = _simple_problem()
    with pytest.raises(ValueError, match="noise_scale must be > 0"):
        trace_energy_at_noise_scale(
            circuit,
            resolver,
            qubits,
            hmat,
            noise_scale=-1.0,
            amp_damp_gamma=0.02,
            phase_damp_gamma=0.02,
            depol_prob=0.01,
            leakage_approx_prob=0.01,
            high_cz_multiplier=2.0,
        )
