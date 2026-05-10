from __future__ import annotations

from pathlib import Path
import sys

import cirq
import numpy as np
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shot_measurement import exact_noiseless_energy_from_statevector, run_shot_zne


def _simple_problem() -> tuple[cirq.Circuit, cirq.ParamResolver, list[cirq.Qid], cirq.PauliSum, float]:
    q0, q1 = cirq.LineQubit.range(2)
    th = sympy.Symbol("th")
    circuit = cirq.Circuit(
        [
            cirq.ry(th).on(q0),
            cirq.rx(0.22).on(q1),
            cirq.CZ(q0, q1),
            cirq.rz(-0.19).on(q1),
        ]
    )
    resolver = cirq.ParamResolver({"th": 0.37})
    qubits = [q0, q1]
    observable = 0.5 * cirq.Z(q0) + 0.4 * cirq.Z(q1) - 0.2 * cirq.X(q0) * cirq.X(q1) + 0.1

    state = cirq.Simulator(seed=3).simulate(
        cirq.resolve_parameters(circuit, resolver), qubit_order=qubits
    ).final_state_vector
    exact = exact_noiseless_energy_from_statevector(state, observable, qubits)
    return circuit, resolver, qubits, observable, float(exact)


def test_shot_zne_runs_and_returns_points() -> None:
    circuit, resolver, qubits, observable, _ = _simple_problem()
    out = run_shot_zne(
        circuit,
        resolver,
        observable,
        qubits,
        noise_scales=[1.0, 2.0, 3.0],
        fit_order=1,
        simulator_seed=23,
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
        num_shots=3000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=31,
    )
    assert len(out["noise_scales"]) == 3
    assert len(out["shot_unmitigated_energies"]) == 3
    assert len(out["shot_rem_energies"]) == 3
    assert len(out["fit_coefficients_unmitigated"]) == 2


def test_shot_zne_unmitigated_reasonable_against_exact() -> None:
    circuit, resolver, qubits, observable, exact = _simple_problem()
    out = run_shot_zne(
        circuit,
        resolver,
        observable,
        qubits,
        noise_scales=[1.0, 2.0, 3.0],
        fit_order=1,
        simulator_seed=29,
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
        num_shots=5000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=41,
    )
    assert abs(float(out["zne_unmitigated"]) - exact) <= 0.35


def test_shot_zne_reproducible_with_fixed_seed() -> None:
    circuit, resolver, qubits, observable, _ = _simple_problem()
    kwargs = dict(
        noise_scales=[1.0, 2.0, 3.0],
        fit_order=1,
        simulator_seed=37,
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
        num_shots=3000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=53,
    )
    out1 = run_shot_zne(circuit, resolver, observable, qubits, **kwargs)
    out2 = run_shot_zne(circuit, resolver, observable, qubits, **kwargs)
    assert out1["shot_unmitigated_energies"] == out2["shot_unmitigated_energies"]
    assert out1["zne_unmitigated"] == out2["zne_unmitigated"]


def test_shot_zne_rem_optional_path() -> None:
    circuit, resolver, qubits, observable, _ = _simple_problem()
    out = run_shot_zne(
        circuit,
        resolver,
        observable,
        qubits,
        noise_scales=[1.0, 2.0, 3.0],
        fit_order=1,
        simulator_seed=43,
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
        num_shots=3000,
        measurement_scheme="direct_pauli",
        p_0_success=np.array([0.92, 0.88]),
        p_1_success=np.array([0.95, 0.91]),
        apply_readout_noise=True,
        apply_rem=True,
        sampling_seed=61,
        extrapolate_rem=True,
    )
    assert "zne_rem" in out
    assert np.isfinite(float(out["zne_rem"]))
