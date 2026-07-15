"""Tests for the isolated selected-CDR training / fitting variant."""

from __future__ import annotations

import cirq
import numpy as np
import sympy

import shot_measurement_select_CDR as selected_measurement


def test_maxmin_selects_both_extremes_around_local_target():
    expectations = np.asarray(
        [[-1.0], [-0.9], [-0.1], [0.0], [0.1], [0.9], [1.0]],
        dtype=float,
    )
    parameters = np.arange(len(expectations), dtype=float).reshape(-1, 1)
    indices = selected_measurement.select_weighted_maxmin_indices(
        expectations,
        np.asarray([1.0]),
        num_select=3,
        parameter_vectors=parameters,
        target_parameter_vector=np.asarray([3.0]),
        local_count=1,
    )

    chosen = expectations[indices, 0]
    assert 3 in indices, "the target-local candidate should be reserved"
    assert float(np.min(chosen)) == -1.0
    assert float(np.max(chosen)) == 1.0


def test_hamiltonian_weight_controls_which_direction_is_covered_first():
    expectations = np.asarray(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [-1.0, 0.0],
            [0.0, 1.0],
            [0.0, -1.0],
        ],
        dtype=float,
    )
    parameters = np.arange(len(expectations), dtype=float).reshape(-1, 1)
    indices = selected_measurement.select_weighted_maxmin_indices(
        expectations,
        np.asarray([100.0, 1.0]),
        num_select=3,
        parameter_vectors=parameters,
        target_parameter_vector=np.asarray([0.0]),
        local_count=1,
    )

    assert indices == [0, 1, 2]


def test_resolver_selection_uses_exact_pauli_spread():
    qubit = cirq.LineQubit(0)
    theta = sympy.Symbol("th_0")
    circuit = cirq.Circuit(cirq.rx(theta).on(qubit))
    observable = cirq.PauliSum.from_pauli_strings([cirq.Z(qubit)])
    angles = np.linspace(0.0, 2.0 * np.pi, 17, endpoint=False)
    pool = [{theta: float(angle)} for angle in angles]

    selected, diagnostics = selected_measurement.select_cdr_resolvers(
        pool,
        ansatz_circuit=circuit,
        observable_h=observable,
        qubits=[qubit],
        symbols=[theta],
        target_params={theta: 0.0},
        num_select=5,
        local_count=1,
        simulator_seed=7,
    )

    assert len(selected) == 5
    assert diagnostics["pool_size"] == 17
    assert diagnostics["selected_count"] == 5
    assert diagnostics["selected_expectation_max_per_term"][0] > 0.99
    assert diagnostics["selected_expectation_min_per_term"][0] < -0.98
    assert diagnostics["selected_weighted_mean_range"] > 1.97


def test_affine_fit_ignores_zero_exact_points_and_keeps_intercept():
    # Many near-zero exact points plus informative ones generated from
    # exact = 1.25 * noisy + 0.08.  The zeros must not pin the intercept to 0.
    noisy_info = np.asarray([0.60, 0.80, -0.55, -0.75], dtype=float)
    exact_info = 1.25 * noisy_info + 0.08
    noisy = np.concatenate([[0.01, -0.02, 0.00], noisy_info])
    exact = np.concatenate([[0.00, 0.00, 0.00], exact_info])

    a, b, info = selected_measurement.robust_per_term_line_fit(
        noisy,
        exact,
        fit_mode="affine",
        exact_tol=0.05,
        min_points=2,
    )

    assert info["fit_used"] == "affine"
    assert info["n_informative"] == 4
    assert abs(a - 1.25) < 1e-9
    assert abs(b - 0.08) < 1e-9


def test_identity_fallback_when_all_exact_values_are_zero():
    noisy = np.asarray([0.05, -0.02, 0.10, -0.08], dtype=float)
    exact = np.zeros_like(noisy)

    a, b, info = selected_measurement.robust_per_term_line_fit(
        noisy,
        exact,
        fit_mode="affine",
        exact_tol=0.05,
        min_points=2,
    )

    assert info["fit_used"] == "identity"
    assert info["n_informative"] == 0
    assert (a, b) == (1.0, 0.0)


def test_default_run_mitigation_refits_affine_without_circuit_pooling():
    qubit = cirq.LineQubit(0)
    theta = sympy.Symbol("th_0")
    circuit = cirq.Circuit(cirq.rx(theta).on(qubit))
    observable = cirq.PauliSum.from_pauli_strings([cirq.Z(qubit)])
    base_measurement = selected_measurement._base
    original_generator = base_measurement.generate_near_clifford_param_sets
    original_train = base_measurement.train_cf_models_per_pauli

    result = selected_measurement.run_mitigation(
        "cdr",
        ansatz_circuit=circuit,
        observable_h=observable,
        qubits=[qubit],
        target_resolver={theta: 0.37},
        target_params={theta: 0.37},
        symbols=[theta],
        base_noise_cfg={
            "two_qubit_depol_prob": 0.01,
            "one_qubit_depol_prob": 0.01,
            "cross_chip_two_qubit_depol_prob": 0.01,
        },
        shot_cfg={
            "num_shots": 256,
            "measurement_scheme": "direct_pauli",
            "apply_readout_noise": False,
            "sampling_seed": 11,
        },
        readout_cal={},
        cdr_cfg={
            "num_circuits": 6,
            "t_max": 0,
            "seed": 13,
            "cdr_fit_scope": "per_pauli",
            # Defaults under test: selection_method=none, per_term_fit=affine.
        },
        simulator_seed=17,
    )

    assert "cdr_selection" not in result
    assert result["cdr_per_term_fit"]["fit_mode"] == "affine"
    training = result["cdr_models"]["training_exact_per_term"]
    assert np.asarray(training).shape == (6, 1)
    coeffs = result["cdr_models"]["coeffs_rem_to_exact_per_term"][0]
    assert len(coeffs) == 2
    assert base_measurement.generate_near_clifford_param_sets is original_generator
    assert base_measurement.train_cf_models_per_pauli is original_train


def test_opt_in_maxmin_selection_still_works():
    qubit = cirq.LineQubit(0)
    theta = sympy.Symbol("th_0")
    circuit = cirq.Circuit(cirq.rx(theta).on(qubit))
    observable = cirq.PauliSum.from_pauli_strings([cirq.Z(qubit)])
    base_measurement = selected_measurement._base
    original_generator = base_measurement.generate_near_clifford_param_sets

    result = selected_measurement.run_mitigation(
        "cdr",
        ansatz_circuit=circuit,
        observable_h=observable,
        qubits=[qubit],
        target_resolver={theta: 0.37},
        target_params={theta: 0.37},
        symbols=[theta],
        base_noise_cfg={
            "two_qubit_depol_prob": 0.01,
            "one_qubit_depol_prob": 0.01,
            "cross_chip_two_qubit_depol_prob": 0.01,
        },
        shot_cfg={
            "num_shots": 256,
            "measurement_scheme": "direct_pauli",
            "apply_readout_noise": False,
            "sampling_seed": 11,
        },
        readout_cal={},
        cdr_cfg={
            "num_circuits": 4,
            "t_max": 0,
            "seed": 13,
            "cdr_fit_scope": "per_pauli",
            "selection_method": "weighted_maxmin",
            "selection_pool_size": 12,
            "selection_local_count": 1,
            "per_term_fit": "affine",
        },
        simulator_seed=17,
    )

    assert result["cdr_selection"]["pool_size"] == 12
    assert result["cdr_selection"]["selected_count"] == 4
    assert result["cdr_per_term_fit"]["fit_mode"] == "affine"
    assert base_measurement.generate_near_clifford_param_sets is original_generator
