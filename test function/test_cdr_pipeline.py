from __future__ import annotations

from pathlib import Path
import sys

import cirq
import numpy as np
import pytest
import sympy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from main_cursor_lib import (
    LocationAwareDecomposedNoise,
    clifford_snap_phi,
    clifford_snap_theta,
    count_non_clifford_ops,
    generate_near_clifford_param_sets,
    generate_random_clifford_analogue_param_sets,
    is_clifford_exponent,
    ordered_parameter_symbols,
    prepare_decomposed_ansatz_cirq,
)
from shot_measurement import (
    apply_cdr_models,
    estimate_energy_from_noisy_rho_shots,
    exact_noiseless_energy_from_statevector,
    exact_pauli_expectation_from_int_row,
    pauli_sum_to_int_observables,
    train_cdr_models,
    train_cf_models_per_pauli,
)


VQE_PARAMETERS_H4_8L = np.array([
    0.62769563, 0.72445584, 0.88167509, 1.21377448, 1.40410528,
    1.54272135, 1.65876094, 0.68703915, 0.53914478, 0.65488307,
    0.38242015, 1.21990629, -0.37586606, -0.55407695, 1.14433297,
    0.93390869, 1.06005155, 1.13875818, 0.31887654, 1.95665943,
    0.92840525, 0.96902297, 0.62448309, 0.85426258, 3.07549546,
    1.06236478, 1.83665633, -0.07343972, 1.05494565, 0.51913651,
    0.8990249, -0.7786491, 0.20801007, 1.33467266, 1.50009365,
    0.75667649, 1.43658698, 1.15788765, 0.93250988, 0.21547095,
    1.14415573, -0.34546883, 1.06523003, 1.45429555, 0.63474669,
    1.32076148, 0.51153084, 1.32344213, 0.64436686, 1.11864704,
    0.46516075, 1.52517135, 1.25912853, 0.6848312, -0.26316329,
    -0.07124165,
])


def _h4_8layer_fixture() -> tuple[cirq.Circuit, list[cirq.Qid], list[sympy.Symbol], dict]:
    circuit, qubits = prepare_decomposed_ansatz_cirq(num_spatial_orbitals=4, num_layers=8)
    symbols = ordered_parameter_symbols(num_spatial_orbitals=4, num_layers=8)
    target = {s: float(v) for s, v in zip(symbols, VQE_PARAMETERS_H4_8L)}
    return circuit, qubits, symbols, target


def _small_problem() -> tuple[cirq.Circuit, list[cirq.Qid], list[sympy.Symbol], dict, cirq.PauliSum]:
    """A 2-qubit, 1-layer mock that mirrors the symbol naming convention."""
    q0, q1 = cirq.LineQubit.range(2)
    th = sympy.Symbol("th_0_0")
    ph = sympy.Symbol("ph_0_1")
    circuit = cirq.Circuit(
        [
            cirq.PhasedXPowGate(phase_exponent=-0.25, exponent=th / sympy.pi).on(q0),
            cirq.ZPowGate(exponent=-ph / (2 * sympy.pi)).on(q1),
            cirq.CZ(q0, q1),
        ]
    )
    target = {th: 0.6, ph: 0.7}
    qubits = [q0, q1]
    observable = 0.6 * cirq.Z(q0) + 0.4 * cirq.X(q1) - 0.2 * cirq.Z(q0) * cirq.Z(q1)
    return circuit, qubits, [th, ph], target, observable


def test_is_clifford_exponent_basic() -> None:
    for e in [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]:
        assert is_clifford_exponent(e), f"expected Clifford for exponent={e}"
    for e in [0.25, 0.7, 1.3, -0.4]:
        assert not is_clifford_exponent(e), f"expected non-Clifford for exponent={e}"


def test_clifford_snap_theta_returns_multiple_of_pi_over_2() -> None:
    step = np.pi / 2.0
    for v in [0.1, 0.4, 1.7, -0.5, 4.0]:
        snapped = clifford_snap_theta(v)
        residual = (snapped / step) - round(snapped / step)
        assert abs(residual) < 1e-9
        assert is_clifford_exponent(snapped / np.pi)


def test_clifford_snap_phi_returns_multiple_of_pi() -> None:
    step = np.pi
    for v in [0.1, 1.0, np.pi - 0.05, -0.4, 7.5]:
        snapped = clifford_snap_phi(v)
        residual = (snapped / step) - round(snapped / step)
        assert abs(residual) < 1e-9
        assert is_clifford_exponent(snapped / (2 * np.pi))


def test_count_non_clifford_ops_zero_on_fully_clifford_resolver() -> None:
    circuit, _, symbols, _ = _h4_8layer_fixture()
    fully_clifford = {s: 0.0 for s in symbols}
    assert count_non_clifford_ops(circuit, fully_clifford) == 0


def test_count_non_clifford_ops_target_has_expected_order_of_magnitude() -> None:
    circuit, _, _, target = _h4_8layer_fixture()
    n = count_non_clifford_ops(circuit, target)
    assert n > 100, f"expected ~192 non-Clifford ops on target; got {n}"
    assert n == 192


def test_generate_near_clifford_param_sets_respects_t_max() -> None:
    circuit, _, symbols, target = _h4_8layer_fixture()
    t_max = 32
    resolvers = generate_near_clifford_param_sets(
        target, symbols, num_circuits=6, t_max=t_max, circuit=circuit, seed=11,
    )
    assert len(resolvers) == 6
    for r in resolvers:
        assert count_non_clifford_ops(circuit, r) <= t_max


def test_generate_near_clifford_param_sets_min_snap_fraction_lower_bound() -> None:
    circuit, _, symbols, target = _h4_8layer_fixture()
    n_total = len(symbols)
    frac = 0.5
    resolvers = generate_near_clifford_param_sets(
        target,
        symbols,
        num_circuits=4,
        t_max=10000,
        circuit=circuit,
        min_snap_fraction=frac,
        seed=23,
    )
    n_min = int(np.ceil(frac * n_total))
    for r in resolvers:
        snapped = sum(
            1 for s in symbols if abs(float(r[s]) - float(target[s])) > 1e-12
        )
        assert snapped >= n_min, f"expected >= {n_min} snapped, got {snapped}"


def test_generate_random_clifford_analogue_param_sets_fully_clifford() -> None:
    circuit, _, symbols, target = _h4_8layer_fixture()
    resolvers = generate_random_clifford_analogue_param_sets(
        target,
        symbols,
        num_circuits=5,
        circuit=circuit,
        seed=99,
    )
    assert len(resolvers) == 5
    for r in resolvers:
        assert count_non_clifford_ops(circuit, r) == 0


def test_exact_pauli_expectation_consistent_with_hamiltonian() -> None:
    circuit, qubits, _, target, obs = _small_problem()
    st = cirq.Simulator(seed=0).simulate(
        cirq.resolve_parameters(circuit, target), qubit_order=qubits
    ).final_state_vector
    e_h = exact_noiseless_energy_from_statevector(st, obs, qubits)
    oi, w, off = pauli_sum_to_int_observables(obs, qubits)
    s = off
    for k in range(len(w)):
        s += w[k] * exact_pauli_expectation_from_int_row(st, oi[k], qubits)
    assert abs(s - e_h) < 1e-8


def test_train_cf_models_per_pauli_smoke() -> None:
    circuit, qubits, symbols, target, obs = _small_problem()
    resolvers = generate_random_clifford_analogue_param_sets(
        target, symbols, num_circuits=4, circuit=circuit, seed=1,
    )
    noise_params = dict(
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
    )
    m = train_cf_models_per_pauli(
        circuit,
        obs,
        qubits,
        resolvers,
        noise_params=noise_params,
        num_shots=2000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
    )
    assert m["fit_scope"] == "per_pauli"
    assert len(m["coeffs_unmit_to_exact_per_term"]) == 3


def test_generate_near_clifford_param_sets_invalid_args_raise() -> None:
    circuit, _, symbols, target = _h4_8layer_fixture()
    with pytest.raises(ValueError, match="num_circuits"):
        generate_near_clifford_param_sets(
            target, symbols, num_circuits=0, t_max=32, circuit=circuit
        )
    with pytest.raises(ValueError, match="t_max"):
        generate_near_clifford_param_sets(
            target, symbols, num_circuits=2, t_max=-1, circuit=circuit
        )
    with pytest.raises(ValueError, match="min_snap_fraction"):
        generate_near_clifford_param_sets(
            target,
            symbols,
            num_circuits=2,
            t_max=32,
            circuit=circuit,
            min_snap_fraction=1.5,
        )


def test_train_cdr_models_returns_expected_keys() -> None:
    circuit, qubits, symbols, target, obs = _small_problem()
    resolvers = generate_near_clifford_param_sets(
        target, symbols, num_circuits=5, t_max=2, circuit=circuit, seed=3,
    )
    noise_params = dict(
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
    )
    models = train_cdr_models(
        circuit,
        obs,
        qubits,
        resolvers,
        noise_params=noise_params,
        simulator_seed=11,
        num_shots=2000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        sampling_seed=17,
    )
    expected_keys = {
        "training_exact_energies",
        "training_unmit_energies",
        "training_rem_energies",
        "training_t_remaining",
        "coeffs_unmit_to_exact",
        "coeffs_rem_to_exact",
    }
    assert expected_keys.issubset(models.keys())
    assert len(models["training_exact_energies"]) == 5
    assert len(models["training_t_remaining"]) == 5
    assert len(models["coeffs_unmit_to_exact"]) == 2
    assert len(models["coeffs_rem_to_exact"]) == 2


def test_apply_cdr_models_outputs_finite() -> None:
    circuit, qubits, symbols, target, obs = _small_problem()
    resolvers = generate_near_clifford_param_sets(
        target, symbols, num_circuits=4, t_max=2, circuit=circuit, seed=5,
    )
    noise_params = dict(
        amp_damp_gamma=0.02,
        phase_damp_gamma=0.02,
        depol_prob=0.01,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
    )
    models = train_cdr_models(
        circuit,
        obs,
        qubits,
        resolvers,
        noise_params=noise_params,
        simulator_seed=13,
        num_shots=2000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        sampling_seed=19,
    )
    corrected = apply_cdr_models(
        target_unmit=models["training_unmit_energies"][0],
        target_rem=models["training_rem_energies"][0],
        models=models,
    )
    assert np.isfinite(corrected["cdr_unmit_corrected"])
    assert np.isfinite(corrected["cdr_rem_corrected"])


def test_cdr_unmit_better_or_equal_baseline_on_small_problem() -> None:
    """Loose CDR quality check: averaged over a few seeds, CDR-corrected target
    should not be markedly worse than the baseline noisy estimate."""
    circuit, qubits, symbols, target, obs = _small_problem()
    noise_params = dict(
        amp_damp_gamma=0.05,
        phase_damp_gamma=0.05,
        depol_prob=0.02,
        leakage_approx_prob=0.02,
        high_cz_multiplier=2.0,
    )

    state = cirq.Simulator(seed=7).simulate(
        cirq.resolve_parameters(circuit, target), qubit_order=qubits
    ).final_state_vector
    exact = exact_noiseless_energy_from_statevector(state, obs, qubits)

    from main_cursor_lib import LocationAwareDecomposedNoise
    rho = cirq.DensityMatrixSimulator(seed=7).simulate(
        cirq.resolve_parameters(circuit.with_noise(LocationAwareDecomposedNoise(**noise_params)), target),
        qubit_order=qubits,
    ).final_density_matrix
    baseline = estimate_energy_from_noisy_rho_shots(
        rho,
        obs,
        qubits,
        num_shots=4000,
        measurement_scheme="direct_pauli",
        apply_readout_noise=False,
        apply_rem=False,
        sampling_seed=29,
    )

    avg_cdr_err = 0.0
    avg_baseline_err = 0.0
    n_trials = 3
    for seed in [1, 2, 3]:
        resolvers = generate_near_clifford_param_sets(
            target, symbols, num_circuits=8, t_max=1, circuit=circuit, seed=seed,
        )
        models = train_cdr_models(
            circuit,
            obs,
            qubits,
            resolvers,
            noise_params=noise_params,
            simulator_seed=7,
            num_shots=4000,
            measurement_scheme="direct_pauli",
            apply_readout_noise=False,
            sampling_seed=29 + seed,
        )
        corrected = apply_cdr_models(
            target_unmit=baseline["energy_unmitigated"],
            target_rem=baseline["energy_rem"],
            models=models,
        )
        avg_cdr_err += abs(corrected["cdr_unmit_corrected"] - exact)
        avg_baseline_err += abs(baseline["energy_unmitigated"] - exact)

    avg_cdr_err /= n_trials
    avg_baseline_err /= n_trials
    assert avg_cdr_err <= avg_baseline_err + 0.5, (
        f"CDR avg err={avg_cdr_err:.4f} should not be much worse than "
        f"baseline avg err={avg_baseline_err:.4f}"
    )


def test_cdr_coefficients_not_fully_degenerate_and_print() -> None:
    """Diagnostic: print fitted CDR coefficients and guard against full degeneracy."""
    circuit, qubits, symbols, target, obs = _small_problem()
    resolvers = generate_near_clifford_param_sets(
        target, symbols, num_circuits=6, t_max=1, circuit=circuit, seed=41,
    )
    noise_params = dict(
        amp_damp_gamma=0.03,
        phase_damp_gamma=0.03,
        depol_prob=0.015,
        leakage_approx_prob=0.01,
        high_cz_multiplier=2.0,
    )
    # Use asymmetric readout calibration so unmit and REM branches are not trivially identical.
    p0 = np.array([0.95, 0.89], dtype=float)
    p1 = np.array([0.91, 0.96], dtype=float)

    total_models = train_cdr_models(
        circuit,
        obs,
        qubits,
        resolvers,
        noise_params=noise_params,
        simulator_seed=23,
        num_shots=6000,
        measurement_scheme="direct_pauli",
        p_0_success=p0,
        p_1_success=p1,
        apply_readout_noise=True,
        sampling_seed=97,
    )
    coeffs_total_u = np.asarray(total_models["coeffs_unmit_to_exact"], dtype=float)
    coeffs_total_r = np.asarray(total_models["coeffs_rem_to_exact"], dtype=float)
    print(f"total-energy fit unmit: a={coeffs_total_u[0]: .8f}, b={coeffs_total_u[1]: .8f}")
    print(f"total-energy fit rem:   a={coeffs_total_r[0]: .8f}, b={coeffs_total_r[1]: .8f}")

    per_pauli_models = train_cf_models_per_pauli(
        circuit,
        obs,
        qubits,
        resolvers,
        noise_params=noise_params,
        simulator_seed=23,
        num_shots=6000,
        measurement_scheme="direct_pauli",
        p_0_success=p0,
        p_1_success=p1,
        apply_readout_noise=True,
        sampling_seed=101,
    )
    coeffs_per_u = np.asarray(per_pauli_models["coeffs_unmit_to_exact_per_term"], dtype=float)
    coeffs_per_r = np.asarray(per_pauli_models["coeffs_rem_to_exact_per_term"], dtype=float)

    for i in range(coeffs_per_u.shape[0]):
        au, bu = float(coeffs_per_u[i, 0]), float(coeffs_per_u[i, 1])
        ar, br = float(coeffs_per_r[i, 0]), float(coeffs_per_r[i, 1])
        print(
            f"per-pauli term {i} unmit: a={au: .8f}, b={bu: .8f} | "
            f"rem: a={ar: .8f}, b={br: .8f}"
        )

    # Print raw training data behind per-Pauli fits: x (noisy term expectation) -> y (exact term expectation).
    observables_int, weights, offset = pauli_sum_to_int_observables(obs, qubits)
    term_strings = [str(term) for term in obs]
    n_terms = len(weights)
    n_train = len(resolvers)
    y_exact = np.zeros((n_train, n_terms), dtype=float)
    x_unmit = np.zeros((n_train, n_terms), dtype=float)
    x_rem = np.zeros((n_train, n_terms), dtype=float)

    noisy_circuit = circuit.with_noise(LocationAwareDecomposedNoise(**noise_params))
    t_remaining = []
    for i, resolver in enumerate(resolvers):
        resolved = cirq.resolve_parameters(circuit, resolver)
        state = cirq.Simulator(seed=23).simulate(resolved, qubit_order=qubits).final_state_vector
        for k in range(n_terms):
            y_exact[i, k] = exact_pauli_expectation_from_int_row(state, observables_int[k], qubits)

        resolved_noisy = cirq.resolve_parameters(noisy_circuit, resolver)
        rho = cirq.DensityMatrixSimulator(seed=23).simulate(
            resolved_noisy, qubit_order=qubits
        ).final_density_matrix
        est = estimate_energy_from_noisy_rho_shots(
            rho,
            obs,
            qubits,
            num_shots=6000,
            measurement_scheme="direct_pauli",
            p_0_success=p0,
            p_1_success=p1,
            apply_rem=True,
            apply_readout_noise=True,
            sampling_seed=1000 + i,
            return_per_term=True,
        )
        x_unmit[i, :] = est["per_term_unmitigated"]
        x_rem[i, :] = est["per_term_rem"]
        t_remaining.append(int(count_non_clifford_ops(circuit, resolver)))

    print("\nper-pauli training details")
    print(f"hamiltonian offset: {float(offset): .8f}")
    for k in range(n_terms):
        print(
            f"term {k}: observable={term_strings[k]} int={observables_int[k].tolist()} coeff={float(weights[k]): .8f}"
        )
        for i in range(n_train):
            print(
                f"  train[{i}] t_remaining={t_remaining[i]:3d} "
                f"x_unmit={x_unmit[i, k]: .8f} x_rem={x_rem[i, k]: .8f} y_exact={y_exact[i, k]: .8f}"
            )

    total_equal = np.allclose(coeffs_total_u, coeffs_total_r, atol=0.0, rtol=0.0)
    per_term_equal = np.allclose(coeffs_per_u, coeffs_per_r, atol=0.0, rtol=0.0)
    assert not (total_equal and per_term_equal), (
        "CDR fits are exactly degenerate (unmit == REM) in both total and per-pauli models. "
        f"total_unmit={coeffs_total_u.tolist()} total_rem={coeffs_total_r.tolist()} "
        f"per_unmit={coeffs_per_u.tolist()} per_rem={coeffs_per_r.tolist()}"
    )
