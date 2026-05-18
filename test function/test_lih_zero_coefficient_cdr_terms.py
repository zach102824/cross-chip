"""Validate LiH per-Pauli CDR terms with fitted a=b≈0 (see full_CDR_details doc)."""

from __future__ import annotations

import sys
from pathlib import Path

import cirq
import numpy as np
import pytest
import sympy

_REPO = Path(__file__).resolve().parents[1]
_LIH_CASE = _REPO / "test_LiH_case"
for _p in (str(_REPO), str(_LIH_CASE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from generate_lih_hamiltonian_paper_lih import build_paper_lih_hamiltonian
from main_cursor_lib_test_LiH import (
    GateArityDepolarizingNoise,
    count_non_clifford_ops,
    qubit_operator_to_pauli_sum,
)
from optimize_lih_fig13_lbfgs import lih_fig13_circuit
from shot_measurement_test_LiH import (
    _generate_near_clifford_resolvers_fallback,
    exact_pauli_expectation_from_int_row,
    estimate_energy_from_noisy_rho_shots,
    int_observable_to_pauli_string,
    pauli_sum_to_int_observables,
    train_cf_models_per_pauli,
)

# Matches docs/full_CDR_details_for_LiH_testing_case.md configuration block.
_LIH_DOC_BOND = 2.2
_LIH_DOC_PARAMS = np.array([-0.444980732142, 0.476365247616, 0.1426860331], dtype=float)
_LIH_DOC_NUM_CIRCUITS = 5
_LIH_DOC_T_MAX = 2
_LIH_DOC_CDR_SEED = 42
_LIH_DOC_NUM_SHOTS = 8192
_LIH_DOC_SAMPLING_SEED = 1234
_LIH_DOC_SIM_SEED = 1234
_LIH_DOC_READOUT_P0 = np.array([0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514], dtype=float)
_LIH_DOC_READOUT_P1 = np.array([0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514], dtype=float)
_LIH_DOC_NOISE = {
    "two_qubit_depol_prob": 0.01,
    "one_qubit_depol_prob": 0.001,
}
_LIH_OGM_FILE = Path(
    "/Users/zacharyhe/shadowgrouping/haozhaowu/LiH/hamil_class/ogm_outputs"
    f"/OGM_ogm_LiH_{_LIH_DOC_BOND:.1f}.txt"
)
_LIH_SHADOWGROUPING_ROOT = Path("/Users/zacharyhe/shadowgrouping")
_LIH_COEFF_ZERO_TOL = 1.0e-10
_LIH_EXACT_ZERO_TOL = 1.0e-10
_LIH_PAULI_MAG_BOUND = 1.0 + 1.0e-9


def _lih_doc_fixture() -> tuple[
    cirq.Circuit,
    list[cirq.Qid],
    list[sympy.Symbol],
    dict[sympy.Symbol, float],
    cirq.PauliSum,
]:
    """FIG.13 compiled LiH ansatz + paper Hamiltonian at bond 2.2 (doc config)."""
    theta1 = sympy.Symbol("theta1")
    theta2 = sympy.Symbol("theta2")
    theta3 = sympy.Symbol("theta3")
    circuit, qubits = lih_fig13_circuit(theta1, theta2, theta3)
    symbols = [theta1, theta2, theta3]
    target = {
        theta1: float(_LIH_DOC_PARAMS[0]),
        theta2: float(_LIH_DOC_PARAMS[1]),
        theta3: float(_LIH_DOC_PARAMS[2]),
    }
    h_op, _meta = build_paper_lih_hamiltonian(_LIH_DOC_BOND)
    pauli_sum = qubit_operator_to_pauli_sum(h_op, list(qubits))
    return circuit, list(qubits), symbols, target, pauli_sum


def _lih_zero_coefficient_term_indices(
    coeffs_unmit: np.ndarray,
    coeffs_rem: np.ndarray,
    *,
    tol: float = _LIH_COEFF_ZERO_TOL,
) -> list[int]:
    """Indices where both unmit and rem affine fits are effectively (a,b)=(0,0)."""
    n = min(len(coeffs_unmit), len(coeffs_rem))
    out: list[int] = []
    for k in range(n):
        au, bu = float(coeffs_unmit[k, 0]), float(coeffs_unmit[k, 1])
        ar, br = float(coeffs_rem[k, 0]), float(coeffs_rem[k, 1])
        if (
            abs(au) <= tol
            and abs(bu) <= tol
            and abs(ar) <= tol
            and abs(br) <= tol
        ):
            out.append(k)
    return out


def _lih_collect_per_term_training_data(
    circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    resolvers: list[dict],
    *,
    noise_params: dict,
    num_shots: int,
    measurement_scheme: str,
    ogm_file: str | Path | None,
    shadowgrouping_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Return per-term training arrays [n_train, n_terms] and Pauli labels."""
    observables_int, weights, _offset = pauli_sum_to_int_observables(observable_h, qubits)
    n_terms = len(weights)
    n_train = len(resolvers)
    y_noiseless = np.zeros((n_train, n_terms), dtype=float)
    x_unmit = np.zeros((n_train, n_terms), dtype=float)
    x_rem = np.zeros((n_train, n_terms), dtype=float)
    labels = [int_observable_to_pauli_string(observables_int[k]) for k in range(n_terms)]

    noise_model = GateArityDepolarizingNoise(**noise_params)
    noisy_template = circuit.with_noise(noise_model)

    for i, resolver in enumerate(resolvers):
        resolved = cirq.resolve_parameters(circuit, resolver)
        state = cirq.Simulator(seed=_LIH_DOC_SIM_SEED).simulate(
            resolved, qubit_order=qubits
        ).final_state_vector
        for k in range(n_terms):
            y_noiseless[i, k] = exact_pauli_expectation_from_int_row(
                state, observables_int[k], qubits
            )

        resolved_noisy = cirq.resolve_parameters(noisy_template, resolver)
        rho = cirq.DensityMatrixSimulator(seed=_LIH_DOC_SIM_SEED).simulate(
            resolved_noisy, qubit_order=qubits
        ).final_density_matrix
        est = estimate_energy_from_noisy_rho_shots(
            rho,
            observable_h,
            qubits,
            num_shots=num_shots,
            measurement_scheme=measurement_scheme,
            p_0_success=_LIH_DOC_READOUT_P0,
            p_1_success=_LIH_DOC_READOUT_P1,
            apply_rem=True,
            apply_readout_noise=True,
            sampling_seed=_LIH_DOC_SAMPLING_SEED + i,
            ogm_file=ogm_file,
            shadowgrouping_root=shadowgrouping_root,
            return_per_term=True,
        )
        x_unmit[i, :] = est["per_term_unmitigated"]
        x_rem[i, :] = est["per_term_rem"]

    return y_noiseless, x_unmit, x_rem, labels


def test_lih_zero_coefficient_terms_have_near_zero_expectations() -> None:
    """Randomly sample 10 per-Pauli terms with a=b≈0 and verify expectations analytically.

    - Strict: noiseless exact training expectations |y_noiseless| < 1e-10 (Clifford circuits).
    - Loose: finite-shot unmit/REM are valid Pauli estimates (|x| ≤ 1); may be non-zero
      even when y_noiseless=0 because depolarizing bias is not shot noise.
    """
    if not _LIH_OGM_FILE.is_file():
        pytest.skip(f"LiH OGM file not found: {_LIH_OGM_FILE}")
    if not _LIH_SHADOWGROUPING_ROOT.is_dir():
        pytest.skip(f"shadowgrouping root not found: {_LIH_SHADOWGROUPING_ROOT}")

    circuit, qubits, symbols, target, pauli_sum = _lih_doc_fixture()
    resolvers = _generate_near_clifford_resolvers_fallback(
        target,
        symbols,
        num_circuits=_LIH_DOC_NUM_CIRCUITS,
        t_max=_LIH_DOC_T_MAX,
        circuit=circuit,
        min_snap_fraction=0.0,
        seed=_LIH_DOC_CDR_SEED,
    )
    assert len(resolvers) == _LIH_DOC_NUM_CIRCUITS

    models = train_cf_models_per_pauli(
        circuit,
        pauli_sum,
        qubits,
        resolvers,
        noise_params=_LIH_DOC_NOISE,
        simulator_seed=_LIH_DOC_SIM_SEED,
        num_shots=_LIH_DOC_NUM_SHOTS,
        measurement_scheme="ogm",
        p_0_success=_LIH_DOC_READOUT_P0,
        p_1_success=_LIH_DOC_READOUT_P1,
        apply_readout_noise=True,
        sampling_seed=_LIH_DOC_SAMPLING_SEED,
        ogm_file=_LIH_OGM_FILE,
        shadowgrouping_root=_LIH_SHADOWGROUPING_ROOT,
    )
    y_noiseless, x_unmit, x_rem, labels = _lih_collect_per_term_training_data(
        circuit,
        pauli_sum,
        qubits,
        resolvers,
        noise_params=_LIH_DOC_NOISE,
        num_shots=_LIH_DOC_NUM_SHOTS,
        measurement_scheme="ogm",
        ogm_file=_LIH_OGM_FILE,
        shadowgrouping_root=_LIH_SHADOWGROUPING_ROOT,
    )

    coeffs_u = np.asarray(models["coeffs_unmit_to_exact_per_term"], dtype=float)
    coeffs_r = np.asarray(models["coeffs_rem_to_exact_per_term"], dtype=float)
    zero_terms = _lih_zero_coefficient_term_indices(coeffs_u, coeffs_r)
    true_zero_noiseless = [
        k
        for k in zero_terms
        if float(np.max(np.abs(y_noiseless[:, k]))) <= _LIH_EXACT_ZERO_TOL
    ]
    assert len(zero_terms) >= 10, (
        f"Need at least 10 zero-coefficient terms for sampling; found {len(zero_terms)}."
    )
    assert len(true_zero_noiseless) >= 10, (
        "Expected >=10 terms with a=b≈0 and all noiseless training ⟨P_k⟩≈0; "
        f"zero_coeff={len(zero_terms)}, true_zero_noiseless={len(true_zero_noiseless)}."
    )

    pool = list(true_zero_noiseless)
    sampled: list[int] = []
    if 0 in pool:
        sampled.append(0)
        pool = [k for k in pool if k != 0]
    need = 10 - len(sampled)
    sampled.extend(list(np.random.choice(pool, size=need, replace=False)))
    print(f"sampled zero-coefficient term indices: {sampled}")

    failures: list[str] = []
    for k in sampled:
        label = labels[k]
        max_abs_yn = float(np.max(np.abs(y_noiseless[:, k])))
        max_abs_xu = float(np.max(np.abs(x_unmit[:, k])))
        max_abs_xr = float(np.max(np.abs(x_rem[:, k])))
        print(
            f"term[{k:03d}] {label}  max|y_noiseless|={max_abs_yn:.3e}  "
            f"max|x_unmit|={max_abs_xu:.3e}  max|x_rem|={max_abs_xr:.3e}"
        )
        for i in range(len(resolvers)):
            t_rem = int(count_non_clifford_ops(circuit, resolvers[i]))
            yn = float(y_noiseless[i, k])
            xu = float(x_unmit[i, k])
            xr = float(x_rem[i, k])
            if abs(yn) > _LIH_EXACT_ZERO_TOL:
                failures.append(
                    f"term[{k}] {label} train[{i}] t_remaining={t_rem}: "
                    f"|y_noiseless|={abs(yn):.3e} > exact_tol={_LIH_EXACT_ZERO_TOL:.1e}"
                )
            if abs(xu) > _LIH_PAULI_MAG_BOUND:
                failures.append(
                    f"term[{k}] {label} train[{i}] t_remaining={t_rem}: "
                    f"|x_unmit|={abs(xu):.3e} exceeds Pauli bound {_LIH_PAULI_MAG_BOUND}"
                )
            if abs(xr) > _LIH_PAULI_MAG_BOUND:
                failures.append(
                    f"term[{k}] {label} train[{i}] t_remaining={t_rem}: "
                    f"|x_rem|={abs(xr):.3e} exceeds Pauli bound {_LIH_PAULI_MAG_BOUND}"
                )

    if failures:
        msg = "Zero-coefficient LiH term validation failed:\n" + "\n".join(failures)
        pytest.fail(msg)
