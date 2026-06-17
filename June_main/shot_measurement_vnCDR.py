from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterable

import cirq
import numpy as np

from main_cursor_vnCDR import (
    CZ_CROSS_CHIP_TAG,
    TWIRL_TAG,
    GateArityDepolarizingNoise,
    ONE_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPOL_PROB,
    count_non_clifford_ops,
    generate_near_clifford_param_sets,
    generate_random_clifford_analogue_param_sets,
    run_trace_zne,
    scale_noise_params_for_zne,
    stable_r2_from_sums,
    trace_energy,
    zne_extrapolate_energy,
)

MEASUREMENT_SCHEMES = {
    "ogm",
    "adaptive_shadows",
    "derandomization",
    "shadow_grouping_bernstein",
    "shadow_grouping_inconfidence",
    "direct_pauli",
}


def ensure_shadowgrouping_importable(shadowgrouping_root: str | Path) -> None:
    root = Path(shadowgrouping_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(
            f"shadowgrouping root does not exist: {root}. Set SHADOWGROUPING_ROOT correctly."
        )
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _load_shadowgrouping_scheme(
    measurement_scheme: str,
    observables_int: np.ndarray,
    weights: np.ndarray,
    epsilon: float,
    ogm_file: str | Path | None,
):
    from shadowgrouping.measurement_schemes import (  # type: ignore
        AdaptiveShadows,
        Derandomization,
        SettingSampler,
        Shadow_Grouping,
    )
    from shadowgrouping.weight_functions import Bernstein_bound, Inconfidence_bound  # type: ignore

    scheme = measurement_scheme.lower()
    if scheme == "ogm":
        if ogm_file is None:
            raise ValueError("measurement_scheme='ogm' requires ogm_file.")
        return SettingSampler(observables_int, weights, str(ogm_file), epsilon=epsilon)
    if scheme == "adaptive_shadows":
        return AdaptiveShadows(observables_int, weights, epsilon=epsilon)
    if scheme == "derandomization":
        return Derandomization(observables_int, weights, epsilon=epsilon, delta=0)
    if scheme == "shadow_grouping_bernstein":
        return Shadow_Grouping(observables_int, weights, epsilon, Bernstein_bound()())
    if scheme == "shadow_grouping_inconfidence":
        return Shadow_Grouping(observables_int, weights, epsilon, Inconfidence_bound()())
    raise ValueError(f"Unsupported measurement scheme: {measurement_scheme}")


def pauli_sum_to_int_observables(
    observable_h: cirq.PauliSum, qubits: list[cirq.Qid]
) -> tuple[np.ndarray, np.ndarray, float]:
    qubit_index = {q: i for i, q in enumerate(qubits)}
    char_to_int = {"I": 0, "X": 1, "Y": 2, "Z": 3}

    obs_rows = []
    weights = []
    offset = 0.0

    for term in observable_h:
        coeff = float(term.coefficient.real)
        if not term.qubits:
            offset += coeff
            continue
        row = np.zeros(len(qubits), dtype=int)
        for q, p in term.items():
            pchar = str(p)
            row[qubit_index[q]] = char_to_int[pchar]
        obs_rows.append(row)
        weights.append(coeff)

    if obs_rows:
        observables = np.stack(obs_rows, axis=0)
        w = np.asarray(weights, dtype=float)
    else:
        observables = np.zeros((0, len(qubits)), dtype=int)
        w = np.zeros((0,), dtype=float)
    return observables, w, float(offset)


def int_observable_to_pauli_string(obs_row: np.ndarray) -> str:
    int_to_char = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    return "".join(int_to_char[int(x)] for x in obs_row.tolist())


def rotation_circuit_for_pauli_string(pauli_str: str, qubits: list[cirq.Qid]) -> cirq.Circuit:
    circuit = cirq.Circuit()
    for q, ch in zip(qubits, pauli_str):
        if ch == "X":
            circuit.append(cirq.H(q))
        elif ch == "Y":
            circuit.append(cirq.rx(np.pi / 2).on(q))
    return circuit


def sample_measurement_basis_from_rho(
    rho: np.ndarray,
    pauli_str: str,
    qubits: list[cirq.Qid],
    num_shots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    rot_circ = rotation_circuit_for_pauli_string(pauli_str, qubits)
    sim = cirq.DensityMatrixSimulator()
    rotated = sim.simulate(rot_circ, initial_state=rho, qubit_order=qubits).final_density_matrix

    probs = np.real(np.diag(rotated))
    probs = np.clip(probs, 0, 1)
    probs /= np.sum(probs)

    sampled_ints = rng.choice(len(probs), size=num_shots, p=probs)
    bitstrings = np.array([list(np.binary_repr(x, width=len(qubits))) for x in sampled_ints], dtype=int)

    mask = np.array([ch != "I" for ch in pauli_str], dtype=bool)
    if np.any(~mask):
        bitstrings[:, ~mask] = 0
    return bitstrings


def apply_asymmetric_readout_noise(
    ideal_bitstrings: np.ndarray,
    p_0_success: np.ndarray,
    p_1_success: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    noisy = ideal_bitstrings.copy()
    num_shots, num_qubits = noisy.shape
    p_flip_0_to_1 = 1.0 - p_0_success
    p_flip_1_to_0 = 1.0 - p_1_success

    for j in range(num_qubits):
        rands = rng.random(num_shots)
        flip01 = (ideal_bitstrings[:, j] == 0) & (rands < p_flip_0_to_1[j])
        flip10 = (ideal_bitstrings[:, j] == 1) & (rands < p_flip_1_to_0[j])
        noisy[flip01, j] = 1
        noisy[flip10, j] = 0
    return noisy


def rem_z_vectors(p_0_success: np.ndarray, p_1_success: np.ndarray) -> np.ndarray:
    p_flip_0_to_1 = 1.0 - p_0_success
    p_flip_1_to_0 = 1.0 - p_1_success
    e_vec = np.array([1.0, 1.0])
    z_matrix = np.array([[1.0, 0.0], [0.0, -1.0]])
    out = []
    for eps, eta in zip(p_flip_0_to_1, p_flip_1_to_0):
        a = np.array([[1.0 - eps, eta], [eps, 1.0 - eta]])
        out.append(e_vec @ z_matrix @ np.linalg.inv(a))
    return np.asarray(out)


def _term_expectation_from_bitstrings(
    bitstrings: np.ndarray,
    obs_row: np.ndarray,
    rem_vectors: np.ndarray | None,
) -> float:
    active = np.where(obs_row != 0)[0]
    if len(active) == 0:
        return 1.0

    if rem_vectors is None:
        zvals = 1 - 2 * bitstrings[:, active]
        vals = np.prod(zvals, axis=1)
        return float(np.mean(vals))

    term_vals = np.ones(bitstrings.shape[0], dtype=float)
    for idx in active:
        mitigated_vals = np.where(bitstrings[:, idx] == 0, rem_vectors[idx, 0], rem_vectors[idx, 1])
        term_vals *= mitigated_vals
    return float(np.mean(term_vals))


def _is_setting_compatible(obs_row: np.ndarray, setting_row: np.ndarray) -> bool:
    # Compatible if every non-identity observable axis is measured in same axis.
    active = obs_row != 0
    if not np.any(active):
        return True
    return bool(np.all(setting_row[active] == obs_row[active]))


def _select_basis_for_observables_from_settings(
    observables_int: np.ndarray, sampled_settings: np.ndarray
) -> list[str]:
    # Assign each Hamiltonian term to a compatible sampled setting when possible;
    # if no compatible setting exists, fall back to direct term-by-term basis.
    bases: list[str] = []
    for obs_row in observables_int:
        compatible = [s for s in sampled_settings if _is_setting_compatible(obs_row, s)]
        if compatible:
            chosen = np.asarray(compatible[0], dtype=int)
        else:
            chosen = np.asarray(obs_row, dtype=int)
        bases.append(int_observable_to_pauli_string(chosen))
    return bases


def _hashable_setting(setting_row: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in setting_row.tolist())


def _unique_settings_with_counts(sampled_settings: np.ndarray) -> tuple[list[np.ndarray], list[int]]:
    counts: dict[tuple[int, ...], int] = {}
    for row in sampled_settings:
        key = _hashable_setting(np.asarray(row, dtype=int))
        counts[key] = counts.get(key, 0) + 1
    unique_rows = [np.asarray(key, dtype=int) for key in counts.keys()]
    unique_counts = [counts[key] for key in counts.keys()]
    return unique_rows, unique_counts


def _build_direct_pauli_settings(observables_int: np.ndarray, num_shots: int) -> np.ndarray:
    """Allocate a fixed global shot budget across direct term-by-term Pauli settings."""
    num_terms = len(observables_int)
    if num_terms == 0:
        return np.zeros((0, 0), dtype=int)
    base = num_shots // num_terms
    rem = num_shots % num_terms
    rows = []
    for i, obs_row in enumerate(observables_int):
        n = base + (1 if i < rem else 0)
        if n > 0:
            rows.extend([np.asarray(obs_row, dtype=int)] * n)
    return np.asarray(rows, dtype=int)


def estimate_energy_from_noisy_rho_shots(
    rho: np.ndarray,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    *,
    num_shots: int = 8192,
    measurement_scheme: str = "ogm",
    p_0_success: Iterable[float] | None = None,
    p_1_success: Iterable[float] | None = None,
    apply_rem: bool = True,
    apply_readout_noise: bool = True,
    sampling_seed: int = 1234,
    epsilon: float = 0.1,
    ogm_file: str | Path | None = None,
    shadowgrouping_root: str | Path | None = None,
    return_per_term: bool = False,
) -> dict[str, Any]:
    scheme = measurement_scheme.lower()
    if scheme not in MEASUREMENT_SCHEMES:
        raise ValueError(f"Unsupported measurement scheme: {measurement_scheme}.")

    num_qubits = len(qubits)
    rng = np.random.default_rng(sampling_seed)
    p0 = np.ones(num_qubits) if p_0_success is None else np.asarray(list(p_0_success), dtype=float)
    p1 = np.ones(num_qubits) if p_1_success is None else np.asarray(list(p_1_success), dtype=float)
    if len(p0) != num_qubits or len(p1) != num_qubits:
        raise ValueError("Readout calibration arrays must match number of qubits.")

    observables_int, weights, offset = pauli_sum_to_int_observables(observable_h, qubits)
    if len(weights) == 0:
        out: dict[str, Any] = {"energy_unmitigated": offset, "energy_rem": offset, "offset": offset}
        if return_per_term:
            out["per_term_unmitigated"] = np.zeros((0,), dtype=float)
            out["per_term_rem"] = np.zeros((0,), dtype=float)
        return out

    rem_vectors = rem_z_vectors(p0, p1) if (apply_readout_noise and apply_rem) else None

    if scheme == "direct_pauli":
        sampled_settings = _build_direct_pauli_settings(observables_int, num_shots)
    else:
        if shadowgrouping_root is None:
            raise ValueError(
                "shadowgrouping_root is required for shadowgrouping schemes. "
                "Set SHADOWGROUPING_ROOT or pass it explicitly."
            )
        ensure_shadowgrouping_importable(shadowgrouping_root)
        method = _load_shadowgrouping_scheme(scheme, observables_int, weights, epsilon, ogm_file)
        settings, _ = method.find_setting(N_samples=num_shots)
        sampled_settings = np.asarray(settings, dtype=int)
        if sampled_settings.ndim == 1:
            sampled_settings = sampled_settings.reshape(1, -1)
    if sampled_settings.shape[0] == 0:
        out = {"energy_unmitigated": float(offset), "energy_rem": float(offset), "offset": float(offset)}
        if return_per_term:
            z = np.zeros(len(weights), dtype=float)
            out["per_term_unmitigated"] = z.copy()
            out["per_term_rem"] = z.copy()
        return out

    unique_settings, unique_counts = _unique_settings_with_counts(sampled_settings)
    basis_samples: dict[tuple[int, ...], np.ndarray] = {}
    for setting_row, count in zip(unique_settings, unique_counts):
        basis = int_observable_to_pauli_string(setting_row)
        ideal = sample_measurement_basis_from_rho(rho, basis, qubits, count, rng)
        if apply_readout_noise:
            noisy = apply_asymmetric_readout_noise(ideal, p0, p1, rng)
        else:
            noisy = ideal
        basis_samples[_hashable_setting(setting_row)] = noisy

    energy_unmit = offset
    energy_rem = offset
    per_term_unmit: list[float] | None = [] if return_per_term else None
    per_term_rem: list[float] | None = [] if return_per_term else None

    for obs_row, coeff in zip(observables_int, weights):
        compatible_keys = [
            _hashable_setting(srow) for srow in unique_settings if _is_setting_compatible(obs_row, srow)
        ]
        if not compatible_keys:
            # Fallback: directly measure this term with a minimal sample.
            direct_basis = int_observable_to_pauli_string(obs_row)
            bits = sample_measurement_basis_from_rho(rho, direct_basis, qubits, 1, rng)
            if apply_readout_noise:
                bits = apply_asymmetric_readout_noise(bits, p0, p1, rng)
            compatible_samples = [bits]
        else:
            compatible_samples = [basis_samples[k] for k in compatible_keys]

        total = sum(arr.shape[0] for arr in compatible_samples)
        unmit_acc = 0.0
        rem_acc = 0.0
        for bits in compatible_samples:
            n = bits.shape[0]
            unmit_acc += n * _term_expectation_from_bitstrings(bits, obs_row, rem_vectors=None)
            rem_acc += n * _term_expectation_from_bitstrings(bits, obs_row, rem_vectors=rem_vectors)
        term_u = unmit_acc / total
        term_r = rem_acc / total
        energy_unmit += coeff * term_u
        energy_rem += coeff * term_r
        if return_per_term:
            assert per_term_unmit is not None and per_term_rem is not None
            per_term_unmit.append(float(term_u))
            per_term_rem.append(float(term_r))

    result: dict[str, Any] = {
        "energy_unmitigated": float(energy_unmit),
        "energy_rem": float(energy_rem),
        "offset": float(offset),
    }
    if return_per_term:
        result["per_term_unmitigated"] = np.asarray(per_term_unmit, dtype=float)
        result["per_term_rem"] = np.asarray(per_term_rem, dtype=float)
    return result


def run_shot_zne(
    ansatz_circuit: cirq.Circuit,
    resolver: cirq.ParamResolver,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    *,
    noise_scales: list[float] | tuple[float, ...] = (1.0, 2.0, 3.0),
    fit_order: int = 1,
    simulator_seed: int = 1234,
    two_qubit_depol_prob: float = TWO_QUBIT_GATE_DEPOL_PROB,
    one_qubit_depol_prob: float = ONE_QUBIT_GATE_DEPOL_PROB,
    num_shots: int = 8192,
    measurement_scheme: str = "ogm",
    p_0_success: Iterable[float] | None = None,
    p_1_success: Iterable[float] | None = None,
    apply_rem: bool = True,
    apply_readout_noise: bool = True,
    sampling_seed: int = 1234,
    epsilon: float = 0.1,
    ogm_file: str | Path | None = None,
    shadowgrouping_root: str | Path | None = None,
    extrapolate_rem: bool = False,
) -> dict[str, object]:
    scales = [float(s) for s in noise_scales]
    if any(scale <= 0 for scale in scales):
        raise ValueError(f"All noise scales must be > 0, got {scales}.")

    shot_unmitigated: list[float] = []
    shot_rem: list[float] = []
    per_scale_results: list[dict[str, float]] = []

    for scale in scales:
        scaled = scale_noise_params_for_zne(
            scale,
            two_qubit_depol_prob=two_qubit_depol_prob,
            one_qubit_depol_prob=one_qubit_depol_prob,
        )
        noise_model = GateArityDepolarizingNoise(**scaled)
        noisy_circuit = ansatz_circuit.with_noise(noise_model)
        resolved_noisy = cirq.resolve_parameters(noisy_circuit, resolver)
        rho_noisy = cirq.DensityMatrixSimulator(seed=simulator_seed).simulate(
            resolved_noisy, qubit_order=qubits
        ).final_density_matrix
        rho_noisy = _sanitize_rho(rho_noisy)

        est = estimate_energy_from_noisy_rho_shots(
            rho_noisy,
            observable_h,
            qubits,
            num_shots=num_shots,
            measurement_scheme=measurement_scheme,
            p_0_success=p_0_success,
            p_1_success=p_1_success,
            apply_rem=apply_rem,
            apply_readout_noise=apply_readout_noise,
            sampling_seed=sampling_seed,
            epsilon=epsilon,
            ogm_file=ogm_file,
            shadowgrouping_root=shadowgrouping_root,
        )
        shot_unmitigated.append(float(est["energy_unmitigated"]))
        shot_rem.append(float(est["energy_rem"]))
        per_scale_results.append(
            {
                "noise_scale": float(scale),
                "energy_unmitigated": float(est["energy_unmitigated"]),
                "energy_rem": float(est["energy_rem"]),
            }
        )

    zne_unmit = zne_extrapolate_energy(scales, shot_unmitigated, fit_order=fit_order)
    out: dict[str, object] = {
        "noise_scales": [float(x) for x in scales],
        "shot_unmitigated_energies": [float(x) for x in shot_unmitigated],
        "shot_rem_energies": [float(x) for x in shot_rem],
        "fit_order": int(fit_order),
        "fit_coefficients_unmitigated": [float(x) for x in zne_unmit["fit_coefficients"]],
        "zne_unmitigated": float(zne_unmit["energy_zne"]),
        "per_scale_results": per_scale_results,
    }
    if extrapolate_rem:
        zne_rem = zne_extrapolate_energy(scales, shot_rem, fit_order=fit_order)
        out["fit_coefficients_rem"] = [float(x) for x in zne_rem["fit_coefficients"]]
        out["zne_rem"] = float(zne_rem["energy_zne"])
    return out


def exact_noiseless_energy_from_statevector(
    state: np.ndarray, observable_h: cirq.PauliSum, qubits: list[cirq.Qid]
) -> float:
    h_matrix = observable_h.matrix(qubits=qubits)
    return float(np.vdot(state, h_matrix @ state).real)


def exact_pauli_expectation_from_int_row(
    state: np.ndarray,
    obs_row: np.ndarray,
    qubits: list[cirq.Qid],
) -> float:
    """Exact ⟨P⟩ for one Pauli product; ``obs_row`` matches ``pauli_sum_to_int_observables`` encoding."""
    if not np.any(obs_row):
        return 1.0
    int_to_gate = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    pstr = cirq.PauliString()
    for i, v in enumerate(obs_row):
        vi = int(v)
        if vi == 0:
            continue
        pstr *= int_to_gate[vi](qubits[i])
    mat = pstr.matrix(qubits=qubits)
    return float(np.vdot(state, mat @ state).real)


def exact_trace_energy_from_density(
    rho: np.ndarray, observable_h: cirq.PauliSum, qubits: list[cirq.Qid]
) -> float:
    h_matrix = observable_h.matrix(qubits=qubits)
    return float(trace_energy(h_matrix, rho))


# ---------------------------------------------------------------------------
# Clifford Data Regression (CDR) orchestration
# ---------------------------------------------------------------------------


def sanitize_density_matrix(rho: np.ndarray) -> np.ndarray:
    """Symmetrize and renormalize a density matrix to absorb tiny floating-point
    deviations from the cirq density-matrix simulator.

    Public alias used by scripts that build rho manually; internal paths call this.
    """
    rho_h = 0.5 * (rho + np.conjugate(rho.T))
    tr = float(np.trace(rho_h).real)
    if tr > 0:
        rho_h = rho_h / tr
    return rho_h


def _sanitize_rho(rho: np.ndarray) -> np.ndarray:
    """Deprecated internal name; prefer :func:`sanitize_density_matrix`."""
    return sanitize_density_matrix(rho)


def _simulate_noisy_rho_for_resolver(
    ansatz_circuit: cirq.Circuit,
    resolver: dict,
    qubits: list[cirq.Qid],
    noise_params: dict,
    *,
    simulator_seed: int,
) -> np.ndarray:
    noise_model = GateArityDepolarizingNoise(**noise_params)
    noisy_circuit = ansatz_circuit.with_noise(noise_model)
    resolved_noisy = cirq.resolve_parameters(noisy_circuit, resolver)
    rho = cirq.DensityMatrixSimulator(seed=simulator_seed).simulate(
        resolved_noisy, qubit_order=qubits
    ).final_density_matrix
    return _sanitize_rho(rho)


def _simulate_noiseless_state_for_resolver(
    ansatz_circuit: cirq.Circuit,
    resolver: dict,
    qubits: list[cirq.Qid],
    *,
    simulator_seed: int,
) -> np.ndarray:
    resolved = cirq.resolve_parameters(ansatz_circuit, resolver)
    return cirq.Simulator(seed=simulator_seed).simulate(
        resolved, qubit_order=qubits
    ).final_state_vector


def train_cdr_models(
    ansatz_circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    resolvers: list[dict],
    *,
    noise_params: dict,
    simulator_seed: int = 1234,
    num_shots: int = 8192,
    measurement_scheme: str = "ogm",
    p_0_success: Iterable[float] | None = None,
    p_1_success: Iterable[float] | None = None,
    apply_readout_noise: bool = True,
    sampling_seed: int = 1234,
    epsilon: float = 0.1,
    ogm_file: str | Path | None = None,
    shadowgrouping_root: str | Path | None = None,
) -> dict[str, object]:
    """Train two linear CDR models from a list of near-Clifford resolvers.

    For each resolver we compute:
      - noiseless statevector exact energy `E_exact_i`,
      - shot-noisy unmitigated energy `E_unmit_i`,
      - shot-noisy REM-corrected energy `E_rem_i`.

    Returns the training data plus the linear fits

      Model B: E_exact ≈ a_B * E_unmit + b_B   (coeffs_unmit_to_exact)
      Model C: E_exact ≈ a_C * E_rem   + b_C   (coeffs_rem_to_exact)
    """
    if not resolvers:
        raise ValueError("At least one resolver is required to train CDR models.")

    h_matrix = observable_h.matrix(qubits=qubits)
    training_exact: list[float] = []
    training_unmit: list[float] = []
    training_rem: list[float] = []
    training_t_remaining: list[int] = []

    for resolver in resolvers:
        state = _simulate_noiseless_state_for_resolver(
            ansatz_circuit, resolver, qubits, simulator_seed=simulator_seed
        )
        exact_energy = float(np.vdot(state, h_matrix @ state).real)

        rho = _simulate_noisy_rho_for_resolver(
            ansatz_circuit, resolver, qubits, noise_params, simulator_seed=simulator_seed
        )

        est = estimate_energy_from_noisy_rho_shots(
            rho,
            observable_h,
            qubits,
            num_shots=num_shots,
            measurement_scheme=measurement_scheme,
            p_0_success=p_0_success,
            p_1_success=p_1_success,
            apply_rem=True,
            apply_readout_noise=apply_readout_noise,
            sampling_seed=sampling_seed,
            epsilon=epsilon,
            ogm_file=ogm_file,
            shadowgrouping_root=shadowgrouping_root,
        )

        training_exact.append(exact_energy)
        training_unmit.append(float(est["energy_unmitigated"]))
        training_rem.append(float(est["energy_rem"]))
        training_t_remaining.append(int(count_non_clifford_ops(ansatz_circuit, resolver)))

    x_unmit = np.asarray(training_unmit, dtype=float)
    x_rem = np.asarray(training_rem, dtype=float)
    y_exact = np.asarray(training_exact, dtype=float)

    if len(resolvers) >= 2 and float(np.std(x_unmit)) > 0.0:
        coeffs_unmit = np.polyfit(x_unmit, y_exact, deg=1)
    else:
        coeffs_unmit = np.array([1.0, float(np.mean(y_exact - x_unmit))])
    if len(resolvers) >= 2 and float(np.std(x_rem)) > 0.0:
        coeffs_rem = np.polyfit(x_rem, y_exact, deg=1)
    else:
        coeffs_rem = np.array([1.0, float(np.mean(y_exact - x_rem))])

    return {
        "training_exact_energies": [float(v) for v in training_exact],
        "training_unmit_energies": [float(v) for v in training_unmit],
        "training_rem_energies": [float(v) for v in training_rem],
        "training_t_remaining": [int(v) for v in training_t_remaining],
        "coeffs_unmit_to_exact": [float(v) for v in coeffs_unmit.tolist()],
        "coeffs_rem_to_exact": [float(v) for v in coeffs_rem.tolist()],
    }


def apply_cdr_models(
    target_unmit: float, target_rem: float, models: dict
) -> dict[str, float]:
    """Apply the two trained linear CDR models to the target circuit's
    unmitigated and REM energies.
    """
    coeffs_unmit = np.asarray(models["coeffs_unmit_to_exact"], dtype=float)
    coeffs_rem = np.asarray(models["coeffs_rem_to_exact"], dtype=float)
    return {
        "cdr_unmit_corrected": float(np.polyval(coeffs_unmit, float(target_unmit))),
        "cdr_rem_corrected": float(np.polyval(coeffs_rem, float(target_rem))),
    }


def train_cf_models_per_pauli(
    ansatz_circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    resolvers: list[dict],
    *,
    noise_params: dict,
    simulator_seed: int = 1234,
    num_shots: int = 8192,
    measurement_scheme: str = "ogm",
    p_0_success: Iterable[float] | None = None,
    p_1_success: Iterable[float] | None = None,
    apply_readout_noise: bool = True,
    sampling_seed: int = 1234,
    epsilon: float = 0.1,
    ogm_file: str | Path | None = None,
    shadowgrouping_root: str | Path | None = None,
) -> dict[str, object]:
    """Per-Pauli affine CF (paper-style): fit ``⟨P_k⟩_exact ≈ a_k ⟨P_k⟩_noisy + b_k`` per term."""
    if not resolvers:
        raise ValueError("At least one resolver is required to train per-Pauli CF models.")

    observables_int, weights, offset = pauli_sum_to_int_observables(observable_h, qubits)
    n_terms = len(weights)
    if n_terms == 0:
        return {
            "fit_scope": "per_pauli",
            "hamiltonian_offset": float(offset),
            "weights": [],
            "coeffs_unmit_to_exact_per_term": [],
            "coeffs_rem_to_exact_per_term": [],
            "r2_rem_to_exact_per_term": [],
            "training_t_remaining": [0] * len(resolvers),
        }

    n = len(resolvers)
    tex_exact = np.zeros((n, n_terms), dtype=float)
    tunmit = np.zeros((n, n_terms), dtype=float)
    trem = np.zeros((n, n_terms), dtype=float)
    t_rem_list: list[int] = []

    for i, resolver in enumerate(resolvers):
        state = _simulate_noiseless_state_for_resolver(
            ansatz_circuit, resolver, qubits, simulator_seed=simulator_seed
        )
        for k in range(n_terms):
            tex_exact[i, k] = exact_pauli_expectation_from_int_row(
                state, observables_int[k], qubits
            )
        rho = _simulate_noisy_rho_for_resolver(
            ansatz_circuit, resolver, qubits, noise_params, simulator_seed=simulator_seed
        )
        est = estimate_energy_from_noisy_rho_shots(
            rho,
            observable_h,
            qubits,
            num_shots=num_shots,
            measurement_scheme=measurement_scheme,
            p_0_success=p_0_success,
            p_1_success=p_1_success,
            apply_rem=True,
            apply_readout_noise=apply_readout_noise,
            sampling_seed=sampling_seed,
            epsilon=epsilon,
            ogm_file=ogm_file,
            shadowgrouping_root=shadowgrouping_root,
            return_per_term=True,
        )
        tunmit[i, :] = est["per_term_unmitigated"]
        trem[i, :] = est["per_term_rem"]
        t_rem_list.append(int(count_non_clifford_ops(ansatz_circuit, resolver)))

    coeffs_unmit: list[list[float]] = []
    coeffs_rem: list[list[float]] = []
    r2_rem: list[float] = []
    for k in range(n_terms):
        xu = tunmit[:, k]
        xr = trem[:, k]
        y = tex_exact[:, k]
        if n >= 2 and float(np.std(xu)) > 0.0:
            cu = np.polyfit(xu, y, deg=1)
        else:
            cu = np.array([1.0, float(np.mean(y - xu))])
        if n >= 2 and float(np.std(xr)) > 0.0:
            cr = np.polyfit(xr, y, deg=1)
        else:
            cr = np.array([1.0, float(np.mean(y - xr))])
        coeffs_unmit.append([float(cu[0]), float(cu[1])])
        coeffs_rem.append([float(cr[0]), float(cr[1])])
        y_pred = float(cr[0]) * xr + float(cr[1])
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2_rem.append(stable_r2_from_sums(ss_res, ss_tot, ss_res_tol=1e-4))

    return {
        "fit_scope": "per_pauli",
        "hamiltonian_offset": float(offset),
        "weights": [float(w) for w in weights],
        "coeffs_unmit_to_exact_per_term": coeffs_unmit,
        "coeffs_rem_to_exact_per_term": coeffs_rem,
        "r2_rem_to_exact_per_term": r2_rem,
        "training_t_remaining": t_rem_list,
    }


def apply_cf_models_per_pauli(
    target_unmit_per_term: np.ndarray,
    target_rem_per_term: np.ndarray,
    models: dict,
) -> dict[str, float]:
    """Apply per-Pauli CF models; energies use ``offset + Σ_k w_k (a_k x_k + b_k)``."""
    offset = float(models["hamiltonian_offset"])
    weights = np.asarray(models["weights"], dtype=float)
    cu = models["coeffs_unmit_to_exact_per_term"]
    cr = models["coeffs_rem_to_exact_per_term"]
    tu = np.asarray(target_unmit_per_term, dtype=float).ravel()
    tr = np.asarray(target_rem_per_term, dtype=float).ravel()
    if len(weights) == 0:
        return {"cdr_unmit_corrected": offset, "cdr_rem_corrected": offset}
    if len(tu) != len(weights) or len(tr) != len(weights):
        raise ValueError(
            f"Per-term target length {len(tu)} / {len(tr)} != number of weights {len(weights)}."
        )
    eu = offset
    er = offset
    for k in range(len(weights)):
        au, bu = float(cu[k][0]), float(cu[k][1])
        ar, br = float(cr[k][0]), float(cr[k][1])
        eu += weights[k] * float(np.polyval(np.array([au, bu]), tu[k]))
        er += weights[k] * float(np.polyval(np.array([ar, br]), tr[k]))
    return {"cdr_unmit_corrected": float(eu), "cdr_rem_corrected": float(er)}


# ---------------------------------------------------------------------------
# vnCDR: noise-amplified (folded) + Pauli-twirled circuit machinery
# ---------------------------------------------------------------------------

_PAULI_1Q_MATRIX: dict[str, np.ndarray] = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}
_PAULI_1Q_GATE: dict[str, cirq.Gate | None] = {
    "I": None,
    "X": cirq.X,
    "Y": cirq.Y,
    "Z": cirq.Z,
}
# All 16 two-qubit Pauli labels "AB" = A on qubit0, B on qubit1.
TWO_QUBIT_PAULI_LABELS: list[str] = [a + b for a in "IXYZ" for b in "IXYZ"]
_PAULI_2Q_MATRIX: dict[str, np.ndarray] = {
    lab: np.kron(_PAULI_1Q_MATRIX[lab[0]], _PAULI_1Q_MATRIX[lab[1]])
    for lab in TWO_QUBIT_PAULI_LABELS
}


def _rzx_unitary(theta: float) -> np.ndarray:
    """RZX(theta) = exp(-i*theta/2 * Z⊗X) as a 4x4 unitary (qubit0=Z, qubit1=X)."""
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    g = np.kron(z, x)
    return np.cos(theta / 2.0) * np.eye(4, dtype=complex) - 1j * np.sin(theta / 2.0) * g


def is_two_qubit_op(op: cirq.Operation) -> bool:
    """A foldable/twirlable two-qubit gate (CZ or the RZX MatrixGate)."""
    return op.gate is not None and len(op.qubits) == 2


def _is_self_inverse_unitary(u: np.ndarray, atol: float = 1e-9) -> bool:
    return bool(np.allclose(u @ u, np.eye(u.shape[0], dtype=complex), atol=atol))


def fold_two_qubit_gates(circuit: cirq.Circuit, c: int) -> cirq.Circuit:
    """Noise-amplify a circuit by physically folding only its two-qubit gates.

    For odd ``c`` each two-qubit gate ``G`` is replaced by ``c`` physical copies whose
    ideal action is still ``G``:
      - self-inverse gates (e.g. CZ): ``G`` repeated ``c`` times,
      - otherwise (e.g. RZX(π/2)): ``G, G†, G, ...`` (``c`` factors) with
        ``G† = MatrixGate(U†)``.
    ``H`` and single-qubit rotations are left untouched. Tags (e.g. the cross-chip
    tag) are preserved on every inserted copy so the per-gate noise model treats each
    folded copy exactly like the original physical gate.
    """
    c = int(c)
    if c < 1 or c % 2 == 0:
        raise ValueError(f"Fold factor c must be a positive odd integer, got {c}.")
    if c == 1:
        return circuit.copy()

    folded = cirq.Circuit()
    for moment in circuit:
        for op in moment.operations:
            if not is_two_qubit_op(op):
                folded.append(op)
                continue
            u = cirq.unitary(op.gate)
            tags = op.tags
            qubits = op.qubits
            if _is_self_inverse_unitary(u):
                for _ in range(c):
                    folded.append(op.gate.on(*qubits).with_tags(*tags))
            else:
                base_name = getattr(op.gate, "_name", None) or "G"
                g_dag = cirq.MatrixGate(u.conj().T, name=f"{base_name}_dag")
                for i in range(c):
                    if i % 2 == 0:
                        folded.append(op.gate.on(*qubits).with_tags(*tags))
                    else:
                        folded.append(g_dag.on(*qubits).with_tags(*tags))
    return folded


def pauli_from_matrix(matrix: np.ndarray, atol: float = 1e-6) -> str:
    """Identify which two-qubit Pauli ``matrix`` equals up to a global phase."""
    for label, pmat in _PAULI_2Q_MATRIX.items():
        overlap = np.trace(pmat.conj().T @ matrix) / 4.0
        if abs(abs(overlap) - 1.0) <= atol:
            return label
    raise ValueError("Matrix is not a two-qubit Pauli (up to global phase).")


_COMPENSATOR_TABLE_CACHE: dict[bytes, dict[str, str]] = {}


def compensator_table_for_unitary(u: np.ndarray) -> dict[str, str]:
    """Map each inserted Pauli ``P`` to its compensator ``R = U·P·U†`` (a Pauli).

    Valid because ``U`` (CZ / RZX) is Clifford, so conjugation maps Paulis to Paulis.
    Cached per distinct gate unitary.
    """
    key = np.round(u, 9).tobytes()
    cached = _COMPENSATOR_TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    table: dict[str, str] = {}
    u_dag = u.conj().T
    for label, pmat in _PAULI_2Q_MATRIX.items():
        r = u @ pmat @ u_dag
        table[label] = pauli_from_matrix(r)
    _COMPENSATOR_TABLE_CACHE[key] = table
    return table


def _pauli_label_ops(label: str, q0: cirq.Qid, q1: cirq.Qid) -> list[cirq.Operation]:
    """Single-qubit Pauli ops for a two-char label (skipping identities), twirl-tagged."""
    ops: list[cirq.Operation] = []
    for ch, q in zip(label, (q0, q1)):
        gate = _PAULI_1Q_GATE[ch]
        if gate is not None:
            ops.append(gate.on(q).with_tags(TWIRL_TAG))
    return ops


def twirl_two_qubit_gates(
    circuit: cirq.Circuit, rng: np.random.Generator
) -> cirq.Circuit:
    """Pauli-twirl every two-qubit gate independently.

    For each two-qubit gate ``G`` a random two-qubit Pauli ``P`` is inserted *before*
    ``G`` and its compensator ``R = G·P·G†`` *after* ``G``. The inserted ``P`` and ``R``
    are ordinary single-qubit Pauli gates (tagged for identification only); they remain
    real, noisy gates and are simplified by the single-qubit fusion pass before noise is
    applied. The ideal action is unchanged because ``R·G·P == G`` up to global phase.
    """
    twirled = cirq.Circuit()
    for moment in circuit:
        for op in moment.operations:
            if not is_two_qubit_op(op):
                twirled.append(op)
                continue
            u = cirq.unitary(op.gate)
            table = compensator_table_for_unitary(u)
            p_label = TWO_QUBIT_PAULI_LABELS[int(rng.integers(0, 16))]
            r_label = table[p_label]
            q0, q1 = op.qubits
            twirled.append(_pauli_label_ops(p_label, q0, q1))
            twirled.append(op)
            twirled.append(_pauli_label_ops(r_label, q0, q1))
    return twirled


def fuse_single_qubit_gates(circuit: cirq.Circuit) -> cirq.Circuit:
    """Compiler-style pass that merges adjacent single-qubit gates and drops identities.

    Runs on a parameter-resolved (numeric) circuit: every maximal run of consecutive
    single-qubit gates on a qubit (ansatz rotations + inserted twirl Paulis) collapses
    into one ``PhasedXZ`` gate, and runs whose product is the identity are removed. This
    mirrors real hardware (single-qubit gates are fused before execution), so a twirl
    Pauli absorbed into a neighbor adds no extra noise channel while the ideal action is
    exactly preserved. Two-qubit gates are untouched, so ``P``/``R`` never fuse across
    the gate they straddle.
    """
    merged = cirq.merge_single_qubit_gates_to_phxz(circuit)
    merged = cirq.drop_negligible_operations(merged)
    merged = cirq.drop_empty_moments(merged)
    return merged


def _simulate_vncdr_noisy_rho(
    prepared_circuit: cirq.Circuit,
    resolver: dict,
    qubits: list[cirq.Qid],
    noise_params: dict,
    *,
    simulator_seed: int,
) -> np.ndarray:
    """Resolve params, fuse single-qubit gates, apply per-gate noise, return rho.

    ``prepared_circuit`` is the (symbolic) folded + twirled circuit. Folding/twirling
    happen on the symbolic ansatz; resolution + fusion + noise happen here so each
    surviving physical gate gets its own faithful coherent + stochastic error.
    """
    resolved = cirq.resolve_parameters(prepared_circuit, resolver)
    fused = fuse_single_qubit_gates(resolved)
    noisy = fused.with_noise(GateArityDepolarizingNoise(**noise_params))
    rho = cirq.DensityMatrixSimulator(seed=simulator_seed).simulate(
        noisy, qubit_order=qubits
    ).final_density_matrix
    return sanitize_density_matrix(rho)


def _vncdr_per_term_levels(
    ansatz_circuit: cirq.Circuit,
    resolver: dict,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    *,
    noise_params: dict,
    noise_levels: tuple[int, ...],
    twirl_samples: int,
    twirl_rng: np.random.Generator,
    simulator_seed: int,
    num_shots: int,
    measurement_scheme: str,
    p_0_success: Iterable[float] | None,
    p_1_success: Iterable[float] | None,
    apply_readout_noise: bool,
    sampling_seed: int,
    epsilon: float,
    ogm_file: str | Path | None,
    shadowgrouping_root: str | Path | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-term noisy ⟨P_k⟩ for one resolver at each folded noise level.

    Returns ``(unmit, rem)`` arrays of shape ``(n_terms, len(noise_levels))``, each entry
    averaged over ``twirl_samples`` independent Pauli twirls.
    """
    observables_int, weights, _ = pauli_sum_to_int_observables(observable_h, qubits)
    n_terms = len(weights)
    n_levels = len(noise_levels)
    unmit = np.zeros((n_terms, n_levels), dtype=float)
    rem = np.zeros((n_terms, n_levels), dtype=float)
    if n_terms == 0:
        return unmit, rem

    samples = max(1, int(twirl_samples))
    for level_idx, c in enumerate(noise_levels):
        folded = fold_two_qubit_gates(ansatz_circuit, int(c))
        acc_u = np.zeros(n_terms, dtype=float)
        acc_r = np.zeros(n_terms, dtype=float)
        for _ in range(samples):
            twirled = twirl_two_qubit_gates(folded, twirl_rng)
            rho = _simulate_vncdr_noisy_rho(
                twirled, resolver, qubits, noise_params, simulator_seed=simulator_seed
            )
            est = estimate_energy_from_noisy_rho_shots(
                rho,
                observable_h,
                qubits,
                num_shots=num_shots,
                measurement_scheme=measurement_scheme,
                p_0_success=p_0_success,
                p_1_success=p_1_success,
                apply_rem=True,
                apply_readout_noise=apply_readout_noise,
                sampling_seed=sampling_seed,
                epsilon=epsilon,
                ogm_file=ogm_file,
                shadowgrouping_root=shadowgrouping_root,
                return_per_term=True,
            )
            acc_u += np.asarray(est["per_term_unmitigated"], dtype=float)
            acc_r += np.asarray(est["per_term_rem"], dtype=float)
        unmit[:, level_idx] = acc_u / samples
        rem[:, level_idx] = acc_r / samples
    return unmit, rem


def train_vncdr_models_per_pauli(
    ansatz_circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    resolvers: list[dict],
    *,
    noise_params: dict,
    noise_levels: tuple[int, ...] = (1, 3, 5),
    twirl_samples: int = 20,
    twirl_seed: int = 0,
    simulator_seed: int = 1234,
    num_shots: int = 8192,
    measurement_scheme: str = "ogm",
    p_0_success: Iterable[float] | None = None,
    p_1_success: Iterable[float] | None = None,
    apply_readout_noise: bool = True,
    sampling_seed: int = 1234,
    epsilon: float = 0.1,
    ogm_file: str | Path | None = None,
    shadowgrouping_root: str | Path | None = None,
) -> dict[str, object]:
    """Per-Pauli vnCDR fit: ``⟨P_k⟩_exact ≈ a_k · x_k`` (vector, NO intercept).

    ``x_k`` is the length-``len(noise_levels)`` vector of folded+twirled noisy
    expectations; ``a_k`` is solved by least squares. Two branches are fit (unmit and
    REM). The exact targets ``y_k`` are the unchanged noise-free statevector values.
    """
    if not resolvers:
        raise ValueError("At least one resolver is required to train vnCDR models.")

    observables_int, weights, offset = pauli_sum_to_int_observables(observable_h, qubits)
    n_terms = len(weights)
    if n_terms == 0:
        return {
            "fit_scope": "vncdr_per_pauli",
            "hamiltonian_offset": float(offset),
            "noise_levels": [int(c) for c in noise_levels],
            "weights": [],
            "coeffs_unmit_to_exact_per_term": [],
            "coeffs_rem_to_exact_per_term": [],
            "r2_rem_to_exact_per_term": [],
            "training_t_remaining": [0] * len(resolvers),
        }

    m = len(resolvers)
    n_levels = len(noise_levels)
    y_exact = np.zeros((m, n_terms), dtype=float)
    x_unmit = np.zeros((m, n_terms, n_levels), dtype=float)
    x_rem = np.zeros((m, n_terms, n_levels), dtype=float)
    t_rem_list: list[int] = []
    twirl_rng = np.random.default_rng(int(twirl_seed))

    for i, resolver in enumerate(resolvers):
        state = _simulate_noiseless_state_for_resolver(
            ansatz_circuit, resolver, qubits, simulator_seed=simulator_seed
        )
        for k in range(n_terms):
            y_exact[i, k] = exact_pauli_expectation_from_int_row(
                state, observables_int[k], qubits
            )
        u_levels, r_levels = _vncdr_per_term_levels(
            ansatz_circuit,
            resolver,
            observable_h,
            qubits,
            noise_params=noise_params,
            noise_levels=noise_levels,
            twirl_samples=twirl_samples,
            twirl_rng=twirl_rng,
            simulator_seed=simulator_seed,
            num_shots=num_shots,
            measurement_scheme=measurement_scheme,
            p_0_success=p_0_success,
            p_1_success=p_1_success,
            apply_readout_noise=apply_readout_noise,
            sampling_seed=sampling_seed,
            epsilon=epsilon,
            ogm_file=ogm_file,
            shadowgrouping_root=shadowgrouping_root,
        )
        x_unmit[i] = u_levels
        x_rem[i] = r_levels
        t_rem_list.append(int(count_non_clifford_ops(ansatz_circuit, resolver)))

    coeffs_unmit: list[list[float]] = []
    coeffs_rem: list[list[float]] = []
    r2_rem: list[float] = []
    for k in range(n_terms):
        xu = x_unmit[:, k, :]
        xr = x_rem[:, k, :]
        y = y_exact[:, k]
        au, _, _, _ = np.linalg.lstsq(xu, y, rcond=None)
        ar, _, _, _ = np.linalg.lstsq(xr, y, rcond=None)
        coeffs_unmit.append([float(v) for v in au.tolist()])
        coeffs_rem.append([float(v) for v in ar.tolist()])
        y_pred = xr @ ar
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2_rem.append(stable_r2_from_sums(ss_res, ss_tot, ss_res_tol=1e-4))

    return {
        "fit_scope": "vncdr_per_pauli",
        "hamiltonian_offset": float(offset),
        "noise_levels": [int(c) for c in noise_levels],
        "weights": [float(w) for w in weights],
        "coeffs_unmit_to_exact_per_term": coeffs_unmit,
        "coeffs_rem_to_exact_per_term": coeffs_rem,
        "r2_rem_to_exact_per_term": r2_rem,
        "training_t_remaining": t_rem_list,
    }


def apply_vncdr_models_per_pauli(
    target_unmit_levels: np.ndarray,
    target_rem_levels: np.ndarray,
    models: dict,
) -> dict[str, float]:
    """Apply vnCDR vector models: ``⟨P_k⟩_mit = a_k · mu_k``; ``E = offset + Σ_k w_k ⟨P_k⟩_mit``."""
    offset = float(models["hamiltonian_offset"])
    weights = np.asarray(models["weights"], dtype=float)
    cu = models["coeffs_unmit_to_exact_per_term"]
    cr = models["coeffs_rem_to_exact_per_term"]
    if len(weights) == 0:
        return {"cdr_unmit_corrected": offset, "cdr_rem_corrected": offset}
    mu_u = np.asarray(target_unmit_levels, dtype=float)
    mu_r = np.asarray(target_rem_levels, dtype=float)
    eu = offset
    er = offset
    for k in range(len(weights)):
        eu += float(weights[k]) * float(np.dot(np.asarray(cu[k], dtype=float), mu_u[k]))
        er += float(weights[k]) * float(np.dot(np.asarray(cr[k], dtype=float), mu_r[k]))
    return {"cdr_unmit_corrected": float(eu), "cdr_rem_corrected": float(er)}


# ---------------------------------------------------------------------------
# Unified mitigation-mode dispatcher
# ---------------------------------------------------------------------------

VALID_MITIGATION_MODES: tuple[str, ...] = ("none", "zne", "cdr", "both")


def _generate_near_clifford_resolvers_fallback(
    target_params: dict,
    symbols: list,
    *,
    num_circuits: int,
    t_max: int,
    circuit: cirq.Circuit,
    min_snap_fraction: float,
    seed: int,
) -> list[dict]:
    """Fallback near-Clifford generator for symbol names outside th_*/ph_*.

    Uses nearest pi/2 snapping for all symbols and the same greedy t_max loop.
    After selecting snapped symbols, randomizes unsnapped symbols so different
    training circuits are not degenerate copies of the target parameters.
    """
    if num_circuits <= 0:
        raise ValueError(f"num_circuits must be > 0, got {num_circuits}.")
    if t_max < 0:
        raise ValueError(f"t_max must be >= 0, got {t_max}.")
    if not (0.0 <= float(min_snap_fraction) <= 1.0):
        raise ValueError(
            f"min_snap_fraction must be in [0, 1], got {min_snap_fraction}."
        )

    def snap_pi_over_2(value: float) -> float:
        step = float(np.pi) / 2.0
        return float(round(float(value) / step) * step)

    target_by_symbol: dict = {}
    for sym in symbols:
        if sym in target_params:
            target_by_symbol[sym] = float(target_params[sym])
        elif str(sym) in target_params:
            target_by_symbol[sym] = float(target_params[str(sym)])
        else:
            raise KeyError(f"Target parameter missing for symbol {sym!s}.")

    resolvers: list[dict] = []
    for circ_idx in range(num_circuits):
        local_rng = np.random.default_rng(int(seed) + 1000 * (circ_idx + 1))
        resolver = dict(target_by_symbol)
        snapped: set = set()

        if min_snap_fraction > 0.0:
            n_min_snap = int(np.ceil(min_snap_fraction * len(symbols)))
            n_min_snap = min(n_min_snap, len(symbols))
            order = list(symbols)
            local_rng.shuffle(order)
            for sym in order[:n_min_snap]:
                resolver[sym] = snap_pi_over_2(target_by_symbol[sym])
                snapped.add(sym)

        guard = 0
        max_iterations = 4 * len(symbols) + 8
        while True:
            t_remaining = count_non_clifford_ops(circuit, resolver)
            if t_remaining <= t_max:
                break
            unsnapped = [s for s in symbols if s not in snapped]
            if not unsnapped:
                break
            pick = unsnapped[int(local_rng.integers(0, len(unsnapped)))]
            resolver[pick] = snap_pi_over_2(target_by_symbol[pick])
            snapped.add(pick)
            guard += 1
            if guard > max_iterations:
                break

        # Keep CDR training diversity: unsnapped symbols should be sampled, not
        # left fixed at target values. This mirrors the two-step near-Clifford
        # construction used by the primary generator.
        for sym in symbols:
            if sym in snapped:
                continue
            resolver[sym] = float(local_rng.uniform(0.0, 2.0 * np.pi))

        resolvers.append(resolver)

    return resolvers


def _baseline_target_energies(
    ansatz_circuit: cirq.Circuit,
    target_resolver: dict,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    *,
    noise_params: dict,
    simulator_seed: int,
    num_shots: int,
    measurement_scheme: str,
    p_0_success: Iterable[float] | None,
    p_1_success: Iterable[float] | None,
    apply_readout_noise: bool,
    sampling_seed: int,
    epsilon: float,
    ogm_file: str | Path | None,
    shadowgrouping_root: str | Path | None,
    return_per_term: bool = False,
) -> dict[str, Any]:
    rho = _simulate_noisy_rho_for_resolver(
        ansatz_circuit, target_resolver, qubits, noise_params, simulator_seed=simulator_seed
    )
    est = estimate_energy_from_noisy_rho_shots(
        rho,
        observable_h,
        qubits,
        num_shots=num_shots,
        measurement_scheme=measurement_scheme,
        p_0_success=p_0_success,
        p_1_success=p_1_success,
        apply_rem=True,
        apply_readout_noise=apply_readout_noise,
        sampling_seed=sampling_seed,
        epsilon=epsilon,
        ogm_file=ogm_file,
        shadowgrouping_root=shadowgrouping_root,
        return_per_term=return_per_term,
    )
    out: dict[str, Any] = {
        "unmit_target": float(est["energy_unmitigated"]),
        "rem_target": float(est["energy_rem"]),
    }
    if return_per_term:
        out["per_term_unmit"] = est["per_term_unmitigated"]
        out["per_term_rem"] = est["per_term_rem"]
    return out


def run_mitigation(
    mode: str,
    *,
    ansatz_circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    target_resolver: dict,
    target_params: dict | None = None,
    symbols: list | None = None,
    base_noise_cfg: dict,
    shot_cfg: dict,
    readout_cal: dict | None = None,
    zne_cfg: dict | None = None,
    cdr_cfg: dict | None = None,
    simulator_seed: int = 1234,
) -> dict[str, object]:
    """Single dispatcher for `none | zne | cdr | both` mitigation pipelines.

    `base_noise_cfg`: dict of `two_qubit_depol_prob`, `one_qubit_depol_prob`.
    `shot_cfg`:      dict of `num_shots`, `measurement_scheme`, `apply_readout_noise`,
                     `sampling_seed`, `epsilon`, `ogm_file`, `shadowgrouping_root`.
    `readout_cal`:   dict with `p_0_success`, `p_1_success` (or None).
    `zne_cfg`:       dict with `noise_scales`, `fit_order` (required for `zne`/`both`).
    `cdr_cfg`:       dict with `num_circuits`, `t_max`, `min_snap_fraction`, `seed`,
                     optional `cdr_training` (`snap_greedy`|`random_clifford`).
                     vnCDR knobs: `noise_levels` (fold factors, default `(1,3,5)`),
                     `twirl_samples` (default `20`), `twirl_seed` (default `seed`).
                     Also carries `target_params` and `symbols` if not provided at the top level.
                     (`cdr` mode performs per-Pauli vnCDR: a vector fit `y = a·x` with no
                     intercept over the folded+twirled noise levels.)
    `target_resolver`: ParamResolver-like dict mapping symbols to floats for the
                       target VQE circuit.
    """
    if mode not in VALID_MITIGATION_MODES:
        raise ValueError(
            f"Unknown mitigation_mode={mode!r}. "
            f"Expected one of {VALID_MITIGATION_MODES}."
        )

    readout_cal = readout_cal or {}
    p_0_success = readout_cal.get("p_0_success")
    p_1_success = readout_cal.get("p_1_success")
    apply_readout_noise = bool(shot_cfg.get("apply_readout_noise", True))

    num_shots = int(shot_cfg.get("num_shots", 8192))
    measurement_scheme = str(shot_cfg.get("measurement_scheme", "ogm"))
    sampling_seed = int(shot_cfg.get("sampling_seed", 1234))
    epsilon = float(shot_cfg.get("epsilon", 0.1))
    ogm_file = shot_cfg.get("ogm_file")
    shadowgrouping_root = shot_cfg.get("shadowgrouping_root")

    base_noise = dict(base_noise_cfg)

    cdr_fit_scope = "per_pauli"
    if mode in ("cdr", "both"):
        if cdr_cfg is None:
            raise ValueError(f"cdr_cfg is required when mitigation_mode={mode!r}.")
        cdr_fit_scope = str(cdr_cfg.get("cdr_fit_scope", "per_pauli"))

    baseline = _baseline_target_energies(
        ansatz_circuit,
        target_resolver,
        observable_h,
        qubits,
        noise_params=base_noise,
        simulator_seed=simulator_seed,
        num_shots=num_shots,
        measurement_scheme=measurement_scheme,
        p_0_success=p_0_success,
        p_1_success=p_1_success,
        apply_readout_noise=apply_readout_noise,
        sampling_seed=sampling_seed,
        epsilon=epsilon,
        ogm_file=ogm_file,
        shadowgrouping_root=shadowgrouping_root,
        return_per_term=False,
    )

    out: dict[str, object] = {
        "mode": mode,
        "unmit_target": baseline["unmit_target"],
        "rem_target": baseline["rem_target"],
    }

    if mode == "none":
        return out

    if mode in ("zne", "both"):
        if zne_cfg is None:
            raise ValueError(f"zne_cfg is required when mitigation_mode={mode!r}.")
        zne_scales = list(zne_cfg["noise_scales"])
        zne_fit_order = int(zne_cfg.get("fit_order", 1))
        if mode == "zne" and len(zne_scales) < 2:
            raise ValueError(
                f"mitigation_mode='zne' requires zne_cfg['noise_scales'] of length >= 2, "
                f"got {zne_scales}."
            )
        if len(zne_scales) >= 2:
            h_matrix = observable_h.matrix(qubits=qubits)
            trace_zne = run_trace_zne(
                ansatz_circuit,
                target_resolver,
                qubits,
                h_matrix,
                noise_scales=zne_scales,
                fit_order=zne_fit_order,
                simulator_seed=simulator_seed,
                **base_noise,
            )
            shot_zne = run_shot_zne(
                ansatz_circuit,
                target_resolver,
                observable_h,
                qubits,
                noise_scales=zne_scales,
                fit_order=zne_fit_order,
                simulator_seed=simulator_seed,
                num_shots=num_shots,
                measurement_scheme=measurement_scheme,
                p_0_success=p_0_success,
                p_1_success=p_1_success,
                apply_rem=True,
                apply_readout_noise=apply_readout_noise,
                sampling_seed=sampling_seed,
                epsilon=epsilon,
                ogm_file=ogm_file,
                shadowgrouping_root=shadowgrouping_root,
                extrapolate_rem=True,
                **base_noise,
            )
            out["trace_zne"] = trace_zne
            out["shot_zne"] = shot_zne

    if mode in ("cdr", "both"):
        cdr_target_params = (
            target_params if target_params is not None else cdr_cfg.get("target_params")
        )
        cdr_symbols = symbols if symbols is not None else cdr_cfg.get("symbols")
        if cdr_target_params is None or cdr_symbols is None:
            raise ValueError(
                "CDR requires `target_params` and `symbols` (top-level or in cdr_cfg)."
            )

        cdr_training = str(cdr_cfg.get("cdr_training", "snap_greedy"))
        if cdr_training == "snap_greedy":
            try:
                resolvers = generate_near_clifford_param_sets(
                    cdr_target_params,
                    list(cdr_symbols),
                    num_circuits=int(cdr_cfg["num_circuits"]),
                    t_max=int(cdr_cfg["t_max"]),
                    circuit=ansatz_circuit,
                    min_snap_fraction=float(cdr_cfg.get("min_snap_fraction", 0.0)),
                    seed=int(cdr_cfg.get("seed", 0)),
                )
            except ValueError as err:
                if "Unrecognized symbol naming convention" not in str(err):
                    raise
                resolvers = _generate_near_clifford_resolvers_fallback(
                    cdr_target_params,
                    list(cdr_symbols),
                    num_circuits=int(cdr_cfg["num_circuits"]),
                    t_max=int(cdr_cfg["t_max"]),
                    circuit=ansatz_circuit,
                    min_snap_fraction=float(cdr_cfg.get("min_snap_fraction", 0.0)),
                    seed=int(cdr_cfg.get("seed", 0)),
                )
        elif cdr_training == "random_clifford":
            resolvers = generate_random_clifford_analogue_param_sets(
                cdr_target_params,
                list(cdr_symbols),
                num_circuits=int(cdr_cfg["num_circuits"]),
                circuit=ansatz_circuit,
                seed=int(cdr_cfg.get("seed", 0)),
            )
        else:
            raise ValueError(
                f"cdr_training={cdr_training!r} must be 'snap_greedy' or 'random_clifford'."
            )

        out["cdr_training"] = cdr_training
        out["cdr_fit_scope"] = "vncdr_per_pauli"

        # vnCDR knobs: noise levels (fold factors) and twirl averaging.
        vncdr_noise_levels = tuple(int(x) for x in cdr_cfg.get("noise_levels", (1, 3, 5)))
        vncdr_twirl_samples = int(cdr_cfg.get("twirl_samples", 20))
        vncdr_twirl_seed = int(cdr_cfg.get("twirl_seed", cdr_cfg.get("seed", 0)))
        out["vncdr_noise_levels"] = [int(c) for c in vncdr_noise_levels]
        out["vncdr_twirl_samples"] = int(vncdr_twirl_samples)

        def _train_and_apply_cdr(scaled_noise: dict) -> tuple[dict[str, object], dict[str, float]]:
            m = train_vncdr_models_per_pauli(
                ansatz_circuit,
                observable_h,
                qubits,
                resolvers,
                noise_params=scaled_noise,
                noise_levels=vncdr_noise_levels,
                twirl_samples=vncdr_twirl_samples,
                twirl_seed=vncdr_twirl_seed,
                simulator_seed=simulator_seed,
                num_shots=num_shots,
                measurement_scheme=measurement_scheme,
                p_0_success=p_0_success,
                p_1_success=p_1_success,
                apply_readout_noise=apply_readout_noise,
                sampling_seed=sampling_seed,
                epsilon=epsilon,
                ogm_file=ogm_file,
                shadowgrouping_root=shadowgrouping_root,
            )
            # Target μ-vectors: same fold+twirl pipeline on the real circuit.
            target_rng = np.random.default_rng(int(vncdr_twirl_seed) + 982451653)
            target_unmit_levels, target_rem_levels = _vncdr_per_term_levels(
                ansatz_circuit,
                target_resolver,
                observable_h,
                qubits,
                noise_params=scaled_noise,
                noise_levels=vncdr_noise_levels,
                twirl_samples=vncdr_twirl_samples,
                twirl_rng=target_rng,
                simulator_seed=simulator_seed,
                num_shots=num_shots,
                measurement_scheme=measurement_scheme,
                p_0_success=p_0_success,
                p_1_success=p_1_success,
                apply_readout_noise=apply_readout_noise,
                sampling_seed=sampling_seed,
                epsilon=epsilon,
                ogm_file=ogm_file,
                shadowgrouping_root=shadowgrouping_root,
            )
            c = apply_vncdr_models_per_pauli(target_unmit_levels, target_rem_levels, m)
            return m, c

        if mode == "cdr":
            models, corrected = _train_and_apply_cdr(base_noise)
            out["cdr_models"] = models
            out["cdr_unmit_corrected"] = corrected["cdr_unmit_corrected"]
            out["cdr_rem_corrected"] = corrected["cdr_rem_corrected"]
            return out

        zne_scales = list(zne_cfg["noise_scales"])  # type: ignore[index]
        zne_fit_order = int(zne_cfg.get("fit_order", 1))  # type: ignore[union-attr]
        per_scale_cdr_unmit: list[float] = []
        per_scale_cdr_rem: list[float] = []
        per_scale_models: list[dict[str, object]] = []
        per_scale_target_unmit: list[float] = []
        per_scale_target_rem: list[float] = []

        for scale in zne_scales:
            scaled_noise = scale_noise_params_for_zne(float(scale), **base_noise)
            target_at_scale = _baseline_target_energies(
                ansatz_circuit,
                target_resolver,
                observable_h,
                qubits,
                noise_params=scaled_noise,
                simulator_seed=simulator_seed,
                num_shots=num_shots,
                measurement_scheme=measurement_scheme,
                p_0_success=p_0_success,
                p_1_success=p_1_success,
                apply_readout_noise=apply_readout_noise,
                sampling_seed=sampling_seed,
                epsilon=epsilon,
                ogm_file=ogm_file,
                shadowgrouping_root=shadowgrouping_root,
                return_per_term=False,
            )
            models_at_scale, corrected = _train_and_apply_cdr(scaled_noise)
            per_scale_cdr_unmit.append(corrected["cdr_unmit_corrected"])
            per_scale_cdr_rem.append(corrected["cdr_rem_corrected"])
            per_scale_models.append(models_at_scale)
            per_scale_target_unmit.append(target_at_scale["unmit_target"])
            per_scale_target_rem.append(target_at_scale["rem_target"])

        scales_arr = np.asarray(zne_scales, dtype=float)
        unmit_arr = np.asarray(per_scale_cdr_unmit, dtype=float)
        rem_arr = np.asarray(per_scale_cdr_rem, dtype=float)

        if zne_fit_order == 0 or len(zne_scales) < 2:
            coeffs_unmit = np.array([float(np.mean(unmit_arr))])
            coeffs_rem = np.array([float(np.mean(rem_arr))])
            zne_unmit = float(coeffs_unmit[0])
            zne_rem = float(coeffs_rem[0])
        else:
            coeffs_unmit = np.polyfit(scales_arr, unmit_arr, deg=zne_fit_order)
            coeffs_rem = np.polyfit(scales_arr, rem_arr, deg=zne_fit_order)
            zne_unmit = float(np.polyval(coeffs_unmit, 0.0))
            zne_rem = float(np.polyval(coeffs_rem, 0.0))

        out["per_scale_cdr_unmit"] = [float(v) for v in per_scale_cdr_unmit]
        out["per_scale_cdr_rem"] = [float(v) for v in per_scale_cdr_rem]
        out["per_scale_target_unmit"] = [float(v) for v in per_scale_target_unmit]
        out["per_scale_target_rem"] = [float(v) for v in per_scale_target_rem]
        out["per_scale_cdr_models"] = per_scale_models
        out["zne_of_cdr_coeffs_unmit"] = [float(v) for v in coeffs_unmit.tolist()]
        out["zne_of_cdr_coeffs_rem"] = [float(v) for v in coeffs_rem.tolist()]
        out["zne_of_cdr_unmit_target"] = zne_unmit
        out["zne_of_cdr_rem_target"] = zne_rem
        out["zne_scales"] = [float(v) for v in zne_scales]
        out["zne_fit_order"] = int(zne_fit_order)

    return out

