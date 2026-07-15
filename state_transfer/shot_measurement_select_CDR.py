"""Selected-CDR variant of :mod:`shot_measurement`.

The original module remains unchanged.  This module re-exports its public API
and replaces only ``run_mitigation``.  Two independent strategies are offered:

1. Per-term robust refit (default, ``per_term_fit='affine'``): the original
   training circuits are kept, but each Pauli term's line
   ``exact ~= a * noisy + b`` is refit using only its informative points
   (exact |<P>| above a threshold).  The intercept ``b`` is free.  Terms with
   no informative training data fall back to the identity map (a=1, b=0),
   passing the measured target value through instead of predicting a constant
   zero.

2. Circuit pool selection (opt-in, ``selection_method='weighted_maxmin'``): a
   larger classically simulated candidate pool is generated and reduced to the
   requested training count using coefficient-weighted max-min sampling.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import threading
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cirq
import numpy as np


_BASE_MODULE_NAME = "_state_transfer_shot_measurement_base"
_BASE_MODULE_PATH = Path(__file__).resolve().with_name("shot_measurement.py")
_base_spec = importlib.util.spec_from_file_location(
    _BASE_MODULE_NAME, _BASE_MODULE_PATH
)
if _base_spec is None or _base_spec.loader is None:
    raise ImportError(f"Could not load local shot measurement module: {_BASE_MODULE_PATH}")
_base = importlib.util.module_from_spec(_base_spec)
sys.modules[_BASE_MODULE_NAME] = _base
_base_spec.loader.exec_module(_base)

# Re-export the sibling module's public API without relying on sys.path.  The
# selected ``run_mitigation`` definition below replaces the copied base one.
for _public_name in dir(_base):
    if not _public_name.startswith("_"):
        globals()[_public_name] = getattr(_base, _public_name)


_BASE_RUN_MITIGATION = _base.run_mitigation
_BASE_NEAR_CLIFFORD_GENERATOR = _base.generate_near_clifford_param_sets
_BASE_RANDOM_CLIFFORD_GENERATOR = _base.generate_random_clifford_analogue_param_sets
_BASE_TRAIN_CF_MODELS_PER_PAULI = _base.train_cf_models_per_pauli
_GENERATOR_PATCH_LOCK = threading.RLock()


def _resolver_value(resolver: dict, symbol: Any) -> float:
    if symbol in resolver:
        return float(resolver[symbol])
    if str(symbol) in resolver:
        return float(resolver[str(symbol)])
    raise KeyError(f"Resolver is missing parameter {symbol!s}.")


def _wrapped_parameter_vectors(
    resolvers: Sequence[dict],
    symbols: Sequence[Any],
    target_params: dict,
) -> tuple[np.ndarray, np.ndarray]:
    vectors = np.asarray(
        [[_resolver_value(resolver, symbol) for symbol in symbols] for resolver in resolvers],
        dtype=float,
    )
    target = np.asarray(
        [_resolver_value(target_params, symbol) for symbol in symbols],
        dtype=float,
    )
    return vectors, target


def select_weighted_maxmin_indices(
    expectations: np.ndarray,
    hamiltonian_weights: np.ndarray,
    *,
    num_select: int,
    parameter_vectors: np.ndarray | None = None,
    target_parameter_vector: np.ndarray | None = None,
    local_count: int = 0,
) -> list[int]:
    """Select diverse candidate rows with Hamiltonian-aware max-min sampling.

    Distances use ``sqrt(abs(weight)) * expectation`` coordinates, making each
    term's contribution to squared distance proportional to its absolute
    Hamiltonian coefficient.  ``local_count`` slots are first reserved for
    candidates nearest to the target in wrapped angular distance; the remaining
    slots greedily maximize distance to the nearest selected candidate.
    """
    values = np.asarray(expectations, dtype=float)
    weights = np.asarray(hamiltonian_weights, dtype=float).reshape(-1)
    if values.ndim != 2:
        raise ValueError(f"expectations must be 2D, got shape {values.shape}.")
    n_candidates, n_terms = values.shape
    if len(weights) != n_terms:
        raise ValueError(
            f"hamiltonian_weights has length {len(weights)}, expected {n_terms}."
        )
    if not 1 <= int(num_select) <= n_candidates:
        raise ValueError(
            f"num_select must be in [1, {n_candidates}], got {num_select}."
        )
    if not 0 <= int(local_count) <= int(num_select):
        raise ValueError(
            f"local_count must be in [0, {num_select}], got {local_count}."
        )
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(weights)):
        raise ValueError("expectations and hamiltonian_weights must be finite.")

    abs_weights = np.abs(weights)
    if n_terms == 0:
        transformed = np.zeros((n_candidates, 0), dtype=float)
    else:
        total_weight = float(np.sum(abs_weights))
        if total_weight <= 0.0:
            abs_weights = np.ones(n_terms, dtype=float)
            total_weight = float(n_terms)
        transformed = values * np.sqrt(abs_weights / total_weight)[None, :]

    selected: list[int] = []
    selected_mask = np.zeros(n_candidates, dtype=bool)

    if local_count:
        if parameter_vectors is None or target_parameter_vector is None:
            raise ValueError(
                "parameter_vectors and target_parameter_vector are required "
                "when local_count > 0."
            )
        params = np.asarray(parameter_vectors, dtype=float)
        target = np.asarray(target_parameter_vector, dtype=float).reshape(-1)
        if params.shape != (n_candidates, len(target)):
            raise ValueError(
                f"parameter_vectors has shape {params.shape}; expected "
                f"({n_candidates}, {len(target)})."
            )
        wrapped_delta = (params - target[None, :] + np.pi) % (2.0 * np.pi) - np.pi
        local_order = np.argsort(
            np.sum(wrapped_delta * wrapped_delta, axis=1), kind="stable"
        )
        for index in local_order[: int(local_count)]:
            idx = int(index)
            selected.append(idx)
            selected_mask[idx] = True

    if not selected:
        # Begin at an informative extreme, then fill the largest uncovered gaps.
        first = int(np.argmax(np.sum(transformed * transformed, axis=1)))
        selected.append(first)
        selected_mask[first] = True

    min_squared_distance = np.full(n_candidates, np.inf, dtype=float)
    for index in selected:
        delta = transformed - transformed[index]
        min_squared_distance = np.minimum(
            min_squared_distance, np.sum(delta * delta, axis=1)
        )
    min_squared_distance[selected_mask] = -np.inf

    while len(selected) < int(num_select):
        next_index = int(np.argmax(min_squared_distance))
        selected.append(next_index)
        selected_mask[next_index] = True
        delta = transformed - transformed[next_index]
        min_squared_distance = np.minimum(
            min_squared_distance, np.sum(delta * delta, axis=1)
        )
        min_squared_distance[selected_mask] = -np.inf

    return selected


def _exact_per_pauli_expectations(
    ansatz_circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    resolvers: Sequence[dict],
    *,
    simulator_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    observables_int, weights, _offset = _base.pauli_sum_to_int_observables(
        observable_h, qubits
    )
    values = np.zeros((len(resolvers), len(weights)), dtype=float)
    for row, resolver in enumerate(resolvers):
        state = _base._simulate_noiseless_state_for_resolver(
            ansatz_circuit,
            resolver,
            qubits,
            simulator_seed=int(simulator_seed),
        )
        for column, observable in enumerate(observables_int):
            values[row, column] = _base.exact_pauli_expectation_from_int_row(
                state, observable, qubits
            )
    return values, np.asarray(weights, dtype=float)


def select_cdr_resolvers(
    resolvers: Sequence[dict],
    *,
    ansatz_circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    symbols: Sequence[Any],
    target_params: dict,
    num_select: int,
    local_count: int,
    simulator_seed: int,
) -> tuple[list[dict], dict[str, Any]]:
    """Select CDR resolvers and return auditable spread diagnostics."""
    if len(resolvers) < int(num_select):
        raise ValueError(
            f"Candidate pool has {len(resolvers)} circuits, fewer than "
            f"num_select={num_select}."
        )
    exact_values, weights = _exact_per_pauli_expectations(
        ansatz_circuit,
        observable_h,
        qubits,
        resolvers,
        simulator_seed=simulator_seed,
    )
    parameter_vectors, target_vector = _wrapped_parameter_vectors(
        resolvers, symbols, target_params
    )
    selected_indices = select_weighted_maxmin_indices(
        exact_values,
        weights,
        num_select=int(num_select),
        parameter_vectors=parameter_vectors,
        target_parameter_vector=target_vector,
        local_count=int(local_count),
    )
    selected_values = exact_values[selected_indices]
    abs_weights = np.abs(weights)
    normalized_weights = (
        abs_weights / float(np.sum(abs_weights))
        if len(weights) and float(np.sum(abs_weights)) > 0.0
        else np.full(len(weights), 1.0 / len(weights)) if len(weights) else np.zeros(0)
    )

    def weighted_mean_spread(values: np.ndarray, statistic: str) -> float:
        if values.shape[1] == 0:
            return 0.0
        if statistic == "std":
            per_term = np.std(values, axis=0)
        else:
            per_term = np.ptp(values, axis=0)
        return float(np.dot(normalized_weights, per_term))

    diagnostics: dict[str, Any] = {
        "method": "weighted_maxmin",
        "pool_size": int(len(resolvers)),
        "selected_count": int(len(selected_indices)),
        "local_count": int(local_count),
        "selected_indices": selected_indices,
        "pool_weighted_mean_std": weighted_mean_spread(exact_values, "std"),
        "selected_weighted_mean_std": weighted_mean_spread(selected_values, "std"),
        "pool_weighted_mean_range": weighted_mean_spread(exact_values, "range"),
        "selected_weighted_mean_range": weighted_mean_spread(selected_values, "range"),
        "pool_expectation_min_per_term": (
            np.min(exact_values, axis=0).tolist() if exact_values.shape[1] else []
        ),
        "pool_expectation_max_per_term": (
            np.max(exact_values, axis=0).tolist() if exact_values.shape[1] else []
        ),
        "selected_expectation_min_per_term": (
            np.min(selected_values, axis=0).tolist() if selected_values.shape[1] else []
        ),
        "selected_expectation_max_per_term": (
            np.max(selected_values, axis=0).tolist() if selected_values.shape[1] else []
        ),
    }
    return [dict(resolvers[index]) for index in selected_indices], diagnostics


def _selection_settings(cdr_cfg: dict, requested_count: int) -> tuple[str, int, int]:
    method = str(
        cdr_cfg.get(
            "selection_method",
            os.environ.get("CDR_SELECTION_METHOD", "none"),
        )
    ).lower()
    pool_size = int(
        cdr_cfg.get(
            "selection_pool_size",
            os.environ.get("CDR_SELECTION_POOL_SIZE", "300"),
        )
    )
    local_count = int(
        cdr_cfg.get(
            "selection_local_count",
            os.environ.get("CDR_SELECTION_LOCAL_COUNT", "10"),
        )
    )
    if method not in {"weighted_maxmin", "none"}:
        raise ValueError(
            f"selection_method={method!r}; expected 'weighted_maxmin' or 'none'."
        )
    if method == "weighted_maxmin" and pool_size < int(requested_count):
        raise ValueError(
            f"selection_pool_size={pool_size} is smaller than num_circuits="
            f"{requested_count}."
        )
    if method == "weighted_maxmin" and not 0 <= local_count <= int(requested_count):
        raise ValueError(
            f"selection_local_count={local_count} must be between 0 and "
            f"num_circuits={requested_count}."
        )
    return method, pool_size, local_count


VALID_PER_TERM_FIT_MODES = ("through_origin", "affine", "none")


def _per_term_fit_settings(cdr_cfg: dict) -> tuple[str, float, int, float]:
    fit_mode = str(
        cdr_cfg.get(
            "per_term_fit",
            os.environ.get("CDR_PER_TERM_FIT", "affine"),
        )
    ).lower()
    exact_tol = float(
        cdr_cfg.get(
            "per_term_exact_tol",
            os.environ.get("CDR_PER_TERM_EXACT_TOL", "0.05"),
        )
    )
    min_points = int(
        cdr_cfg.get(
            "per_term_min_points",
            os.environ.get("CDR_PER_TERM_MIN_POINTS", "2"),
        )
    )
    slope_max = float(
        cdr_cfg.get(
            "per_term_slope_max",
            os.environ.get("CDR_PER_TERM_SLOPE_MAX", "10.0"),
        )
    )
    if fit_mode not in VALID_PER_TERM_FIT_MODES:
        raise ValueError(
            f"per_term_fit={fit_mode!r}; expected one of {VALID_PER_TERM_FIT_MODES}."
        )
    if exact_tol < 0.0:
        raise ValueError(f"per_term_exact_tol must be >= 0, got {exact_tol}.")
    if min_points < 1:
        raise ValueError(f"per_term_min_points must be >= 1, got {min_points}.")
    if slope_max <= 0.0:
        raise ValueError(f"per_term_slope_max must be > 0, got {slope_max}.")
    return fit_mode, exact_tol, min_points, slope_max


def _fit_through_origin(xs: np.ndarray, ys: np.ndarray, slope_max: float) -> tuple[float, float] | None:
    denom = float(np.dot(xs, xs))
    if denom <= 1e-12:
        return None
    slope = float(np.dot(xs, ys) / denom)
    if 0.0 < slope <= float(slope_max):
        return slope, 0.0
    return None


def _fit_affine(xs: np.ndarray, ys: np.ndarray, slope_max: float) -> tuple[float, float] | None:
    if float(np.std(xs)) <= 1e-9:
        return None
    coeffs = np.polyfit(xs, ys, deg=1)
    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    if 0.0 < slope <= float(slope_max):
        return slope, intercept
    return None


def robust_per_term_line_fit(
    noisy: np.ndarray,
    exact: np.ndarray,
    *,
    fit_mode: str = "affine",
    exact_tol: float = 0.05,
    min_points: int = 2,
    slope_max: float = 10.0,
) -> tuple[float, float, dict[str, Any]]:
    """Fit ``exact ~= a * noisy + b`` for one Pauli term using informative points.

    Training circuits are near-Clifford, so most exact expectations sit at
    exactly 0 and carry no slope information.  Points with ``|exact| >
    exact_tol`` are kept; the exact values are classically computed, so this
    filter is noise-free and hardware-honest.

    - ``affine`` (default): ordinary two-parameter fit on the informative
      points.  Intercept ``b`` is free.  If the affine fit is degenerate,
      falls back to a through-origin slope before identity.
    - ``through_origin``: least squares with ``b`` forced to 0.

    Whenever the informative data is missing, degenerate, or produces an
    implausible slope (a <= 0 or a > slope_max), the identity map (a=1, b=0)
    is returned so the target's measured value passes through unchanged
    instead of being replaced by a constant.
    """
    x = np.asarray(noisy, dtype=float).ravel()
    y = np.asarray(exact, dtype=float).ravel()
    if x.shape != y.shape:
        raise ValueError(f"noisy/exact length mismatch: {x.shape} vs {y.shape}.")
    informative = np.abs(y) > float(exact_tol)
    info: dict[str, Any] = {
        "n_points": int(len(x)),
        "n_informative": int(np.count_nonzero(informative)),
        "fit_used": "identity",
    }
    a, b = 1.0, 0.0
    if fit_mode != "none" and info["n_informative"] >= int(min_points):
        xs = x[informative]
        ys = y[informative]
        if fit_mode == "affine":
            fitted = _fit_affine(xs, ys, slope_max)
            if fitted is not None:
                a, b = fitted
                info["fit_used"] = "affine"
            else:
                # Degenerate x-spread: a multiplicative slope is still usable.
                fitted = _fit_through_origin(xs, ys, slope_max)
                if fitted is not None:
                    a, b = fitted
                    info["fit_used"] = "through_origin"
        elif fit_mode == "through_origin":
            fitted = _fit_through_origin(xs, ys, slope_max)
            if fitted is not None:
                a, b = fitted
                info["fit_used"] = "through_origin"
        else:
            raise ValueError(f"Unknown fit_mode={fit_mode!r}.")
    return float(a), float(b), info


def _refit_models_per_pauli(
    models: dict,
    *,
    fit_mode: str,
    exact_tol: float,
    min_points: int,
    slope_max: float,
) -> dict[str, Any]:
    """Replace per-term CDR coefficients in ``models`` with robust refits."""
    exact_matrix = np.asarray(models["training_exact_per_term"], dtype=float)
    summary: dict[str, Any] = {
        "fit_mode": fit_mode,
        "exact_tol": float(exact_tol),
        "min_points": int(min_points),
        "slope_max": float(slope_max),
        "n_terms": 0,
    }
    if exact_matrix.ndim != 2 or exact_matrix.shape[1] == 0:
        return summary
    n_terms = int(exact_matrix.shape[1])
    summary["n_terms"] = n_terms

    branches = {
        "unmit": ("training_unmit_per_term", "coeffs_unmit_to_exact_per_term"),
        "rem": ("training_rem_per_term", "coeffs_rem_to_exact_per_term"),
    }
    for branch_name, (training_key, coeff_key) in branches.items():
        noisy_matrix = np.asarray(models[training_key], dtype=float)
        coeffs: list[list[float]] = []
        fit_used: list[str] = []
        n_informative: list[int] = []
        for k in range(n_terms):
            a, b, info = robust_per_term_line_fit(
                noisy_matrix[:, k],
                exact_matrix[:, k],
                fit_mode=fit_mode,
                exact_tol=exact_tol,
                min_points=min_points,
                slope_max=slope_max,
            )
            coeffs.append([a, b])
            fit_used.append(str(info["fit_used"]))
            n_informative.append(int(info["n_informative"]))
        models[coeff_key] = coeffs
        summary[f"{branch_name}_fit_used_counts"] = {
            name: fit_used.count(name) for name in sorted(set(fit_used))
        }
        summary[f"{branch_name}_fit_used_per_term"] = fit_used
        if branch_name == "rem":
            summary["n_informative_per_term"] = n_informative

    rem_matrix = np.asarray(models["training_rem_per_term"], dtype=float)
    r2_values: list[float] = []
    for k in range(n_terms):
        a, b = models["coeffs_rem_to_exact_per_term"][k]
        y = exact_matrix[:, k]
        prediction = float(a) * rem_matrix[:, k] + float(b)
        ss_res = float(np.sum((y - prediction) ** 2))
        ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
        r2_values.append(_base.stable_r2_from_sums(ss_res, ss_tot, ss_res_tol=1e-4))
    models["r2_rem_to_exact_per_term"] = r2_values
    models["per_term_fit"] = dict(summary)
    return summary


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
    """Run the original mitigation pipeline with selected-CDR strategies."""
    if mode not in ("cdr", "both") or cdr_cfg is None:
        return _BASE_RUN_MITIGATION(
            mode,
            ansatz_circuit=ansatz_circuit,
            observable_h=observable_h,
            qubits=qubits,
            target_resolver=target_resolver,
            target_params=target_params,
            symbols=symbols,
            base_noise_cfg=base_noise_cfg,
            shot_cfg=shot_cfg,
            readout_cal=readout_cal,
            zne_cfg=zne_cfg,
            cdr_cfg=cdr_cfg,
            simulator_seed=simulator_seed,
        )

    requested_count = int(cdr_cfg["num_circuits"])
    method, pool_size, local_count = _selection_settings(cdr_cfg, requested_count)
    fit_mode, exact_tol, min_points, slope_max = _per_term_fit_settings(cdr_cfg)

    cdr_target_params = target_params if target_params is not None else cdr_cfg.get("target_params")
    cdr_symbols = symbols if symbols is not None else cdr_cfg.get("symbols")
    if cdr_target_params is None or cdr_symbols is None:
        raise ValueError("Selected CDR requires target_params and symbols.")

    selection_diagnostics: dict[str, Any] = {}
    refit_summary: dict[str, Any] = {}

    def selected_generator(
        pool_generator: Callable[..., list[dict]],
        generator_kwargs: dict[str, Any],
    ) -> list[dict]:
        pool = pool_generator(num_circuits=pool_size, **generator_kwargs)
        selected, diagnostics = select_cdr_resolvers(
            pool,
            ansatz_circuit=ansatz_circuit,
            observable_h=observable_h,
            qubits=qubits,
            symbols=list(cdr_symbols),
            target_params=cdr_target_params,
            num_select=requested_count,
            local_count=local_count,
            simulator_seed=simulator_seed,
        )
        selection_diagnostics.clear()
        selection_diagnostics.update(diagnostics)
        return selected

    def near_clifford_wrapper(
        generated_target_params: dict,
        generated_symbols: list,
        *,
        num_circuits: int,
        t_max: int,
        circuit: cirq.Circuit,
        min_snap_fraction: float = 0.0,
        seed: int = 0,
    ) -> list[dict]:
        if int(num_circuits) != requested_count:
            raise ValueError("Unexpected CDR training count passed by base pipeline.")
        return selected_generator(
            _BASE_NEAR_CLIFFORD_GENERATOR,
            {
                "target_params": generated_target_params,
                "symbols": generated_symbols,
                "t_max": int(t_max),
                "circuit": circuit,
                "min_snap_fraction": float(min_snap_fraction),
                "seed": int(seed),
            },
        )

    def random_clifford_wrapper(
        generated_target_params: dict,
        generated_symbols: list,
        *,
        num_circuits: int,
        circuit: cirq.Circuit,
        seed: int = 0,
    ) -> list[dict]:
        if int(num_circuits) != requested_count:
            raise ValueError("Unexpected CDR training count passed by base pipeline.")
        return selected_generator(
            _BASE_RANDOM_CLIFFORD_GENERATOR,
            {
                "target_params": generated_target_params,
                "symbols": generated_symbols,
                "circuit": circuit,
                "seed": int(seed),
            },
        )

    def train_cf_models_per_pauli_wrapper(*args: Any, **kwargs: Any) -> dict:
        models = _BASE_TRAIN_CF_MODELS_PER_PAULI(*args, **kwargs)
        summary = _refit_models_per_pauli(
            models,
            fit_mode=fit_mode,
            exact_tol=exact_tol,
            min_points=min_points,
            slope_max=slope_max,
        )
        refit_summary.clear()
        refit_summary.update(summary)
        return models

    # The base module resolves these callables from its own globals, so replace
    # those references temporarily while retaining its orchestration code.
    with _GENERATOR_PATCH_LOCK:
        previous_near = _base.generate_near_clifford_param_sets
        previous_random = _base.generate_random_clifford_analogue_param_sets
        previous_train = _base.train_cf_models_per_pauli
        if method == "weighted_maxmin":
            _base.generate_near_clifford_param_sets = near_clifford_wrapper
            _base.generate_random_clifford_analogue_param_sets = random_clifford_wrapper
        if fit_mode != "none":
            _base.train_cf_models_per_pauli = train_cf_models_per_pauli_wrapper
        try:
            result = _BASE_RUN_MITIGATION(
                mode,
                ansatz_circuit=ansatz_circuit,
                observable_h=observable_h,
                qubits=qubits,
                target_resolver=target_resolver,
                target_params=target_params,
                symbols=symbols,
                base_noise_cfg=base_noise_cfg,
                shot_cfg=shot_cfg,
                readout_cal=readout_cal,
                zne_cfg=zne_cfg,
                cdr_cfg=cdr_cfg,
                simulator_seed=simulator_seed,
            )
        finally:
            _base.generate_near_clifford_param_sets = previous_near
            _base.generate_random_clifford_analogue_param_sets = previous_random
            _base.train_cf_models_per_pauli = previous_train

    if method == "weighted_maxmin":
        result["cdr_selection"] = dict(selection_diagnostics)
    if fit_mode != "none" and refit_summary:
        result["cdr_per_term_fit"] = dict(refit_summary)
    if bool(int(os.environ.get("CDR_SELECTION_VERBOSE", "0"))):
        if method == "weighted_maxmin":
            print(
                "[CDR-select] "
                f"selected={selection_diagnostics.get('selected_count')} "
                f"from pool={selection_diagnostics.get('pool_size')}; "
                f"local={selection_diagnostics.get('local_count')}; "
                "weighted range="
                f"{selection_diagnostics.get('selected_weighted_mean_range', float('nan')):.6f}"
            )
        if fit_mode != "none" and refit_summary:
            print(
                "[CDR-select] per-term refit "
                f"mode={refit_summary.get('fit_mode')} "
                f"rem_used={refit_summary.get('rem_fit_used_counts')}"
            )
    return result

