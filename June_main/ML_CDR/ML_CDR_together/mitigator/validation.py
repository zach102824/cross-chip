"""Validation checks to run BEFORE the live VQE loop (Section 9).

- holdout: single GP vs unmitigated vs a plain linear-CDR baseline.
- linear-recovery (Design-2-specific): confirm the slope implied by the fitted
  k_lin term roughly matches a standalone CDR line -> evidence the GP ABSORBED CDR.
- interpolation/extrapolation split: train on one angle sub-range, test on another.
- anchor: a fully-Clifford theta with a known exact energy.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .config import MitigatorConfig
from .features import build_feature_matrix, feature_index_map
from .gp_model import SingleGPMitigatorModel


def _xy(rows: list[dict], config: MitigatorConfig) -> tuple[np.ndarray, np.ndarray]:
    X = build_feature_matrix(rows, config)
    y = np.array([float(r["o_ideal"]) for r in rows], dtype=float)
    return X, y


def _mae(a, b) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    return float(np.mean(np.abs(a - b)))


def _split_rows_by_circuit(rows: list[dict], holdout_fraction: float, seed: int):
    """Split by unique circuit (theta), so holdout circuits are fully unseen."""
    thetas = [tuple(np.round(np.asarray(r["theta"], dtype=float), 9).tolist()) for r in rows]
    uniq = sorted(set(thetas))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(uniq))
    n_hold = max(1, int(round(holdout_fraction * len(uniq))))
    hold_keys = set(uniq[i] for i in perm[:n_hold].tolist())
    train, hold = [], []
    for r, key in zip(rows, thetas):
        (hold if key in hold_keys else train).append(r)
    return train, hold


def _plain_cdr_per_pauli(train: list[dict], hold: list[dict]) -> np.ndarray:
    """Standalone per-Pauli linear CDR baseline; returns predicted o_ideal on hold."""
    def by_pauli(rows):
        groups: dict = {}
        for r in rows:
            groups.setdefault(r["pauli"], []).append(r)
        return groups

    train_g = by_pauli(train)
    coeffs: dict = {}
    for pauli, rs in train_g.items():
        xn = np.array([r["o_noisy"] for r in rs], dtype=float)
        yi = np.array([r["o_ideal"] for r in rs], dtype=float)
        if len(rs) >= 2 and float(np.std(xn)) > 0.0:
            coeffs[pauli] = np.polyfit(xn, yi, deg=1)
        else:
            coeffs[pauli] = np.array([1.0, float(np.mean(yi - xn))])
    preds = []
    for r in hold:
        c = coeffs.get(r["pauli"], np.array([1.0, 0.0]))
        preds.append(float(np.polyval(c, r["o_noisy"])))
    return np.asarray(preds, dtype=float)


def holdout_validation(
    rows: list[dict], config: MitigatorConfig, *, seed: int | None = None
) -> dict:
    """Compare single GP vs unmitigated vs plain CDR on held-out circuits."""
    seed = int(config.rng_seed if seed is None else seed)
    train, hold = _split_rows_by_circuit(rows, config.holdout_fraction, seed)
    if not hold or not train:
        raise ValueError("Not enough distinct circuits to form a holdout split.")

    Xtr, ytr = _xy(train, config)
    Xho, yho = _xy(hold, config)

    gp = SingleGPMitigatorModel(config).fit(Xtr, ytr)
    gp_mean, _ = gp.predict(Xho)

    o_noisy_ho = np.array([float(r["o_noisy"]) for r in hold], dtype=float)
    cdr_pred = _plain_cdr_per_pauli(train, hold)

    return {
        "n_train_rows": len(train),
        "n_holdout_rows": len(hold),
        "mae_gp": _mae(gp_mean, yho),
        "mae_unmitigated": _mae(o_noisy_ho, yho),
        "mae_plain_cdr": _mae(cdr_pred, yho),
    }


def linear_recovery_check(
    rows: list[dict], config: MitigatorConfig, *, seed: int | None = None
) -> dict:
    """Confirm the GP's effective linear slope matches a standalone CDR line.

    Fits a k_lin-only GP (use_rbf_kernel=False, linear_times_obs=False) and probes
    d(mean)/d(o_noisy) numerically, comparing to a global least-squares slope. A
    close match is the evidence the GP ABSORBED CDR rather than fighting it.
    """
    seed = int(config.rng_seed if seed is None else seed)
    lin_cfg = replace(config, use_rbf_kernel=False, linear_times_obs=False)
    X, y = _xy(rows, lin_cfg)
    gp = SingleGPMitigatorModel(lin_cfg).fit(X, y)

    idx = feature_index_map(lin_cfg)
    noisy_col = idx["noisy"].start
    o_noisy = X[:, noisy_col]
    lo, hi = float(np.percentile(o_noisy, 10)), float(np.percentile(o_noisy, 90))
    if hi - lo < 1e-9:
        lo, hi = float(np.min(o_noisy)) - 0.5, float(np.max(o_noisy)) + 0.5

    # Probe slope at a representative feature row (median row), varying only o_noisy.
    base = np.median(X, axis=0, keepdims=True)
    x_lo = base.copy(); x_lo[0, noisy_col] = lo
    x_hi = base.copy(); x_hi[0, noisy_col] = hi
    m_lo, _ = gp.predict(x_lo)
    m_hi, _ = gp.predict(x_hi)
    gp_slope = float((m_hi[0] - m_lo[0]) / (hi - lo))

    if float(np.std(o_noisy)) > 0.0:
        ls_slope, ls_intercept = np.polyfit(o_noisy, y, deg=1)
    else:
        ls_slope, ls_intercept = 1.0, float(np.mean(y - o_noisy))

    rel_err = abs(gp_slope - float(ls_slope)) / (abs(float(ls_slope)) + 1e-9)
    return {
        "gp_linear_slope": gp_slope,
        "least_squares_slope": float(ls_slope),
        "least_squares_intercept": float(ls_intercept),
        "relative_slope_error": float(rel_err),
        "passed": bool(rel_err < 0.5),
    }


def extrapolation_split(
    rows: list[dict], config: MitigatorConfig, *, axis: int = 0
) -> dict:
    """Train on the lower half of an angle axis, test on the upper half."""
    vals = np.array([float(np.asarray(r["theta"], dtype=float)[axis]) for r in rows])
    median = float(np.median(vals))
    train = [r for r, v in zip(rows, vals) if v <= median]
    test = [r for r, v in zip(rows, vals) if v > median]
    if not train or not test:
        return {"skipped": True, "reason": "degenerate split along axis"}

    Xtr, ytr = _xy(train, config)
    Xte, yte = _xy(test, config)
    gp = SingleGPMitigatorModel(config).fit(Xtr, ytr)
    gp_mean, _ = gp.predict(Xte)
    o_noisy_te = np.array([float(r["o_noisy"]) for r in test], dtype=float)
    return {
        "axis": int(axis),
        "split_value": median,
        "n_train_rows": len(train),
        "n_test_rows": len(test),
        "mae_gp_extrapolation": _mae(gp_mean, yte),
        "mae_unmitigated_extrapolation": _mae(o_noisy_te, yte),
    }


def anchor_check(mitigator, backend, config: MitigatorConfig, *, seed: int | None = None) -> dict:
    """Fully-Clifford theta with a known exact energy; confirm the pipeline lands on it."""
    seed = int(config.rng_seed if seed is None else seed)
    theta0 = (
        np.asarray(config.theta_init, dtype=float).reshape(config.n_params)
        if config.theta_init is not None
        else np.zeros(config.n_params, dtype=float)
    )
    # Snap all angles to Clifford (n_nonclifford=0) -> one fully-Clifford circuit.
    resolver, theta_vec = backend.generate_near_clifford(
        theta=theta0,
        n_circuits=1,
        n_nonclifford=0,
        snap_step=float(config.clifford_snap_step),
        spread=0.0,
        seed=seed,
    )[0]

    ideal = backend.simulate_ideal(resolver)
    noisy = backend.run_noisy(resolver, shots=int(config.shots), sampling_seed=seed)

    offset = getattr(backend, "hamiltonian_offset", 0.0)
    e_exact = float(offset) + sum(c * float(ideal[p]) for c, p in mitigator.hamiltonian)
    e_unmit = float(offset) + sum(c * float(noisy[p]) for c, p in mitigator.hamiltonian)
    o_mit = mitigator.mitigate(theta_vec, noisy)
    e_mit = mitigator.energy(o_mit)

    return {
        "theta_clifford": theta_vec.tolist(),
        "energy_exact": float(e_exact),
        "energy_unmitigated": float(e_unmit),
        "energy_mitigated": float(e_mit),
        "abs_error_mitigated": float(abs(e_mit - e_exact)),
        "abs_error_unmitigated": float(abs(e_unmit - e_exact)),
    }


def run_all(mitigator, backend, rows: list[dict], config: MitigatorConfig) -> dict:
    """Convenience: run every validation check and return a combined report."""
    return {
        "holdout": holdout_validation(rows, config),
        "linear_recovery": linear_recovery_check(rows, config),
        "extrapolation": extrapolation_split(rows, config),
        "anchor": anchor_check(mitigator, backend, config),
    }
