"""Fit ansatz parameters to maximize overlap with a target doubles state."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize


@dataclass
class FitResult:
    overlap: float
    nfev: int
    success: bool
    x: np.ndarray
    n2: int
    n1_params: int
    n2_params: int


def bind(qc, params: list[Parameter], x: np.ndarray):
    return qc.assign_parameters(dict(zip(params, x)))


def overlap_with_target(qc, params, x, target: Statevector) -> float:
    sv = Statevector.from_instruction(bind(qc, params, x))
    return float(np.abs(np.vdot(target.data, sv.data)) ** 2)


def fit_one(
    qc,
    all_params: list[Parameter],
    p2: list[Parameter],
    p1: list[Parameter],
    target: Statevector,
    *,
    mode: str = "joint",
    fixed_2q: float | None = None,
    x0: np.ndarray | None = None,
    maxiter: int = 400,
    seed: int = 0,
) -> FitResult:
    """Maximize |⟨target|U(x)|0⟩|².

    mode:
      joint     — optimize all params
      only_1q   — freeze 2q at fixed_2q (default π/4), optimize 1q only
    """
    rng = np.random.default_rng(seed)
    n2_count = qc.num_nonlocal_gates()

    if mode == "only_1q":
        w = np.pi / 4 if fixed_2q is None else float(fixed_2q)
        # all_params order is not guaranteed; build full vector via names
        p2_set = set(p2)
        n_all = len(all_params)

        def full_from_1q(y):
            y = np.asarray(y, dtype=float)
            out = np.zeros(n_all)
            i1 = 0
            for i, p in enumerate(all_params):
                if p in p2_set:
                    out[i] = w
                else:
                    out[i] = y[i1]
                    i1 += 1
            return out

        n1 = len(p1)
        y0 = x0 if x0 is not None else rng.normal(0.0, 0.15, size=n1)

        def cost(y):
            return -overlap_with_target(qc, all_params, full_from_1q(y), target)

        res = minimize(cost, y0, method="L-BFGS-B", options={"maxiter": maxiter})
        ov = -float(res.fun)
        return FitResult(ov, int(res.nfev), bool(res.success), full_from_1q(res.x),
                         n2_count, n1, len(p2))

    # joint
    n = len(all_params)
    x_init = x0 if x0 is not None else rng.normal(0.0, 0.15, size=n)

    def cost(x):
        return -overlap_with_target(qc, all_params, x, target)

    res = minimize(cost, x_init, method="L-BFGS-B", options={"maxiter": maxiter})
    return FitResult(-float(res.fun), int(res.nfev), bool(res.success), np.asarray(res.x),
                     n2_count, len(p1), len(p2))


def fit_multistart(
    qc, all_params, p2, p1, target, *, mode="joint", fixed_2q=None,
    n_starts=4, maxiter=400, seed=0,
) -> FitResult:
    best = None
    for s in range(n_starts):
        r = fit_one(
            qc, all_params, p2, p1, target,
            mode=mode, fixed_2q=fixed_2q, maxiter=maxiter, seed=seed + 17 * s,
        )
        if best is None or r.overlap > best.overlap:
            best = r
    return best
