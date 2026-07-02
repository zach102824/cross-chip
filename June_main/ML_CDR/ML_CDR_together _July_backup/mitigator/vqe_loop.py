"""VQE loop integration (Section 8).

A bounded-step / trust-region gradient descent so ``optimizer_step_max`` is
meaningful (it sets the top-up radius). Per iteration:

  1. Measure the REAL circuit's noisy per-Pauli values for all Hamiltonian Paulis.
  2. O_mit, std = mitigator.predict_with_uncertainty(theta, o_noisy_by_pauli).
  3. If needs_topup: sample local rows, update_with_rows, re-predict.
  4. Energy = sum_i c_i * O_mit[P_i]; take a bounded gradient step.
  5. Log top-up frequency vs iteration (it should taper toward zero).
"""

from __future__ import annotations

import numpy as np

from .config import MitigatorConfig
from .mitigator import Mitigator
from .topup import aggregate_uncertainty, energy_std, needs_topup, sample_local_rows


def _resolver_for(backend, theta):
    if hasattr(backend, "theta_to_resolver"):
        return backend.theta_to_resolver(theta)
    raise AttributeError("backend must provide theta_to_resolver(theta) for the VQE loop.")


def measure_noisy(backend, theta, shots: int, sampling_seed: int) -> dict:
    """Measure the real circuit once at ``theta`` -> {pauli: o_noisy}."""
    return backend.run_noisy(_resolver_for(backend, theta), shots=int(shots), sampling_seed=int(sampling_seed))


def mitigated_energy(
    mitigator: Mitigator, backend, theta, shots: int, sampling_seed: int
) -> float:
    """Mitigated energy at ``theta`` (measure real circuit, transform with the GP)."""
    noisy = measure_noisy(backend, theta, shots, sampling_seed)
    o_mit = mitigator.mitigate(theta, noisy)
    return mitigator.energy(o_mit)


def _bounded_step(grad: np.ndarray, lr: float, step_max: float) -> np.ndarray:
    step = -float(lr) * np.asarray(grad, dtype=float)
    largest = float(np.max(np.abs(step))) if step.size else 0.0
    if largest > step_max > 0.0:
        step = step * (step_max / largest)
    return step


def run_vqe(
    mitigator: Mitigator,
    backend,
    config: MitigatorConfig,
    theta_init: np.ndarray | None = None,
    *,
    learning_rate: float = 0.1,
    gradient: str = "parameter_shift",
    max_iterations: int | None = None,
    verbose: bool = True,
) -> dict:
    """Run the GP-mitigated, uncertainty-gated, bounded-step VQE.

    Returns a history dict with per-iteration energy, energy std, top-up flags,
    cumulative top-ups, and GP size.
    """
    n_params = config.n_params
    theta = (
        np.asarray(theta_init, dtype=float).reshape(n_params)
        if theta_init is not None
        else (
            np.asarray(config.theta_init, dtype=float).reshape(n_params)
            if config.theta_init is not None
            else np.zeros(n_params, dtype=float)
        )
    )
    iters = int(max_iterations if max_iterations is not None else config.max_vqe_iterations)
    shift = np.pi / 2 if gradient == "parameter_shift" else 2e-2
    ham = mitigator.hamiltonian

    seed_counter = int(config.rng_seed) + 1
    n_topups = 0
    history: dict[str, list] = {
        "iter": [],
        "energy": [],
        "energy_std": [],
        "agg_uncertainty": [],
        "did_topup": [],
        "cum_topups": [],
        "n_rows": [],
        "theta": [],
        "step_max": [],
    }

    def _next_seed() -> int:
        nonlocal seed_counter
        seed_counter += 1
        return seed_counter

    def _energy(th) -> float:
        return mitigated_energy(mitigator, backend, th, config.shots, _next_seed())

    for it in range(1, iters + 1):
        # 1-3: measure at theta, predict with uncertainty, top-up if needed.
        noisy = measure_noisy(backend, theta, config.shots, _next_seed())
        o_mit, std = mitigator.predict_with_uncertainty(theta, noisy)
        agg = aggregate_uncertainty(std, ham)
        did_topup = False
        if needs_topup(std, ham, config):
            rows = sample_local_rows(backend, theta, config, seed=_next_seed())
            mitigator.update_with_rows(rows)
            o_mit, std = mitigator.predict_with_uncertainty(theta, noisy)
            agg = aggregate_uncertainty(std, ham)
            did_topup = True
            n_topups += 1

        E = mitigator.energy(o_mit)
        e_std = energy_std(std, ham)

        # 4: bounded-step gradient descent on the mitigated energy.
        grad = np.zeros(n_params, dtype=float)
        for j in range(n_params):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] += shift
            tm[j] -= shift
            grad[j] = (_energy(tp) - _energy(tm)) / (2.0 if gradient == "parameter_shift" else 2.0 * shift)
        step = _bounded_step(grad, learning_rate, config.optimizer_step_max)
        theta = theta + step

        history["iter"].append(it)
        history["energy"].append(float(E))
        history["energy_std"].append(float(e_std))
        history["agg_uncertainty"].append(float(agg))
        history["did_topup"].append(bool(did_topup))
        history["cum_topups"].append(int(n_topups))
        history["n_rows"].append(int(mitigator.n_rows))
        history["theta"].append(theta.copy())
        history["step_max"].append(float(np.max(np.abs(step))) if step.size else 0.0)

        if verbose:
            print(
                f"[GP-VQE] iter={it:02d}  E={E:.8f}  E_std={e_std:.3e}  "
                f"agg_unc={agg:.3e}  topup={'Y' if did_topup else '.'}  "
                f"cum_topups={n_topups}  rows={mitigator.n_rows}  "
                f"step_max={history['step_max'][-1]:.3e}"
            )

    history["theta_final"] = theta.copy()
    history["n_topups"] = n_topups
    return history
