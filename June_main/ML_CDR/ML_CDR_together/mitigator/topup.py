"""Uncertainty-gated top-up controller (Section 7).

During VQE, if the GP is confident at the current theta we spend zero new circuits;
if uncertain we sample a small batch of LOCAL near-Clifford circuits, refit, and
continue. Per-iteration cost tapers toward zero as the optimizer settles.
"""

from __future__ import annotations

import numpy as np

from .adapters import PauliObservable, QuantumBackendAdapter
from .config import MitigatorConfig


def aggregate_uncertainty(std_by_pauli: dict, hamiltonian) -> float:
    """|c_i|-weighted aggregate of per-Pauli predicted std (energy-scale)."""
    total = 0.0
    for coeff, obs in hamiltonian:
        total += abs(float(coeff)) * float(std_by_pauli.get(obs, 0.0))
    return float(total)


def energy_std(std_by_pauli: dict, hamiltonian) -> float:
    """Predicted energy standard deviation sqrt(sum_i (c_i * std_i)^2)."""
    acc = 0.0
    for coeff, obs in hamiltonian:
        acc += (float(coeff) * float(std_by_pauli.get(obs, 0.0))) ** 2
    return float(np.sqrt(acc))


def needs_topup(std_by_pauli: dict, hamiltonian, config: MitigatorConfig) -> bool:
    """True if the |c_i|-weighted predicted std exceeds the threshold."""
    return aggregate_uncertainty(std_by_pauli, hamiltonian) > float(config.uncertainty_threshold)


def sample_local_rows(
    backend: QuantumBackendAdapter,
    theta: np.ndarray,
    config: MitigatorConfig,
    seed: int | None = None,
) -> list[dict]:
    """Generate ``topup_batch_size`` near-Clifford circuits within ``topup_radius``
    of ``theta`` (radius defaults to ``optimizer_step_max``), measure all
    observables, and return training rows {theta, pauli, o_noisy, o_ideal}.
    """
    theta = np.asarray(theta, dtype=float).reshape(config.n_params)
    radius = config.effective_topup_radius
    base_seed = int(config.rng_seed if seed is None else seed)

    circuits = backend.generate_near_clifford(
        theta=theta,
        n_circuits=int(config.topup_batch_size),
        n_nonclifford=int(config.n_nonclifford_gates),
        snap_step=float(config.clifford_snap_step),
        spread=float(radius),
        seed=base_seed,
    )
    observables = backend.observables()
    rows: list[dict] = []
    for i, (resolver, theta_vec) in enumerate(circuits):
        ideal = backend.simulate_ideal(resolver)
        noisy = backend.run_noisy(
            resolver, shots=int(config.shots), sampling_seed=base_seed + 104729 * (i + 1)
        )
        for obs in observables:
            rows.append(
                {
                    "theta": np.asarray(theta_vec, dtype=float),
                    "pauli": obs,
                    "o_noisy": float(noisy[obs]),
                    "o_ideal": float(ideal[obs]),
                }
            )
    return rows
