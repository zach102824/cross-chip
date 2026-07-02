"""Configuration for the single-GP error mitigator (Design 2).

CDR is absorbed inside the GP kernel (a linear kernel on the ``o_noisy`` feature),
so there is NO separate linear CDR fit. The values below are DEFAULTS, not
constants -- everything is an argument that follows from the problem.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MitigatorConfig:
    # --- Problem / ansatz ---
    n_params: int = 3                       # variational parameters in the ansatz
    n_qubits: int = 4                       # number of qubits
    theta_init: "np.ndarray | None" = None  # warm-start center, len == n_params

    # --- Near-Clifford data generation ---
    n_warmstart_circuits: int = 30          # one-time warm-start near-Clifford circuits
    n_nonclifford_gates: int = 2            # non-Clifford gates kept per near-Clifford circuit
    warmstart_spread: float = 0.3           # angle spread (radians) around theta_init
    clifford_snap_step: float = np.pi / 2   # angles snapped near multiples of this
    shots: int = 10000                      # shots per noisy expectation value
    rng_seed: int = 0                       # reproducibility

    # --- Observables (mostly derived from the Hamiltonian, kept for clarity) ---
    n_observables: int = 100                # informational; real list comes from the Hamiltonian

    # --- Feature encoding ---
    max_harmonic: int = 1                   # highest k in (cos k*theta, sin k*theta) per angle
    include_noisy_feature: bool = True      # REQUIRED in Design 2: o_noisy is the linear-kernel feature
    pauli_onehot: bool = True               # per-qubit [I,X,Y,Z] one-hot block
    pauli_summaries: bool = True            # weight, #XY, #Z summary features

    # --- Single-GP kernel (CDR lives here) ---
    use_linear_kernel: bool = True          # k_lin on o_noisy -> the "CDR-inside-GP" term
    use_rbf_kernel: bool = True             # k_rbf/matern on (angles, Pauli) -> coherent residual
    linear_times_obs: bool = True           # k_lin x k_obs -> per-observable effective slope
    kernel_type: str = "matern"             # "matern" or "rbf" for the angle/Pauli block
    matern_nu: float = 2.5                  # smoothness if kernel_type == "matern"
    use_ard: bool = True                    # one lengthscale per input feature
    noise_variance_init: float = 1e-3       # white-noise term init (shot noise)
    normalize_targets: bool = True          # standardize O_ideal targets before GP fit

    # --- Top-up controller ---
    uncertainty_threshold: float = 0.05     # |c_i|-weighted predicted std that triggers a top-up
    topup_batch_size: int = 12              # local near-Clifford circuits added per top-up
    topup_radius: "float | None" = None     # local sampling radius; None -> tie to optimizer_step_max
    drop_faraway_points: bool = True        # prune distant points to keep the GP fast
    max_gp_points: int = 5000               # cap on stored rows before pruning

    # --- Optimizer ---
    optimizer_step_max: float = 0.2         # max move per iteration (trust region); sets topup_radius
    max_vqe_iterations: int = 100

    # --- Validation ---
    holdout_fraction: float = 0.2           # fraction of warm-start circuits held out
    target_energy_accuracy: float = 0.05    # accuracy goal; can derive uncertainty_threshold

    def __post_init__(self) -> None:
        # Design-2 invariants: CDR is represented by the linear kernel on o_noisy.
        # Disabling either of these removes the CDR backbone entirely.
        if not self.include_noisy_feature:
            raise ValueError(
                "Design 2 requires include_noisy_feature=True (o_noisy is the linear-kernel feature)."
            )
        if not self.use_linear_kernel:
            raise ValueError(
                "Design 2 requires use_linear_kernel=True (this is how CDR lives inside the GP)."
            )
        if self.theta_init is not None:
            self.theta_init = np.asarray(self.theta_init, dtype=float).reshape(self.n_params)

    @property
    def effective_topup_radius(self) -> float:
        """Local sampling radius; falls back to the optimizer trust-region step."""
        return float(self.topup_radius if self.topup_radius is not None else self.optimizer_step_max)
