"""Combined CDR + Gaussian-Process Error Mitigator for VQE (Design 2).

Unlike the two-stage design (separate per-Pauli affine CDR backbone + a GP on the
residual), here ONE Gaussian Process maps the feature row directly to the ideal
expectation:

    O_mit(theta, P_i) = GP( features(theta, P_i, O_noisy) )   ~= O_ideal

The classic CDR line ``a * O_noisy + b`` is NOT fit separately; it is absorbed
INTO the GP kernel as a linear (DotProduct) term on the ``O_noisy`` feature
column, with a Matern/RBF term on the angle+Pauli features capturing the coherent,
angle-dependent residual on top:

    k = c_lin * k_lin(O_noisy) [* k_obs(pauli)]      # the learned CDR slope/bias
      + c_base * k_base(angle, pauli)                # coherent / angle residual
      + WhiteKernel                                  # shot noise

The GP returns a variance at every prediction, which drives an uncertainty-gated
top-up loop during VQE.

This module is a thin layer on top of the user's EXISTING cirq codebase
(``main_cursor_lib_ML_tog`` and ``shot_measurement_ML_tog``). It does NOT
re-implement circuit construction, simulation, or shot estimation; it binds the
adapter methods to those existing functions. A plain per-Pauli CDR backbone is
still fit, but ONLY as a baseline reference for validation/plots -- it is not part
of the mitigated value.

The public API (config, adapter, Mitigator, validation, VQE loops) mirrors the
two-stage ``gp_mitigator_ML_sep`` module so the two notebooks differ only in the
error-mitigation core.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Keep this experiment folder self-contained: import the LOCAL ``*_ML_tog`` copies
# of the cirq pipeline that live next to this module, regardless of the caller's
# working directory. These are the editable "together" copies for the ML-CDR
# experiments; the June_main originals are left untouched.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from main_cursor_lib_ML_tog import (
    clifford_snap_value_for_symbol,
)
from shot_measurement_ML_tog import (
    _simulate_noiseless_state_for_resolver,
    _simulate_noisy_rho_for_resolver,
    estimate_energy_from_noisy_rho_shots,
    exact_pauli_expectation_from_int_row,
    int_observable_to_pauli_string,
    pauli_sum_to_int_observables,
)

try:  # scikit-learn is the GP backend (the single combined GP).
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
        DotProduct,
        Hyperparameter,
        Kernel,
        Matern,
        RBF,
        WhiteKernel,
    )

    _SKLEARN_AVAILABLE = True
except Exception as _exc:  # pragma: no cover - surfaced lazily when GP is used.
    _SKLEARN_AVAILABLE = False
    _SKLEARN_IMPORT_ERROR = _exc


# ---------------------------------------------------------------------------
# Section 1 - Global configuration (defaults, not constants)
# ---------------------------------------------------------------------------


@dataclass
class MitigatorConfig:
    # --- Problem / ansatz ---
    n_params: int = 3
    n_qubits: int = 8
    theta_init: "np.ndarray | None" = None  # warm-start center, len == n_params

    # --- Near-Clifford data generation ---
    n_warmstart_circuits: int = 30
    n_nonclifford_gates: int = 2
    warmstart_spread: float = 0.3  # angle spread (radians) around theta_init
    clifford_snap_step: float = float(np.pi) / 2.0
    shots: int = 10000
    rng_seed: int = 0

    # --- Observables (real list comes from the Hamiltonian) ---
    n_observables: int = 100  # informational only

    # --- Feature encoding ---
    max_harmonic: int = 1
    include_noisy_feature_in_gp: bool = True
    pauli_onehot: bool = True
    pauli_summaries: bool = True

    # --- Baseline per-Pauli CDR backbone (reference only; NOT part of the
    #     combined mitigated value, which is the single GP below). Kept so the
    #     validation/plots can show "plain CDR" alongside the combined method. ---
    affine_regularization: float = 0.0
    refit_affine_on_topup: bool = True

    # --- Combined single-GP kernel (CDR absorbed in the kernel) ---
    kernel_type: str = "matern"  # "matern" or "rbf" for the coherent-residual term
    matern_nu: float = 2.5
    use_ard: bool = True
    # use_product_kernel here means: multiply the linear CDR term k_lin(o_noisy) by
    # a per-observable kernel k_obs(pauli) -> a learned, observable-dependent CDR
    # slope (True), instead of a single shared slope (False).
    use_product_kernel: bool = True
    noise_variance_init: float = 1e-3
    normalize_targets: bool = True
    gp_n_restarts: int = 2  # marginal-likelihood restarts (not in spec; small default)
    # Cap on rows actually used to FIT the exact GP (the baseline backbone still
    # uses all rows). Exact-GP hyperparameter optimization with ARD builds an (N, N, n_hyper)
    # gradient tensor, so N must stay bounded; rows are randomly subsampled down to
    # this many before fitting. ~1000-2000 is plenty for an exact GP.
    max_gp_train_points: int = 1500

    # --- Top-up controller ---
    uncertainty_threshold: float = 0.05
    topup_batch_size: int = 12
    topup_radius: "float | None" = None  # None -> tie to optimizer_step_max
    drop_faraway_points: bool = True
    max_gp_points: int = 5000
    # (1) Trust-region trigger: top up whenever theta has moved more than
    #     ``topup_move_fraction * effective_topup_radius`` from the center of the
    #     last top-up, so the surrogate stays valid as the optimizer travels --
    #     destination-agnostic, fires even when the (overconfident) uncertainty
    #     gate stays silent off the near-Clifford cloud.
    topup_on_move: bool = True
    topup_move_fraction: float = 1.0
    # (2) Mandatory periodic top-up every ``topup_every`` iterations (0 = off).
    #     Cheap insurance against a frozen, overconfident GP trapping the optimizer.
    topup_every: int = 0
    # (3) Top-up-until-satisfied: after a top-up fires, keep adding local batches
    #     while the |c|-weighted predicted std stays above ``uncertainty_threshold``,
    #     up to ``max_topup_retries`` extra batches (0 = off -> a single top-up).
    #     Bounds the shot cost while forcing the surrogate to be trustworthy at
    #     the current theta before the gradient/step are computed.
    max_topup_retries: int = 0
    # (4) Convergence validation: when |grad| < ``convergence_grad_tol`` fire one
    #     REAL near-Clifford top-up at the current theta and re-check before
    #     declaring convergence; only stop if the energy then moves less than
    #     ``convergence_energy_tol`` and the gradient is still below tolerance.
    #     ``convergence_grad_tol = 0`` disables early stopping (run the full
    #     iteration budget -- the legacy behavior).
    convergence_grad_tol: float = 0.0
    convergence_energy_tol: float = 1e-3

    # --- Optimizer ---
    optimizer_step_max: float = 0.2
    max_vqe_iterations: int = 100

    # --- Validation ---
    holdout_fraction: float = 0.2
    target_energy_accuracy: float = 0.05

    # --- Branch selection (this repo exposes both an unmitigated and a
    #     readout-error-mitigated noisy expectation value per term). The
    #     notebook's live objective is ``cdr_rem_corrected`` so we default to the
    #     REM branch; flip to fit/predict on the raw unmitigated branch. ---
    use_rem_branch: bool = True

    def effective_topup_radius(self) -> float:
        return float(self.topup_radius if self.topup_radius is not None else self.optimizer_step_max)


# ---------------------------------------------------------------------------
# Section 4 - Feature construction
# ---------------------------------------------------------------------------


def encode_angles(theta: np.ndarray, max_harmonic: int) -> np.ndarray:
    """Fourier angle block: ``[cos(k t_j), sin(k t_j) for k in 1..max_harmonic]``.

    Encoding sinusoids (never raw angles) gives the GP the correct periodic
    inductive bias; an ordinary RBF/Matern kernel on these features is then
    automatically periodic in theta. Length = ``n_params * 2 * max_harmonic``.
    """
    theta = np.asarray(theta, dtype=float).ravel()
    if max_harmonic < 1:
        raise ValueError(f"max_harmonic must be >= 1, got {max_harmonic}.")
    feats: list[float] = []
    for t in theta:
        for k in range(1, int(max_harmonic) + 1):
            feats.append(float(np.cos(k * t)))
            feats.append(float(np.sin(k * t)))
    return np.asarray(feats, dtype=float)


def encode_pauli(
    pauli: str, n_qubits: int, onehot: bool = True, summaries: bool = True
) -> np.ndarray:
    """Pauli block from a per-qubit ``{I,X,Y,Z}`` label string.

    - one-hot: per qubit a 4-dim indicator over [I, X, Y, Z]  (length 4 * n_qubits)
    - summaries: ``[weight, n_XY, n_Z]`` where weight = #non-identity factors.
    """
    label = str(pauli)
    if len(label) != n_qubits:
        raise ValueError(
            f"Pauli label '{label}' has length {len(label)} but n_qubits={n_qubits}."
        )
    idx = {"I": 0, "X": 1, "Y": 2, "Z": 3}
    feats: list[float] = []
    if onehot:
        for ch in label:
            block = [0.0, 0.0, 0.0, 0.0]
            block[idx[ch]] = 1.0
            feats.extend(block)
    if summaries:
        weight = float(sum(1 for ch in label if ch != "I"))
        n_xy = float(sum(1 for ch in label if ch in ("X", "Y")))
        n_z = float(sum(1 for ch in label if ch == "Z"))
        feats.extend([weight, n_xy, n_z])
    return np.asarray(feats, dtype=float)


def feature_index_map(config: MitigatorConfig) -> dict[str, list[int]]:
    """Record which feature indices belong to which block.

    Blocks: ``angle`` (Fourier), ``noisy`` (optional O_noisy scalar), ``pauli``.
    The product kernel groups (angle + noisy) as the continuous block and pauli
    as the observable block.
    """
    n_angle = int(config.n_params) * 2 * int(config.max_harmonic)
    n_noisy = 1 if config.include_noisy_feature_in_gp else 0
    n_pauli = (4 * int(config.n_qubits) if config.pauli_onehot else 0) + (
        3 if config.pauli_summaries else 0
    )
    angle_idx = list(range(0, n_angle))
    noisy_idx = list(range(n_angle, n_angle + n_noisy))
    pauli_idx = list(range(n_angle + n_noisy, n_angle + n_noisy + n_pauli))
    return {
        "angle": angle_idx,
        "noisy": noisy_idx,
        "pauli": pauli_idx,
        "continuous": angle_idx + noisy_idx,  # angle + noisy for the product kernel
    }


def build_feature_row(row: dict, config: MitigatorConfig) -> np.ndarray:
    """Build the GP input vector for one row: ``[angle | (o_noisy) | pauli]``."""
    parts = [encode_angles(row["theta"], config.max_harmonic)]
    if config.include_noisy_feature_in_gp:
        parts.append(np.asarray([float(row["o_noisy"])], dtype=float))
    parts.append(
        encode_pauli(
            row["pauli"],
            config.n_qubits,
            onehot=config.pauli_onehot,
            summaries=config.pauli_summaries,
        )
    )
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Section 2 - Integration adapter (binds spec methods to existing repo code)
# ---------------------------------------------------------------------------


class CirqBackendAdapter:
    """Binds the spec's ``QuantumBackendAdapter`` to the existing cirq pipeline.

    The ansatz is a fixed symbolic ``cirq.Circuit``; "building" it for a given
    ``theta`` is just constructing a resolver mapping symbols -> floats.
    """

    def __init__(
        self,
        *,
        circuit,
        qubits,
        symbols,
        pauli_sum,
        base_noise_cfg: dict,
        shot_cfg: dict,
        readout_cal: dict | None = None,
        simulator_seed: int = 1234,
        use_rem_branch: bool = True,
    ) -> None:
        self.circuit = circuit
        self.qubits = list(qubits)
        self.symbols = list(symbols)
        self.pauli_sum = pauli_sum
        self.base_noise_cfg = dict(base_noise_cfg)
        self.shot_cfg = dict(shot_cfg)
        self.readout_cal = dict(readout_cal or {})
        self.simulator_seed = int(simulator_seed)
        self.use_rem_branch = bool(use_rem_branch)

        # The two-stage CDR + GP mitigator measures every circuit with the OGM
        # (operator-grouping) scheme. The per-Pauli (``direct_pauli``) fallback is
        # intentionally disabled: if the OGM basis file or the shadowgrouping
        # backend is unavailable we raise instead of silently degrading.
        self._require_ogm_available()

        observables_int, weights, offset = pauli_sum_to_int_observables(
            pauli_sum, self.qubits
        )
        self.observables_int = observables_int
        self.weights = np.asarray(weights, dtype=float)
        self.offset = float(offset)
        self.obs_labels = [
            int_observable_to_pauli_string(row) for row in observables_int
        ]
        self.coeff_by_pauli = {
            label: float(w) for label, w in zip(self.obs_labels, self.weights)
        }

        # --- Shot accounting -------------------------------------------------
        # EVERY quantum execution on this backend flows through ``run_noisy``
        # (warm start, per-iteration energy, parameter-shift gradient and
        # top-ups all call it), so counting those calls gives the exact hardware
        # cost. Each call consumes ``shots`` shots (the OGM scheme spreads that
        # budget across measurement groups internally). ``ideal_energy`` /
        # ``simulate_ideal`` are noiseless statevector evaluations and cost ZERO
        # shots, so they are deliberately not counted.
        self.n_circuit_evals = 0
        self.total_shots = 0

    def reset_shot_counter(self) -> None:
        """Zero the run_noisy call / shot counters (e.g. before a fresh run)."""
        self.n_circuit_evals = 0
        self.total_shots = 0

    def shot_report(self) -> dict:
        """Current empirical shot usage: ``run_noisy`` calls and total shots."""
        return {
            "circuit_evaluations": int(self.n_circuit_evals),
            "total_shots": int(self.total_shots),
        }

    # -- OGM availability (no silent per-Pauli fallback) ------------------
    def _require_ogm_available(self) -> None:
        """Raise unless the OGM measurement backend is fully available.

        The mitigator does not fall back to per-Pauli (``direct_pauli``)
        measurement; OGM is mandatory so the assessed performance reflects the
        real grouped-measurement layout.
        """
        scheme = str(self.shot_cfg.get("measurement_scheme", "ogm")).lower()
        if scheme == "direct_pauli":
            raise ValueError(
                "The two-stage CDR+GP mitigator requires the OGM measurement "
                "scheme; the per-Pauli 'direct_pauli' fallback is disabled. "
                "Set shot_cfg['measurement_scheme'] = 'ogm'."
            )
        ogm_file = self.shot_cfg.get("ogm_file")
        if not ogm_file or not Path(str(ogm_file)).is_file():
            raise FileNotFoundError(
                f"OGM measurement-basis file is not available: {ogm_file!r}. "
                "Generate the OGM basis for this molecule/bond "
                "(June_main/OGM_measurement_basis/OGM_<MOLECULE>_bond_<bond>.txt) "
                "before running the two-stage CDR+GP mitigator."
            )
        sg_root = self.shot_cfg.get("shadowgrouping_root")
        if not sg_root or not Path(str(sg_root)).is_dir():
            raise FileNotFoundError(
                f"shadowgrouping_root is not available: {sg_root!r}. The OGM "
                "scheme needs the shadowgrouping package; set "
                "shot_cfg['shadowgrouping_root'] to its location."
            )

    # -- ansatz / parameter helpers --------------------------------------
    def resolver_from_theta(self, theta: np.ndarray) -> dict:
        theta = np.asarray(theta, dtype=float).ravel()
        return {sym: float(theta[j]) for j, sym in enumerate(self.symbols)}

    def theta_from_resolver(self, resolver: dict) -> np.ndarray:
        return np.asarray([float(resolver[sym]) for sym in self.symbols], dtype=float)

    def build_ansatz(self, theta: np.ndarray) -> dict:
        """Spec ``build_ansatz``: returns the resolver for the fixed circuit."""
        return self.resolver_from_theta(theta)

    # -- near-Clifford generation (LOCAL around theta) -------------------
    def generate_near_clifford(
        self,
        theta: np.ndarray,
        n_circuits: int,
        n_nonclifford: int,
        snap_step: float,
        spread: float,
        seed: int,
    ) -> list[dict]:
        """Local near-Clifford resolvers around ``theta``.

        Most parameters are snapped to the nearest Clifford grid; a random subset
        of size ``n_nonclifford`` is left non-Clifford by jittering within
        ``spread`` radians of ``theta``. Locality matters because the GP uses the
        angles as features (warm-start spread and top-up radius both flow here).
        """
        theta = np.asarray(theta, dtype=float).ravel()
        n_params = len(self.symbols)
        n_nc = int(max(0, min(int(n_nonclifford), n_params)))
        rng = np.random.default_rng(int(seed))
        resolvers: list[dict] = []
        for _ in range(int(n_circuits)):
            if n_nc > 0:
                nc_idx = set(
                    int(j)
                    for j in rng.choice(n_params, size=n_nc, replace=False).tolist()
                )
            else:
                nc_idx = set()
            resolver: dict = {}
            for j, sym in enumerate(self.symbols):
                tj = float(theta[j])
                if j in nc_idx:
                    resolver[sym] = tj + float(rng.uniform(-spread, spread))
                else:
                    resolver[sym] = _snap_to_clifford(sym, tj, snap_step)
            resolvers.append(resolver)
        return resolvers

    # -- simulation (classical exact) ------------------------------------
    def simulate_ideal(self, resolver: dict) -> dict:
        state = _simulate_noiseless_state_for_resolver(
            self.circuit, resolver, self.qubits, simulator_seed=self.simulator_seed
        )
        return {
            label: float(
                exact_pauli_expectation_from_int_row(state, obs_row, self.qubits)
            )
            for label, obs_row in zip(self.obs_labels, self.observables_int)
        }

    # -- noisy execution (shots) -----------------------------------------
    def run_noisy(self, resolver: dict, shots: int, sampling_seed: int) -> dict:
        # Shot accounting: one physical circuit execution at ``shots`` shots.
        self.n_circuit_evals += 1
        self.total_shots += int(shots)
        rho = _simulate_noisy_rho_for_resolver(
            self.circuit,
            resolver,
            self.qubits,
            self.base_noise_cfg,
            simulator_seed=self.simulator_seed,
        )
        est = estimate_energy_from_noisy_rho_shots(
            rho,
            self.pauli_sum,
            self.qubits,
            num_shots=int(shots),
            measurement_scheme=str(self.shot_cfg.get("measurement_scheme", "direct_pauli")),
            p_0_success=self.readout_cal.get("p_0_success"),
            p_1_success=self.readout_cal.get("p_1_success"),
            apply_rem=True,
            apply_readout_noise=bool(self.shot_cfg.get("apply_readout_noise", True)),
            sampling_seed=int(sampling_seed),
            epsilon=float(self.shot_cfg.get("epsilon", 0.1)),
            ogm_file=self.shot_cfg.get("ogm_file"),
            shadowgrouping_root=self.shot_cfg.get("shadowgrouping_root"),
            return_per_term=True,
        )
        per_term_unmit = np.asarray(est["per_term_unmitigated"], dtype=float).ravel()
        per_term_rem = np.asarray(est["per_term_rem"], dtype=float).ravel()
        key = "per_term_rem" if self.use_rem_branch else "per_term_unmitigated"
        per_term = per_term_rem if key == "per_term_rem" else per_term_unmit
        primary = {label: float(per_term[k]) for k, label in enumerate(self.obs_labels)}
        unmit_by_pauli = {
            label: float(per_term_unmit[k]) for k, label in enumerate(self.obs_labels)
        }
        rem_by_pauli = {
            label: float(per_term_rem[k]) for k, label in enumerate(self.obs_labels)
        }
        return {
            "primary": primary,
            "unmit_by_pauli": unmit_by_pauli,
            "rem_by_pauli": rem_by_pauli,
            "energy_unmitigated": float(est["energy_unmitigated"]),
            "energy_rem": float(est["energy_rem"]),
        }

    # -- row collection (one row per (circuit, observable)) ---------------
    def collect_rows(
        self, resolvers: list[dict], *, shots: int, seed_base: int
    ) -> list[dict]:
        rows: list[dict] = []
        for i, resolver in enumerate(resolvers):
            theta = self.theta_from_resolver(resolver)
            ideal = self.simulate_ideal(resolver)
            noisy = self.run_noisy(resolver, shots=shots, sampling_seed=seed_base + i)
            measured = noisy["primary"]
            for label in self.obs_labels:
                rows.append(
                    {
                        "theta": theta,
                        "pauli": label,
                        "o_noisy": float(measured[label]),
                        "o_ideal": float(ideal[label]),
                    }
                )
        return rows

    # -- energy assembly --------------------------------------------------
    def energy_from_values(self, value_by_pauli: dict) -> float:
        e = self.offset
        for label, w in zip(self.obs_labels, self.weights):
            e += float(w) * float(value_by_pauli[label])
        return float(e)

    def ideal_energy(self, theta: np.ndarray) -> float:
        return self.energy_from_values(
            self.simulate_ideal(self.resolver_from_theta(theta))
        )


def _snap_to_clifford(symbol, value: float, snap_step: float) -> float:
    """Snap ``value`` to the nearest Clifford-equivalent angle for ``symbol``.

    Uses the repo's th_*/ph_* aware rule when possible; otherwise falls back to
    the nearest multiple of ``snap_step``.
    """
    try:
        return float(clifford_snap_value_for_symbol(symbol, float(value)))
    except Exception:
        step = float(snap_step)
        return float(round(float(value) / step) * step)


# ---------------------------------------------------------------------------
# Section 5 - Baseline per-Pauli CDR backbone (reference only)
# ---------------------------------------------------------------------------


class CDRBackbone:
    """Per-observable affine CDR fit: ``O_ideal ~= a_i * O_noisy + b_i``.

    In this COMBINED design the CDR line is absorbed inside the single GP kernel,
    so this standalone backbone is NOT used to form the mitigated value. It is fit
    purely as a "plain CDR" baseline so the validation tables and the VQE plots can
    show classic CDR next to the combined GP. Same per-Pauli least-squares the repo
    performs in ``train_cf_models_per_pauli``.
    """

    def __init__(self, affine_regularization: float = 0.0) -> None:
        self.affine_regularization = float(affine_regularization)
        self.coeffs: dict[str, tuple[float, float]] = {}

    def fit(self, rows: list[dict]) -> None:
        by_pauli: dict[str, list[tuple[float, float]]] = {}
        for row in rows:
            by_pauli.setdefault(row["pauli"], []).append(
                (float(row["o_noisy"]), float(row["o_ideal"]))
            )
        self.coeffs = {}
        lam = self.affine_regularization
        for pauli, pairs in by_pauli.items():
            x = np.asarray([p[0] for p in pairs], dtype=float)
            y = np.asarray([p[1] for p in pairs], dtype=float)
            if len(x) >= 2 and float(np.std(x)) > 0.0:
                # Ridge on the slope only (penalize a, never the offset b).
                a_design = np.stack([x, np.ones_like(x)], axis=1)
                gram = a_design.T @ a_design
                gram[0, 0] += lam
                rhs = a_design.T @ y
                a, b = np.linalg.solve(gram, rhs)
            else:
                a, b = 1.0, float(np.mean(y - x))
            self.coeffs[pauli] = (float(a), float(b))

    def apply(self, pauli: str, o_noisy: float) -> float:
        a, b = self.coeffs.get(pauli, (1.0, 0.0))
        return float(a) * float(o_noisy) + float(b)

    def residual(self, row: dict) -> float:
        return float(row["o_ideal"]) - self.apply(row["pauli"], row["o_noisy"])


# ---------------------------------------------------------------------------
# Section 6 - Combined single GP (CDR absorbed in the kernel)
# ---------------------------------------------------------------------------


if _SKLEARN_AVAILABLE:

    class _SubsetKernel(Kernel):
        """Apply an inner kernel to a fixed subset of feature columns.

        Lets us form the product kernel ``k_angle(continuous) * k_obs(pauli)`` with
        scikit-learn, which otherwise applies every kernel to all columns.
        Hyperparameters are delegated to the inner kernel.
        """

        def __init__(self, kernel, indices):
            # Keep the exact objects passed in (no copying): sklearn.clone checks
            # parameter identity after re-instantiating from get_params.
            self.kernel = kernel
            self.indices = indices

        def get_params(self, deep=True):
            params = dict(kernel=self.kernel, indices=self.indices)
            if deep:
                for key, val in self.kernel.get_params().items():
                    params["kernel__" + key] = val
            return params

        @property
        def hyperparameters(self):
            out = []
            for hp in self.kernel.hyperparameters:
                out.append(
                    Hyperparameter(
                        "kernel__" + hp.name,
                        hp.value_type,
                        hp.bounds,
                        hp.n_elements,
                    )
                )
            return out

        @property
        def theta(self):
            return self.kernel.theta

        @theta.setter
        def theta(self, theta):
            self.kernel.theta = theta

        @property
        def bounds(self):
            return self.kernel.bounds

        def __call__(self, X, Y=None, eval_gradient=False):
            Xs = np.asarray(X)[:, self.indices]
            Ys = None if Y is None else np.asarray(Y)[:, self.indices]
            return self.kernel(Xs, Ys, eval_gradient=eval_gradient)

        def diag(self, X):
            return self.kernel.diag(np.asarray(X)[:, self.indices])

        def is_stationary(self):
            return self.kernel.is_stationary()


def _make_base_kernel(n_dims: int, config: MitigatorConfig):
    length_scale = np.ones(n_dims) if config.use_ard else 1.0
    if config.kernel_type.lower() == "rbf":
        return RBF(length_scale=length_scale, length_scale_bounds=(1e-3, 1e3))
    return Matern(
        length_scale=length_scale,
        length_scale_bounds=(1e-3, 1e3),
        nu=float(config.matern_nu),
    )


# ---------------------------------------------------------------------------
# Analytic input-gradient of the (fitted) kernel: d k(x*, x_n) / d x*.
#
# This is the purely CLASSICAL half of the analytic VQE gradient. The GP mean is
#     m(x*) = y_mean + y_std * sum_n alpha_n * k(x*, x_n),
# so  dm/dx* = y_std * sum_n alpha_n * dk(x*, x_n)/dx*.
# The quantum device is never involved here: kernel, training points x_n, dual
# weights alpha_n and hyperparameters all live on the classical machine after
# the fit. The only device-supplied quantity (d O_noisy / d theta, via a
# parameter-shift on the real measurement) is chained in later, in
# ``Mitigator.energy_gradient``.
# ---------------------------------------------------------------------------


def _leaf_kernel_value_and_input_grad(kernel, xs: np.ndarray, Xs: np.ndarray):
    """Value and d/dx* gradient of a leaf kernel restricted to its own columns.

    ``xs`` is the (sliced) query point, shape (k,); ``Xs`` the (sliced) training
    points, shape (m, k). Returns ``(v, g)`` with ``v`` shape (m,) the kernel
    values ``k(xs, Xs[n])`` and ``g`` shape (m, k) the derivative w.r.t. ``xs``.
    Supports the exact leaf kernels used in ``SingleGPModel.build_kernel``.
    """
    m, k = Xs.shape
    if isinstance(kernel, DotProduct):
        sigma0 = float(kernel.sigma_0)
        v = sigma0 * sigma0 + Xs @ xs
        # d/dx*_d ( sigma0^2 + sum_d x*_d * x_n,d ) = x_n,d
        return v, Xs.copy()
    if isinstance(kernel, (Matern, RBF)):
        ls = np.asarray(kernel.length_scale, dtype=float)
        if ls.ndim == 0:
            ls = np.full(k, float(ls))
        diff = xs[None, :] - Xs  # (m, k) = x* - x_n
        scaled = diff / ls[None, :]
        # NOTE: sklearn's ``Matern`` is a SUBCLASS of ``RBF``, so the Matern
        # branch must run first -- ``isinstance(matern, RBF)`` is True.
        if not isinstance(kernel, Matern):
            r2 = np.sum(scaled * scaled, axis=1)
            v = np.exp(-0.5 * r2)
            # d/dx*_d exp(-1/2 sum ((x*-x_n)/l)^2) = -k * (x*_d - x_n,d)/l_d^2
            g = -v[:, None] * (diff / (ls[None, :] ** 2))
            return v, g
        # Matern: k(r) with r = ||(x*-x_n)/l||_2.
        r = np.sqrt(np.sum(scaled * scaled, axis=1))
        nu = float(kernel.nu)
        if abs(nu - 0.5) < 1e-8:
            v = np.exp(-r)
            dvdr = -np.exp(-r)
        elif abs(nu - 1.5) < 1e-8:
            a = np.sqrt(3.0) * r
            v = (1.0 + a) * np.exp(-a)
            dvdr = -3.0 * r * np.exp(-a)
        elif abs(nu - 2.5) < 1e-8:
            a = np.sqrt(5.0) * r
            v = (1.0 + a + (5.0 / 3.0) * r * r) * np.exp(-a)
            dvdr = -(5.0 / 3.0) * r * (1.0 + a) * np.exp(-a)
        else:
            raise ValueError(
                f"Analytic Matern input-gradient supports nu in {{0.5,1.5,2.5}}, got nu={nu}."
            )
        # dr/dx*_d = (x*_d - x_n,d)/(l_d^2 * r); at r=0 the gradient is 0 for
        # these nu (dvdr carries a factor r), so guard the division.
        safe_r = np.where(r > 0.0, r, 1.0)
        g = (dvdr / safe_r)[:, None] * (diff / (ls[None, :] ** 2))
        g = np.where((r > 0.0)[:, None], g, 0.0)
        return v, g
    if isinstance(kernel, ConstantKernel):
        return np.full(m, float(kernel.constant_value)), np.zeros((m, k))
    if isinstance(kernel, WhiteKernel):
        # White contributes 0 to the cross-covariance k(x*, x_train) for x* not
        # exactly equal to a training point, hence 0 to the predictive mean.
        return np.zeros(m), np.zeros((m, k))
    raise ValueError(
        f"Unsupported leaf kernel for analytic input gradient: {type(kernel).__name__}."
    )


def _kernel_value_and_input_grad(kernel, x_star: np.ndarray, X_train: np.ndarray):
    """Recursively compute ``(k(x*, X_train), d k/d x*)`` for a composite kernel.

    Handles ``Sum``/``Product`` of the kernels assembled in ``build_kernel``
    (ConstantKernel, WhiteKernel, and ``_SubsetKernel``-wrapped DotProduct /
    Matern / RBF). ``x_star`` is shape (D,), ``X_train`` shape (m, D); returns
    ``v`` shape (m,) and ``g`` shape (m, D) (gradient w.r.t. ``x_star``).
    """
    from sklearn.gaussian_process.kernels import Product, Sum

    m, D = X_train.shape
    if isinstance(kernel, Sum):
        v1, g1 = _kernel_value_and_input_grad(kernel.k1, x_star, X_train)
        v2, g2 = _kernel_value_and_input_grad(kernel.k2, x_star, X_train)
        return v1 + v2, g1 + g2
    if isinstance(kernel, Product):
        v1, g1 = _kernel_value_and_input_grad(kernel.k1, x_star, X_train)
        v2, g2 = _kernel_value_and_input_grad(kernel.k2, x_star, X_train)
        return v1 * v2, v1[:, None] * g2 + v2[:, None] * g1
    if isinstance(kernel, ConstantKernel):
        return np.full(m, float(kernel.constant_value)), np.zeros((m, D))
    if isinstance(kernel, WhiteKernel):
        return np.zeros(m), np.zeros((m, D))
    if isinstance(kernel, _SubsetKernel):
        idx = list(kernel.indices)
        v, g_sub = _leaf_kernel_value_and_input_grad(
            kernel.kernel, np.asarray(x_star)[idx], np.asarray(X_train)[:, idx]
        )
        g = np.zeros((m, D))
        g[:, idx] = g_sub
        return v, g
    # Bare leaf kernel applied to all dimensions (not used by build_kernel, but
    # handled for completeness).
    return _leaf_kernel_value_and_input_grad(kernel, np.asarray(x_star), np.asarray(X_train))


class SingleGPModel:
    """One GP over all observables; target = raw ``O_ideal`` (no separate CDR).

    The CDR line is built INTO the kernel: a linear ``DotProduct`` term on the
    ``o_noisy`` feature column reproduces ``a * o_noisy + b`` (the slope/bias are
    learned as kernel hyperparameters), optionally multiplied by a per-observable
    kernel so the effective CDR slope can vary by Pauli. A Matern/RBF term on the
    angle+pauli block then captures the coherent, angle-dependent residual, and a
    WhiteKernel absorbs shot noise. ``predict`` returns ``(mean = O_mit, std)``.
    """

    def __init__(self, config: MitigatorConfig) -> None:
        if not _SKLEARN_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for SingleGPModel. Install it into the "
                f"active environment. Original import error: {_SKLEARN_IMPORT_ERROR!r}"
            )
        self.config = config
        self.gp: "GaussianProcessRegressor | None" = None
        self._index_map = feature_index_map(config)

    def build_kernel(self, feature_index_map_: dict, config: MitigatorConfig):
        noisy_idx = feature_index_map_["noisy"]
        angle_idx = feature_index_map_["angle"]
        obs_idx = feature_index_map_["pauli"]
        base_idx = angle_idx + obs_idx

        if not config.include_noisy_feature_in_gp or len(noisy_idx) == 0:
            raise ValueError(
                "The combined (Design 2) mitigator requires an o_noisy feature "
                "column for the in-kernel CDR term; set "
                "include_noisy_feature_in_gp=True."
            )

        white = WhiteKernel(
            noise_level=float(config.noise_variance_init),
            noise_level_bounds=(1e-8, 1e2),
        )

        # k_lin: DotProduct on o_noisy reproduces the CDR line a*o_noisy + b
        # (slope via the ConstantKernel amplitude, bias via DotProduct sigma_0).
        k_lin = ConstantKernel(1.0, (1e-3, 1e3)) * _SubsetKernel(
            DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e2)), noisy_idx
        )
        # Optionally make the effective CDR slope observable-dependent.
        if config.use_product_kernel and len(obs_idx) > 0:
            k_obs = _SubsetKernel(_make_base_kernel(len(obs_idx), config), obs_idx)
            k_lin = k_lin * k_obs

        kernel = k_lin

        # k_base: coherent / angle-dependent residual over (angles, pauli).
        if len(base_idx) > 0:
            k_base = ConstantKernel(1.0, (1e-3, 1e3)) * _SubsetKernel(
                _make_base_kernel(len(base_idx), config), base_idx
            )
            kernel = kernel + k_base

        return kernel + white

    def fit(self, X: np.ndarray, y_ideal: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y_ideal, dtype=float).ravel()
        cap = int(getattr(self.config, "max_gp_train_points", 0) or 0)
        if cap > 0 and X.shape[0] > cap:
            # Subsample rows so the exact-GP gradient tensor stays bounded.
            sub = np.random.default_rng(int(self.config.rng_seed)).choice(
                X.shape[0], size=cap, replace=False
            )
            X = X[sub]
            y = y[sub]
        kernel = self.build_kernel(self._index_map, self.config)
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-10,  # WhiteKernel carries the (learned) shot noise.
            normalize_y=bool(self.config.normalize_targets),
            n_restarts_optimizer=int(self.config.gp_n_restarts),
            random_state=int(self.config.rng_seed),
        )
        self.gp.fit(X, y)

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.gp is None:
            raise RuntimeError("SingleGPModel.fit must be called before predict.")
        X_star = np.asarray(X_star, dtype=float)
        mean, std = self.gp.predict(X_star, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)

    def predict_input_gradient(self, X_star: np.ndarray) -> np.ndarray:
        """Analytic ``d mean / d x*`` of the GP posterior mean (per query row).

        The mean is ``m(x*) = y_mean + y_std * sum_n alpha_n k(x*, x_n)``, so
        ``dm/dx* = y_std * sum_n alpha_n dk(x*, x_n)/dx*``. Returns shape
        ``(n_queries, n_features)``. Purely classical (no device access).
        """
        if self.gp is None:
            raise RuntimeError("SingleGPModel.fit must be called before predict_input_gradient.")
        X_star = np.asarray(X_star, dtype=float)
        gp = self.gp
        kernel = gp.kernel_
        X_train = np.asarray(gp.X_train_, dtype=float)
        alpha = np.asarray(gp.alpha_, dtype=float).ravel()
        y_std = getattr(gp, "_y_train_std", 1.0)
        y_std_arr = np.asarray(y_std, dtype=float).ravel()
        y_std_f = float(y_std_arr[0]) if y_std_arr.size else 1.0
        out = np.zeros((X_star.shape[0], X_star.shape[1]), dtype=float)
        for i, x in enumerate(X_star):
            _, kg = _kernel_value_and_input_grad(kernel, x, X_train)  # (m, D)
            out[i] = y_std_f * (kg.T @ alpha)
        return out


# ---------------------------------------------------------------------------
# Section 7 - Combined mitigator (public API)
# ---------------------------------------------------------------------------


class Mitigator:
    def __init__(
        self, backend: CirqBackendAdapter, hamiltonian, config: MitigatorConfig
    ) -> None:
        self.adapter = backend
        self.hamiltonian = hamiltonian  # kept for parity with the spec signature
        self.config = config
        # The single GP IS the mitigator; the backbone is a plain-CDR baseline only.
        self.gp = SingleGPModel(config)
        self.backbone = CDRBackbone(config.affine_regularization)
        self.rows: list[dict] = []
        self._current_theta: np.ndarray | None = None
        if config.theta_init is None:
            config.theta_init = np.zeros(int(config.n_params), dtype=float)

    # -- convenience ------------------------------------------------------
    @property
    def coeff_by_pauli(self) -> dict:
        return self.adapter.coeff_by_pauli

    def _feature_matrix(self, rows: list[dict]) -> np.ndarray:
        return np.stack([build_feature_row(r, self.config) for r in rows], axis=0)

    def _fit_gp_from_rows(self) -> None:
        # Combined design: the GP maps features -> O_ideal directly (CDR is in the
        # kernel). The baseline backbone is also (re)fit for reference only.
        X = self._feature_matrix(self.rows)
        y = np.asarray([float(r["o_ideal"]) for r in self.rows], dtype=float)
        self.gp.fit(X, y)
        self.backbone.fit(self.rows)

    # -- warm start -------------------------------------------------------
    def warmstart(self) -> None:
        theta0 = np.asarray(self.config.theta_init, dtype=float).ravel()
        self._current_theta = theta0
        resolvers = self.adapter.generate_near_clifford(
            theta0,
            n_circuits=int(self.config.n_warmstart_circuits),
            n_nonclifford=int(self.config.n_nonclifford_gates),
            snap_step=float(self.config.clifford_snap_step),
            spread=float(self.config.warmstart_spread),
            seed=int(self.config.rng_seed),
        )
        self.rows = self.adapter.collect_rows(
            resolvers, shots=int(self.config.shots), seed_base=int(self.config.rng_seed) + 1
        )
        self._fit_gp_from_rows()  # fits the combined GP (and the baseline backbone)

    # -- prediction -------------------------------------------------------
    def mitigate(self, theta, o_noisy_by_pauli: dict) -> dict:
        o_mit, _ = self.predict_with_uncertainty(theta, o_noisy_by_pauli)
        return o_mit

    def predict_with_uncertainty(
        self, theta, o_noisy_by_pauli: dict
    ) -> tuple[dict, dict]:
        theta = np.asarray(theta, dtype=float).ravel()
        labels = list(o_noisy_by_pauli.keys())
        rows = [
            {"theta": theta, "pauli": label, "o_noisy": float(o_noisy_by_pauli[label])}
            for label in labels
        ]
        X = self._feature_matrix(rows)
        mean, std = self.gp.predict(X)  # mean IS the mitigated O_ideal (CDR in-kernel)
        o_mit: dict = {}
        std_by: dict = {}
        for k, label in enumerate(labels):
            o_mit[label] = float(mean[k])
            std_by[label] = float(std[k])
        return o_mit, std_by

    # -- online updates ---------------------------------------------------
    def update_with_rows(self, new_rows: list[dict], current_theta=None) -> None:
        if current_theta is not None:
            self._current_theta = np.asarray(current_theta, dtype=float).ravel()
        self.rows.extend(new_rows)
        if (
            self.config.drop_faraway_points
            and len(self.rows) > int(self.config.max_gp_points)
        ):
            self._prune_rows()
        self._fit_gp_from_rows()  # refits the combined GP (and baseline backbone)

    def _prune_rows(self) -> None:
        cap = int(self.config.max_gp_points)
        if self._current_theta is not None:
            dist = np.asarray(
                [
                    float(np.linalg.norm(np.asarray(r["theta"]) - self._current_theta))
                    for r in self.rows
                ],
                dtype=float,
            )
            keep = np.argsort(dist)[:cap]
            self.rows = [self.rows[i] for i in sorted(keep.tolist())]
        else:
            self.rows = self.rows[-cap:]

    def mitigated_energy(self, theta, o_noisy_by_pauli: dict) -> float:
        o_mit = self.mitigate(theta, o_noisy_by_pauli)
        return self.adapter.energy_from_values(o_mit)

    # -- analytic energy gradient ----------------------------------------
    def energy_gradient(
        self, theta, o_noisy_by_pauli: dict, do_noisy_dtheta: dict
    ) -> np.ndarray:
        """Analytic ``dE_mit/dtheta`` at the CURRENT theta (in-distribution).

        Chains the classical GP-mean input gradient with the feature Jacobian:

            dE/dtheta_j = sum_P w_P * [ d mean_P/d(angle) . d(angle)/dtheta_j
                                        + d mean_P/d(o_noisy) * d o_noisy_P/dtheta_j ]

        - ``d(angle)/dtheta_j`` (Fourier features) is closed form (classical).
        - ``d o_noisy_P/dtheta_j`` is the only device-supplied term and must be
          passed in (parameter-shift on the real noisy measurement). It maps
          ``pauli label -> np.ndarray`` of length ``n_params``.

        Pauli features are constant in theta, so they contribute nothing.
        """
        theta = np.asarray(theta, dtype=float).ravel()
        labels = list(self.adapter.obs_labels)
        rows = [
            {"theta": theta, "pauli": label, "o_noisy": float(o_noisy_by_pauli[label])}
            for label in labels
        ]
        X = self._feature_matrix(rows)
        dmean = self.gp.predict_input_gradient(X)  # (P, D), classical

        idx = feature_index_map(self.config)
        angle_idx = idx["angle"]
        noisy_idx = idx["noisy"]
        n_params = int(self.config.n_params)
        n_harm = int(self.config.max_harmonic)

        # d(angle features)/d theta_j: nonzero only for the angle dims of param j.
        n_angle = len(angle_idx)
        dangle = np.zeros((n_angle, n_params), dtype=float)
        for i in range(n_params):
            ti = float(theta[i])
            base = i * 2 * n_harm
            for kk in range(1, n_harm + 1):
                c = base + (kk - 1) * 2
                dangle[c, i] = -kk * np.sin(kk * ti)      # d cos(k t_i)/dt_i
                dangle[c + 1, i] = kk * np.cos(kk * ti)   # d sin(k t_i)/dt_i

        # angle contribution: (P, n_angle) @ (n_angle, n_params) -> (P, n_params)
        d_omit_dtheta = dmean[:, angle_idx] @ dangle

        # device-supplied noisy-feature contribution
        if len(noisy_idx) > 0:
            do = np.stack(
                [np.asarray(do_noisy_dtheta[label], dtype=float).ravel() for label in labels],
                axis=0,
            )  # (P, n_params)
            d_omit_dtheta = d_omit_dtheta + dmean[:, noisy_idx[0]][:, None] * do

        weights = np.asarray(self.adapter.weights, dtype=float)  # aligned with obs_labels
        grad = weights @ d_omit_dtheta  # (n_params,)
        return np.asarray(grad, dtype=float).ravel()


# ---------------------------------------------------------------------------
# Section 8 - Top-up controller
# ---------------------------------------------------------------------------


def needs_topup(std_by_pauli: dict, coeff_by_pauli: dict, config: MitigatorConfig) -> bool:
    return weighted_uncertainty(std_by_pauli, coeff_by_pauli) > float(
        config.uncertainty_threshold
    )


def weighted_uncertainty(std_by_pauli: dict, coeff_by_pauli: dict) -> float:
    """|c_i|-weighted aggregate of predicted std (an energy-scale uncertainty)."""
    total = 0.0
    for label, std in std_by_pauli.items():
        total += abs(float(coeff_by_pauli.get(label, 0.0))) * float(std)
    return float(total)


def sample_local_rows(
    adapter: CirqBackendAdapter, theta, config: MitigatorConfig, seed: int
) -> list[dict]:
    theta = np.asarray(theta, dtype=float).ravel()
    resolvers = adapter.generate_near_clifford(
        theta,
        n_circuits=int(config.topup_batch_size),
        n_nonclifford=int(config.n_nonclifford_gates),
        snap_step=float(config.clifford_snap_step),
        spread=float(config.effective_topup_radius()),
        seed=int(seed),
    )
    return adapter.collect_rows(resolvers, shots=int(config.shots), seed_base=int(seed) + 1)


# ---------------------------------------------------------------------------
# Section 9 - VQE loop integration (bounded-step / trust region)
# ---------------------------------------------------------------------------


def run_vqe_with_mitigator(
    mitigator: Mitigator,
    theta_init: np.ndarray,
    *,
    max_iters: int | None = None,
    step_max: float | None = None,
    learning_rate: float = 0.5,
    fd_eps: float = 0.05,
    gradient_mode: str = "parameter_shift",
    ideal_energy_fn: Callable[[np.ndarray], float] | None = None,
    verbose: bool = True,
) -> dict:
    """Fixed-learning-rate gradient descent on the mitigated energy.

    Matches the optimizer in ``June_main/main_HF.ipynb``: a **parameter-shift**
    gradient (exact +/- pi/2 shifts, prefactor 1/2 -- each ansatz angle drives a
    single RX/RZX with eigenvalues +/-1) and a **full fixed-LR update**
    ``theta <- theta - lr * grad`` on every coordinate, for a fixed number of
    iterations. Set ``gradient_mode='finite_difference'`` to use central
    differences with ``fd_eps`` instead.

    Per iteration:
      1. measure the REAL circuit's noisy expectation values for all Paulis,
      2. predict mitigated values + uncertainty,
      3. top up locally if uncertain, then re-predict,
      4. energy = sum_i c_i * O_mit[P_i]; gradient; fixed-LR step.

    ``step_max`` is OFF by default (None) to match main_HF's un-clipped fixed-LR
    update; pass a float to re-enable trust-region step clipping. Top-up frequency
    should taper toward zero as the optimizer settles.
    """
    cfg = mitigator.config
    adapter = mitigator.adapter
    max_iters = int(cfg.max_vqe_iterations if max_iters is None else max_iters)
    # step_max stays None unless explicitly provided -> no clipping (main_HF style).

    mode = str(gradient_mode).strip().lower()
    if mode == "gp_analytic":
        mode = "analytic"
    if mode not in {"parameter_shift", "finite_difference", "analytic"}:
        raise ValueError(
            "gradient_mode must be 'parameter_shift', 'finite_difference' or "
            f"'analytic', got {gradient_mode!r}."
        )
    shift = float(np.pi) / 2.0 if mode == "parameter_shift" else float(fd_eps)
    grad_scale = 0.5 if mode == "parameter_shift" else 1.0 / (2.0 * float(fd_eps))

    theta = np.asarray(theta_init, dtype=float).ravel()
    rng = np.random.default_rng(int(cfg.rng_seed) + 777)

    def measure(th: np.ndarray) -> dict:
        return adapter.run_noisy(
            adapter.resolver_from_theta(th),
            shots=int(cfg.shots),
            sampling_seed=int(rng.integers(1, 2**31 - 1)),
        )

    def mitigated_energy(th: np.ndarray) -> float:
        bundle = measure(th)
        return adapter.energy_from_values(mitigator.mitigate(th, bundle["primary"]))

    def compute_grad(th: np.ndarray, measured_th: dict) -> np.ndarray:
        if mode == "analytic":
            # Quantum part: parameter-shift on the REAL noisy measurement gives
            # d O_noisy_P/d theta_j (exact for single-generator gates, up to shot
            # noise). Same device cost as parameter_shift -- the difference is the
            # GP gradient is evaluated analytically at the current (in-distribution)
            # theta instead of querying the surrogate at the far-away +/- pi/2 points.
            half = float(np.pi) / 2.0
            do_noisy = {label: np.zeros(len(th)) for label in measured_th}
            for j in range(len(th)):
                tp = th.copy()
                tm = th.copy()
                tp[j] += half
                tm[j] -= half
                bp = measure(tp)["primary"]
                bm = measure(tm)["primary"]
                for label in measured_th:
                    do_noisy[label][j] = 0.5 * (bp[label] - bm[label])
            return mitigator.energy_gradient(th, measured_th, do_noisy)
        g = np.zeros_like(th)
        for j in range(len(th)):
            tp = th.copy()
            tm = th.copy()
            tp[j] += shift
            tm[j] -= shift
            g[j] = (mitigated_energy(tp) - mitigated_energy(tm)) * grad_scale
        return g

    def do_topup(th: np.ndarray, it_seed: int) -> tuple[dict, dict]:
        new_rows = sample_local_rows(adapter, th, cfg, seed=int(it_seed))
        mitigator.update_with_rows(new_rows, current_theta=th)
        return mitigator.predict_with_uncertainty(th, measured)

    history: list[dict] = []
    topup_total = 0
    # Shot accounting: snapshot the backend counters so the caller gets the exact
    # number of circuit executions (and shots) this optimization consumed.
    evals_at_start = int(adapter.n_circuit_evals)
    shots_at_start = int(adapter.total_shots)
    # Center of the most recent top-up (trust region). Seeded at theta_init since
    # the warm-start cloud is built around it.
    last_topup_theta = theta.copy()
    converged = False
    for it in range(max_iters):
        bundle = measure(theta)
        measured = bundle["primary"]
        o_mit, std = mitigator.predict_with_uncertainty(theta, measured)

        # --- Decide whether to top up (uncertainty OR trust-region OR periodic) ---
        unc_trigger = needs_topup(std, mitigator.coeff_by_pauli, cfg)
        moved = float(np.linalg.norm(theta - last_topup_theta))
        move_trigger = bool(cfg.topup_on_move) and moved > (
            float(cfg.topup_move_fraction) * cfg.effective_topup_radius()
        )
        periodic_trigger = (
            int(cfg.topup_every) > 0 and it > 0 and (it % int(cfg.topup_every) == 0)
        )
        topped = 0
        iter_topups = 0  # number of top-up batches this iter (for shot accounting)
        if unc_trigger or move_trigger or periodic_trigger:
            o_mit, std = do_topup(theta, int(cfg.rng_seed) + 10_000 + it)
            last_topup_theta = theta.copy()
            topped = 1
            iter_topups += 1
            topup_total += 1
            # (3) Top-up-until-satisfied: keep adding local batches while the GP is
            #     still too uncertain at theta, bounded by ``max_topup_retries`` so
            #     the shot cost stays capped. Forces a trustworthy surrogate before
            #     the gradient/step are taken (the fix for runaway extrapolation).
            retries = 0
            while (
                int(cfg.max_topup_retries) > 0
                and weighted_uncertainty(std, mitigator.coeff_by_pauli)
                > float(cfg.uncertainty_threshold)
                and retries < int(cfg.max_topup_retries)
            ):
                o_mit, std = do_topup(
                    theta, int(cfg.rng_seed) + 30_000 + it * 1000 + retries
                )
                iter_topups += 1
                topup_total += 1
                retries += 1

        energy = adapter.energy_from_values(o_mit)
        unc = weighted_uncertainty(std, mitigator.coeff_by_pauli)
        backbone_by_pauli = {
            label: mitigator.backbone.apply(label, measured[label])
            for label in measured
        }

        grad = compute_grad(theta, measured)
        grad_evals = 1  # gradient evaluations this iter (each = 2*n_params circuits)

        # --- (4) Convergence validation: a small surrogate gradient is only
        #     trustworthy if a REAL near-Clifford batch at theta doesn't move the
        #     answer. If we haven't already refit here, top up once and re-check. ---
        if float(cfg.convergence_grad_tol) > 0 and float(np.linalg.norm(grad)) < float(
            cfg.convergence_grad_tol
        ):
            if topped == 0:
                o_mit, std = do_topup(theta, int(cfg.rng_seed) + 20_000 + it)
                last_topup_theta = theta.copy()
                topped = 1
                iter_topups += 1
                topup_total += 1
                energy_new = adapter.energy_from_values(o_mit)
                unc = weighted_uncertainty(std, mitigator.coeff_by_pauli)
                grad = compute_grad(theta, measured)
                grad_evals = 2  # re-evaluated the gradient after the validation top-up
                energy_shift = abs(energy_new - energy)
                energy = energy_new
                if float(np.linalg.norm(grad)) < float(
                    cfg.convergence_grad_tol
                ) and energy_shift < float(cfg.convergence_energy_tol):
                    converged = True
            else:
                # Surrogate was just refit with real data this iter -> trust it.
                converged = True

        step = -learning_rate * grad
        if step_max is not None:
            max_comp = float(np.max(np.abs(step))) if step.size else 0.0
            if max_comp > step_max and max_comp > 0:
                step *= step_max / max_comp

        rec = {
            "iter": it,
            "theta": theta.copy(),
            "energy_mitigated": float(energy),
            "energy_unmitigated": float(bundle["energy_unmitigated"]),
            "energy_rem": float(bundle["energy_rem"]),
            "energy_backbone": float(adapter.energy_from_values(backbone_by_pauli)),
            "weighted_uncertainty": float(unc),
            "topped_up": int(topped),
            "n_topups": int(iter_topups),
            "topup_cum": int(topup_total),
            "grad_evals": int(grad_evals),
            "grad_norm": float(np.linalg.norm(grad)),
            "n_rows": int(len(mitigator.rows)),
        }
        if ideal_energy_fn is not None:
            rec["energy_ideal"] = float(ideal_energy_fn(theta))
        history.append(rec)
        if verbose:
            extra = (
                f"  E_ideal={rec['energy_ideal']: .6f}"
                if "energy_ideal" in rec
                else ""
            )
            print(
                f"[GP-VQE] iter={it:03d}  E_mit={energy: .6f}{extra}  "
                f"unc={unc: .4e}  topup={iter_topups} (cum={topup_total})  "
                f"|g|={rec['grad_norm']: .3e}  rows={rec['n_rows']}"
            )

        if converged:
            if verbose:
                print(
                    f"[GP-VQE] converged at iter={it:03d}: |g|={rec['grad_norm']:.3e} "
                    f"< {float(cfg.convergence_grad_tol):.3e} and validated with a "
                    f"real near-Clifford top-up."
                )
            break

        theta = theta + step

    return {
        "theta_final": theta,
        "history": history,
        "topup_total": int(topup_total),
        "converged": bool(converged),
        # Empirical (numerical) shot cost of this optimization, measured straight
        # from the backend counters (excludes warm start, which happens before).
        "circuit_evals": int(adapter.n_circuit_evals - evals_at_start),
        "total_shots": int(adapter.total_shots - shots_at_start),
    }


def gradient_circuit_evals(config: MitigatorConfig) -> int:
    """Circuit executions per gradient evaluation.

    All gradient modes cost the same: each of the ``n_params`` angles needs a
    ``+`` and a ``-`` shift (parameter-shift +/- pi/2, finite-difference +/- eps,
    or the analytic mode's parameter-shift on the real device for
    ``dO_noisy/dtheta``) -> ``2 * n_params`` physical circuit executions.
    """
    return 2 * int(config.n_params)


def analytic_shot_count(
    config: MitigatorConfig,
    history: list[dict],
    *,
    include_warmstart: bool = True,
) -> dict:
    """Closed-form (analytic) reconstruction of the hardware shot cost.

    Counts EVERYTHING that touches the device, exactly as the real run does, so
    it can be cross-checked against the empirical backend counter
    (``adapter.shot_report()`` / ``vqe_out['total_shots']``):

      * warm start          : ``n_warmstart_circuits`` executions (once),
      * per VQE iteration    : 1 energy measurement
                               + ``grad_evals * 2 * n_params`` gradient circuits
                               + ``n_topups * topup_batch_size`` top-up circuits.

    ``grad_evals`` (1, or 2 on a convergence-validation iteration) and
    ``topped_up`` (0/1) are read back from ``history`` so the data-dependent
    top-up / early-stop behavior is reflected exactly. Every execution consumes
    ``config.shots`` shots, so ``total_shots = circuit_evals * shots``.
    """
    grad_evals_cost = gradient_circuit_evals(config)
    shots = int(config.shots)

    warmstart_evals = int(config.n_warmstart_circuits) if include_warmstart else 0
    energy_evals = 0
    gradient_evals = 0
    topup_evals = 0
    for rec in history:
        energy_evals += 1
        gradient_evals += int(rec.get("grad_evals", 1)) * grad_evals_cost
        # ``n_topups`` counts every batch this iter (top-up-until-satisfied can fire
        # several); fall back to the 0/1 ``topped_up`` flag for legacy histories.
        n_topups = int(rec.get("n_topups", rec.get("topped_up", 0)))
        topup_evals += n_topups * int(config.topup_batch_size)

    breakdown = {
        "warm_start": warmstart_evals,
        "energy": energy_evals,
        "gradient": gradient_evals,
        "topup": topup_evals,
    }
    circuit_evals = warmstart_evals + energy_evals + gradient_evals + topup_evals
    return {
        "shots_per_circuit": shots,
        "circuit_evals": int(circuit_evals),
        "circuit_evals_breakdown": breakdown,
        "total_shots": int(circuit_evals * shots),
        "total_shots_breakdown": {k: int(v * shots) for k, v in breakdown.items()},
        "n_iterations": len(history),
        "n_params": int(config.n_params),
    }


def run_noiseless_reference_vqe(
    adapter: CirqBackendAdapter,
    theta_init: np.ndarray,
    *,
    max_iters: int,
    learning_rate: float = 0.5,
    gradient_mode: str = "parameter_shift",
    fd_eps: float = 0.05,
) -> dict:
    """Decoupled PURE-noiseless optimizer reference.

    Mirrors ``_build_noiseless_reference_curve`` in ``June_main/main_HF.ipynb``:
    a fixed-learning-rate gradient descent on the EXACT statevector energy
    (``adapter.ideal_energy``) via the parameter-shift rule, with NO noise, NO
    shots, and NO mitigation. This is the curve that should reach chemical
    accuracy; it is independent of the noisy/mitigated optimization so it is an
    honest "what the ansatz+optimizer can do" baseline.

    Returns per-iteration ideal energies and thetas, indexed ``0..max_iters``
    (entry 0 is ``theta_init`` before any update).
    """
    mode = str(gradient_mode).strip().lower()
    if mode not in {"parameter_shift", "finite_difference"}:
        raise ValueError(
            f"gradient_mode must be 'parameter_shift' or 'finite_difference', got {gradient_mode!r}."
        )
    shift = float(np.pi) / 2.0 if mode == "parameter_shift" else float(fd_eps)
    grad_scale = 0.5 if mode == "parameter_shift" else 1.0 / (2.0 * float(fd_eps))

    theta = np.asarray(theta_init, dtype=float).ravel()
    energies = [float(adapter.ideal_energy(theta))]
    thetas = [theta.copy()]
    for _ in range(int(max_iters)):
        grad = np.zeros_like(theta)
        for j in range(len(theta)):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] += shift
            tm[j] -= shift
            grad[j] = (adapter.ideal_energy(tp) - adapter.ideal_energy(tm)) * grad_scale
        theta = theta - float(learning_rate) * grad
        energies.append(float(adapter.ideal_energy(theta)))
        thetas.append(theta.copy())
    return {
        "energies": energies,
        "thetas": thetas,
        "theta_final": theta,
        "energy_final": float(energies[-1]),
    }


# ---------------------------------------------------------------------------
# Section 10 - Validation (run BEFORE the live loop)
# ---------------------------------------------------------------------------


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def holdout_validation(
    adapter: CirqBackendAdapter, config: MitigatorConfig, *, seed: int | None = None
) -> dict:
    """Hold out a fraction of warm-start circuits; compare per-row prediction of
    O_ideal by: unmitigated, plain CDR (baseline), and the combined single GP.
    """
    seed = int(config.rng_seed if seed is None else seed)
    theta0 = (
        np.zeros(int(config.n_params))
        if config.theta_init is None
        else np.asarray(config.theta_init, dtype=float)
    )
    resolvers = adapter.generate_near_clifford(
        theta0,
        n_circuits=int(config.n_warmstart_circuits),
        n_nonclifford=int(config.n_nonclifford_gates),
        snap_step=float(config.clifford_snap_step),
        spread=float(config.warmstart_spread),
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(resolvers))
    n_hold = max(1, int(round(config.holdout_fraction * len(resolvers))))
    hold_set = set(order[:n_hold].tolist())
    train_res = [resolvers[i] for i in range(len(resolvers)) if i not in hold_set]
    hold_res = [resolvers[i] for i in range(len(resolvers)) if i in hold_set]

    train_rows = adapter.collect_rows(train_res, shots=int(config.shots), seed_base=seed + 1)
    hold_rows = adapter.collect_rows(hold_res, shots=int(config.shots), seed_base=seed + 5000)

    backbone = CDRBackbone(config.affine_regularization)  # plain-CDR baseline
    backbone.fit(train_rows)
    gp = SingleGPModel(config)
    Xtr = np.stack([build_feature_row(r, config) for r in train_rows], axis=0)
    ytr = np.asarray([r["o_ideal"] for r in train_rows], dtype=float)  # target = O_ideal
    gp.fit(Xtr, ytr)

    Xho = np.stack([build_feature_row(r, config) for r in hold_rows], axis=0)
    gp_pred, _ = gp.predict(Xho)  # combined GP mean IS the mitigated value
    o_ideal = np.asarray([r["o_ideal"] for r in hold_rows], dtype=float)
    o_noisy = np.asarray([r["o_noisy"] for r in hold_rows], dtype=float)
    backbone_pred = np.asarray(
        [backbone.apply(r["pauli"], r["o_noisy"]) for r in hold_rows], dtype=float
    )

    return {
        "n_train_circuits": len(train_res),
        "n_holdout_circuits": len(hold_res),
        "n_holdout_rows": len(hold_rows),
        "rmse_unmitigated": _rmse(o_noisy, o_ideal),
        "rmse_cdr_only": _rmse(backbone_pred, o_ideal),
        "rmse_single_gp": _rmse(gp_pred, o_ideal),
    }


def extrapolation_validation(
    adapter: CirqBackendAdapter,
    config: MitigatorConfig,
    *,
    train_spread: float = 0.3,
    test_spread: float = 0.8,
    seed: int | None = None,
) -> dict:
    """Train on a tight angle sub-range, test on a wider one, to measure the
    near-Clifford -> arbitrary-angle generalization gap (the real risk).
    """
    seed = int(config.rng_seed if seed is None else seed)
    theta0 = (
        np.zeros(int(config.n_params))
        if config.theta_init is None
        else np.asarray(config.theta_init, dtype=float)
    )

    train_res = adapter.generate_near_clifford(
        theta0,
        n_circuits=int(config.n_warmstart_circuits),
        n_nonclifford=int(config.n_nonclifford_gates),
        snap_step=float(config.clifford_snap_step),
        spread=float(train_spread),
        seed=seed,
    )
    test_res = adapter.generate_near_clifford(
        theta0,
        n_circuits=max(8, int(config.topup_batch_size)),
        n_nonclifford=int(config.n_nonclifford_gates),
        snap_step=float(config.clifford_snap_step),
        spread=float(test_spread),
        seed=seed + 999,
    )
    train_rows = adapter.collect_rows(train_res, shots=int(config.shots), seed_base=seed + 1)
    test_rows = adapter.collect_rows(test_res, shots=int(config.shots), seed_base=seed + 7000)

    backbone = CDRBackbone(config.affine_regularization)  # plain-CDR baseline
    backbone.fit(train_rows)
    gp = SingleGPModel(config)
    Xtr = np.stack([build_feature_row(r, config) for r in train_rows], axis=0)
    ytr = np.asarray([r["o_ideal"] for r in train_rows], dtype=float)  # target = O_ideal
    gp.fit(Xtr, ytr)

    Xte = np.stack([build_feature_row(r, config) for r in test_rows], axis=0)
    gp_pred, gp_std = gp.predict(Xte)  # combined GP mean IS the mitigated value
    o_ideal = np.asarray([r["o_ideal"] for r in test_rows], dtype=float)
    o_noisy = np.asarray([r["o_noisy"] for r in test_rows], dtype=float)
    backbone_pred = np.asarray(
        [backbone.apply(r["pauli"], r["o_noisy"]) for r in test_rows], dtype=float
    )
    return {
        "train_spread": float(train_spread),
        "test_spread": float(test_spread),
        "rmse_unmitigated": _rmse(o_noisy, o_ideal),
        "rmse_cdr_only": _rmse(backbone_pred, o_ideal),
        "rmse_single_gp": _rmse(gp_pred, o_ideal),
        "mean_gp_std": float(np.mean(gp_std)),
    }


def make_eval_set(
    adapter: CirqBackendAdapter,
    center: np.ndarray,
    *,
    radius: float,
    n: int,
    seed: int,
    shots: int,
) -> dict:
    """Build a fixed evaluation set of thetas around ``center`` (the region a VQE
    would explore), with their exact ideal energies and a single noisy measurement
    each. Reused across configs so model comparisons are apples-to-apples and no
    extra circuits are spent per config at eval time.
    """
    center = np.asarray(center, dtype=float).ravel()
    rng = np.random.default_rng(int(seed))
    thetas: list[np.ndarray] = []
    ideal_energies: list[float] = []
    measured_list: list[dict] = []
    for i in range(int(n)):
        theta = center + rng.uniform(-radius, radius, size=center.shape[0])
        thetas.append(theta)
        ideal_energies.append(float(adapter.ideal_energy(theta)))
        measured_list.append(
            adapter.run_noisy(
                adapter.resolver_from_theta(theta),
                shots=int(shots),
                sampling_seed=int(seed) + 100_000 + i,
            )["primary"]
        )
    return {
        "thetas": thetas,
        "ideal_energies": ideal_energies,
        "measured_list": measured_list,
    }


def evaluate_mitigator_on_thetas(mitigator: Mitigator, eval_set: dict) -> dict:
    """Energy-level MAE over the eval set for: unmitigated, backbone-only, full."""
    adapter = mitigator.adapter
    full_err: list[float] = []
    bb_err: list[float] = []
    noisy_err: list[float] = []
    unc_list: list[float] = []
    for theta, e_ideal, measured in zip(
        eval_set["thetas"], eval_set["ideal_energies"], eval_set["measured_list"]
    ):
        o_mit, std = mitigator.predict_with_uncertainty(theta, measured)
        e_full = adapter.energy_from_values(o_mit)
        bb = {lab: mitigator.backbone.apply(lab, measured[lab]) for lab in measured}
        e_bb = adapter.energy_from_values(bb)
        e_noisy = adapter.energy_from_values(measured)
        full_err.append(abs(e_full - e_ideal))
        bb_err.append(abs(e_bb - e_ideal))
        noisy_err.append(abs(e_noisy - e_ideal))
        unc_list.append(weighted_uncertainty(std, mitigator.coeff_by_pauli))
    return {
        "mae_full": float(np.mean(full_err)),
        "mae_backbone": float(np.mean(bb_err)),
        "mae_noisy": float(np.mean(noisy_err)),
        "median_weighted_uncertainty": float(np.median(unc_list)),
        "warmstart_circuits": int(mitigator.config.n_warmstart_circuits),
    }


def clifford_anchor_check(
    mitigator: Mitigator, theta_clifford: np.ndarray | None = None
) -> dict:
    """A fully-Clifford ``theta`` has a classically-known exact energy; confirm the
    mitigated pipeline lands on it.
    """
    cfg = mitigator.config
    adapter = mitigator.adapter
    if theta_clifford is None:
        base = (
            np.zeros(int(cfg.n_params))
            if cfg.theta_init is None
            else np.asarray(cfg.theta_init, dtype=float)
        )
        theta_clifford = np.asarray(
            [_snap_to_clifford(s, float(base[j]), cfg.clifford_snap_step)
             for j, s in enumerate(adapter.symbols)],
            dtype=float,
        )
    bundle = adapter.run_noisy(
        adapter.resolver_from_theta(theta_clifford),
        shots=int(cfg.shots),
        sampling_seed=int(cfg.rng_seed) + 31,
    )
    measured = bundle["primary"]
    e_mit = mitigator.mitigated_energy(theta_clifford, measured)
    e_ideal = adapter.ideal_energy(theta_clifford)
    e_noisy = adapter.energy_from_values(measured)
    return {
        "theta_clifford": np.asarray(theta_clifford, dtype=float),
        "energy_ideal": float(e_ideal),
        "energy_noisy": float(e_noisy),
        "energy_mitigated": float(e_mit),
        "abs_error_mitigated": float(abs(e_mit - e_ideal)),
        "abs_error_noisy": float(abs(e_noisy - e_ideal)),
    }
