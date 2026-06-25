"""Two-Stage CDR + Gaussian-Process Error Mitigator for VQE (Design 1).

Stage 1 (CDR backbone): per-observable linear Clifford-Data-Regression,
    O_ideal ~= a_i * O_noisy + b_i   (classic CDR, one (a_i, b_i) per Pauli).
Stage 2 (GP residual): ONE shared Gaussian Process that learns the leftover
    y = O_ideal - (a_i * O_noisy + b_i) the linear fit cannot capture.

Final mitigated value:
    O_mit(theta, P_i) = a_i * O_noisy + b_i  +  GP_residual(features(theta, P_i, O_noisy))

The GP also returns a variance at every prediction, which drives an
uncertainty-gated top-up loop during VQE.

This module is a thin layer on top of the user's EXISTING cirq codebase
(``main_cursor_lib`` and ``shot_measurement`` under ``June_main``). It does NOT
re-implement circuit construction, simulation, or shot estimation; it binds the
spec's adapter methods to those existing functions. Stage 1 reuses the same
per-Pauli least-squares the repo already performs in
``train_cf_models_per_pauli``.

Build/verify order mirrors the design spec. See the accompanying notebook cells.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

# These resolve to June_main/main_cursor_lib.py and June_main/shot_measurement.py
# (the same modules the notebook imports), so cirq class identities -- e.g. the
# RZXGate used inside count_non_clifford_ops -- match the loaded ansatz circuit.
from main_cursor_lib import (
    clifford_snap_value_for_symbol,
)
from shot_measurement import (
    _simulate_noiseless_state_for_resolver,
    _simulate_noisy_rho_for_resolver,
    estimate_energy_from_noisy_rho_shots,
    exact_pauli_expectation_from_int_row,
    int_observable_to_pauli_string,
    pauli_sum_to_int_observables,
)

try:  # scikit-learn is the GP backend (see plan: Stage 2 == sklearn GPR).
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ConstantKernel,
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

    # --- Stage 1: CDR linear backbone ---
    affine_regularization: float = 0.0
    refit_affine_on_topup: bool = True

    # --- Stage 2: GP residual kernel ---
    kernel_type: str = "matern"  # "matern" or "rbf"
    matern_nu: float = 2.5
    use_ard: bool = True
    use_product_kernel: bool = True  # k_angle * k_obs (True) vs k_angle + k_obs (False)
    noise_variance_init: float = 1e-3
    normalize_targets: bool = True
    gp_n_restarts: int = 2  # marginal-likelihood restarts (not in spec; small default)

    # --- Top-up controller ---
    uncertainty_threshold: float = 0.05
    topup_batch_size: int = 12
    topup_radius: "float | None" = None  # None -> tie to optimizer_step_max
    drop_faraway_points: bool = True
    max_gp_points: int = 5000

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
        key = "per_term_rem" if self.use_rem_branch else "per_term_unmitigated"
        per_term = np.asarray(est[key], dtype=float).ravel()
        return {label: float(per_term[k]) for k, label in enumerate(self.obs_labels)}

    # -- row collection (one row per (circuit, observable)) ---------------
    def collect_rows(
        self, resolvers: list[dict], *, shots: int, seed_base: int
    ) -> list[dict]:
        rows: list[dict] = []
        for i, resolver in enumerate(resolvers):
            theta = self.theta_from_resolver(resolver)
            ideal = self.simulate_ideal(resolver)
            noisy = self.run_noisy(resolver, shots=shots, sampling_seed=seed_base + i)
            for label in self.obs_labels:
                rows.append(
                    {
                        "theta": theta,
                        "pauli": label,
                        "o_noisy": float(noisy[label]),
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
# Section 5 - Stage 1: CDR linear backbone
# ---------------------------------------------------------------------------


class CDRBackbone:
    """Per-observable affine CDR fit: ``O_ideal ~= a_i * O_noisy + b_i``.

    Same per-Pauli least-squares the repo performs in ``train_cf_models_per_pauli``,
    but operating on the row schema so Stage 2 can read residuals and we can refit
    cheaply on top-up.
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
# Section 6 - Stage 2: shared GP on the residual
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


class GPResidualModel:
    """One shared GP over all observables; target = Stage-1 residual."""

    def __init__(self, config: MitigatorConfig) -> None:
        if not _SKLEARN_AVAILABLE:  # pragma: no cover
            raise ImportError(
                "scikit-learn is required for GPResidualModel. Install it into the "
                f"active environment. Original import error: {_SKLEARN_IMPORT_ERROR!r}"
            )
        self.config = config
        self.gp: "GaussianProcessRegressor | None" = None
        self._index_map = feature_index_map(config)

    def build_kernel(self, feature_index_map_: dict, config: MitigatorConfig):
        cont_idx = feature_index_map_["continuous"]
        obs_idx = feature_index_map_["pauli"]
        white = WhiteKernel(
            noise_level=float(config.noise_variance_init),
            noise_level_bounds=(1e-8, 1e2),
        )
        if config.use_product_kernel and len(obs_idx) > 0 and len(cont_idx) > 0:
            k_cont = ConstantKernel(1.0, (1e-3, 1e3)) * _SubsetKernel(
                _make_base_kernel(len(cont_idx), config), cont_idx
            )
            k_obs = _SubsetKernel(_make_base_kernel(len(obs_idx), config), obs_idx)
            return k_cont * k_obs + white
        if len(obs_idx) > 0 and len(cont_idx) > 0:
            # Additive structure: k_angle + k_obs.
            k_cont = ConstantKernel(1.0, (1e-3, 1e3)) * _SubsetKernel(
                _make_base_kernel(len(cont_idx), config), cont_idx
            )
            k_obs = ConstantKernel(1.0, (1e-3, 1e3)) * _SubsetKernel(
                _make_base_kernel(len(obs_idx), config), obs_idx
            )
            return k_cont + k_obs + white
        # Single block present: plain ARD kernel on all dims.
        all_idx = cont_idx + obs_idx
        return (
            ConstantKernel(1.0, (1e-3, 1e3)) * _make_base_kernel(len(all_idx), config)
            + white
        )

    def fit(self, X: np.ndarray, y_residual: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y_residual, dtype=float).ravel()
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
            raise RuntimeError("GPResidualModel.fit must be called before predict.")
        X_star = np.asarray(X_star, dtype=float)
        mean, std = self.gp.predict(X_star, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float)


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
        self.backbone = CDRBackbone(config.affine_regularization)
        self.gp = GPResidualModel(config)
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
        X = self._feature_matrix(self.rows)
        y = np.asarray([self.backbone.residual(r) for r in self.rows], dtype=float)
        self.gp.fit(X, y)

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
        self.backbone.fit(self.rows)
        self._fit_gp_from_rows()

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
        mean, std = self.gp.predict(X)
        o_mit: dict = {}
        std_by: dict = {}
        for k, label in enumerate(labels):
            base = self.backbone.apply(label, float(o_noisy_by_pauli[label]))
            o_mit[label] = float(base + mean[k])
            std_by[label] = float(std[k])
        return o_mit, std_by

    # -- online updates ---------------------------------------------------
    def update_with_rows(self, new_rows: list[dict], current_theta=None) -> None:
        if current_theta is not None:
            self._current_theta = np.asarray(current_theta, dtype=float).ravel()
        self.rows.extend(new_rows)
        if self.config.refit_affine_on_topup:
            self.backbone.fit(self.rows)
        if (
            self.config.drop_faraway_points
            and len(self.rows) > int(self.config.max_gp_points)
        ):
            self._prune_rows()
        self._fit_gp_from_rows()

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
    learning_rate: float = 0.3,
    fd_eps: float = 0.05,
    ideal_energy_fn: Callable[[np.ndarray], float] | None = None,
    verbose: bool = True,
) -> dict:
    """Bounded-step gradient descent on the mitigated energy.

    Per iteration:
      1. measure the REAL circuit's noisy expectation values for all Paulis,
      2. predict mitigated values + uncertainty,
      3. top up locally if uncertain, then re-predict,
      4. energy = sum_i c_i * O_mit[P_i]; finite-difference gradient; clipped step.

    Top-up frequency should taper toward zero as the optimizer settles.
    """
    cfg = mitigator.config
    adapter = mitigator.adapter
    max_iters = int(cfg.max_vqe_iterations if max_iters is None else max_iters)
    step_max = float(cfg.optimizer_step_max if step_max is None else step_max)

    theta = np.asarray(theta_init, dtype=float).ravel()
    rng = np.random.default_rng(int(cfg.rng_seed) + 777)

    def measure(th: np.ndarray) -> dict:
        return adapter.run_noisy(
            adapter.resolver_from_theta(th),
            shots=int(cfg.shots),
            sampling_seed=int(rng.integers(1, 2**31 - 1)),
        )

    def mitigated_energy(th: np.ndarray) -> float:
        return adapter.energy_from_values(mitigator.mitigate(th, measure(th)))

    history: list[dict] = []
    topup_total = 0
    for it in range(max_iters):
        measured = measure(theta)
        o_mit, std = mitigator.predict_with_uncertainty(theta, measured)
        topped = 0
        if needs_topup(std, mitigator.coeff_by_pauli, cfg):
            new_rows = sample_local_rows(
                adapter, theta, cfg, seed=int(cfg.rng_seed) + 10_000 + it
            )
            mitigator.update_with_rows(new_rows, current_theta=theta)
            o_mit, std = mitigator.predict_with_uncertainty(theta, measured)
            topped = 1
            topup_total += 1

        energy = adapter.energy_from_values(o_mit)
        unc = weighted_uncertainty(std, mitigator.coeff_by_pauli)

        grad = np.zeros_like(theta)
        for j in range(len(theta)):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] += fd_eps
            tm[j] -= fd_eps
            grad[j] = (mitigated_energy(tp) - mitigated_energy(tm)) / (2.0 * fd_eps)

        step = -learning_rate * grad
        max_comp = float(np.max(np.abs(step))) if step.size else 0.0
        if max_comp > step_max and max_comp > 0:
            step *= step_max / max_comp

        rec = {
            "iter": it,
            "theta": theta.copy(),
            "energy_mitigated": float(energy),
            "weighted_uncertainty": float(unc),
            "topped_up": int(topped),
            "topup_cum": int(topup_total),
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
                f"unc={unc: .4e}  topup={topped} (cum={topup_total})  "
                f"|g|={rec['grad_norm']: .3e}  rows={rec['n_rows']}"
            )

        theta = theta + step

    return {
        "theta_final": theta,
        "history": history,
        "topup_total": int(topup_total),
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
    O_ideal by: unmitigated, backbone-only (plain CDR), and backbone+GP.
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

    backbone = CDRBackbone(config.affine_regularization)
    backbone.fit(train_rows)
    gp = GPResidualModel(config)
    Xtr = np.stack([build_feature_row(r, config) for r in train_rows], axis=0)
    ytr = np.asarray([backbone.residual(r) for r in train_rows], dtype=float)
    gp.fit(Xtr, ytr)

    Xho = np.stack([build_feature_row(r, config) for r in hold_rows], axis=0)
    gp_mean, _ = gp.predict(Xho)
    o_ideal = np.asarray([r["o_ideal"] for r in hold_rows], dtype=float)
    o_noisy = np.asarray([r["o_noisy"] for r in hold_rows], dtype=float)
    backbone_pred = np.asarray(
        [backbone.apply(r["pauli"], r["o_noisy"]) for r in hold_rows], dtype=float
    )
    full_pred = backbone_pred + gp_mean

    return {
        "n_train_circuits": len(train_res),
        "n_holdout_circuits": len(hold_res),
        "n_holdout_rows": len(hold_rows),
        "rmse_unmitigated": _rmse(o_noisy, o_ideal),
        "rmse_backbone_only": _rmse(backbone_pred, o_ideal),
        "rmse_backbone_plus_gp": _rmse(full_pred, o_ideal),
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

    backbone = CDRBackbone(config.affine_regularization)
    backbone.fit(train_rows)
    gp = GPResidualModel(config)
    Xtr = np.stack([build_feature_row(r, config) for r in train_rows], axis=0)
    ytr = np.asarray([backbone.residual(r) for r in train_rows], dtype=float)
    gp.fit(Xtr, ytr)

    Xte = np.stack([build_feature_row(r, config) for r in test_rows], axis=0)
    gp_mean, gp_std = gp.predict(Xte)
    o_ideal = np.asarray([r["o_ideal"] for r in test_rows], dtype=float)
    o_noisy = np.asarray([r["o_noisy"] for r in test_rows], dtype=float)
    backbone_pred = np.asarray(
        [backbone.apply(r["pauli"], r["o_noisy"]) for r in test_rows], dtype=float
    )
    full_pred = backbone_pred + gp_mean
    return {
        "train_spread": float(train_spread),
        "test_spread": float(test_spread),
        "rmse_unmitigated": _rmse(o_noisy, o_ideal),
        "rmse_backbone_only": _rmse(backbone_pred, o_ideal),
        "rmse_backbone_plus_gp": _rmse(full_pred, o_ideal),
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
            )
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
    measured = adapter.run_noisy(
        adapter.resolver_from_theta(theta_clifford),
        shots=int(cfg.shots),
        sampling_seed=int(cfg.rng_seed) + 31,
    )
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
