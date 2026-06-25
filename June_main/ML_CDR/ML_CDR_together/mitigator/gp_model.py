"""Single Gaussian Process with a composite kernel -- CDR LIVES HERE (Section 5).

ONE GP over all observables maps feature rows -> ``o_ideal`` DIRECTLY (standardized).
The kernel is the heart of Design 2:

    k = k_lin(o_noisy)            # the learned CDR line a*o_noisy + b
      + k_base(angles, pauli)     # coherent / angle-dependent residual
      + white_noise               # shot noise

scikit-learn kernels have no native ``active_dims``, so :class:`OnColumns` wraps a
base kernel and restricts it to a fixed set of feature columns (using the index
map from ``features.feature_index_map``). This is how each sub-kernel is applied
to the correct block.
"""

from __future__ import annotations

import numpy as np
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

from .config import MitigatorConfig
from .features import feature_index_map


class OnColumns(Kernel):
    """Apply ``kernel`` only to the given feature ``columns`` (active_dims).

    Modeled on sklearn's ``Exponentiation`` so the hyperparameter / ``theta``
    plumbing (needed by the marginal-likelihood optimizer) forwards to the wrapped
    kernel. Column selection is a constant transform, so the kernel gradient passes
    through unchanged.
    """

    def __init__(self, kernel: Kernel, columns):
        # NOTE: store params unmodified so sklearn.base.clone (identity check) works.
        self.kernel = kernel
        self.columns = columns

    def _slice(self, X):
        if X is None:
            return None
        return np.asarray(X)[:, list(self.columns)]

    def get_params(self, deep=True):
        params = dict(kernel=self.kernel, columns=self.columns)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(("kernel__" + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        return [
            Hyperparameter("kernel__" + h.name, h.value_type, h.bounds, h.n_elements)
            for h in self.kernel.hyperparameters
        ]

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
        return self.kernel(self._slice(X), self._slice(Y), eval_gradient=eval_gradient)

    def diag(self, X):
        return self.kernel.diag(self._slice(X))

    def is_stationary(self):
        return self.kernel.is_stationary()

    def __repr__(self):
        return f"OnColumns({self.kernel!r}, columns={self.columns})"


def _slice_to_columns(s: slice, total: int) -> list[int]:
    return list(range(*s.indices(total)))


def _base_kernel(kernel_type: str, n_dims: int, use_ard: bool, matern_nu: float) -> Kernel:
    ls = np.ones(n_dims) if (use_ard and n_dims > 0) else 1.0
    # Wide upper bound: constant one-hot Pauli columns push their lengthscale large
    # (= "feature irrelevant"), which is correct and should not warn.
    bounds = (1e-3, 1e5)
    if kernel_type == "matern":
        return Matern(length_scale=ls, length_scale_bounds=bounds, nu=float(matern_nu))
    if kernel_type == "rbf":
        return RBF(length_scale=ls, length_scale_bounds=bounds)
    raise ValueError(f"kernel_type must be 'matern' or 'rbf', got {kernel_type!r}.")


class SingleGPMitigatorModel:
    """One GP, target = raw o_ideal (standardized). predict -> (mean, variance)."""

    def __init__(self, config: MitigatorConfig):
        self.config = config
        self._index_map = feature_index_map(config)
        self._gp: GaussianProcessRegressor | None = None
        self._n_features: int | None = None

    def build_kernel(self, feature_index_map: dict | None = None, config: MitigatorConfig | None = None) -> Kernel:
        config = config or self.config
        idx = feature_index_map or self._index_map
        total = idx["pauli"].stop  # last column boundary == feature dim

        noisy_cols = _slice_to_columns(idx["noisy"], total)
        pauli_cols = _slice_to_columns(idx["pauli"], total)
        angle_cols = _slice_to_columns(idx["angle"], total)
        base_cols = angle_cols + pauli_cols

        if not config.include_noisy_feature or not noisy_cols:
            raise ValueError("Design 2 requires an o_noisy feature column for k_lin.")

        # k_lin: DotProduct on o_noisy reproduces a*o_noisy + b (bias via sigma_0).
        k_lin = ConstantKernel(1.0, (1e-3, 1e3)) * OnColumns(
            DotProduct(sigma_0=1.0, sigma_0_bounds=(1e-5, 1e2)), noisy_cols
        )
        # linear_times_obs: per-observable effective slope via k_lin x k_obs(pauli).
        if config.linear_times_obs and pauli_cols:
            k_obs = OnColumns(
                _base_kernel(config.kernel_type, len(pauli_cols), config.use_ard, config.matern_nu),
                pauli_cols,
            )
            k_lin = k_lin * k_obs

        kernel = k_lin

        # k_base: coherent / angle-dependent residual over (angles, pauli).
        if config.use_rbf_kernel and base_cols:
            k_base = ConstantKernel(1.0, (1e-3, 1e3)) * OnColumns(
                _base_kernel(config.kernel_type, len(base_cols), config.use_ard, config.matern_nu),
                base_cols,
            )
            kernel = kernel + k_base

        # white_noise: shot noise.
        kernel = kernel + WhiteKernel(
            noise_level=float(config.noise_variance_init), noise_level_bounds=(1e-8, 1e1)
        )
        return kernel

    def fit(self, X: np.ndarray, y_ideal: np.ndarray) -> "SingleGPMitigatorModel":
        """Maximize marginal likelihood once, shared across all observables."""
        X = np.asarray(X, dtype=float)
        y = np.asarray(y_ideal, dtype=float).ravel()
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f"Shape mismatch: X={X.shape}, y={y.shape}.")
        self._n_features = X.shape[1]
        kernel = self.build_kernel()
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=bool(self.config.normalize_targets),
            alpha=1e-10,  # white kernel models the shot noise; keep jitter tiny
            n_restarts_optimizer=2,
            random_state=int(self.config.rng_seed),
        )
        self._gp.fit(X, y)
        return self

    def predict(self, X_star: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean = mitigated O_ideal, variance). Closed-form once fit."""
        if self._gp is None:
            raise RuntimeError("Model is not fit yet; call fit() first.")
        X_star = np.asarray(X_star, dtype=float)
        mean, std = self._gp.predict(X_star, return_std=True)
        return np.asarray(mean, dtype=float), np.asarray(std, dtype=float) ** 2

    @property
    def kernel_(self):
        return None if self._gp is None else self._gp.kernel_

    @property
    def log_marginal_likelihood_(self):
        if self._gp is None:
            return None
        return float(self._gp.log_marginal_likelihood(self._gp.kernel_.theta))
