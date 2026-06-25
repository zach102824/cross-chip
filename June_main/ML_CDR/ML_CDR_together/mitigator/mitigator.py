"""The Mitigator -- public API (Section 6).

ONE Gaussian Process learns (angle features, o_noisy, Pauli features) -> o_ideal.
The mitigated value IS the GP mean; there is no separate backbone to add on.
"""

from __future__ import annotations

import numpy as np

from .adapters import PauliObservable, QuantumBackendAdapter
from .config import MitigatorConfig
from .features import build_feature_matrix
from .gp_model import SingleGPMitigatorModel


class Mitigator:
    def __init__(
        self,
        backend: QuantumBackendAdapter,
        hamiltonian=None,
        config: MitigatorConfig | None = None,
    ):
        self.backend = backend
        self.config = config or MitigatorConfig()
        # Hamiltonian as [(coefficient, Pauli), ...]; default from the backend.
        if hamiltonian is None:
            hamiltonian = (
                backend.hamiltonian()
                if hasattr(backend, "hamiltonian")
                else [(o.coefficient, o) for o in backend.observables()]
            )
        self.hamiltonian = list(hamiltonian)
        self.observables = [p for _, p in self.hamiltonian]

        self.model = SingleGPMitigatorModel(self.config)
        self._rows: list[dict] = []
        self._theta_ref: np.ndarray = (
            np.asarray(self.config.theta_init, dtype=float).reshape(self.config.n_params)
            if self.config.theta_init is not None
            else np.zeros(self.config.n_params, dtype=float)
        )

    # --- data collection ---

    def collect_rows(
        self, theta: np.ndarray, n_circuits: int, n_nonclifford: int, spread: float, seed: int
    ) -> list[dict]:
        """Generate near-Clifford circuits around ``theta`` and build training rows.

        One row per (circuit, observable): {theta, pauli, o_noisy, o_ideal}. All
        observables ride along the same circuits for free.
        """
        cfg = self.config
        circuits = self.backend.generate_near_clifford(
            theta=np.asarray(theta, dtype=float),
            n_circuits=int(n_circuits),
            n_nonclifford=int(n_nonclifford),
            snap_step=float(cfg.clifford_snap_step),
            spread=float(spread),
            seed=int(seed),
        )
        rows: list[dict] = []
        for i, (resolver, theta_vec) in enumerate(circuits):
            ideal = self.backend.simulate_ideal(resolver)
            noisy = self.backend.run_noisy(
                resolver, shots=int(cfg.shots), sampling_seed=int(seed) + 7919 * (i + 1)
            )
            for obs in self.observables:
                rows.append(
                    {
                        "theta": np.asarray(theta_vec, dtype=float),
                        "pauli": obs,
                        "o_noisy": float(noisy[obs]),
                        "o_ideal": float(ideal[obs]),
                    }
                )
        return rows

    # --- public API ---

    def warmstart(self) -> "Mitigator":
        """One-time near-Clifford warm-start around ``theta_init``; fit the single GP."""
        cfg = self.config
        theta0 = self._theta_ref
        rows = self.collect_rows(
            theta0,
            n_circuits=cfg.n_warmstart_circuits,
            n_nonclifford=cfg.n_nonclifford_gates,
            spread=cfg.warmstart_spread,
            seed=cfg.rng_seed,
        )
        self._rows = rows
        self._refit()
        return self

    def mitigate(self, theta, o_noisy_by_pauli: dict) -> dict:
        """Return {pauli: O_mit} where O_mit is the GP mean for the real circuit."""
        means, _ = self._predict(theta, o_noisy_by_pauli)
        return means

    def predict_with_uncertainty(self, theta, o_noisy_by_pauli: dict) -> tuple[dict, dict]:
        """Return (O_mit_by_pauli, std_by_pauli)."""
        return self._predict(theta, o_noisy_by_pauli)

    def update_with_rows(self, new_rows: list[dict]) -> "Mitigator":
        """Append local near-Clifford rows; prune to max_gp_points; refit the GP."""
        if new_rows:
            self._rows.extend(new_rows)
            # Reference point for pruning = the latest sampled theta.
            self._theta_ref = np.asarray(new_rows[-1]["theta"], dtype=float).reshape(
                self.config.n_params
            )
        self._prune()
        self._refit()
        return self

    # --- internals ---

    def _predict(self, theta, o_noisy_by_pauli: dict) -> tuple[dict, dict]:
        theta = np.asarray(theta, dtype=float).reshape(self.config.n_params)
        paulis = list(o_noisy_by_pauli.keys())
        rows = [
            {"theta": theta, "pauli": p, "o_noisy": float(o_noisy_by_pauli[p])} for p in paulis
        ]
        X = build_feature_matrix(rows, self.config)
        mean, var = self.model.predict(X)
        std = np.sqrt(np.clip(var, 0.0, None))
        means = {p: float(mean[k]) for k, p in enumerate(paulis)}
        stds = {p: float(std[k]) for k, p in enumerate(paulis)}
        return means, stds

    def _refit(self) -> None:
        if not self._rows:
            raise RuntimeError("No training rows collected; call warmstart() first.")
        X = build_feature_matrix(self._rows, self.config)
        y = np.array([r["o_ideal"] for r in self._rows], dtype=float)
        self.model.fit(X, y)

    def _prune(self) -> None:
        cfg = self.config
        if not cfg.drop_faraway_points or len(self._rows) <= cfg.max_gp_points:
            return
        thetas = np.stack([np.asarray(r["theta"], dtype=float) for r in self._rows], axis=0)
        dist = np.linalg.norm(thetas - self._theta_ref[None, :], axis=1)
        keep = np.argsort(dist)[: cfg.max_gp_points]
        keep_set = set(int(i) for i in keep.tolist())
        self._rows = [r for i, r in enumerate(self._rows) if i in keep_set]

    # --- diagnostics ---

    @property
    def n_rows(self) -> int:
        return len(self._rows)

    def energy(self, o_mit_by_pauli: dict) -> float:
        """E = offset + sum_i c_i * O_mit[P_i]."""
        offset = getattr(self.backend, "hamiltonian_offset", 0.0)
        e = float(offset)
        for coeff, obs in self.hamiltonian:
            e += float(coeff) * float(o_mit_by_pauli[obs])
        return e
