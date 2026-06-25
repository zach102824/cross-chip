"""Integration adapters: bind the mitigator to the user's EXISTING cirq/CDR code.

The single-GP mitigator never builds circuits or runs simulations itself. It calls
the methods of a ``QuantumBackendAdapter``. ``CirqCDRAdapter`` is the concrete
binding to this repo's functions in ``main_cursor_lib`` and ``shot_measurement``:

  build_ansatz            -> cirq.resolve_parameters(circuit, resolver_from_params)
  generate_near_clifford  -> local near-Clifford sampling (snap most angles, perturb a few)
                             reusing clifford_snap_value_for_symbol / count_non_clifford_ops
  simulate_ideal          -> _simulate_noiseless_state_for_resolver + exact per-Pauli value
  run_noisy               -> _simulate_noisy_rho_for_resolver + estimate_energy_from_noisy_rho_shots

A "Pauli" here is a hashable :class:`PauliObservable` whose per-qubit ``label`` is a
string over {I, X, Y, Z}; this is the dict key used everywhere downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import cirq
import numpy as np

from main_cursor_lib import (
    clifford_snap_value_for_symbol,
    count_non_clifford_ops,
)
from shot_measurement import (
    _simulate_noiseless_state_for_resolver,
    _simulate_noisy_rho_for_resolver,
    estimate_energy_from_noisy_rho_shots,
    exact_pauli_expectation_from_int_row,
    pauli_sum_to_int_observables,
)

_CHAR_TO_INT = {"I": 0, "X": 1, "Y": 2, "Z": 3}
_INT_TO_CHAR = {0: "I", 1: "X", 2: "Y", 3: "Z"}


@dataclass(frozen=True)
class PauliObservable:
    """A single Hamiltonian Pauli term ``c_i * P_i``.

    ``label`` is the per-qubit {I,X,Y,Z} string (length n_qubits) and is the
    hashable identity used as a dict key. ``coefficient`` is the Hamiltonian
    weight c_i. The integer-row encoding matches ``pauli_sum_to_int_observables``.
    """

    label: str
    coefficient: float

    @property
    def int_row(self) -> np.ndarray:
        return np.array([_CHAR_TO_INT[ch] for ch in self.label], dtype=int)

    @property
    def weight(self) -> int:
        return sum(1 for ch in self.label if ch != "I")


@runtime_checkable
class QuantumBackendAdapter(Protocol):
    """Thin interface the mitigator depends on (Section 2 of the spec)."""

    def observables(self) -> list[PauliObservable]: ...

    def build_ansatz(self, theta: np.ndarray) -> cirq.Circuit: ...

    def generate_near_clifford(
        self,
        theta: np.ndarray,
        n_circuits: int,
        n_nonclifford: int,
        snap_step: float,
        spread: float,
        seed: int,
    ) -> list[tuple[dict, np.ndarray]]: ...

    def simulate_ideal(self, resolver: dict) -> dict[PauliObservable, float]: ...

    def run_noisy(
        self, resolver: dict, shots: int, sampling_seed: int | None = None
    ) -> dict[PauliObservable, float]: ...


@dataclass
class CirqCDRAdapter:
    """Concrete adapter binding to this repo's cirq + CDR helpers.

    Parameters
    ----------
    circuit:
        The (symbolic) ansatz ``cirq.Circuit`` (prep + ansatz), as built in the
        notebook. Angles are sympy symbols resolved per ``theta``.
    qubits:
        Ordered list of ``cirq.Qid`` matching the Hamiltonian convention.
    symbols:
        Ordered list of sympy symbols; ``theta[j]`` binds ``symbols[j]``.
    pauli_sum:
        ``cirq.PauliSum`` Hamiltonian H = sum_i c_i P_i (identity offset allowed).
    base_noise_cfg:
        dict of two_qubit_depol_prob / one_qubit_depol_prob / cross_chip_two_qubit_depol_prob.
    shot_cfg:
        dict of num_shots, measurement_scheme, apply_readout_noise, sampling_seed,
        epsilon, ogm_file, shadowgrouping_root.
    readout_cal:
        dict with p_0_success, p_1_success (or empty for ideal readout).
    use_rem_noisy_feature:
        If True (default) the ``o_noisy`` feature is the REM-corrected per-term value
        (matches the existing CDR pipeline). Set False to feed raw unmitigated values.
    """

    circuit: cirq.Circuit
    qubits: list
    symbols: list
    pauli_sum: cirq.PauliSum
    base_noise_cfg: dict
    shot_cfg: dict = field(default_factory=dict)
    readout_cal: dict = field(default_factory=dict)
    simulator_seed: int = 1234
    use_rem_noisy_feature: bool = True

    _observables: list[PauliObservable] = field(default_factory=list, init=False, repr=False)
    _offset: float = field(default=0.0, init=False, repr=False)

    def __post_init__(self) -> None:
        obs_int, weights, offset = pauli_sum_to_int_observables(self.pauli_sum, self.qubits)
        self._offset = float(offset)
        self._observables = [
            PauliObservable(
                label="".join(_INT_TO_CHAR[int(v)] for v in row.tolist()),
                coefficient=float(w),
            )
            for row, w in zip(obs_int, weights)
        ]

    # --- introspection ---

    @property
    def n_params(self) -> int:
        return len(self.symbols)

    @property
    def n_qubits(self) -> int:
        return len(self.qubits)

    @property
    def hamiltonian_offset(self) -> float:
        """Identity-term coefficient: E = offset + sum_i c_i <P_i>."""
        return self._offset

    def observables(self) -> list[PauliObservable]:
        return list(self._observables)

    def hamiltonian(self) -> list[tuple[float, PauliObservable]]:
        """[(coefficient, Pauli), ...] convenience view of H (no identity offset)."""
        return [(obs.coefficient, obs) for obs in self._observables]

    # --- theta <-> resolver ---

    def theta_to_resolver(self, theta: np.ndarray) -> dict:
        t = np.asarray(theta, dtype=float).reshape(self.n_params)
        return {self.symbols[j]: float(t[j]) for j in range(self.n_params)}

    def resolver_to_theta(self, resolver: dict) -> np.ndarray:
        return np.array([float(resolver[self.symbols[j]]) for j in range(self.n_params)], dtype=float)

    # --- adapter methods ---

    def build_ansatz(self, theta: np.ndarray) -> cirq.Circuit:
        return cirq.resolve_parameters(self.circuit, self.theta_to_resolver(theta))

    def _snap(self, symbol, value: float, snap_step: float) -> float:
        try:
            return clifford_snap_value_for_symbol(symbol, value)
        except ValueError:
            step = float(snap_step)
            return float(round(float(value) / step) * step)

    def generate_near_clifford(
        self,
        theta: np.ndarray,
        n_circuits: int,
        n_nonclifford: int,
        snap_step: float,
        spread: float,
        seed: int,
    ) -> list[tuple[dict, np.ndarray]]:
        """Local near-Clifford circuits around ``theta``.

        Most angles are snapped to the nearest Clifford value (of ``theta``); a
        random subset of ``n_nonclifford`` angles is kept non-Clifford by
        perturbing ``theta`` within +/- ``spread`` radians. This LOCAL sampling
        (vs the global [0, 2pi] sampling in ``generate_near_clifford_param_sets``)
        is what makes top-ups concentrate near the current optimizer point.

        Returns [(resolver, exact_theta_array), ...].
        """
        if n_circuits <= 0:
            raise ValueError(f"n_circuits must be > 0, got {n_circuits}.")
        theta = np.asarray(theta, dtype=float).reshape(self.n_params)
        n_keep = int(min(max(n_nonclifford, 0), self.n_params))

        out: list[tuple[dict, np.ndarray]] = []
        for c in range(int(n_circuits)):
            rng = np.random.default_rng(int(seed) + 1000 * (c + 1))
            if n_keep > 0:
                keep_idx = set(
                    int(i) for i in rng.choice(self.n_params, size=n_keep, replace=False).tolist()
                )
            else:
                keep_idx = set()

            vec = np.empty(self.n_params, dtype=float)
            for j, sym in enumerate(self.symbols):
                if j in keep_idx:
                    vec[j] = float(theta[j] + rng.uniform(-spread, spread))
                else:
                    vec[j] = self._snap(sym, float(theta[j]), snap_step)
            out.append((self.theta_to_resolver(vec), vec))
        return out

    def count_non_clifford(self, resolver: dict) -> int:
        return int(count_non_clifford_ops(self.circuit, resolver))

    def simulate_ideal(self, resolver: dict) -> dict[PauliObservable, float]:
        state = _simulate_noiseless_state_for_resolver(
            self.circuit, resolver, self.qubits, simulator_seed=self.simulator_seed
        )
        return {
            obs: float(exact_pauli_expectation_from_int_row(state, obs.int_row, self.qubits))
            for obs in self._observables
        }

    def run_noisy(
        self, resolver: dict, shots: int, sampling_seed: int | None = None
    ) -> dict[PauliObservable, float]:
        rho = _simulate_noisy_rho_for_resolver(
            self.circuit, resolver, self.qubits, self.base_noise_cfg, simulator_seed=self.simulator_seed
        )
        p0 = self.readout_cal.get("p_0_success")
        p1 = self.readout_cal.get("p_1_success")
        apply_readout_noise = bool(self.shot_cfg.get("apply_readout_noise", True))
        est = estimate_energy_from_noisy_rho_shots(
            rho,
            self.pauli_sum,
            self.qubits,
            num_shots=int(shots),
            measurement_scheme=str(self.shot_cfg.get("measurement_scheme", "direct_pauli")),
            p_0_success=p0,
            p_1_success=p1,
            apply_rem=True,
            apply_readout_noise=apply_readout_noise,
            sampling_seed=int(
                sampling_seed
                if sampling_seed is not None
                else self.shot_cfg.get("sampling_seed", 1234)
            ),
            epsilon=float(self.shot_cfg.get("epsilon", 0.1)),
            ogm_file=self.shot_cfg.get("ogm_file"),
            shadowgrouping_root=self.shot_cfg.get("shadowgrouping_root"),
            return_per_term=True,
        )
        branch = est["per_term_rem"] if self.use_rem_noisy_feature else est["per_term_unmitigated"]
        branch = np.asarray(branch, dtype=float).ravel()
        return {obs: float(branch[k]) for k, obs in enumerate(self._observables)}
