"""Monte Carlo statevector trajectories for gate-local depolarizing noise.

Replaces full density-matrix evolution while preserving the same
``GateArityDepolarizingNoise`` channel semantics used by the DM backend:

- after each 1Q gate: single-qubit depolarize(p1)
- after each 2Q gate: independent single-qubit depolarize(p) on each qubit
  (cross-chip tagged ops use the higher cross-chip probability)
- readout noise stays classical (handled by ``shot_measurement``)

Each physical shot gets an independent noise trajectory + one computational
measurement in the requested OGM / Pauli basis.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import cirq
import numpy as np

from main_cursor_lib import GateArityDepolarizingNoise

# Pauli matrices for single-qubit depolarizing sampling.
_PAULIS = (
    np.eye(2, dtype=np.complex128),
    np.array([[0, 1], [1, 0]], dtype=np.complex128),  # X
    np.array([[0, -1j], [1j, 0]], dtype=np.complex128),  # Y
    np.array([[1, 0], [0, -1]], dtype=np.complex128),  # Z
)


def _qubit_index_map(qubits: list[cirq.Qid]) -> dict[cirq.Qid, int]:
    return {q: i for i, q in enumerate(qubits)}


def _apply_unitary(
    state: np.ndarray,
    matrix: np.ndarray,
    targets: list[int],
    n_qubits: int,
) -> None:
    """Apply a unitary in-place matching ``cirq.Simulator`` qubit_order layout.

    With ``qubit_order=[q0,...,q_{n-1}]``, axis ``i`` of the reshaped
    ``(2,)*n`` tensor is qubit ``i`` (q0 = most-significant index bit).
    ``matrix`` / ``targets`` follow Cirq operation order (first target = MSB
    of the subspace matrix).
    """
    matrix = np.asarray(matrix, dtype=np.complex128)
    k = len(targets)
    if k == 0:
        return
    if k == 1:
        # Fast path: qubit t is axis t; pack as (2^t, 2, 2^{n-t-1}).
        t = int(targets[0])
        left = 2**t
        mid = 2
        right = 2 ** (n_qubits - t - 1)
        view = state.reshape(left, mid, right)
        # out[:, a, :] = sum_b U[a,b] in[:, b, :]
        np.einsum("ab,ibj->iaj", matrix, view, optimize=True, out=view)
        state.shape = (2**n_qubits,)
        return

    tensor = state.reshape((2,) * n_qubits)
    target_axes = list(targets)
    other_axes = [ax for ax in range(n_qubits) if ax not in target_axes]
    # Append targets in op order so the first target is the slowest / MSB axis.
    perm = other_axes + target_axes
    moved = np.transpose(tensor, axes=perm)
    flat = moved.reshape(2 ** (n_qubits - k), 2**k)
    updated = flat @ matrix.T
    moved = updated.reshape([2] * (n_qubits - k) + [2] * k)
    inv = np.argsort(perm)
    restored = np.transpose(moved, axes=inv)
    state[:] = restored.reshape(2**n_qubits)


def _compile_noisy_ops(
    ansatz_circuit: cirq.Circuit,
    resolver: dict,
    qubits: list[cirq.Qid],
    noise_params: dict,
) -> list[tuple[str, Any]]:
    """Compile resolved noisy circuit into (kind, payload) ops for fast replay."""
    noise_model = GateArityDepolarizingNoise(**noise_params)
    noisy = ansatz_circuit.with_noise(noise_model)
    resolved = cirq.resolve_parameters(noisy, resolver)
    qmap = _qubit_index_map(qubits)
    compiled: list[tuple[str, Any]] = []

    for op in resolved.all_operations():
        if isinstance(op.gate, cirq.MeasurementGate):
            continue
        if isinstance(op.gate, cirq.DepolarizingChannel):
            if len(op.qubits) != 1:
                raise ValueError("Only single-qubit DepolarizingChannel is supported.")
            p = float(op.gate.p)
            compiled.append(("depol", (qmap[op.qubits[0]], p)))
            continue
        if cirq.has_unitary(op):
            matrix = np.asarray(cirq.unitary(op), dtype=np.complex128)
            targets = [qmap[q] for q in op.qubits]
            compiled.append(("unitary", (matrix, targets)))
            continue
        raise TypeError(f"Unsupported noisy op for trajectory backend: {op!r}")

    return compiled


def evolve_noisy_trajectory(
    compiled_ops: list[tuple[str, Any]],
    n_qubits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Evolve |0...0> through compiled noisy ops; return statevector."""
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[0] = 1.0
    for kind, payload in compiled_ops:
        if kind == "unitary":
            matrix, targets = payload
            _apply_unitary(state, matrix, targets, n_qubits)
        elif kind == "depol":
            q, p = payload
            p = float(min(1.0, max(0.0, p)))
            # P(I)=1-p, P(X)=P(Y)=P(Z)=p/3
            probs = np.array([1.0 - p, p / 3.0, p / 3.0, p / 3.0], dtype=float)
            choice = int(rng.choice(4, p=probs))
            if choice != 0:
                _apply_unitary(state, _PAULIS[choice], [q], n_qubits)
        else:
            raise RuntimeError(f"Unknown compiled op kind {kind!r}")
    return state


def _basis_rotation_ops(pauli_str: str, n_qubits: int) -> list[tuple[str, Any]]:
    """Single-qubit basis-change unitaries matching ``rotation_circuit_for_pauli_string``."""
    ops: list[tuple[str, Any]] = []
    h = np.asarray(cirq.unitary(cirq.H), dtype=np.complex128)
    rx = np.asarray(cirq.unitary(cirq.rx(np.pi / 2)), dtype=np.complex128)
    for i, ch in enumerate(pauli_str):
        if ch == "X":
            ops.append(("unitary", (h, [i])))
        elif ch == "Y":
            ops.append(("unitary", (rx, [i])))
    _ = n_qubits
    return ops


def _sample_z_bitstring(state: np.ndarray, n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    probs = np.real(state * np.conjugate(state))
    probs = np.clip(probs, 0.0, None)
    total = float(np.sum(probs))
    if total <= 0.0:
        probs = np.full(2**n_qubits, 1.0 / (2**n_qubits))
    else:
        probs = probs / total
    idx = int(rng.choice(2**n_qubits, p=probs))
    bits = np.array(list(np.binary_repr(idx, width=n_qubits)), dtype=int)
    return bits


def _shot_worker_count(num_shots: int) -> int:
    """Optional shot-level threads (``TRAJECTORY_SHOT_WORKERS``, default 1)."""
    raw = str(os.environ.get("TRAJECTORY_SHOT_WORKERS", "1")).strip()
    try:
        workers = int(raw)
    except ValueError:
        workers = 1
    return max(1, min(workers, int(num_shots)))


def sample_measurement_basis_from_trajectories(
    compiled_ops: list[tuple[str, Any]],
    pauli_str: str,
    n_qubits: int,
    num_shots: int,
    noise_rng: np.random.Generator,
    measure_rng: np.random.Generator,
) -> np.ndarray:
    """Independent noisy trajectories measured in ``pauli_str`` basis."""
    rot_ops = _basis_rotation_ops(pauli_str, n_qubits)
    num_shots = int(num_shots)
    out = np.zeros((num_shots, n_qubits), dtype=int)
    mask = np.array([ch != "I" for ch in pauli_str], dtype=bool)
    workers = _shot_worker_count(num_shots)

    def _one_shot(noise_seed: int, measure_seed: int) -> np.ndarray:
        local_noise = np.random.default_rng(noise_seed)
        local_measure = np.random.default_rng(measure_seed)
        state = evolve_noisy_trajectory(compiled_ops, n_qubits, local_noise)
        for kind, payload in rot_ops:
            matrix, targets = payload
            _apply_unitary(state, matrix, targets, n_qubits)
        bits = _sample_z_bitstring(state, n_qubits, local_measure)
        if np.any(~mask):
            bits = bits.copy()
            bits[~mask] = 0
        return bits

    if workers == 1:
        for s in range(num_shots):
            out[s] = _one_shot(
                int(noise_rng.integers(0, 2**31 - 1)),
                int(measure_rng.integers(0, 2**31 - 1)),
            )
        return out

    noise_seeds = noise_rng.integers(0, 2**31 - 1, size=num_shots)
    measure_seeds = measure_rng.integers(0, 2**31 - 1, size=num_shots)
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="trajshot") as pool:
        rows = list(pool.map(_one_shot, noise_seeds.tolist(), measure_seeds.tolist()))
    for s, bits in enumerate(rows):
        out[s] = bits
    return out


def _use_stim_clifford() -> bool:
    """Stim near-Clifford path (default on when Stim is installed)."""
    raw = str(os.environ.get("USE_STIM_CLIFFORD", "1")).strip().lower()
    return raw not in ("0", "false", "no", "off")


def estimate_energy_from_noisy_circuit_shots(
    ansatz_circuit: cirq.Circuit,
    resolver: dict,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    noise_params: dict,
    *,
    simulator_seed: int = 1234,
    num_shots: int = 8192,
    measurement_scheme: str = "ogm",
    p_0_success: Iterable[float] | None = None,
    p_1_success: Iterable[float] | None = None,
    apply_rem: bool = True,
    apply_readout_noise: bool = True,
    sampling_seed: int = 1234,
    epsilon: float = 0.1,
    ogm_file: str | Path | None = None,
    shadowgrouping_root: str | Path | None = None,
    return_per_term: bool = False,
) -> dict[str, Any]:
    """Trajectory analogue of ``estimate_energy_from_noisy_rho_shots``.

    Uses the same OGM / direct-Pauli setting allocation and REM/readout path as
    the density-matrix backend; only the noisy-state + measurement sampling changes.
    """
    # Import from shot_measurement lazily to reuse setting / REM helpers without
    # creating an import cycle at module import time.
    import shot_measurement as sm

    scheme = measurement_scheme.lower()
    if scheme not in sm.MEASUREMENT_SCHEMES:
        raise ValueError(f"Unsupported measurement scheme: {measurement_scheme}.")

    num_qubits = len(qubits)
    measure_rng = np.random.default_rng(sampling_seed)
    noise_rng = np.random.default_rng(simulator_seed)
    p0 = np.ones(num_qubits) if p_0_success is None else np.asarray(list(p_0_success), dtype=float)
    p1 = np.ones(num_qubits) if p_1_success is None else np.asarray(list(p_1_success), dtype=float)
    if len(p0) != num_qubits or len(p1) != num_qubits:
        raise ValueError("Readout calibration arrays must match number of qubits.")

    observables_int, weights, offset = sm.pauli_sum_to_int_observables(observable_h, qubits)
    if len(weights) == 0:
        out: dict[str, Any] = {"energy_unmitigated": offset, "energy_rem": offset, "offset": offset}
        if return_per_term:
            out["per_term_unmitigated"] = np.zeros((0,), dtype=float)
            out["per_term_rem"] = np.zeros((0,), dtype=float)
        return out

    rem_vectors = sm.rem_z_vectors(p0, p1) if (apply_readout_noise and apply_rem) else None

    if scheme == "direct_pauli":
        sampled_settings = sm._build_direct_pauli_settings(observables_int, num_shots)
    elif scheme == "ogm":
        sampled_settings = sm._sample_settings_from_ogm_file(
            ogm_file,
            observables_int,
            weights,
            num_shots=num_shots,
            epsilon=epsilon,
        )
    else:
        if shadowgrouping_root is None:
            raise ValueError(
                "shadowgrouping_root is required for shadowgrouping schemes. "
                "Set SHADOWGROUPING_ROOT or pass it explicitly."
            )
        sm.ensure_shadowgrouping_importable(shadowgrouping_root)
        method = sm._load_shadowgrouping_scheme(scheme, observables_int, weights, epsilon, ogm_file)
        settings, _ = method.find_setting(N_samples=num_shots)
        sampled_settings = np.asarray(settings, dtype=int)
        if sampled_settings.ndim == 1:
            sampled_settings = sampled_settings.reshape(1, -1)

    if sampled_settings.shape[0] == 0:
        out = {
            "energy_unmitigated": float(offset),
            "energy_rem": float(offset),
            "offset": float(offset),
        }
        if return_per_term:
            z = np.zeros(len(weights), dtype=float)
            out["per_term_unmitigated"] = z.copy()
            out["per_term_rem"] = z.copy()
        return out

    stim_compiled = None
    backend_tag = "trajectory"
    if _use_stim_clifford():
        try:
            from stim_clifford import (  # local export2cloud module
                compile_stim_hybrid_ops,
                count_non_clifford_compiled,
                sample_measurement_basis_stim_hybrid,
                stim_available,
            )

            if stim_available():
                # Stim helps most for near-Clifford trainers (t_max≈2 → ~1 non-Clifford).
                # General VQE targets with many non-Cliffords are faster on NumPy traj.
                max_non = int(os.environ.get("STIM_MAX_NON_CLIFFORD", "2"))
                candidate = compile_stim_hybrid_ops(
                    ansatz_circuit, resolver, qubits, noise_params
                )
                if candidate is not None:
                    n_non = count_non_clifford_compiled(candidate)
                    if n_non <= max_non:
                        stim_compiled = candidate
                        backend_tag = (
                            "stim_clifford" if n_non == 0 else f"stim_hybrid_t{n_non}"
                        )
        except Exception:
            stim_compiled = None

    compiled = None
    if stim_compiled is None:
        compiled = _compile_noisy_ops(ansatz_circuit, resolver, qubits, noise_params)

    def _sample_basis(basis: str, count: int) -> np.ndarray:
        if stim_compiled is not None:
            return sample_measurement_basis_stim_hybrid(
                stim_compiled,
                basis,
                num_qubits,
                count,
                noise_rng,
                measure_rng,
                apply_unitary=_apply_unitary,
                sample_z_bitstring=_sample_z_bitstring,
                paulis=_PAULIS,
                basis_rotation_ops=_basis_rotation_ops,
            )
        assert compiled is not None
        return sample_measurement_basis_from_trajectories(
            compiled,
            basis,
            num_qubits,
            count,
            noise_rng,
            measure_rng,
        )

    unique_settings, unique_counts = sm._unique_settings_with_counts(sampled_settings)
    basis_samples: dict[tuple[int, ...], np.ndarray] = {}
    for setting_row, count in zip(unique_settings, unique_counts):
        basis = sm.int_observable_to_pauli_string(setting_row)
        ideal = _sample_basis(basis, count)
        if apply_readout_noise:
            noisy = sm.apply_asymmetric_readout_noise(ideal, p0, p1, measure_rng)
        else:
            noisy = ideal
        basis_samples[sm._hashable_setting(setting_row)] = noisy

    energy_unmit = offset
    energy_rem = offset
    per_term_unmit: list[float] | None = [] if return_per_term else None
    per_term_rem: list[float] | None = [] if return_per_term else None

    for obs_row, coeff in zip(observables_int, weights):
        compatible_keys = [
            sm._hashable_setting(srow)
            for srow in unique_settings
            if sm._is_setting_compatible(obs_row, srow)
        ]
        if not compatible_keys:
            direct_basis = sm.int_observable_to_pauli_string(obs_row)
            bits = _sample_basis(direct_basis, 1)
            if apply_readout_noise:
                bits = sm.apply_asymmetric_readout_noise(bits, p0, p1, measure_rng)
            compatible_samples = [bits]
        else:
            compatible_samples = [basis_samples[k] for k in compatible_keys]

        total = sum(arr.shape[0] for arr in compatible_samples)
        unmit_acc = 0.0
        rem_acc = 0.0
        for bits in compatible_samples:
            n = bits.shape[0]
            unmit_acc += n * sm._term_expectation_from_bitstrings(bits, obs_row, rem_vectors=None)
            rem_acc += n * sm._term_expectation_from_bitstrings(bits, obs_row, rem_vectors=rem_vectors)
        term_u = unmit_acc / total
        term_r = rem_acc / total
        energy_unmit += coeff * term_u
        energy_rem += coeff * term_r
        if return_per_term:
            assert per_term_unmit is not None and per_term_rem is not None
            per_term_unmit.append(float(term_u))
            per_term_rem.append(float(term_r))

    result: dict[str, Any] = {
        "energy_unmitigated": float(energy_unmit),
        "energy_rem": float(energy_rem),
        "offset": float(offset),
        "noisy_backend": backend_tag,
    }
    if return_per_term:
        result["per_term_unmitigated"] = np.asarray(per_term_unmit, dtype=float)
        result["per_term_rem"] = np.asarray(per_term_rem, dtype=float)
    return result
