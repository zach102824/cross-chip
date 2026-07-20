"""Stim-accelerated sampling for Clifford / near-Clifford noisy circuits.

Matches ``GateArityDepolarizingNoise`` semantics used by the trajectory backend:

* after each 1Q gate: ``DEPOLARIZE1(p1)``
* after each 2Q gate: independent ``DEPOLARIZE1(p)`` on each qubit
  (cross-chip tagged ops use the higher cross-chip probability)

Fully Clifford resolved circuits are sampled with Stim's compiled sampler
(all shots at once). Near-Clifford circuits (few non-Clifford unitaries) use a
hybrid TableauSimulator → statevector handoff, then finish with the NumPy
trajectory kernel.
"""

from __future__ import annotations

from typing import Any

import cirq
import numpy as np

try:
    import stim
except ImportError:  # pragma: no cover
    stim = None  # type: ignore

from main_cursor_lib import GateArityDepolarizingNoise


def stim_available() -> bool:
    return stim is not None


def _qubit_index_map(qubits: list[cirq.Qid]) -> dict[cirq.Qid, int]:
    return {q: i for i, q in enumerate(qubits)}


def _try_clifford_tableau(matrix: np.ndarray, *, atol: float = 1e-6):
    """Return a Stim Tableau if ``matrix`` is Clifford, else None.

    Stim's ``from_unitary_matrix`` can return a *nearest* Clifford for some
    non-Clifford inputs; we reject those by checking unitary reconstruction
    (up to global phase).
    """
    if stim is None:
        return None
    mat = np.asarray(matrix, dtype=np.complex128)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        return None
    dim = mat.shape[0]
    n = int(round(np.log2(dim)))
    if 2**n != dim or n not in (1, 2):
        return None
    try:
        # Cirq unitaries use the first listed qubit as MSB ↔ Stim endian='big'.
        tab = stim.Tableau.from_unitary_matrix(mat, endian="big")
    except (ValueError, RuntimeError, TypeError):
        return None
    recon = np.asarray(tab.to_unitary_matrix(endian="big"), dtype=np.complex128)
    overlap = complex(np.vdot(recon.ravel(), mat.ravel()))
    if abs(overlap) < 1e-12:
        return None
    phase = overlap / abs(overlap)
    err = float(np.linalg.norm(mat - phase * recon))
    if err > atol * dim:
        return None
    return tab


def _remap_tableau_circuit(tab, targets: list[int]) -> "stim.Circuit":
    """Expand a Tableau on logical qubits 0..k-1 onto physical ``targets``."""
    assert stim is not None
    remapped = stim.Circuit()
    for inst in tab.to_circuit():
        gate_name = inst.name
        args = list(inst.gate_args_copy())
        qs = [int(targets[int(t.value)]) for t in inst.targets_copy()]
        if args:
            remapped.append(gate_name, qs, args)
        else:
            remapped.append(gate_name, qs)
    return remapped


def compile_stim_hybrid_ops(
    ansatz_circuit: cirq.Circuit,
    resolver: dict,
    qubits: list[cirq.Qid],
    noise_params: dict,
) -> list[tuple[str, Any]] | None:
    """Compile resolved noisy circuit into Stim hybrid ops.

    Returns ``None`` if Stim is unavailable or an unsupported op appears.
    Op kinds:
      - ``("tab", (tableau, targets))`` Clifford unitary
      - ``("depol", (q, p))`` single-qubit depolarize (Stim or NumPy)
      - ``("unitary", (matrix, targets))`` non-Clifford unitary (NumPy)
    """
    if stim is None:
        return None

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
                return None
            compiled.append(("depol", (qmap[op.qubits[0]], float(op.gate.p))))
            continue
        if not cirq.has_unitary(op):
            return None
        matrix = np.asarray(cirq.unitary(op), dtype=np.complex128)
        targets = [qmap[q] for q in op.qubits]
        tab = _try_clifford_tableau(matrix)
        if tab is not None:
            compiled.append(("tab", (tab, targets)))
        else:
            compiled.append(("unitary", (matrix, targets)))
    return compiled


def count_non_clifford_compiled(compiled: list[tuple[str, Any]]) -> int:
    return sum(1 for kind, _ in compiled if kind == "unitary")


def _append_basis_rotations(circuit: "stim.Circuit", pauli_str: str) -> None:
    """Append Clifford basis changes matching trajectory / DM backends."""
    for i, ch in enumerate(pauli_str):
        if ch == "X":
            circuit.append("H", [i])
        elif ch == "Y":
            # Rx(π/2) ≡ SQRT_X
            circuit.append("SQRT_X", [i])


def sample_fully_clifford_stim(
    compiled: list[tuple[str, Any]],
    pauli_str: str,
    n_qubits: int,
    num_shots: int,
    seed: int,
) -> np.ndarray:
    """Sample computational bitstrings for a fully Clifford noisy circuit."""
    assert stim is not None
    circuit = stim.Circuit()
    for kind, payload in compiled:
        if kind == "tab":
            tab, targets = payload
            circuit += _remap_tableau_circuit(tab, targets)
        elif kind == "depol":
            q, p = payload
            p = float(min(1.0, max(0.0, p)))
            if p > 0.0:
                circuit.append("DEPOLARIZE1", [int(q)], p)
        else:
            raise RuntimeError("fully Clifford sampler saw non-Clifford op")

    _append_basis_rotations(circuit, pauli_str)
    circuit.append("M", list(range(n_qubits)))
    samples = circuit.compile_sampler(seed=int(seed)).sample(int(num_shots))
    bits = np.asarray(samples, dtype=int)
    mask = np.array([ch != "I" for ch in pauli_str], dtype=bool)
    if np.any(~mask):
        bits = bits.copy()
        bits[:, ~mask] = 0
    return bits


def _evolve_hybrid_one_shot(
    compiled: list[tuple[str, Any]],
    n_qubits: int,
    noise_rng: np.random.Generator,
    *,
    apply_unitary,
    paulis,
) -> np.ndarray:
    """Evolve one noisy trajectory; return Cirq-layout statevector."""
    assert stim is not None
    sim = stim.TableauSimulator(seed=int(noise_rng.integers(0, 2**31 - 1)))
    sim.set_num_qubits(n_qubits)
    state: np.ndarray | None = None

    for kind, payload in compiled:
        if state is None:
            if kind == "tab":
                tab, targets = payload
                sim.do_tableau(tab, targets)
            elif kind == "depol":
                q, p = payload
                p = float(min(1.0, max(0.0, p)))
                if p > 0.0:
                    sim.depolarize1(int(q), p=p)
            elif kind == "unitary":
                state = np.asarray(sim.state_vector(endian="big"), dtype=np.complex128)
                matrix, targets = payload
                apply_unitary(state, matrix, targets, n_qubits)
            else:
                raise RuntimeError(kind)
        else:
            if kind == "tab":
                tab, targets = payload
                matrix = np.asarray(tab.to_unitary_matrix(endian="big"), dtype=np.complex128)
                apply_unitary(state, matrix, targets, n_qubits)
            elif kind == "depol":
                q, p = payload
                p = float(min(1.0, max(0.0, p)))
                probs = np.array([1.0 - p, p / 3.0, p / 3.0, p / 3.0], dtype=float)
                choice = int(noise_rng.choice(4, p=probs))
                if choice != 0:
                    apply_unitary(state, paulis[choice], [int(q)], n_qubits)
            elif kind == "unitary":
                matrix, targets = payload
                apply_unitary(state, matrix, targets, n_qubits)
            else:
                raise RuntimeError(kind)

    if state is None:
        state = np.asarray(sim.state_vector(endian="big"), dtype=np.complex128)
    return state


def sample_measurement_basis_stim_hybrid(
    compiled: list[tuple[str, Any]],
    pauli_str: str,
    n_qubits: int,
    num_shots: int,
    noise_rng: np.random.Generator,
    measure_rng: np.random.Generator,
    *,
    apply_unitary,
    sample_z_bitstring,
    paulis,
    basis_rotation_ops,
) -> np.ndarray:
    """Sample bitstrings using Stim hybrid evolution + Cirq-layout measurement."""
    n_non = count_non_clifford_compiled(compiled)
    if n_non == 0:
        seed = int(noise_rng.integers(0, 2**31 - 1))
        return sample_fully_clifford_stim(compiled, pauli_str, n_qubits, num_shots, seed)

    rot_ops = basis_rotation_ops(pauli_str, n_qubits)
    out = np.zeros((int(num_shots), n_qubits), dtype=int)
    mask = np.array([ch != "I" for ch in pauli_str], dtype=bool)
    for s in range(int(num_shots)):
        state = _evolve_hybrid_one_shot(
            compiled,
            n_qubits,
            noise_rng,
            apply_unitary=apply_unitary,
            paulis=paulis,
        )
        for kind, payload in rot_ops:
            matrix, targets = payload
            apply_unitary(state, matrix, targets, n_qubits)
        bits = sample_z_bitstring(state, n_qubits, measure_rng)
        if np.any(~mask):
            bits = bits.copy()
            bits[~mask] = 0
        out[s] = bits
    return out
