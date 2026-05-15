from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import cirq
import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from decompose_fsim_gate import (  # noqa: E402
    decompose_fsim_phi_only,
    decompose_fsim_theta_only,
    is_cz_plus_single_qubit,
)


def random_normalized_state_for_n_qubits(
    n_qubits: int, rng: np.random.Generator
) -> np.ndarray:
    dim = 2**n_qubits
    state = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    return state / np.linalg.norm(state)


def states_equal_up_to_global_phase(
    lhs: np.ndarray, rhs: np.ndarray, atol: float = 1e-7
) -> bool:
    return np.isclose(abs(np.vdot(lhs, rhs)), 1.0, atol=atol)


def choose_disjoint_adjacent_pairs(
    n_qubits: int, rng: np.random.Generator
) -> list[tuple[int, int]]:
    start = int(rng.integers(0, 2))
    return [(i, i + 1) for i in range(start, n_qubits - 1, 2)]


def build_original_and_decomposed_circuits(
    n_qubits: int,
    depth: int,
    rng: np.random.Generator,
    decompose_gate: Callable[[float, cirq.Qid, cirq.Qid], list[cirq.Operation]],
    make_original_gate: Callable[[float], cirq.Gate],
    angle_low: float = -np.pi,
    angle_high: float = np.pi,
) -> tuple[cirq.Circuit, cirq.Circuit, list[cirq.Qid]]:
    qubits = list(cirq.LineQubit.range(n_qubits))
    original = cirq.Circuit()
    decomposed = cirq.Circuit()

    for _ in range(depth):
        pairs = choose_disjoint_adjacent_pairs(n_qubits, rng)
        for left, right in pairs:
            q0 = qubits[left]
            q1 = qubits[right]
            angle = float(rng.uniform(angle_low, angle_high))
            original.append(make_original_gate(angle).on(q0, q1))
            decomposed.append(decompose_gate(angle, q0, q1))

    if not is_cz_plus_single_qubit(decomposed.all_operations()):
        raise AssertionError("Decomposed random circuit contains non CZ+1Q operations.")

    return original, decomposed, qubits


def build_mixed_original_and_decomposed_circuit(
    n_qubits: int,
    depth: int,
    rng: np.random.Generator,
    angle_low: float = -np.pi,
    angle_high: float = np.pi,
) -> tuple[cirq.Circuit, cirq.Circuit, list[cirq.Qid], int, int]:
    qubits = list(cirq.LineQubit.range(n_qubits))
    original = cirq.Circuit()
    decomposed = cirq.Circuit()
    theta_gate_count = 0
    phi_gate_count = 0

    for _ in range(depth):
        pairs = choose_disjoint_adjacent_pairs(n_qubits, rng)
        for left, right in pairs:
            q0 = qubits[left]
            q1 = qubits[right]
            angle = float(rng.uniform(angle_low, angle_high))

            if bool(rng.integers(0, 2)):
                original.append(cirq.FSimGate(theta=angle, phi=0.0).on(q0, q1))
                decomposed.append(decompose_fsim_theta_only(angle, q0, q1))
                theta_gate_count += 1
            else:
                original.append(cirq.FSimGate(theta=0.0, phi=angle).on(q0, q1))
                decomposed.append(decompose_fsim_phi_only(angle, q0, q1))
                phi_gate_count += 1

    if not is_cz_plus_single_qubit(decomposed.all_operations()):
        raise AssertionError("Decomposed mixed circuit contains non CZ+1Q operations.")

    return original, decomposed, qubits, theta_gate_count, phi_gate_count


def build_notebook_style_original_and_decomposed_circuit(
    n_qubits: int,
    ansatz_layers: int,
    rng: np.random.Generator,
    angle_low: float = -np.pi,
    angle_high: float = np.pi,
) -> tuple[cirq.Circuit, cirq.Circuit, list[cirq.Qid], int, int]:
    """Build random circuits with the same layer structure as prepare_original_fsim_ansatz_cirq.

    Structure per layer:
    1) even-odd theta-only couplers on both halves
    2) odd-even theta-only couplers on both halves
    3) onsite phi-only couplers between halves
    """
    if n_qubits % 2 != 0:
        raise ValueError(f"Notebook-style ansatz requires even n_qubits, got {n_qubits}.")
    num_spatial_orbitals = n_qubits // 2
    if num_spatial_orbitals < 2:
        raise ValueError(
            "Notebook-style ansatz requires at least 2 spatial orbitals (n_qubits >= 4)."
        )

    qubits = list(cirq.LineQubit.range(n_qubits))
    original = cirq.Circuit()
    decomposed = cirq.Circuit()
    theta_gate_count = 0
    phi_gate_count = 0

    for _ in range(ansatz_layers):
        even_odd_moments = []
        even_odd_decomp = []
        for i in range(0, num_spatial_orbitals - 1, 2):
            theta = float(rng.uniform(angle_low, angle_high))
            q_top0, q_top1 = qubits[i], qubits[i + 1]
            q_bot0, q_bot1 = qubits[i + num_spatial_orbitals], qubits[i + 1 + num_spatial_orbitals]
            even_odd_moments.append(cirq.FSimGate(theta, 0.0).on(q_top0, q_top1))
            even_odd_moments.append(cirq.FSimGate(theta, 0.0).on(q_bot0, q_bot1))
            even_odd_decomp.extend(decompose_fsim_theta_only(theta, q_top0, q_top1))
            even_odd_decomp.extend(decompose_fsim_theta_only(theta, q_bot0, q_bot1))
            theta_gate_count += 2
        original.append(even_odd_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        decomposed.append(even_odd_decomp, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        odd_even_moments = []
        odd_even_decomp = []
        for i in range(1, num_spatial_orbitals - 1, 2):
            theta = float(rng.uniform(angle_low, angle_high))
            q_top0, q_top1 = qubits[i], qubits[i + 1]
            q_bot0, q_bot1 = qubits[i + num_spatial_orbitals], qubits[i + 1 + num_spatial_orbitals]
            odd_even_moments.append(cirq.FSimGate(theta, 0.0).on(q_top0, q_top1))
            odd_even_moments.append(cirq.FSimGate(theta, 0.0).on(q_bot0, q_bot1))
            odd_even_decomp.extend(decompose_fsim_theta_only(theta, q_top0, q_top1))
            odd_even_decomp.extend(decompose_fsim_theta_only(theta, q_bot0, q_bot1))
            theta_gate_count += 2
        original.append(odd_even_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        decomposed.append(odd_even_decomp, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

        onsite_moments = []
        onsite_decomp = []
        for i in range(num_spatial_orbitals):
            phi = float(rng.uniform(angle_low, angle_high))
            q_top = qubits[i]
            q_bot = qubits[i + num_spatial_orbitals]
            onsite_moments.append(cirq.FSimGate(0.0, phi).on(q_top, q_bot))
            onsite_decomp.extend(decompose_fsim_phi_only(phi, q_top, q_bot))
            phi_gate_count += 1
        original.append(onsite_moments, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
        decomposed.append(onsite_decomp, strategy=cirq.InsertStrategy.NEW_THEN_INLINE)

    if not is_cz_plus_single_qubit(decomposed.all_operations()):
        raise AssertionError("Notebook-style decomposed circuit contains non CZ+1Q operations.")

    return original, decomposed, qubits, theta_gate_count, phi_gate_count


def test_fsim_decomposition_random_circuits(
    min_qubits: int = 4,
    max_qubits: int = 10,
    max_depth: int = 10,
    circuits_per_config: int = 6,
    trials_per_circuit: int = 4,
    atol: float = 1e-7,
    seed: int = 12345,
) -> None:
    rng = np.random.default_rng(seed)
    sim = cirq.Simulator()

    for n_qubits in range(min_qubits, max_qubits + 1):
        for depth in range(1, max_depth + 1):
            for _ in range(circuits_per_config):
                theta_original, theta_decomposed, qubits = build_original_and_decomposed_circuits(
                    n_qubits=n_qubits,
                    depth=depth,
                    rng=rng,
                    decompose_gate=decompose_fsim_theta_only,
                    make_original_gate=lambda angle: cirq.FSimGate(theta=angle, phi=0.0),
                )
                phi_original, phi_decomposed, _ = build_original_and_decomposed_circuits(
                    n_qubits=n_qubits,
                    depth=depth,
                    rng=rng,
                    decompose_gate=decompose_fsim_phi_only,
                    make_original_gate=lambda angle: cirq.FSimGate(theta=0.0, phi=angle),
                )

                for _ in range(trials_per_circuit):
                    initial_state = random_normalized_state_for_n_qubits(n_qubits, rng)

                    theta_result_original = sim.simulate(
                        theta_original,
                        qubit_order=qubits,
                        initial_state=initial_state.copy(),
                    ).final_state_vector
                    theta_result_decomposed = sim.simulate(
                        theta_decomposed,
                        qubit_order=qubits,
                        initial_state=initial_state.copy(),
                    ).final_state_vector
                    if not states_equal_up_to_global_phase(
                        theta_result_original, theta_result_decomposed, atol=atol
                    ):
                        raise AssertionError(
                            f"theta-only mismatch for n_qubits={n_qubits}, depth={depth}."
                        )

                    phi_result_original = sim.simulate(
                        phi_original,
                        qubit_order=qubits,
                        initial_state=initial_state.copy(),
                    ).final_state_vector
                    phi_result_decomposed = sim.simulate(
                        phi_decomposed,
                        qubit_order=qubits,
                        initial_state=initial_state.copy(),
                    ).final_state_vector
                    if not states_equal_up_to_global_phase(
                        phi_result_original, phi_result_decomposed, atol=atol
                    ):
                        raise AssertionError(
                            f"phi-only mismatch for n_qubits={n_qubits}, depth={depth}."
                        )


def test_mixed_fsim_decomposition_random_circuits(
    n_qubits: int = 10,
    depth: int = 5,
    circuits: int = 5,
    trials_per_circuit: int = 4,
    atol: float = 1e-7,
    seed: int = 12345,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    sim = cirq.Simulator()
    total_theta = 0
    total_phi = 0

    for _ in range(circuits):
        original, decomposed, qubits, theta_count, phi_count = (
            build_mixed_original_and_decomposed_circuit(
                n_qubits=n_qubits,
                depth=depth,
                rng=rng,
            )
        )
        total_theta += theta_count
        total_phi += phi_count

        for _ in range(trials_per_circuit):
            initial_state = random_normalized_state_for_n_qubits(n_qubits, rng)
            result_original = sim.simulate(
                original,
                qubit_order=qubits,
                initial_state=initial_state.copy(),
            ).final_state_vector
            result_decomposed = sim.simulate(
                decomposed,
                qubit_order=qubits,
                initial_state=initial_state.copy(),
            ).final_state_vector
            if not states_equal_up_to_global_phase(
                result_original, result_decomposed, atol=atol
            ):
                raise AssertionError(
                    f"mixed mismatch for n_qubits={n_qubits}, depth={depth}."
                )

    return total_theta, total_phi


def test_notebook_style_random_fsim_decomposition(
    n_qubits: int = 8,
    ansatz_layers: int = 5,
    circuits: int = 5,
    trials_per_circuit: int = 4,
    atol: float = 1e-7,
    seed: int = 12345,
) -> tuple[int, int]:
    rng = np.random.default_rng(seed)
    sim = cirq.Simulator()
    total_theta = 0
    total_phi = 0

    for _ in range(circuits):
        original, decomposed, qubits, theta_count, phi_count = (
            build_notebook_style_original_and_decomposed_circuit(
                n_qubits=n_qubits,
                ansatz_layers=ansatz_layers,
                rng=rng,
            )
        )
        total_theta += theta_count
        total_phi += phi_count

        for _ in range(trials_per_circuit):
            initial_state = random_normalized_state_for_n_qubits(n_qubits, rng)
            result_original = sim.simulate(
                original,
                qubit_order=qubits,
                initial_state=initial_state.copy(),
            ).final_state_vector
            result_decomposed = sim.simulate(
                decomposed,
                qubit_order=qubits,
                initial_state=initial_state.copy(),
            ).final_state_vector
            if not states_equal_up_to_global_phase(result_original, result_decomposed, atol=atol):
                raise AssertionError(
                    f"notebook-style mismatch for n_qubits={n_qubits}, ansatz_layers={ansatz_layers}."
                )

    return total_theta, total_phi


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test analytical FSim decompositions on random multi-qubit circuits."
    )
    parser.add_argument("--min-qubits", type=int, default=2, help="Minimum qubit count")
    parser.add_argument("--max-qubits", type=int, default=5, help="Maximum qubit count")
    parser.add_argument("--max-depth", type=int, default=8, help="Maximum random depth")
    parser.add_argument(
        "--circuits-per-config",
        type=int,
        default=6,
        help="Random circuits per (qubit_count, depth)",
    )
    parser.add_argument(
        "--trials-per-circuit",
        type=int,
        default=4,
        help="Random states tested per random circuit",
    )
    parser.add_argument("--atol", type=float, default=1e-7, help="Abs tolerance")
    parser.add_argument("--seed", type=int, default=12345, help="RNG seed")
    parser.add_argument(
        "--mixed",
        action="store_true",
        help="Run mixed theta-only/phi-only FSim random-circuit test",
    )
    parser.add_argument(
        "--mixed-circuits",
        type=int,
        default=5,
        help="Number of mixed random circuits",
    )
    parser.add_argument(
        "--ansatz-qubits",
        type=int,
        default=8,
        help="Qubit count for notebook-style ansatz random test (must be even)",
    )
    parser.add_argument(
        "--ansatz-layers",
        type=int,
        default=5,
        help="Layer count for notebook-style ansatz random test",
    )
    parser.add_argument(
        "--ansatz-circuits",
        type=int,
        default=5,
        help="Number of notebook-style random ansatz circuits",
    )
    args = parser.parse_args()

    if args.mixed:
        theta_total, phi_total = test_mixed_fsim_decomposition_random_circuits(
            n_qubits=args.max_qubits,
            depth=args.max_depth,
            circuits=args.mixed_circuits,
            trials_per_circuit=args.trials_per_circuit,
            atol=args.atol,
            seed=args.seed,
        )
        print("PASS: Mixed theta-only/phi-only FSim circuits match decomposition.")
        print(f"Qubits: {args.max_qubits}")
        print(f"Theta-only FSim gates: {theta_total}")
        print(f"Phi-only FSim gates: {phi_total}")
        print(f"Total FSim gates: {theta_total + phi_total}")
    else:
        test_fsim_decomposition_random_circuits(
            min_qubits=args.min_qubits,
            max_qubits=args.max_qubits,
            max_depth=args.max_depth,
            circuits_per_config=args.circuits_per_config,
            trials_per_circuit=args.trials_per_circuit,
            atol=args.atol,
            seed=args.seed,
        )
        print(
            "PASS: Random multi-qubit/depth circuits with theta-only and phi-only FSim "
            "match their analytical CZ+1Q decompositions up to global phase."
        )

    theta_total, phi_total = test_notebook_style_random_fsim_decomposition(
        n_qubits=args.ansatz_qubits,
        ansatz_layers=args.ansatz_layers,
        circuits=args.ansatz_circuits,
        trials_per_circuit=args.trials_per_circuit,
        atol=args.atol,
        seed=args.seed,
    )
    print("PASS: Notebook-style random FSim ansatz circuits match decomposition.")
    print(f"Notebook-style qubits: {args.ansatz_qubits}")
    print(f"Notebook-style ansatz layers: {args.ansatz_layers}")
    print(f"Notebook-style theta-only FSim gates: {theta_total}")
    print(f"Notebook-style phi-only FSim gates: {phi_total}")
    print(f"Notebook-style total FSim gates: {theta_total + phi_total}")


if __name__ == "__main__":
    main()
