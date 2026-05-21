#!/usr/bin/env python3
"""Optimize LiH compiled FIG.13 ansatz (three RX angles) with L-BFGS-B.

Circuit and Hamiltonian match ``lih_fig13_compiled_ansatz.ipynb``: three independent
``theta1``, ``theta2``, ``theta3`` on ``LineQubit(1)``; Pauli sum from
``Pauli_Ham/LiH_bond_<bond>.txt``.

Example::

    python optimize_lih_fig13_lbfgs.py --bond 1.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cirq
import numpy as np
import sympy
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, CASE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

def load_pauli_sum_from_numbered_file(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    idx_to_pauli = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    out = cirq.PauliSum()

    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            coeff = float(parts[0])
            pauli_codes = [int(x) for x in parts[1:]]
            if len(pauli_codes) != len(qubits):
                raise ValueError(
                    f"{path}:{lineno} has {len(pauli_codes)} Pauli indices, expected {len(qubits)}."
                )

            pauli_string = cirq.PauliString()
            for q, code in zip(qubits, pauli_codes):
                if code == 0:
                    continue
                if code not in idx_to_pauli:
                    raise ValueError(
                        f"{path}:{lineno} has invalid Pauli code {code}; expected 0/1/2/3."
                    )
                pauli_string *= idx_to_pauli[code](q)

            out += coeff * pauli_string

    return out


def lih_fig13_circuit(
    theta1: sympy.Symbol,
    theta2: sympy.Symbol,
    theta3: sympy.Symbol,
) -> tuple[cirq.Circuit, list[cirq.LineQubit]]:
    """Compiled LiH ansatz (FIG. 13): three independent RX angles on q[1].

    Same gate order as ``lih_fig13_compiled_ansatz.ipynb`` (multi-reference prep, not bare HF).
    """

    q = cirq.LineQubit.range(6)
    q0, q1, q2, q3, q4, q5 = q
    c = cirq.Circuit()
    c.append(cirq.ry(-0.1).on(q0))
    c.append(cirq.H(q1))
    c.append(cirq.CZ(q0, q1))
    c.append(cirq.H(q1))
    c.append(cirq.H(q4))
    c.append(cirq.CZ(q1, q4))
    c.append(cirq.H(q4))
    c.append(cirq.H(q3))
    c.append(cirq.CZ(q4, q3))
    c.append(cirq.H(q3))
    c.append(cirq.X(q0))
    c.append(cirq.X(q3))
    c.append(cirq.identity_each(*q))  # visual barrier: multi-ref prep | FIG. 13 ansatz

    c.append(cirq.rx(np.pi / 2).on(q0))
    c.append(cirq.H(q2))
    c.append(cirq.H(q3))
    c.append(cirq.H(q4))
    c.append(cirq.H(q5))

    c.append([cirq.CZ(q0, q1), cirq.CZ(q3, q4)])
    c.append([cirq.CZ(q4, q5), cirq.H(q4)])

    c.append(cirq.CZ(q1, q4))
    c.append(cirq.rx(theta1).on(q1))
    c.append(cirq.CZ(q1, q4))

    c.append(cirq.CZ(q0, q1))
    c.append(cirq.H(q1))
    c.append(cirq.CZ(q0, q1))
    c.append(cirq.CZ(q1, q2))

    c.append(cirq.CZ(q1, q4))
    c.append(cirq.rx(theta2).on(q1))
    c.append(cirq.CZ(q1, q4))

    c.append(cirq.H(q4))
    c.append(cirq.CZ(q4, q5))
    c.append(cirq.H(q5))
    c.append(cirq.CZ(q3, q4))
    c.append(cirq.H(q4))
    c.append(cirq.CZ(q3, q4))
    c.append(cirq.H(q4))

    c.append(cirq.CZ(q1, q4))
    c.append(cirq.rx(theta3).on(q1))
    c.append(cirq.CZ(q1, q4))

    c.append(cirq.H(q4))
    c.append(cirq.CZ(q1, q2))
    c.append(cirq.H(q2))
    c.append(cirq.CZ(q0, q1))
    c.append(cirq.CZ(q3, q4))
    c.append(cirq.rx(-np.pi / 2).on(q0))
    c.append([cirq.H(q1), cirq.H(q3)])

    return c, q


def expectation_energy(
    pauli_sum: cirq.PauliSum,
    qubits: list[cirq.LineQubit],
    circuit: cirq.Circuit,
) -> float:
    sim = cirq.Simulator()
    result = sim.simulate(circuit, qubit_order=qubits)
    psi = np.asarray(result.final_state_vector, dtype=np.complex128)
    qubit_map = {q: i for i, q in enumerate(qubits)}
    return float(np.real(pauli_sum.expectation_from_state_vector(psi, qubit_map=qubit_map)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=float, default=1.5, help="Li–H bond length (Å)")
    parser.add_argument("--maxiter", type=int, default=200, help="L-BFGS-B max iterations")
    parser.add_argument("--ftol", type=float, default=1e-12, help="L-BFGS-B ftol")
    parser.add_argument("--gtol", type=float, default=1e-8, help="L-BFGS-B gtol")
    args = parser.parse_args()

    theta1 = sympy.Symbol("theta1")
    theta2 = sympy.Symbol("theta2")
    theta3 = sympy.Symbol("theta3")
    circuit_template, qubits = lih_fig13_circuit(theta1, theta2, theta3)

    ham_path = REPO_ROOT / "Pauli_Ham" / f"LiH_bond_{args.bond:.1f}.txt"
    if not ham_path.is_file():
        raise FileNotFoundError(f"Hamiltonian file not found: {ham_path}")
    pauli_sum = load_pauli_sum_from_numbered_file(ham_path, list(qubits))

    x0 = np.array([0.0, 0.0, 0.0], dtype=float)

    def objective(x: np.ndarray) -> float:
        resolver = cirq.ParamResolver(
            {theta1: float(x[0]), theta2: float(x[1]), theta3: float(x[2])}
        )
        resolved = cirq.resolve_parameters(circuit_template, resolver)
        return expectation_energy(pauli_sum, qubits, resolved)

    e0 = objective(x0)
    print(f"bond length: {args.bond} Å")
    print(f"Hamiltonian source: {ham_path}")
    print(f"initial θ = [{x0[0]:.10f}, {x0[1]:.10f}, {x0[2]:.10f}]")
    print(f"initial ⟨H⟩ = {e0:.12f} Eh")

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        options={"maxiter": args.maxiter, "ftol": args.ftol, "gtol": args.gtol},
    )

    xf = np.asarray(result.x, dtype=float)
    ef = float(result.fun)
    print(f"\noptimized θ = [{xf[0]:.12f}, {xf[1]:.12f}, {xf[2]:.12f}]")
    print(f"optimized ⟨H⟩ = {ef:.12f} Eh")
    print(f"success={result.success}  status={result.status}  nit={result.nit}")
    print(f"message: {result.message}")


if __name__ == "__main__":
    main()
