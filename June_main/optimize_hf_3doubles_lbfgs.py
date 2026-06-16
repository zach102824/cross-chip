#!/usr/bin/env python3
"""Optimize the HF (8-qubit, 3-doubles) cross-chip ansatz noiselessly with L-BFGS-B.

Matches the circuit about to be simulated in ``June_main/main.ipynb``:

  * bare ansatz loaded from
    ``June_main/circuits2read/HF_8q_3doubles_decomposed_simplified.json``
    (three parameters ``t0``, ``t1``, ``t2``; cross-chip CZs decomposed to RZX),
  * Hartree-Fock determinant as the initial state (X on the occupied
    spin-orbitals of the 6e/4o active space),
  * Hamiltonian from ``Pauli_Ham/HF_bond_1.4.txt``.

A noiseless ``cirq.Simulator`` evaluates ``<H>``; L-BFGS-B minimizes it over the
three angles. The script reports the optimized energy and how close it gets to
the exact ground-state energy.

Example::

    python optimize_hf_3doubles_lbfgs.py
    python optimize_hf_3doubles_lbfgs.py --bond 1.4 --restarts 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cirq
import numpy as np
import sympy
from scipy.linalg import expm as _expm
from scipy.optimize import minimize

REPO_ROOT = Path(__file__).resolve().parents[1]
JUNE_DIR = Path(__file__).resolve().parent
for _p in (REPO_ROOT, JUNE_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Active space for the HF molecule (must match UCCSD_Mole/HF.ipynb and main.ipynb).
N_ACTIVE_ELECTRONS = 6
N_SPATIAL_ORBITALS = 4
N_QUBITS = 2 * N_SPATIAL_ORBITALS
ETA = N_ACTIVE_ELECTRONS // 2

CIRCUIT_NAME = "HF_8q_3doubles_decomposed_simplified"
CZ_CROSS_CHIP_TAG = "cz_cross_chip"


def _rzx_unitary(theta: float) -> np.ndarray:
    """RZX(theta) = exp(-i*theta/2 * Z⊗X) as a 4x4 unitary (qubit0=Z, qubit1=X)."""
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Xm = np.array([[0, 1], [1, 0]], dtype=complex)
    return _expm(-0.5j * float(theta) * np.kron(Z, Xm))


def load_circuit_from_json(path: Path):
    """Build a cirq.Circuit (LineQubit layout) from a saved UCCSD circuit JSON.

    Returns (circuit, qubits, symbols, meta). Same loader as ``main.ipynb``.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    n = int(data["num_qubits"])
    q = cirq.LineQubit.range(n)
    param_names = list(data["param_names"])
    sym = {name: sympy.Symbol(f"th_{i}") for i, name in enumerate(param_names)}
    syms = [sym[name] for name in param_names]

    c = cirq.Circuit()
    for g in data["gates"]:
        op = g["op"]
        qs = [q[i] for i in g["qubits"]]
        if op == "h":
            c.append(cirq.H(qs[0]))
        elif op == "x":
            c.append(cirq.X(qs[0]))
        elif op == "ry":
            c.append(cirq.ry(float(g["value"])).on(qs[0]))
        elif op == "cx":
            c.append(cirq.CNOT(qs[0], qs[1]))
        elif op == "cz":
            cz = cirq.CZ(qs[0], qs[1])
            if g.get("cross_chip"):
                cz = cz.with_tags(CZ_CROSS_CHIP_TAG)
            c.append(cz)
        elif op == "rzx":
            rzx_op = cirq.MatrixGate(
                _rzx_unitary(float(g.get("angle", g.get("value", 0.0)))), name="RZX"
            ).on(qs[0], qs[1])
            if g.get("cross_chip"):
                rzx_op = rzx_op.with_tags(CZ_CROSS_CHIP_TAG)
            c.append(rzx_op)
        elif op in ("rx", "rz"):
            angle = float(g["coeff"]) * sym[g["param"]] if "param" in g else float(g["value"])
            gate = cirq.rx if op == "rx" else cirq.rz
            c.append(gate(angle).on(qs[0]))
        else:
            raise ValueError(f"Unhandled op in circuit JSON: {op!r}")
    return c, list(q), syms, data


def load_pauli_sum_from_numbered_file(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    idx_to_pauli = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    out = cirq.PauliSum()

    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
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


def hf_prep_circuit(qubits: list[cirq.Qid]) -> cirq.Circuit:
    """Hartree-Fock determinant: X on each occupied spin-orbital of the active space."""
    half = N_SPATIAL_ORBITALS
    occupied = list(range(ETA)) + list(range(half, half + ETA))
    return cirq.Circuit([cirq.X(qubits[k]) for k in occupied])


def expectation_energy(
    pauli_sum: cirq.PauliSum,
    qubits: list[cirq.Qid],
    circuit: cirq.Circuit,
    simulator: cirq.Simulator,
) -> float:
    result = simulator.simulate(circuit, qubit_order=qubits)
    psi = np.asarray(result.final_state_vector, dtype=np.complex128)
    qubit_map = {q: i for i, q in enumerate(qubits)}
    return float(np.real(pauli_sum.expectation_from_state_vector(psi, qubit_map=qubit_map)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bond", type=float, default=1.4, help="H–F bond length (Å)")
    parser.add_argument("--restarts", type=int, default=8, help="random restarts (plus the all-zero start)")
    parser.add_argument("--maxiter", type=int, default=400, help="L-BFGS-B max iterations")
    parser.add_argument("--ftol", type=float, default=1e-12, help="L-BFGS-B ftol")
    parser.add_argument("--gtol", type=float, default=1e-9, help="L-BFGS-B gtol")
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed for restarts")
    args = parser.parse_args()

    circuit_path = JUNE_DIR / "circuits2read" / f"{CIRCUIT_NAME}.json"
    if not circuit_path.is_file():
        raise FileNotFoundError(f"Circuit JSON not found: {circuit_path}")
    ansatz, qubits, symbols, meta = load_circuit_from_json(circuit_path)
    n_params = len(symbols)

    assert len(qubits) == N_QUBITS, (
        f"Circuit has {len(qubits)} qubits but active space implies N_QUBITS={N_QUBITS}."
    )

    # Full circuit = HF prep + bare ansatz (matches main.ipynb INIT_STATE_METHOD='hf').
    full_circuit = hf_prep_circuit(qubits) + ansatz

    ham_path = REPO_ROOT / "Pauli_Ham" / f"HF_bond_{args.bond:.1f}.txt"
    if not ham_path.is_file():
        raise FileNotFoundError(f"Hamiltonian file not found: {ham_path}")
    pauli_sum = load_pauli_sum_from_numbered_file(ham_path, list(qubits))
    qubit_map = {q: i for i, q in enumerate(qubits)}

    # Exact ground-state energy via dense diagonalization.
    e_gs = float(np.linalg.eigvalsh(pauli_sum.matrix(qubits=qubits))[0].real)

    # Double precision is REQUIRED: cirq's default simulator is complex64, whose
    # ~1e-7 noise floor is larger than the L-BFGS-B finite-difference step, making
    # the numerical gradient collapse to zero (optimizer never leaves theta=0).
    sim = cirq.Simulator(dtype=np.complex128)
    psi_hf = np.asarray(
        sim.simulate(hf_prep_circuit(qubits), qubit_order=qubits).final_state_vector,
        dtype=np.complex128,
    )
    e_hf_ref = float(np.real(pauli_sum.expectation_from_state_vector(psi_hf, qubit_map=qubit_map)))

    def objective(x: np.ndarray) -> float:
        resolver = cirq.ParamResolver({symbols[i]: float(x[i]) for i in range(n_params)})
        resolved = cirq.resolve_parameters(full_circuit, resolver)
        return expectation_energy(pauli_sum, qubits, resolved, sim)

    print(f"molecule: HF   bond length: {args.bond} Å   (active space: "
          f"{N_ACTIVE_ELECTRONS}e, {N_SPATIAL_ORBITALS}o -> {N_QUBITS} qubits)")
    print(f"circuit source     : {circuit_path}")
    print(f"Hamiltonian source : {ham_path}")
    print(f"initial state      : Hartree-Fock determinant")
    print(f"free parameters    : {n_params}  ({', '.join(meta['param_names'])})")
    print(f"⟨H⟩ HF determinant : {e_hf_ref:.10f} Eh")
    print(f"exact ground state : {e_gs:.10f} Eh")
    print()

    rng = np.random.default_rng(args.seed)
    starts = [np.zeros(n_params, dtype=float)]
    for _ in range(max(0, args.restarts)):
        starts.append(rng.uniform(-np.pi, np.pi, size=n_params))

    best = None
    for i, x0 in enumerate(starts):
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            options={"maxiter": args.maxiter, "ftol": args.ftol, "gtol": args.gtol},
        )
        tag = "zero-start" if i == 0 else f"restart {i}"
        print(f"  [{tag:>10}] E = {float(result.fun):.10f} Eh   (nit={result.nit}, success={result.success})")
        if best is None or float(result.fun) < best.fun:
            best = result

    xf = np.asarray(best.x, dtype=float)
    ef = float(best.fun)
    gap = ef - e_gs

    print()
    print(f"optimized θ        : [{', '.join(f'{v:.10f}' for v in xf)}]")
    print(f"optimized ⟨H⟩      : {ef:.10f} Eh")
    print(f"exact ground state : {e_gs:.10f} Eh")
    print(f"energy gap to GS   : {gap:.3e} Eh  ({gap * 1e3:.6f} mEh)")
    if abs(e_hf_ref - e_gs) > 1e-12:
        recovered = 100.0 * (e_hf_ref - ef) / (e_hf_ref - e_gs)
        print(f"correlation energy recovered vs HF: {recovered:.4f}%")
    chem_acc = 1.6e-3  # chemical accuracy (1 kcal/mol ≈ 0.0016 Eh)
    print(f"within chemical accuracy (<{chem_acc} Eh): {gap < chem_acc}")


if __name__ == "__main__":
    main()
