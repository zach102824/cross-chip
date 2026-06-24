#!/usr/bin/env python3
"""Noiseless verification of a saved Cl2 UCCSD-style circuit.

What this script does
----------------------
1. Reads a hardware circuit JSON (default:
   ``June_main/circuits2read/Cl2_12q_5doubles_hardware.json``) and rebuilds it
   as a parameterized ``cirq.Circuit`` on ``LineQubit`` wires.
2. Prepends the Hartree-Fock reference state: an ``X`` on every occupied
   spin-orbital, i.e. all qubits set to ``1`` *except the highest spatial
   orbital in each spin sector* (qubits ``n_spatial-1`` and ``2*n_spatial-1``).
3. Loads the qubit Hamiltonian from the numbered-Pauli text file
   (default ``Pauli_Ham/Cl2_bond_<bond>.txt``) and minimizes
   ``<psi(theta)|H|psi(theta)>`` with a noiseless statevector simulator using
   SciPy's BFGS optimizer.
4. Diagonalizes the same Hamiltonian to obtain the exact ground-state energy
   AND eigenvector, computes the fidelity of the optimized state with that
   eigenvector, and cross-checks the eigenvalue against the published
   ``Cl2_bond_scan_summary.txt`` value.
5. Saves the optimized thetas, final energy, and fidelity next to this script.

Everything is exact / noiseless: the energy comes straight from the
statevector expectation value (no OGM, no shots, no error mitigation).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cirq
import numpy as np
import sympy
from scipy.optimize import minimize

# Numbered-Pauli convention used across the repo: 0=I, 1=X, 2=Y, 3=Z.
IDX_TO_PAULI = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Circuit loading
# ---------------------------------------------------------------------------
def load_circuit_from_json(
    path: Path,
) -> tuple[cirq.Circuit, list[cirq.LineQubit], list[sympy.Symbol], dict]:
    """Build the bare ansatz (no state prep) from a saved circuit JSON.

    Parameterized ``rz`` gates reference a name in ``param_names``; each such
    name becomes one free sympy symbol so the optimizer has one knob per name.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    num_qubits = int(data["num_qubits"])
    qubits = [cirq.LineQubit(i) for i in range(num_qubits)]

    param_names = list(data["param_names"])
    sym = {name: sympy.Symbol(name) for name in param_names}
    symbols = [sym[name] for name in param_names]

    circuit = cirq.Circuit()
    for g in data["gates"]:
        op = g["op"]
        qs = [qubits[i] for i in g["qubits"]]
        if op == "h":
            circuit.append(cirq.H(qs[0]))
        elif op == "x":
            circuit.append(cirq.X(qs[0]))
        elif op == "s":
            circuit.append(cirq.S(qs[0]))
        elif op == "sdg":
            circuit.append(cirq.S(qs[0]) ** -1)
        elif op == "cx":
            circuit.append(cirq.CNOT(qs[0], qs[1]))
        elif op == "cz":
            circuit.append(cirq.CZ(qs[0], qs[1]))
        elif op in ("rx", "ry", "rz"):
            if "param" in g:
                angle = float(g.get("coeff", 1.0)) * sym[g["param"]]
            else:
                angle = float(g.get("value", g.get("angle", 0.0)))
            gate = {"rx": cirq.rx, "ry": cirq.ry, "rz": cirq.rz}[op]
            circuit.append(gate(angle).on(qs[0]))
        else:
            raise ValueError(f"Unhandled op in circuit JSON: {op!r}")

    return circuit, qubits, symbols, data


def hf_occupied_orbitals(n_spatial: int, n_electrons: int) -> list[int]:
    """Occupied spin-orbital (qubit) indices for the HF determinant.

    Qubits are in spin-block order ``[alpha 0..n-1, beta 0..n-1]``. With
    ``eta = n_electrons // 2`` electrons per spin sector, the lowest ``eta``
    orbitals are filled in each block, leaving the highest spatial orbital of
    each spin sector (qubits ``n_spatial-1`` and ``2*n_spatial-1``) empty.
    """
    eta = n_electrons // 2
    if eta > n_spatial:
        raise ValueError(
            f"eta={eta} electrons/spin exceeds n_spatial={n_spatial} orbitals."
        )
    alpha = list(range(eta))
    beta = list(range(n_spatial, n_spatial + eta))
    return alpha + beta


def hf_prep_circuit(qubits: list[cirq.LineQubit], occupied: list[int]) -> cirq.Circuit:
    """HF determinant: an X on each occupied spin-orbital qubit."""
    return cirq.Circuit([cirq.X(qubits[k]) for k in occupied])


# ---------------------------------------------------------------------------
# Hamiltonian loading
# ---------------------------------------------------------------------------
def load_pauli_sum_from_numbered_file(
    path: Path, qubits: list[cirq.Qid]
) -> cirq.PauliSum:
    """Parse a numbered-Pauli Hamiltonian: ``coeff p0 p1 ... p_{n-1}`` per line."""
    out = cirq.PauliSum()
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            coeff = float(parts[0])
            pauli_codes = [int(x) for x in parts[1:]]
            if len(pauli_codes) != len(qubits):
                raise ValueError(
                    f"{path}:{lineno} has {len(pauli_codes)} Pauli codes; "
                    f"expected {len(qubits)}."
                )
            pauli_string = cirq.PauliString()
            for q, code in zip(qubits, pauli_codes):
                if code == 0:
                    continue
                if code not in IDX_TO_PAULI:
                    raise ValueError(
                        f"{path}:{lineno} invalid Pauli code {code}; expected 0/1/2/3."
                    )
                pauli_string *= IDX_TO_PAULI[code](q)
            out += coeff * pauli_string
    return out


def bond_token(bond: float) -> str:
    """Render a bond length the way the Hamiltonian filenames do (2.2 -> '2.2')."""
    return f"{bond:.10g}".rstrip("0").rstrip(".") if isinstance(bond, float) else str(bond)


def read_summary_gs_energy(summary_path: Path, bond: float) -> float | None:
    """Return E_GS_Ha for ``bond`` from the bond-scan summary, or None."""
    if not summary_path.exists():
        return None
    token = bond_token(bond)
    for raw in summary_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("bond_angstrom"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if bond_token(float(parts[0])) == token:
            try:
                return float(parts[2])
            except ValueError:
                return None
    return None


# ---------------------------------------------------------------------------
# Main verification routine
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bond-length",
        type=float,
        default=2.2,
        help="Bond length in Angstrom (selects the Hamiltonian file). Default 2.2.",
    )
    parser.add_argument(
        "--circuit",
        type=Path,
        default=REPO_ROOT / "June_main/circuits2read/Cl2_12q_5doubles_hardware.json",
        help="Path to the hardware circuit JSON.",
    )
    parser.add_argument(
        "--hamiltonian",
        type=Path,
        default=None,
        help="Override Hamiltonian text path. Default Pauli_Ham/Cl2_bond_<bond>.txt.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "Pauli_Ham/Cl2_bond_scan_summary.txt",
        help="Bond-scan summary used to cross-check the exact ground-state energy.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=2000, help="BFGS max iterations. Default 2000."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for any random initial-parameter noise (0 -> exact HF start).",
    )
    parser.add_argument(
        "--init-kick",
        type=float,
        default=0.5,
        help="Std of the random kick added to the all-zero (HF) start when "
        "--seed != 0. Default 0.5 rad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR,
        help="Where to save the results JSON. Defaults to this script's folder.",
    )
    args = parser.parse_args()

    bond = float(args.bond_length)
    ham_path = args.hamiltonian or (
        REPO_ROOT / "Pauli_Ham" / f"Cl2_bond_{bond_token(bond)}.txt"
    )

    # --- Load circuit + HF prep ------------------------------------------------
    ansatz, qubits, symbols, meta = load_circuit_from_json(args.circuit)
    n_spatial = int(meta["n_spatial"])
    n_electrons = int(meta["n_electrons"])
    occupied = hf_occupied_orbitals(n_spatial, n_electrons)
    prep = hf_prep_circuit(qubits, occupied)
    full_circuit = prep + ansatz

    qubit_map = {q: i for i, q in enumerate(qubits)}
    n_params = len(symbols)

    print(f"Circuit       : {args.circuit}")
    print(f"  qubits={len(qubits)}, params={n_params}, "
          f"n_spatial={n_spatial}, n_electrons={n_electrons}")
    print(f"  HF occupied spin-orbitals: {occupied}")
    unoccupied = [q for q in range(len(qubits)) if q not in occupied]
    print(f"  HF empty spin-orbitals    : {unoccupied} "
          f"(highest orbital per spin sector)")
    print(f"Hamiltonian   : {ham_path}")

    # --- Hamiltonian + exact diagonalization ----------------------------------
    pauli_sum = load_pauli_sum_from_numbered_file(ham_path, qubits)
    h_matrix = pauli_sum.matrix(qubits=qubits)
    eigvals, eigvecs = np.linalg.eigh(h_matrix)
    e_gs_exact = float(eigvals[0].real)
    gs_vector = eigvecs[:, 0].astype(np.complex128)

    e_gs_summary = read_summary_gs_energy(args.summary, bond)

    # --- Noiseless statevector energy objective -------------------------------
    simulator = cirq.Simulator(dtype=np.complex128)

    def state_vector(theta: np.ndarray) -> np.ndarray:
        resolver = {symbols[i]: float(theta[i]) for i in range(n_params)}
        resolved = cirq.resolve_parameters(full_circuit, resolver)
        return simulator.simulate(resolved, qubit_order=qubits).final_state_vector

    def energy(theta: np.ndarray) -> float:
        psi = state_vector(theta).astype(np.complex128)
        return float(
            np.real(
                pauli_sum.expectation_from_state_vector(psi, qubit_map=qubit_map)
            )
        )

    # Start from the HF state (all theta = 0). Optional tiny random kick via seed.
    if args.seed:
        rng = np.random.default_rng(args.seed)
        x0 = rng.normal(0.0, float(args.init_kick), size=n_params)
    else:
        x0 = np.zeros(n_params, dtype=float)
    e_hf = energy(x0)
    print(f"\nHF / initial energy : {e_hf:.12f} Eh")

    # --- BFGS optimization -----------------------------------------------------
    result = minimize(
        energy,
        x0,
        method="BFGS",
        options={"maxiter": int(args.max_iter)},
    )
    theta_opt = np.asarray(result.x, dtype=float)
    e_final = float(result.fun)

    # --- Fidelity with the exact ground-state eigenvector ---------------------
    psi_opt = state_vector(theta_opt).astype(np.complex128)
    fidelity = float(np.abs(np.vdot(gs_vector, psi_opt)) ** 2)

    # --- Report ----------------------------------------------------------------
    print("\n=== Optimization result (noiseless statevector, BFGS) ===")
    print(f"  converged           : {bool(result.success)} ({result.message})")
    print(f"  iterations          : {result.nit}, fn evals: {result.nfev}")
    print(f"  optimized thetas    : {theta_opt.tolist()}")
    print(f"  final energy        : {e_final:.12f} Eh")
    print(f"  exact GS (eig)      : {e_gs_exact:.12f} Eh")
    if e_gs_summary is not None:
        print(f"  GS from summary     : {e_gs_summary:.12f} Eh")
        print(f"  |eig - summary|     : {abs(e_gs_exact - e_gs_summary):.3e} Eh")
        agree = abs(e_gs_exact - e_gs_summary) < 1e-6
        print(f"  eigenvalue agrees   : {agree}")
    else:
        agree = None
        print("  GS from summary     : (bond not found in summary)")
    print(f"  energy above GS     : {(e_final - e_gs_exact) * 1e3:.6f} mEh")
    print(f"  fidelity |<gs|psi>|^2: {fidelity:.10f}")

    # --- Save results ----------------------------------------------------------
    out = {
        "molecule": meta.get("molecule", "Cl2"),
        "bond_length": bond,
        "circuit_json": str(args.circuit),
        "hamiltonian_file": str(ham_path),
        "num_qubits": len(qubits),
        "n_spatial": n_spatial,
        "n_electrons": n_electrons,
        "hf_occupied_spin_orbitals": occupied,
        "hf_empty_spin_orbitals": unoccupied,
        "param_names": list(meta.get("param_names", [])),
        "optimized_thetas": theta_opt.tolist(),
        "hf_initial_energy_Ha": e_hf,
        "final_energy_Ha": e_final,
        "exact_ground_state_energy_Ha": e_gs_exact,
        "summary_ground_state_energy_Ha": e_gs_summary,
        "eigenvalue_matches_summary": agree,
        "energy_above_gs_mHa": (e_final - e_gs_exact) * 1e3,
        "fidelity_with_ground_state": fidelity,
        "bfgs_converged": bool(result.success),
        "bfgs_iterations": int(result.nit),
        "bfgs_message": str(result.message),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / f"verify_cl2_bond_{bond_token(bond)}.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nSaved results: {out_path}")


if __name__ == "__main__":
    main()
