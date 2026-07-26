#!/usr/bin/env python3
"""Noiseless statevector check for exploring/HF_6q_indep_pairs_freeze.json.

Builds the frozen independent-pair ansatz + tapered HF prep, minimizes ⟨H⟩
with BFGS, and reports how close the optimum is to the exact ground-state
energy of the tapered HF Hamiltonian.

Usage (from repo root):

    .venv_h4_tencirchem/bin/python exploring/test_hf_freeze_noiseless.py
    .venv_h4_tencirchem/bin/python exploring/test_hf_freeze_noiseless.py --bond 1.0
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_STATE = _ROOT / "state_transfer"

sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_STATE))

from flexible_compile import gates_to_qc  # noqa: E402

CIRCUIT_JSON = _HERE / "HF_6q_indep_pairs_freeze.json"
BASELINE_JSON = _STATE / "circuits2read" / "HF_tapered_6q_3doubles_rzx.json"
CHEMICAL_ACCURACY_HA = 1.6e-3


def bond_token(bond: float) -> str:
    return f"{bond:.10g}".rstrip("0").rstrip(".") or "0"


def load_numbered_hamiltonian(path: Path, n_qubits: int) -> QubitOperator:
    code_to_pauli = {1: "X", 2: "Y", 3: "Z"}
    op = QubitOperator()
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        coeff = float(parts[0])
        codes = [int(x) for x in parts[1:]]
        if len(codes) != n_qubits:
            raise ValueError(f"{path}: expected {n_qubits} Pauli codes, got {len(codes)}")
        term = tuple(
            (q, code_to_pauli[c]) for q, c in enumerate(codes) if c != 0
        )
        op += QubitOperator(term, coeff)
    return op


def qiskit_to_openfermion_sv(circuit: QuantumCircuit) -> np.ndarray:
    """Qiskit little-endian → openfermion qubit-0-as-MSB ordering."""
    data = Statevector(circuit).data
    n = circuit.num_qubits
    out = np.empty_like(data)
    for index, amp in enumerate(data):
        rev = int(format(index, f"0{n}b")[::-1], 2)
        out[rev] = amp
    return out


def hf_prep(n_qubits: int, init_bits: str) -> QuantumCircuit:
    prep = QuantumCircuit(n_qubits)
    for q, bit in enumerate(init_bits):
        if bit == "1":
            prep.x(q)
    return prep


def optimize_circuit_energy(
    gates,
    n_qubits: int,
    init_bits: str,
    h_dense: np.ndarray,
    *,
    label: str,
    gtol: float = 1e-7,
    maxiter: int = 200,
) -> dict:
    ansatz = gates_to_qc(gates, n_qubits)
    params = sorted(ansatz.parameters, key=lambda p: p.name)
    prep = hf_prep(n_qubits, init_bits)
    n_params = len(params)

    def energy(theta) -> float:
        bound = ansatz.assign_parameters(
            {p: float(v) for p, v in zip(params, theta)}
        )
        psi = qiskit_to_openfermion_sv(prep.compose(bound))
        return float(np.real(np.vdot(psi, h_dense @ psi)))

    def grad(theta) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        g = np.zeros_like(theta)
        for i in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += 0.5 * np.pi
            minus[i] -= 0.5 * np.pi
            g[i] = 0.5 * (energy(plus) - energy(minus))
        return g

    theta0 = np.zeros(n_params, dtype=float)
    e0 = energy(theta0)
    history = [e0]

    def cb(theta):
        history.append(energy(theta))

    print(f"\n=== {label}: noiseless BFGS ===")
    print(f"  n_params = {n_params}")
    print(f"  E(θ=0)   = {e0:.12f} Ha")

    res = minimize(
        energy,
        theta0,
        method="BFGS",
        jac=grad,
        callback=cb,
        options={"gtol": gtol, "maxiter": maxiter},
    )
    e_opt = float(res.fun)
    print(f"  E_opt    = {e_opt:.12f} Ha")
    print(f"  θ_opt    = {np.asarray(res.x).round(8).tolist()}")
    print(f"  success  = {bool(res.success)}  nit={res.nit}  nfev={res.nfev}")
    return {
        "label": label,
        "e_theta0": e0,
        "e_opt": e_opt,
        "theta_opt": np.asarray(res.x, dtype=float),
        "success": bool(res.success),
        "nit": int(res.nit),
        "history": history,
    }


def available_bonds() -> list[float]:
    bonds = []
    for path in sorted((_STATE / "Pauli_Ham").glob("HF_tapered_bond_*.txt")):
        # HF_tapered_bond_1.txt / HF_tapered_bond_1.2.txt (skip scan_summary, powers, …)
        token = path.stem.split("HF_tapered_bond_", 1)[1]
        try:
            bond = float(token)
        except ValueError:
            continue
        if not (_STATE / "Pauli_Ham" / f"HF_tapered_bond_{bond_token(bond)}_meta.json").is_file():
            continue
        bonds.append(bond)
    return sorted(bonds)


def run_one_bond(
    data: dict,
    bond: float,
    *,
    compare_baseline: bool,
    quiet: bool = False,
) -> dict:
    n_qubits = int(data["num_qubits"])
    token = bond_token(bond)
    meta_path = _STATE / "Pauli_Ham" / f"HF_tapered_bond_{token}_meta.json"
    ham_path = _STATE / "Pauli_Ham" / f"HF_tapered_bond_{token}.txt"
    if not ham_path.is_file():
        raise FileNotFoundError(
            f"missing {ham_path}; run state_transfer/generate_tapered_hamiltonians.py"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    init_bits = str(meta["hf_bitstring_tapered"])
    if len(init_bits) != n_qubits:
        raise ValueError(
            f"hf_bitstring_tapered length {len(init_bits)} != num_qubits {n_qubits}"
        )

    ham = load_numbered_hamiltonian(ham_path, n_qubits)
    h_dense = get_sparse_operator(ham, n_qubits=n_qubits).toarray()
    e_gs = float(np.linalg.eigvalsh(h_dense)[0].real)
    prep = hf_prep(n_qubits, init_bits)
    psi_hf = qiskit_to_openfermion_sv(prep)
    e_hf = float(np.real(np.vdot(psi_hf, h_dense @ psi_hf)))

    if not quiet:
        print("HF freeze circuit — noiseless energy vs ground state")
        print(f"  circuit   : {CIRCUIT_JSON.name}")
        print(f"  method    : {data.get('method')}")
        print(f"  bond      : {bond} Å")
        print(f"  qubits    : {n_qubits}")
        print(f"  init bits : {init_bits}")
        print(f"  RZX pairs : {data.get('rzx_pairs')}")
        print(f"  frozen    : {data.get('frozen_strings')}")
        print(f"  original  : {data.get('original_strings')}")
        print(f"  E_HF      : {e_hf:.12f} Ha")
        print(f"  E_GS      : {e_gs:.12f} Ha")
        print(f"  |E_HF-GS| : {(e_hf - e_gs) * 1e3:.6f} mHa")

    import contextlib
    import io

    sink = io.StringIO() if quiet else None
    ctx = contextlib.redirect_stdout(sink) if quiet else contextlib.nullcontext()
    with ctx:
        freeze = optimize_circuit_energy(
            data["gates"],
            n_qubits,
            init_bits,
            h_dense,
            label="freeze (indep_pairs)",
        )
        baseline = None
        if compare_baseline and BASELINE_JSON.is_file():
            base = json.loads(BASELINE_JSON.read_text(encoding="utf-8"))
            baseline = optimize_circuit_energy(
                base["gates"],
                n_qubits,
                init_bits,
                h_dense,
                label="baseline (tapered RZX)",
            )

    err = freeze["e_opt"] - e_gs
    out = {
        "bond": bond,
        "e_hf": e_hf,
        "e_gs": e_gs,
        "e_freeze": freeze["e_opt"],
        "e_baseline": None if baseline is None else baseline["e_opt"],
        "err_mha": err * 1e3,
        "ok": err < CHEMICAL_ACCURACY_HA,
        "freeze": freeze,
        "baseline": baseline,
    }

    if not quiet:
        print("\n=== Summary vs ground state ===")
        print(f"{'circuit':<28} {'E_opt (Ha)':>16} {'|E-GS| (mHa)':>14} {'chem.acc.?':>10}")
        rows = [("freeze (indep_pairs)", freeze["e_opt"])]
        if baseline is not None:
            rows.append(("baseline (tapered RZX)", baseline["e_opt"]))
        for label, e in rows:
            err_mha = (e - e_gs) * 1e3
            ok = "yes" if (e - e_gs) < CHEMICAL_ACCURACY_HA else "no"
            print(f"{label:<28} {e:16.12f} {err_mha:14.6f} {ok:>10}")
        print(f"{'exact GS':<28} {e_gs:16.12f} {0.0:14.6f} {'—':>10}")
        print(f"{'HF determinant':<28} {e_hf:16.12f} {(e_hf - e_gs) * 1e3:14.6f} {'—':>10}")
        if baseline is not None:
            d = freeze["e_opt"] - baseline["e_opt"]
            print(
                f"\nfreeze − baseline optimum = {d:+.3e} Ha "
                f"({d * 1e3:+.6f} mHa)"
            )
        print(
            f"\nVerdict: freeze |E_opt − E_GS| = {err * 1e3:.6f} mHa "
            f"(chemical accuracy {CHEMICAL_ACCURACY_HA * 1e3:.1f} mHa)"
        )
        if out["ok"]:
            print("  OK: within chemical accuracy of GS.")
        else:
            print("  WARNING: outside chemical accuracy of GS.")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bond",
        type=float,
        default=1.0,
        help="bond length (Å) for tapered HF Hamiltonian (default: 1.0)",
    )
    parser.add_argument(
        "--all-bonds",
        action="store_true",
        help="scan every available HF_tapered_bond_*.txt",
    )
    parser.add_argument(
        "--no-baseline",
        action="store_true",
        help="skip comparison against the original tapered RZX baseline",
    )
    parser.add_argument(
        "--circuit",
        type=Path,
        default=CIRCUIT_JSON,
        help="path to freeze-circuit JSON",
    )
    args = parser.parse_args(argv)

    data = json.loads(args.circuit.read_text(encoding="utf-8"))
    compare_baseline = not args.no_baseline

    if not args.all_bonds:
        rec = run_one_bond(data, float(args.bond), compare_baseline=compare_baseline)
        return 0 if rec["ok"] else 1

    bonds = available_bonds()
    if not bonds:
        raise FileNotFoundError("no HF_tapered_bond_*.txt found under state_transfer/Pauli_Ham")

    print("HF freeze — noiseless bond scan")
    print(f"  circuit : {args.circuit.name}")
    print(f"  method  : {data.get('method')}")
    print(f"  bonds   : {bonds}")
    print(f"  chem.acc. threshold = {CHEMICAL_ACCURACY_HA * 1e3:.1f} mHa\n")
    hdr = (
        f"{'bond':>6} {'E_HF':>14} {'E_freeze':>14} {'E_base':>14} {'E_GS':>14} "
        f"{'frz-GS':>10} {'frz-base':>12} {'ok?':>6}"
    )
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for bond in bonds:
        rec = run_one_bond(
            data, bond, compare_baseline=compare_baseline, quiet=True
        )
        rows.append(rec)
        e_base = rec["e_baseline"]
        e_base_s = f"{e_base:14.8f}" if e_base is not None else f"{'—':>14}"
        dbase = (
            (rec["e_freeze"] - e_base) * 1e3 if e_base is not None else float("nan")
        )
        dbase_s = f"{dbase:+12.3e}" if e_base is not None else f"{'—':>12}"
        print(
            f"{bond:6.1f} {rec['e_hf']:14.8f} {rec['e_freeze']:14.8f} {e_base_s} "
            f"{rec['e_gs']:14.8f} {rec['err_mha']:10.4f} {dbase_s} "
            f"{'yes' if rec['ok'] else 'NO':>6}"
        )

    print("-" * len(hdr))
    n_ok = sum(1 for r in rows if r["ok"])
    all_ok = n_ok == len(rows)
    print(
        f"{'ALL OK' if all_ok else 'SOME FAILED'}: "
        f"{n_ok}/{len(rows)} bonds within chemical accuracy"
    )
    # Freeze vs baseline: freeze should not cost accuracy relative to the ansatz.
    if compare_baseline and all(r["e_baseline"] is not None for r in rows):
        max_diff = max(abs(r["e_freeze"] - r["e_baseline"]) for r in rows)
        print(f"max |freeze − baseline| = {max_diff:.3e} Ha")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
