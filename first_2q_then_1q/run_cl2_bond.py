#!/usr/bin/env python3
"""Cl2 10q at one stretched bond: how close can 2q+1q get to the doubles?

1. Load H at the chosen bond (default 2.8 Å).
2. VQE-optimize the exploring UCCSD doubles θ on |HF⟩.
3. Fit sparse RZX/CZ + U3 scaffolds to that target state.
4. Report overlap vs doubles and energies vs E_HF / E_GS.

  .venv_h4_tencirchem/bin/python first_2q_then_1q/run_cl2_bond.py
  .venv_h4_tencirchem/bin/python first_2q_then_1q/run_cl2_bond.py --bond 3.0 --quick
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from openfermion import QubitOperator
from openfermion.linalg import get_sparse_operator
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "exploring"))

from ansatz import AnsatzSpec, PAIRS_CL2, build_circuit  # noqa: E402
from fit import bind, fit_multistart, overlap_with_target  # noqa: E402
from targets import (  # noqa: E402
    load_cl2_10q,
    prep_hf,
    target_statevector,
)
from flexible_compile import gates_to_qc  # noqa: E402

# From Pauli_Ham/Cl2_bond_scan_summary.txt
SCAN = {
    2.8: {"E_HF": -908.996506785220, "E_GS": -909.100254358093},
    3.0: {"E_HF": -908.962927789720, "E_GS": -909.091900895864},
}


def load_numbered_h(path: Path, n_qubits: int) -> np.ndarray:
    """Dense H in Qiskit bit order (qubit 0 = LSB).

    openfermion's get_sparse_operator uses qubit 0 = MSB; remap by bit-reversal.
    """
    code = {1: "X", 2: "Y", 3: "Z"}
    op = QubitOperator()
    for raw in path.read_text().splitlines():
        if not raw.strip():
            continue
        parts = raw.split()
        coeff = float(parts[0])
        codes = [int(v) for v in parts[1:]]
        assert len(codes) == n_qubits
        term = tuple((q, code[c]) for q, c in enumerate(codes) if c != 0)
        op += QubitOperator(term, coeff)
    h_of = get_sparse_operator(op, n_qubits=n_qubits).toarray()
    dim = 2**n_qubits
    perm = np.array([
        int(format(i, f"0{n_qubits}b")[::-1], 2) for i in range(dim)
    ])
    return h_of[np.ix_(perm, perm)]


def energy_of_sv(h: np.ndarray, sv: Statevector) -> float:
    v = sv.data
    return float(np.real(np.vdot(v, h @ v)))


def corr_pct(e, e_hf, e_gs) -> float:
    denom = e_hf - e_gs
    if abs(denom) < 1e-15:
        return 0.0
    return float(100.0 * (e_hf - e) / denom)


def cl2_specs(occupied, quick: bool) -> list[AnsatzSpec]:
    n = 10
    lib = PAIRS_CL2
    specs = [
        AnsatzSpec("vert3|rzx|pre+u3", n, lib["vert3"], "rzx", True, 1, occupied),
        AnsatzSpec("vert|rzx|pre+u3", n, lib["vert"], "rzx", True, 1, occupied),
        AnsatzSpec("nn+vert3|mix|pre+u3", n, lib["nn+vert3"], "rzx_on_vert_else_cz", True, 1, occupied),
        AnsatzSpec("nn+chord+vert3|mix|pre+u3", n, lib["nn+chord+vert3"], "rzx_on_vert_else_cz", True, 1, occupied),
    ]
    if not quick:
        specs.extend([
            AnsatzSpec("nn+chord+vert3|rzx|pre+u3", n, lib["nn+chord+vert3"], "rzx", True, 1, occupied),
            AnsatzSpec("nn+chord+vert3|mix|u3x2", n, lib["nn+chord+vert3"], "rzx_on_vert_else_cz", False, 2, occupied),
            AnsatzSpec("vert3|rzx|pre+u3x2", n, lib["vert3"], "rzx", True, 2, occupied),
        ])
    return specs


def optimize_doubles_theta(gates, n, hf_bits, h, n_params=3, maxiter=200, seed=0):
    rng = np.random.default_rng(seed)
    best = None
    for s in range(4):
        x0 = rng.uniform(-0.5, 0.5, size=n_params)

        def cost(th):
            sv = target_statevector(gates, n, th, hf_bits)
            return energy_of_sv(h, sv)

        res = minimize(cost, x0, method="L-BFGS-B", options={"maxiter": maxiter})
        if best is None or res.fun < best["E"]:
            best = {"E": float(res.fun), "theta": res.x.tolist(), "nfev": int(res.nfev)}
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bond", type=float, default=2.8)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--maxiter", type=int, default=None)
    args = ap.parse_args()
    bond = args.bond
    maxiter = args.maxiter or (120 if args.quick else 350)
    n_starts = 2 if args.quick else 4

    problem = load_cl2_10q()
    n = problem["n_qubits"]
    hf_bits = problem["hf_bits"]
    occupied = [q for q, b in enumerate(hf_bits) if b == "1"]

    ham_path = REPO / "Pauli_Ham" / f"Cl2_bond_{bond:g}.txt"
    if not ham_path.is_file():
        raise FileNotFoundError(ham_path)
    print(f"loading {ham_path.name} …")
    h = load_numbered_h(ham_path, n)

    # Prefer scan table; also check dense diagonalization for sanity
    e_hf_sv = energy_of_sv(h, Statevector.from_instruction(prep_hf(n, hf_bits)))
    if bond in SCAN:
        e_hf, e_gs = SCAN[bond]["E_HF"], SCAN[bond]["E_GS"]
    else:
        e_hf = e_hf_sv
        e_gs = float(np.linalg.eigvalsh(h)[0])
    print(f"bond={bond}  E_HF={e_hf:.8f} (sv {e_hf_sv:.8f})  E_GS={e_gs:.8f}  "
          f"gap={(e_hf-e_gs)*1e3:.1f} mHa")

    # Target = exploring fully-frozen doubles at VQE-optimal θ
    print("optimizing doubles θ …")
    t0 = time.time()
    doubles = optimize_doubles_theta(
        problem["gates_frozen"], n, hf_bits, h,
        n_params=3, maxiter=maxiter, seed=1,
    )
    print(f"  doubles: E={doubles['E']:.8f}  corr={corr_pct(doubles['E'], e_hf, e_gs):.1f}%  "
          f"θ={np.round(doubles['theta'], 4).tolist()}  ({time.time()-t0:.1f}s)")

    target = target_statevector(problem["gates_frozen"], n, doubles["theta"], hf_bits)
    # also check winner circuit at same θ
    winner_sv = target_statevector(problem["gates_winner"], n, doubles["theta"], hf_bits)
    ov_win = float(np.abs(np.vdot(target.data, winner_sv.data)) ** 2)
    e_win = energy_of_sv(h, winner_sv)
    print(f"  winner@sameθ overlap vs frozen={ov_win:.6f}  E={e_win:.8f}")

    rows = []
    for spec in cl2_specs(occupied, args.quick):
        qc, all_p, p2, p1 = build_circuit(spec)
        for mode in ("joint", "only_1q"):
            t1 = time.time()
            if mode == "only_1q":
                best = None
                for w in (0.2, np.pi / 6, float(np.mean(np.abs(doubles["theta"]))), np.pi / 4):
                    r = fit_multistart(
                        qc, all_p, p2, p1, target,
                        mode="only_1q", fixed_2q=w,
                        n_starts=max(1, n_starts // 2),
                        maxiter=maxiter, seed=int(1000 * w),
                    )
                    if best is None or r.overlap > best.overlap:
                        best = r
                        best_w = w
            else:
                best = fit_multistart(
                    qc, all_p, p2, p1, target,
                    mode="joint", n_starts=n_starts,
                    maxiter=maxiter, seed=0,
                )
                best_w = None
            sv = Statevector.from_instruction(bind(qc, all_p, best.x))
            e = energy_of_sv(h, sv)
            row = {
                "spec": spec.name,
                "mode": mode,
                "fixed_2q": best_w,
                "n2": best.n2,
                "n_params_1q": best.n1_params,
                "n_params_2q": best.n2_params,
                "overlap": best.overlap,
                "E": e,
                "err_mHa": (e - e_gs) * 1e3,
                "corr_pct": corr_pct(e, e_hf, e_gs),
                "seconds": round(time.time() - t1, 2),
            }
            rows.append(row)
            print(
                f"  {spec.name:36s} {mode:8s}  ov={row['overlap']:.6f}  "
                f"E={e:.8f}  corr={row['corr_pct']:.1f}%  n2={row['n2']}  "
                f"({row['seconds']}s)"
            )

    out = {
        "bond": bond,
        "E_HF": e_hf,
        "E_GS": e_gs,
        "gap_mHa": (e_hf - e_gs) * 1e3,
        "doubles": {
            **doubles,
            "corr_pct": corr_pct(doubles["E"], e_hf, e_gs),
            "err_mHa": (doubles["E"] - e_gs) * 1e3,
            "ref_n2_frozen": sum(len(g["qubits"]) == 2 for g in problem["gates_frozen"]),
            "ref_n2_winner": sum(len(g["qubits"]) == 2 for g in problem["gates_winner"]),
            "frozen_strings": problem["frozen_strings"],
        },
        "winner_at_doubles_theta": {"overlap_vs_frozen": ov_win, "E": e_win},
        "results": sorted(rows, key=lambda r: -r["overlap"]),
    }
    stem = f"Cl2_10q_bond_{bond:g}_approx_doubles"
    if args.quick:
        stem += "_quick"
    path = HERE / f"{stem}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {path.name}")
    print("\n=== top by overlap ===")
    for r in out["results"][:6]:
        print(
            f"  ov={r['overlap']:.6f}  corr={r['corr_pct']:.1f}%  "
            f"{r['spec']} [{r['mode']}] n2={r['n2']}"
        )


if __name__ == "__main__":
    main()
