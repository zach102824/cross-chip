#!/usr/bin/env python3
"""Noiseless BFGS VQE bond scan for Cl2 (1.6–3.0 Å) on the disjoint-RZX circuit.

Uses ``exploring/Cl2_10q_disjoint_rzx.json``, HF prep, SciPy BFGS.
Prints live status; saves SciPy results under ``exploring/``.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))

from approx_omit_boundary_cz_vqe import load_hamiltonian, hf_prep, expect_h  # noqa: E402
from qiskit.circuit import Parameter  # noqa: E402

CIRCUIT_JSON = _HERE / "Cl2_10q_disjoint_rzx.json"
HAM_DIR = _ROOT / "Pauli_Ham"
OUT_JSON = _HERE / "Cl2_bond_scan_bfgs_results.json"
CHEM_ACC_mHa = 1.6

BONDS = [1.6, 1.8, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
N_ELECTRONS = 8


def bond_token(bond: float) -> str:
    return f"{float(bond):.10g}".rstrip("0").rstrip(".")


def ham_path(bond: float) -> Path:
    return HAM_DIR / f"Cl2_bond_{bond_token(bond)}.txt"


def build_energy_fn(gates, n, ne, ham):
    names = []
    for g in gates:
        if "param" in g and g["param"] not in names:
            names.append(g["param"])

    def energy(x):
        qc = hf_prep(n, ne)
        params = {}
        for g in gates:
            op = g["op"].lower()
            qs = g["qubits"]
            if op == "h":
                qc.h(qs[0])
            elif op == "cz":
                qc.cz(qs[0], qs[1])
            elif op == "rx":
                if "param" in g:
                    p = params.setdefault(g["param"], Parameter(g["param"]))
                    qc.rx(float(g.get("coeff", 1.0)) * p, qs[0])
                else:
                    qc.rx(float(g["value"]), qs[0])
            elif op == "rzx":
                p = params.setdefault(g["param"], Parameter(g["param"]))
                qc.rzx(float(g.get("coeff", 1.0)) * p, qs[0], qs[1])
        qc = qc.assign_parameters({params[nm]: float(v) for nm, v in zip(names, x)})
        return expect_h(qc, ham)

    return energy, names


def scipy_result_to_dict(res) -> dict:
    """Serialize SciPy OptimizeResult (BFGS) to JSON-friendly dict."""
    out = {
        "success": bool(res.success),
        "status": int(res.status),
        "message": str(res.message),
        "fun": float(res.fun),
        "x": np.asarray(res.x, dtype=float).tolist(),
        "nit": int(getattr(res, "nit", -1)),
        "nfev": int(getattr(res, "nfev", -1)),
        "njev": int(getattr(res, "njev", -1)),
    }
    if getattr(res, "jac", None) is not None:
        out["jac"] = np.asarray(res.jac, dtype=float).tolist()
    if getattr(res, "hess_inv", None) is not None:
        try:
            h = res.hess_inv
            # BFGS may store hess_inv as LinearOperator or ndarray
            if hasattr(h, "todense"):
                out["hess_inv"] = np.asarray(h.todense(), dtype=float).tolist()
            else:
                out["hess_inv"] = np.asarray(h, dtype=float).tolist()
        except Exception as e:
            out["hess_inv_error"] = str(e)
    return out


def run_bond(bond: float, gates, n, ne, x0_warm=None):
    path = ham_path(bond)
    if not path.exists():
        raise FileNotFoundError(path)
    print(f"\n{'='*70}", flush=True)
    print(f"Cl2 bond = {bond} Å   H = {path.name}", flush=True)

    t0 = time.time()
    ham = load_hamiltonian(path, n)
    e_gs = float(np.linalg.eigvalsh(ham.to_matrix())[0].real)
    e_hf = expect_h(hf_prep(n, ne), ham)
    print(f"  E_GS = {e_gs:.12f} Eh", flush=True)
    print(f"  E_HF = {e_hf:.12f} Eh   (HF−GS = {(e_hf - e_gs)*1e3:.4f} mHa)", flush=True)

    energy, names = build_energy_fn(gates, n, ne, ham)
    n_p = len(names)
    print(f"  params = {names}   optimizer = BFGS", flush=True)

    # Warm-start from previous bond if available; else 0 + a few random starts
    starts = []
    if x0_warm is not None and len(x0_warm) == n_p:
        starts.append(np.asarray(x0_warm, dtype=float))
    starts.append(np.zeros(n_p))
    rng = np.random.default_rng(int(bond * 1000) % (2**31))
    starts += [rng.uniform(-0.4, 0.4, size=n_p) for _ in range(3)]

    best = None
    all_runs = []
    for i, x0 in enumerate(starts):
        history = []

        def callback(xk):
            f = energy(xk)
            history.append({"nfev_at_callback": len(history) + 1, "fun": float(f),
                            "x": np.asarray(xk, dtype=float).tolist()})
            if len(history) == 1 or len(history) % 5 == 0:
                print(
                    f"    [start {i}] iter~{len(history):3d}  E={f:.12f}  "
                    f"ΔGS={(f - e_gs)*1e3:+.4f} mHa  x={np.array2string(xk, precision=5)}",
                    flush=True,
                )

        print(f"  -- BFGS start {i}/{len(starts)-1}  x0={np.array2string(x0, precision=5)}", flush=True)
        res = minimize(
            energy,
            x0,
            method="BFGS",
            callback=callback,
            options={"maxiter": 200, "gtol": 1e-8, "disp": False},
        )
        run = {
            "start_index": i,
            "x0": np.asarray(x0, dtype=float).tolist(),
            "scipy": scipy_result_to_dict(res),
            "callback_history": history,
        }
        all_runs.append(run)
        print(
            f"    done: success={res.success}  nit={res.nit}  nfev={res.nfev}  "
            f"E={res.fun:.12f}  msg={res.message}",
            flush=True,
        )
        if best is None or res.fun < best.fun:
            best = res

    e_vqe = float(best.fun)
    err_mHa = (e_vqe - e_gs) * 1e3
    corr_pct = (e_hf - e_vqe) / (e_hf - e_gs) * 100 if e_hf != e_gs else float("nan")
    chem = err_mHa <= CHEM_ACC_mHa
    elapsed = time.time() - t0

    print(f"  BEST E_VQE = {e_vqe:.12f} Eh", flush=True)
    print(f"  |E_VQE−E_GS| = {err_mHa:.4f} mHa   chem_acc({CHEM_ACC_mHa})? {chem}", flush=True)
    print(f"  corr recovered = {corr_pct:.2f}%   theta* = {best.x}", flush=True)
    print(f"  wall time = {elapsed:.1f}s", flush=True)

    return {
        "bond": bond,
        "ham_file": str(path),
        "e_gs": e_gs,
        "e_hf": e_hf,
        "e_vqe": e_vqe,
        "err_mHa": err_mHa,
        "chem_acc": chem,
        "corr_recovered_pct": corr_pct,
        "theta_star": np.asarray(best.x, dtype=float).tolist(),
        "param_names": names,
        "best_scipy": scipy_result_to_dict(best),
        "all_bfgs_runs": all_runs,
        "wall_time_s": elapsed,
    }


def main():
    data = json.loads(CIRCUIT_JSON.read_text())
    gates = data["gates"]
    n = int(data["num_qubits"])
    print("Cl2 noiseless BFGS bond scan", flush=True)
    print(f"circuit = {CIRCUIT_JSON.name}", flush=True)
    print(f"bonds   = {BONDS}", flush=True)
    print(f"chem_acc threshold = {CHEM_ACC_mHa} mHa", flush=True)

    rows = []
    x_warm = None
    for bond in BONDS:
        row = run_bond(bond, gates, n, N_ELECTRONS, x0_warm=x_warm)
        rows.append(row)
        x_warm = row["theta_star"]
        # incremental save
        payload = {
            "molecule": "Cl2",
            "circuit": str(CIRCUIT_JSON),
            "optimizer": "BFGS",
            "chem_acc_mHa": CHEM_ACC_mHa,
            "bonds": BONDS,
            "results": rows,
        }
        OUT_JSON.write_text(json.dumps(payload, indent=2))
        print(f"  (saved checkpoint → {OUT_JSON.name})", flush=True)

    print(f"\n{'='*70}", flush=True)
    print("SUMMARY", flush=True)
    print(f"{'bond':>6}  {'E_VQE':>16}  {'E_GS':>16}  {'err_mHa':>9}  {'chem?':>6}  {'corr%':>7}", flush=True)
    for r in rows:
        print(
            f"{r['bond']:6.1f}  {r['e_vqe']:16.10f}  {r['e_gs']:16.10f}  "
            f"{r['err_mHa']:9.4f}  {str(r['chem_acc']):>6}  {r['corr_recovered_pct']:7.2f}",
            flush=True,
        )
    print(f"\nFull SciPy BFGS results → {OUT_JSON}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
