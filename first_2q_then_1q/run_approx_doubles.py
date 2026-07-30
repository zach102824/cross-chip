#!/usr/bin/env python3
"""Approximate exploring UCCSD doubles with 2q-scaffold + 1q-only adjust.

Question: how close can RZX/CZ + free single-qubit layers get to the
compiled doubles U(θ)|HF⟩ (statevector overlap)?

Run from repo root:
  .venv_h4_tencirchem/bin/python first_2q_then_1q/run_approx_doubles.py
  .venv_h4_tencirchem/bin/python first_2q_then_1q/run_approx_doubles.py --case HF_6q --quick
  .venv_h4_tencirchem/bin/python first_2q_then_1q/run_approx_doubles.py --case Cl2_10q
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ansatz import AnsatzSpec, default_specs, build_circuit  # noqa: E402
from fit import fit_multistart, overlap_with_target  # noqa: E402
from targets import (  # noqa: E402
    load_cl2_10q,
    load_hf6q,
    sample_thetas,
    target_statevector,
)


def occupied_from_bits(hf_bits: str) -> list[int]:
    return [q for q, b in enumerate(hf_bits) if b == "1"]


def n2_of_gates(gates) -> int:
    return sum(len(g["qubits"]) == 2 for g in gates)


def evaluate_case(problem: dict, *, n_theta: int, n_starts: int, maxiter: int,
                  modes: list[str], target_key: str = "gates_frozen"):
    n = problem["n_qubits"]
    hf_bits = problem["hf_bits"]
    occupied = occupied_from_bits(hf_bits)
    gates_ref = problem[target_key]
    thetas = sample_thetas(problem["n_params"], n_theta, seed=0, scale=0.3)

    # Sanity: fully-frozen vs full JW should agree on |HF⟩
    sanity = []
    for th in thetas:
        a = target_statevector(problem["gates_frozen"], n, th, hf_bits)
        b = target_statevector(problem["gates_full"], n, th, hf_bits)
        sanity.append(float(np.abs(np.vdot(a.data, b.data)) ** 2))

    specs = default_specs(problem["name"], occupied)
    rows = []
    for spec in specs:
        qc, all_p, p2, p1 = build_circuit(spec)
        for mode in modes:
            ovs = []
            t0 = time.time()
            for i, th in enumerate(thetas):
                target = target_statevector(gates_ref, n, th, hf_bits)
                # for only_1q try a few fixed weights
                if mode == "only_1q":
                    best_ov = -1.0
                    for w in (0.15, np.pi / 6, np.pi / 4, 0.5):
                        r = fit_multistart(
                            qc, all_p, p2, p1, target,
                            mode="only_1q", fixed_2q=w,
                            n_starts=max(1, n_starts // 2),
                            maxiter=maxiter, seed=100 * i + int(10 * w),
                        )
                        best_ov = max(best_ov, r.overlap)
                    ovs.append(best_ov)
                else:
                    r = fit_multistart(
                        qc, all_p, p2, p1, target,
                        mode="joint", n_starts=n_starts,
                        maxiter=maxiter, seed=100 * i,
                    )
                    ovs.append(r.overlap)
            elapsed = time.time() - t0
            row = {
                "spec": spec.name,
                "mode": mode,
                "n2": qc.num_nonlocal_gates(),
                "n_params_1q": len(p1),
                "n_params_2q": len(p2),
                "overlap_mean": float(np.mean(ovs)),
                "overlap_min": float(np.min(ovs)),
                "overlap_max": float(np.max(ovs)),
                "overlaps": [float(x) for x in ovs],
                "seconds": round(elapsed, 2),
            }
            rows.append(row)
            print(
                f"  {spec.name:40s} {mode:8s}  "
                f"⟨ov⟩={row['overlap_mean']:.6f}  min={row['overlap_min']:.6f}  "
                f"n2={row['n2']}  ({elapsed:.1f}s)"
            )
    return {
        "case": problem["name"],
        "target": target_key,
        "ref_n2_frozen": n2_of_gates(problem["gates_frozen"]),
        "ref_n2_winner": n2_of_gates(problem["gates_winner"]),
        "ref_n2_full": n2_of_gates(problem["gates_full"]),
        "frozen_strings": problem["frozen_strings"],
        "winner_vs_frozen_overlap_min": float(min(sanity)),
        "frozen_vs_full_overlap_min": float(min(sanity)),
        "n_theta": n_theta,
        "thetas": thetas.tolist(),
        "results": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=["HF_6q", "Cl2_10q", "both"], default="both")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--maxiter", type=int, default=None)
    ap.add_argument("--target", default="gates_frozen",
                    choices=["gates_frozen", "gates_winner", "gates_full"])
    args = ap.parse_args()

    n_theta = 2 if args.quick else 4
    n_starts = 2 if args.quick else 4
    maxiter = args.maxiter or (150 if args.quick else 400)
    modes = ["joint", "only_1q"]

    cases = []
    if args.case in ("HF_6q", "both"):
        cases.append(load_hf6q())
    if args.case in ("Cl2_10q", "both"):
        cases.append(load_cl2_10q())

    all_out = []
    for problem in cases:
        print(f"\n=== {problem['name']}  target={args.target}  "
              f"frozen={problem['frozen_strings']} ===")
        print(f"  ref 2q: full={n2_of_gates(problem['gates_full'])}  "
              f"frozen={n2_of_gates(problem['gates_frozen'])}  "
              f"winner={n2_of_gates(problem['gates_winner'])}")
        out = evaluate_case(
            problem, n_theta=n_theta, n_starts=n_starts,
            maxiter=maxiter, modes=modes, target_key=args.target,
        )
        all_out.append(out)
        stem = f"{problem['name']}_approx_doubles"
        if args.quick:
            stem += "_quick"
        path = HERE / f"{stem}.json"
        path.write_text(json.dumps(out, indent=2))
        print(f"  wrote {path.name}")

    # Short ranking
    print("\n=== ranking by mean overlap ===")
    for out in all_out:
        ranked = sorted(out["results"], key=lambda r: -r["overlap_mean"])
        print(f"{out['case']}:")
        for r in ranked[:6]:
            print(
                f"  {r['overlap_mean']:.6f}  min={r['overlap_min']:.6f}  "
                f"{r['spec']} [{r['mode']}] n2={r['n2']}"
            )


if __name__ == "__main__":
    main()
