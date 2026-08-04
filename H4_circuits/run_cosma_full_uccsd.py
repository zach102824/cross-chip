#!/usr/bin/env python3
"""CLI: full-linked H4 UCCSD through COSMA-faithful PPTT mappings.

Example:
  .venv_py311/bin/python H4_circuits/run_cosma_full_uccsd.py
  .venv_py311/bin/python H4_circuits/run_cosma_full_uccsd.py --encodings JW PE --quick
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from cosma_uccsd.emit import build_logical_circuit, best_score_over_seeds  # noqa: E402
from cosma_uccsd.optimize import compile_path, optimize_h4  # noqa: E402
from cosma_uccsd.mapping import build_mapping  # noqa: E402
from cosma_uccsd.workload import load_h4_workload  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--encodings",
        nargs="+",
        default=["JW", "PE", "JKMN", "TREE_GA"],
        help="Fermion-to-qubit encodings to compare",
    )
    ap.add_argument("--ga-population", type=int, default=16)
    ap.add_argument("--ga-generations", type=int, default=8)
    ap.add_argument("--ga-seed", type=int, default=42)
    ap.add_argument("--transpile-seeds", type=int, default=6)
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Smaller GA + fewer transpile seeds",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=HERE / "H4_cosma_full_uccsd_results.json",
    )
    ap.add_argument(
        "--write-best-circuits",
        action="store_true",
        help="Write best 8q/6q QPY circuits next to the JSON",
    )
    args = ap.parse_args()

    if args.quick:
        args.ga_population = 8
        args.ga_generations = 4
        args.transpile_seeds = 3

    print("Running COSMA-faithful full-linked H4 UCCSD optimization...", flush=True)
    print(f"  encodings={args.encodings}", flush=True)
    if args.quick:
        print(
            f"  quick mode: ga_pop={args.ga_population} ga_gen={args.ga_generations} "
            f"transpile_seeds={args.transpile_seeds}",
            flush=True,
        )
    summary = optimize_h4(
        encodings=args.encodings,
        ga_population=args.ga_population,
        ga_generations=args.ga_generations,
        ga_seed=args.ga_seed,
        transpile_seeds=range(args.transpile_seeds),
    )

    args.output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}", flush=True)

    def _print_best(tag: str, row: dict) -> None:
        sc = row["score"]
        print(
            f"best {tag}: mapping={row['mapping_name']} "
            f"cross={sc['cross_chip_2q']} total2q={sc['total_2q']} "
            f"depth={sc['depth']} factors={row['n_factors']} "
            f"exact={row['exact_per_excitation']}"
        )

    _print_best("8q", summary["best_8q"])
    _print_best("6q", summary["best_6q"])
    base = summary.get("approx_8gadget_baseline") or {}
    if base:
        print(
            f"approx 8-gadget baseline: cross={base.get('cross_chip_2q')} "
            f"total2q={base.get('routed_2q')} depth={base.get('depth')} "
            f"({base.get('note')})"
        )

    if args.write_best_circuits:
        try:
            from qiskit import qpy
        except ImportError:
            qpy = None
        workload = load_h4_workload()
        for tag, best in (("8q", summary["best_8q"]), ("6q", summary["best_6q"])):
            nq = 8 if tag == "8q" else 6
            # Recompile winner mapping for circuit export
            name = best["mapping_name"]
            if name == "TREE_GA":
                # Rebuild from stored tree is limited; re-run compile via JW/PE/JKMN only
                # For TREE_GA, reconstruct factors from saved paulis
                from cosma_uccsd.schedule import PauliFactor

                factors = [
                    PauliFactor(
                        label=lab,
                        pid=pid,
                        ex_op_index=0,
                        pauli=p,
                        angle_sign=ang,
                        theta_name=f"t{pid}",
                        coeff=0.0j,
                    )
                    for lab, pid, p, ang in zip(
                        best["factor_labels"],
                        best["pids"],
                        best["paulis"],
                        best["angle_signs"],
                    )
                ]
            else:
                basis = build_mapping(name, 8)
                res = compile_path(workload, basis, n_qubits=nq, path_name=tag, seeds=range(args.transpile_seeds))
                factors = res.factors
            score, logical, routed, seed = best_score_over_seeds(
                factors, nq, seeds=range(args.transpile_seeds)
            )
            out_json = HERE / f"H4_cosma_full_{tag}_best.json"
            out_json.write_text(
                json.dumps(
                    {
                        "path": tag,
                        "mapping": best["mapping_name"],
                        "score": score.to_dict(),
                        "seed": seed,
                        "paulis": [f.pauli for f in factors],
                        "angle_signs": [f.angle_sign for f in factors],
                        "pids": [f.pid for f in factors],
                    },
                    indent=2,
                )
                + "\n"
            )
            if qpy is not None:
                with open(HERE / f"H4_cosma_full_{tag}_logical.qpy", "wb") as f:
                    qpy.dump(logical, f)
                with open(HERE / f"H4_cosma_full_{tag}_routed.qpy", "wb") as f:
                    qpy.dump(routed, f)
            print(f"wrote circuits for {tag} (seed={seed})")


if __name__ == "__main__":
    main()
