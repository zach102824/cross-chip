#!/usr/bin/env python3
"""Find a Cl2 winner under the updated rules:

1. Every qubit must be used.
2. RZX pairs are any α↔β but vertex-disjoint (no shared index / no pile-up).
3. Within-spin CZ: nearest-neighbour only (+ chords 0–3 and 5–8).
   Cross-spin CZ forbidden — only RZX couples α↔β.

Keeps searching (wider keep-masks) until error < Cl2 baseline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "UCCSD circuit"))

from constraints import (  # noqa: E402
    allowed_cz_edges,
    cz_pairs,
    qubits_used,
    rzx_pairs,
    satisfies_rules,
)
from error_budget import cross_from_list, score_gates, spin_split_cross_pairs  # noqa: E402
from flexible_compile import compile_strings, gates_to_qc, gen, statevector_overlap  # noqa: E402
from methods import method_disjoint_rzx_all_qubits  # noqa: E402

CL2_BASELINE = _ROOT / "June_main/circuits2read/Cl2_10q_3doubles_rzx.json"
OUT_JSON = _HERE / "Cl2_10q_spatial_premerge.json"  # replace prior Cl2 winner path
OUT_PNG = _HERE / "Cl2_10q_spatial_premerge_circuit.png"
# also write a clearly named copy
OUT_JSON2 = _HERE / "Cl2_10q_disjoint_rzx.json"
OUT_PNG2 = _HERE / "Cl2_10q_disjoint_rzx_circuit.png"


def main():
    base = json.loads(CL2_BASELINE.read_text())
    base_bud = score_gates(
        base["gates"], cross_from_list([(3, 8), (0, 1), (5, 6)])
    )
    print(f"Cl2 baseline: fid={base_bud.fidelity:.6f} err={base_bud.error:.6f}")
    print(f"allowed CZ edges: {sorted(tuple(sorted(e)) for e in allowed_cz_edges(10))}")

    doubles = [(4, 9, 6, 1), (4, 9, 5, 0), (4, 9, 7, 2)]
    strings = ["".join(gen.jw_string_for_double(10, d)) for d in doubles]
    signs = [1, 1, 1]
    print("original:", strings)

    spin_cross = spin_split_cross_pairs(10)
    best = None
    for max_opts in (4, 6, 8, 12, 16):
        print(f"\n--- search max_mask_opts={max_opts} ---")
        hits = method_disjoint_rzx_all_qubits(
            strings, signs=signs, cross_pairs=spin_cross, max_mask_opts=max_opts
        )
        print(f"  legal candidates: {len(hits)}")
        for h in hits[:5]:
            b = h["budget"]
            print(
                f"  err={b.error:.6f} fid={b.fidelity:.6f} "
                f"1q={b.n1} on={b.n_onchip} x={b.n_cross} "
                f"rzx={rzx_pairs(h['gates'])} cz={cz_pairs(h['gates'])} "
                f"frozen={h['frozen_strings']} keep={h['keep_pairs']} "
                f"sched={h['schedule']}"
            )
            if best is None or b.error < best["budget"].error:
                best = h
        if best is not None and best["budget"].error < base_bud.error - 1e-12:
            print("  beat baseline — stop widening")
            break

    if best is None:
        raise SystemExit("no legal Cl2 circuit found under new rules")

    # Verify freeze/selective strings vs original on HF subspace
    ref = compile_strings(strings, signs=signs, order="given", fuse=True)
    new = compile_strings(
        best["frozen_strings"], signs=signs, order="given",
        hub_schedule=None, fuse=True,
    )
    # Use the actual winning gates/schedule for overlap vs original
    from flexible_compile import compile_strings as CS
    order = best["order"]
    ordered = [best["frozen_strings"][i] if False else None for i in order]
    # recompile with winning schedule (schedule aligned to ordered strings)
    win = CS(
        [best["strings"][i] for i in range(len(best["strings"]))],
        signs=[signs[best["order"][i]] for i in range(len(signs))],
        order="given",
        hub_schedule=best["schedule"],
        fuse=True,
    )
    # best["strings"] is already ordered; signs need remapping
    ord_idx = best["order"]
    ord_strings = best["frozen_strings"]  # wait - search reorders; frozen in rec is pre-order
    # In method_disjoint, frozen_strings is pre-order (combo order), but
    # schedule/strings in hit are post-order from search_flexible_hubs.
    # Prefer the gates already in best.
    gates = best["gates"]
    ok, why = satisfies_rules(gates, 10)
    assert ok, why
    assert len(rzx_pairs(gates)) == 3
    used = sorted(qubits_used(gates))
    assert used == list(range(10)), used

    rng = np.random.default_rng(0)
    init = "".join(
        "1" if q in (0, 1, 2, 3, 5, 6, 7, 8) else "0" for q in range(10)
    )
    # Compare winning circuit to original-string circuit with same theta
    # Need matching theta index: both use t0,t1,t2 for doubles 0,1,2
    ovs = []
    for _ in range(5):
        th = rng.uniform(-0.3, 0.3, 3)
        qa = gates_to_qc(ref["gates"], 10, th)
        qb = gates_to_qc(gates, 10, th)
        ovs.append(statevector_overlap(qa, qb, 10, init_bits=init))
    print(f"\nWinner: err={best['budget'].error:.6f} fid={best['budget'].fidelity:.6f}")
    print(f"  vs baseline Δerr={base_bud.error - best['budget'].error:.6f}")
    print(f"  rzx pairs: {rzx_pairs(gates)}")
    print(f"  qubits used: {used}")
    print(f"  frozen: {best['frozen_strings']}")
    print(f"  keep: {best['keep_pairs']}")
    print(f"  schedule: {best['schedule']}")
    print(f"  overlap min/mean: {min(ovs):.6f} / {np.mean(ovs):.6f}")

    if best["budget"].error >= base_bud.error:
        raise SystemExit(
            f"best legal err {best['budget'].error} does not beat baseline "
            f"{base_bud.error}"
        )
    if min(ovs) < 0.999:
        print("WARNING: overlap < 0.999 — selective freeze may need recheck")

    payload = {
        "molecule": "Cl2",
        "num_qubits": 10,
        "method": "disjoint_rzx_all_qubits",
        "original_strings": strings,
        "frozen_strings": best["frozen_strings"],
        "keep_pairs": best["keep_pairs"],
        "hub_schedule": [list(p) for p in best["schedule"]],
        "order": list(best["order"]),
        "signs": signs,
        "doubles": doubles,
        "budget": best["budget"].as_dict(),
        "baseline_budget": base_bud.as_dict(),
        "rzx_pairs": [list(p) for p in rzx_pairs(gates)],
        "qubits_used": used,
        "overlap_min": float(min(ovs)),
        "gates": gates,
        "cross_model": "spin_split_alpha_beta",
        "rules": [
            "all qubits used",
            "RZX pairs vertex-disjoint (any α↔β, no shared index)",
            "within-spin CZ: NN + chords (0,3)/(5,8); no cross-spin CZ",
        ],
        "cz_pairs": [list(p) for p in cz_pairs(gates)],
        "allowed_cz_edges": [list(sorted(e)) for e in sorted(allowed_cz_edges(10), key=lambda e: tuple(sorted(e)))],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    OUT_JSON2.write_text(json.dumps(payload, indent=2))

    qc = gates_to_qc(gates, 10)
    fig = qc.draw(output="mpl", fold=-1, style=gen.IQP_STYLE, idle_wires=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_PNG2, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT_JSON2} and {OUT_PNG2}")
    print(qc.draw(output="text", fold=120))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
