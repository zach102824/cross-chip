#!/usr/bin/env python3
"""Export HF + Cl2 disjoint-RZX winners.

Naming (consistent across molecules):
  {Molecule}_{nq}q_disjoint_rzx.{json,png}
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "state_transfer"))
sys.path.insert(0, str(_ROOT / "UCCSD circuit"))

from constraints import rzx_pairs, satisfies_rules  # noqa: E402
from error_budget import cross_from_list, score_gates, spin_split_cross_pairs  # noqa: E402
from flexible_compile import (  # noqa: E402
    candidate_bridges,
    compile_strings,
    gates_to_qc,
    gen,
    statevector_overlap,
)
from freeze import all_keep_masks  # noqa: E402
from methods import method_independent_pairs  # noqa: E402
import taper_lib  # noqa: E402

# stem = {Molecule}_{nq}q_disjoint_rzx
STEM = {
    "HF": "HF_6q_disjoint_rzx",
    "Cl2": "Cl2_10q_disjoint_rzx",
}


def hf_case():
    taper = taper_lib.build_taper_data(n_spatial=4, n_electrons=6)
    doubles = [(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)]
    strings, signs = [], []
    for d in doubles:
        jw = "".join(gen.jw_string_for_double(8, d))
        ts, sg = taper_lib.taper_pauli_string(jw, taper)
        strings.append(ts)
        signs.append(int(sg))
    hits = method_independent_pairs(
        strings, 6, signs=signs, cross_pairs=spin_split_cross_pairs(6)
    )
    best = hits[0]
    ok, why = satisfies_rules(best["gates"], 6)
    assert ok, why
    ref = compile_strings(strings, signs=signs, order="given", fuse=True)
    rng = np.random.default_rng(0)
    ovs = []
    for _ in range(6):
        th = rng.uniform(-0.3, 0.3, 3)
        qa = gates_to_qc(ref["gates"], 6, th)
        qb = gates_to_qc(best["gates"], 6, th)
        ovs.append(statevector_overlap(qa, qb, 6, init_bits=taper.hf_bitstring_tapered))
    return {
        "molecule": "HF",
        "num_qubits": 6,
        "method": best["name"],
        "original_strings": strings,
        "frozen_strings": best["frozen_strings"],
        "signs": signs,
        "doubles_full": doubles,
        "budget": best["budget"].as_dict(),
        "rzx_pairs": [list(p) for p in rzx_pairs(best["gates"])],
        "overlap_min": float(min(ovs)),
        "gates": best["gates"],
        "cross_model": "spin_split_alpha_beta",
        "rules": [
            "all qubits used",
            "RZX pairs vertex-disjoint (any α↔β, no shared index)",
            "within-spin CZ: NN + chords (0,3)/(half,half+3); no cross-spin CZ",
        ],
    }


def _cl2_exhaustive_min(strings, signs, n=10):
    """Exhaustive keep-mask × vertex-disjoint bridge scan.

    Picks the constraint-legal circuit minimizing (total 2q, depth, total 1q).
    """
    mask_lists = [list(all_keep_masks(s)) for s in strings]
    best = None
    for combo in itertools.product(*mask_lists):
        frozen = [c[1] for c in combo]
        keeps = [sorted(c[0]) for c in combo]
        cands = [candidate_bridges(s, n) for s in frozen]

        def schedules(i, used, acc):
            if i == len(cands):
                yield list(acc)
                return
            for a, b in cands[i]:
                if a not in used and b not in used:
                    yield from schedules(i + 1, used | {a, b}, acc + [(a, b)])

        for sched in schedules(0, set(), []):
            try:
                out = compile_strings(
                    frozen, signs=signs, order="given",
                    hub_schedule=sched, fuse=True,
                )
            except Exception:
                continue
            gates = out["gates"]
            if not satisfies_rules(gates, n)[0]:
                continue
            n2 = sum(len(g["qubits"]) == 2 for g in gates)
            n1 = sum(len(g["qubits"]) == 1 for g in gates)
            key = (n2, gates_to_qc(gates, n).depth(), n1)
            if best is None or key < best["key"]:
                best = {
                    "key": key,
                    "name": "disjoint_rzx_all_qubits",
                    "gates": gates,
                    "frozen_strings": frozen,
                    "keep_pairs": keeps,
                    "schedule": sched,
                    "budget": score_gates(gates, spin_split_cross_pairs(n)),
                }
    return best


def cl2_case():
    # Order chosen so the (2q, depth, 1q)-optimal schedule keeps t_k ↔ doubles[k].
    doubles = [(4, 9, 5, 0), (4, 9, 6, 1), (4, 9, 7, 2)]
    strings = ["".join(gen.jw_string_for_double(10, d)) for d in doubles]
    signs = [1, 1, 1]
    best = _cl2_exhaustive_min(strings, signs)
    assert best, "no Cl2 circuit under disjoint/all-qubit rules"
    ok, why = satisfies_rules(best["gates"], 10)
    assert ok, why
    ref = compile_strings(strings, signs=signs, order="given", fuse=True)
    rng = np.random.default_rng(0)
    init = "".join("1" if q in (0, 1, 2, 3, 5, 6, 7, 8) else "0" for q in range(10))
    ovs = []
    for _ in range(6):
        th = rng.uniform(-0.3, 0.3, 3)
        qa = gates_to_qc(ref["gates"], 10, th)
        qb = gates_to_qc(best["gates"], 10, th)
        ovs.append(statevector_overlap(qa, qb, 10, init_bits=init))
    return {
        "molecule": "Cl2",
        "num_qubits": 10,
        "method": best["name"],
        "original_strings": strings,
        "frozen_strings": best["frozen_strings"],
        "keep_pairs": best["keep_pairs"],
        "hub_schedule": [list(p) for p in best["schedule"]],
        "signs": signs,
        "doubles": doubles,
        "budget": best["budget"].as_dict(),
        "depth": gates_to_qc(best["gates"], 10).depth(),
        "rzx_pairs": [list(p) for p in rzx_pairs(best["gates"])],
        "overlap_min": float(min(ovs)),
        "gates": best["gates"],
        "cross_model": "spin_split_alpha_beta",
        "rules": [
            "all qubits used",
            "RZX pairs vertex-disjoint (any α↔β, no shared index)",
            "within-spin CZ: NN + chords (0,3)/(5,8); no cross-spin CZ",
        ],
    }


def main():
    hf = hf_case()
    cl2 = cl2_case()
    base_hf = json.loads(
        (_ROOT / "state_transfer/circuits2read/HF_tapered_6q_3doubles_rzx.json").read_text()
    )
    base_cl2 = json.loads(
        (_ROOT / "June_main/circuits2read/Cl2_10q_3doubles_rzx.json").read_text()
    )
    hf_base = score_gates(base_hf["gates"], cross_from_list([(2, 5)])).as_dict()
    cl2_base = score_gates(
        base_cl2["gates"], cross_from_list([(3, 8), (0, 1), (5, 6)])
    ).as_dict()

    summary = {
        "baselines": {"HF_6q": hf_base, "Cl2_10q": cl2_base},
        "winners": {"HF_6q": hf, "Cl2_10q": cl2},
        "verdict": {
            "HF_error_reduction": hf_base["error"] - hf["budget"]["error"],
            "Cl2_error_reduction": cl2_base["error"] - cl2["budget"]["error"],
            "both_beat_baseline": (
                hf["budget"]["error"] < hf_base["error"]
                and cl2["budget"]["error"] < cl2_base["error"]
            ),
            "cl2_rules": cl2["rules"],
        },
    }
    (_HERE / "winners.json").write_text(json.dumps(summary, indent=2))

    # Drop legacy HF name if present
    for legacy in (
        "HF_6q_indep_pairs_freeze.json",
        "HF_6q_indep_pairs_freeze_circuit.png",
    ):
        p = _HERE / legacy
        if p.exists():
            p.unlink()

    for stem, data, n in (
        (STEM["HF"], hf, 6),
        (STEM["Cl2"], cl2, 10),
    ):
        (_HERE / f"{stem}.json").write_text(json.dumps(data, indent=2))
        qc = gates_to_qc(data["gates"], n)
        fig = qc.draw(output="mpl", fold=-1, style=gen.IQP_STYLE, idle_wires=True)
        fig.savefig(_HERE / f"{stem}_circuit.png", dpi=160, bbox_inches="tight")
        print(f"wrote {stem}.json / {stem}_circuit.png")

    print("HF:", hf["budget"], "rzx", hf["rzx_pairs"], "overlap", hf["overlap_min"])
    print("Cl2:", cl2["budget"], "rzx", cl2["rzx_pairs"], "overlap", cl2["overlap_min"])
    print("Verdict:", summary["verdict"])
    assert summary["verdict"]["both_beat_baseline"]


if __name__ == "__main__":
    main()
