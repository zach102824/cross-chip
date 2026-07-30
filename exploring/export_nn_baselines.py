#!/usr/bin/env python3
"""Export NN-only baselines for HF + Cl2.

HF: same selective-freeze + *fixed* RZX bridge (2,5) as
``state_transfer/.../HF_tapered_6q_3doubles_freeze_fixed_rzx``, but within-spin
CZ is NN-only — no direct (0,2)/(3,5).  Couriers ``keep_pairs=[{1,2},{2},∅]``
restore Steiner nodes so fan-in can walk 0–1–2 and 3–4–5.

Cl2: same freeze + disjoint-RZX strategy as the exploring winner, but no
(0,3)/(5,8) chords.

Writes:
  HF_6q_freeze_fixed_rzx.{json,png}            # with (0,2)/(3,5)
  HF_6q_freeze_fixed_rzx_nn_only.{json,png}    # NN-only baseline
  Cl2_10q_2doubles_disjoint_rzx_nn_only.{json,png}
  chord_savings.json
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

from constraints import cz_pairs, rzx_pairs, satisfies_rules  # noqa: E402
from error_budget import score_gates, spin_split_cross_pairs  # noqa: E402
from flexible_compile import (  # noqa: E402
    compile_flexible,
    compile_strings,
    gates_to_qc,
    gen,
    prog_to_gates,
    statevector_overlap,
)
from freeze import freeze_z_pairs  # noqa: E402
from methods import method_disjoint_rzx_all_qubits  # noqa: E402

HF_BRIDGE = (2, 5)
# Couriers so NN fan-in 0–1–2 / 3–4–5 reaches the fixed hub (2,5).
HF_KEEP_NN = [{1, 2}, {2}, set()]
# Same keep as state_transfer freeze_fixed (allows direct 0–2 / 3–5).
HF_KEEP_FIXED = [{2}, {2}, set()]

CL2_DOUBLES = [(4, 9, 6, 1), (4, 9, 5, 0)]
CL2_ALPHA_HUBS = {0, 1, 2}
CL2_BETA_HUBS = {5, 6, 7}

RULES_HF_NN = [
    "selective Z-pair freeze + fixed RZX bridge (2,5)",
    "within-spin CZ: NN only — no (0,2)/(3,5); no cross-spin CZ",
]
RULES_CL2_NN = [
    "all qubits used",
    "RZX pairs vertex-disjoint (any α↔β, no shared index)",
    "within-spin CZ: NN only (no chords (0,3)/(5,8)); no cross-spin CZ",
]


def _overlap_vs_ref(ref_gates, cand_gates, n, n_params, init_bits, seed=0, trials=6):
    rng = np.random.default_rng(seed)
    ovs = []
    for _ in range(trials):
        th = rng.uniform(-0.3, 0.3, n_params)
        qa = gates_to_qc(ref_gates, n, th)
        qb = gates_to_qc(cand_gates, n, th)
        ovs.append(statevector_overlap(qa, qb, n, init_bits=init_bits))
    return float(min(ovs))


def _compile_fixed_hub(strings, signs, hub, n, *, chords: bool | None):
    """Fixed α–β hub. chords=None → legacy unconstrained row fan-in (_row_chain)."""
    ha, hb = hub
    use_graph = chords is not None
    prog, expected, bridges = [], [], []
    for d, s in enumerate(strings):
        kw = dict(hub_a=ha, hub_b=hb, hub_hint=ha, use_cz_graph=use_graph)
        if use_graph:
            kw["chords"] = bool(chords)
        prefix, pivot, ph, bridge = compile_flexible(s, n, **kw)
        bridges.append(bridge)
        prog += prefix
        prog.append(("ROT", pivot, d, signs[d] * ph))
        prog += gen._invert(prefix)
        expected.append((s, ph))
    prog = gen._peephole(prog)
    gen._verify_program(prog, n, expected)
    return gen.fuse_cz_rot_cz_to_rzx(prog_to_gates(prog)), bridges


def _hf_meta():
    base = json.loads(
        (_ROOT / "state_transfer/circuits2read/HF_tapered_6q_3doubles_rzx.json").read_text()
    )
    strings = list(base["tapered_strings"])
    signs = [int(s) for s in base["signs"]]
    return base, strings, signs


def hf_freeze_fixed(*, nn_only: bool):
    """Freeze + fixed RZX (2,5); nn_only forbids direct (0,2)/(3,5)."""
    base, strings, signs = _hf_meta()
    n = 6
    keep = HF_KEEP_NN if nn_only else HF_KEEP_FIXED
    frozen = [freeze_z_pairs(s, k) for s, k in zip(strings, keep)]
    if nn_only:
        gates, bridges = _compile_fixed_hub(
            frozen, signs, HF_BRIDGE, n, chords=False
        )
    else:
        # Match state_transfer/generate_frozen_fixed_rzx.py (unconstrained fan-in).
        gates, bridges = _compile_fixed_hub(
            frozen, signs, HF_BRIDGE, n, chords=None
        )

    assert all(b == HF_BRIDGE for b in bridges), bridges
    assert all(tuple(sorted(p)) == HF_BRIDGE for p in rzx_pairs(gates)), rzx_pairs(gates)
    if nn_only:
        for a, b in ((0, 2), (3, 5)):
            assert (a, b) not in cz_pairs(gates), f"NN-only still has CZ {(a, b)}"
        ok, why = satisfies_rules(gates, n, chords=False)
        # Fixed RZX reuses (2,5) → rzx_overlap is expected; only CZ legality matters.
        assert why in ("ok", "rzx_overlap"), why
        assert all(
            frozenset(p) in {frozenset(e) for e in ((0, 1), (1, 2), (3, 4), (4, 5))}
            for p in cz_pairs(gates)
        ), cz_pairs(gates)

    depth = gates_to_qc(gates, n).depth()
    bud = score_gates(gates, spin_split_cross_pairs(n), depth=depth)
    ref = compile_strings(
        strings, signs=signs, order="given",
        hub_schedule=[HF_BRIDGE] * 3, fuse=True,
    )
    ov = _overlap_vs_ref(ref["gates"], gates, n, 3, base["hf_bitstring_tapered"])

    return {
        "molecule": "HF",
        "num_qubits": n,
        "method": "freeze_fixed_rzx",
        "connectivity": "nn_only" if nn_only else "row_chain_with_skip",
        "chords": False if nn_only else None,
        "original_strings": strings,
        "frozen_strings": frozen,
        "keep_pairs": [sorted(k) for k in keep],
        "bridge_pair": list(HF_BRIDGE),
        "hub_schedule": [list(HF_BRIDGE)] * 3,
        "signs": signs,
        "doubles_full": base["doubles"],
        "budget": bud.as_dict(),
        "depth": depth,
        "rzx_pairs": [list(p) for p in rzx_pairs(gates)],
        "cz_pairs": [list(p) for p in cz_pairs(gates)],
        "overlap_min": ov,
        "gates": gates,
        "cross_model": "spin_split_alpha_beta",
        "hf_bitstring_tapered": base["hf_bitstring_tapered"],
        "note": (
            "NN-only: keep Z1Z4+Z2Z5 couriers so fan-in walks 0–1–2 / 3–4–5; "
            "no direct CZ(0,2)/(3,5)."
            if nn_only else
            "Unconstrained row fan-in (same as state_transfer freeze_fixed); "
            "allows skip edges CZ(0,2)/(3,5)."
        ),
        "rules": RULES_HF_NN if nn_only else [
            "selective Z-pair freeze + fixed RZX bridge (2,5)",
            "within-spin CZ: unconstrained row fan-in (allows (0,2)/(3,5))",
        ],
    }


def _cl2_hub_ok(schedule) -> bool:
    for a, b in schedule:
        if a is None:
            continue
        if a not in CL2_ALPHA_HUBS or b not in CL2_BETA_HUBS:
            return False
    return True


def cl2_nn_baseline():
    n = 10
    signs = [1, 1]
    strings = ["".join(gen.jw_string_for_double(n, d)) for d in CL2_DOUBLES]
    hits = method_disjoint_rzx_all_qubits(
        strings, signs=signs, cross_pairs=spin_split_cross_pairs(n),
        max_mask_opts=8, chords=False,
    )
    restricted = [h for h in hits if _cl2_hub_ok(h["schedule"])]
    pool = restricted if restricted else hits
    assert pool, "no NN-only Cl2 circuit under freeze+disjoint-RZX"
    best = pool[0]
    gates = best["gates"]
    ok, why = satisfies_rules(gates, n, chords=False)
    assert ok, why
    for a, b in ((0, 3), (5, 8)):
        for g in gates:
            if g["op"].lower() == "cz" and set(g["qubits"]) == {a, b}:
                raise AssertionError(f"chord CZ {(a, b)} present in NN-only circuit")

    bud = score_gates(gates, spin_split_cross_pairs(n))
    depth = gates_to_qc(gates, n).depth()
    ref = compile_strings(strings, signs=signs, order="given", fuse=True)
    init = "".join("1" if q in (0, 1, 2, 3, 5, 6, 7, 8) else "0" for q in range(n))
    ov = _overlap_vs_ref(ref["gates"], gates, n, 2, init)
    return {
        "molecule": "Cl2",
        "num_qubits": n,
        "method": "disjoint_rzx_all_qubits",
        "connectivity": "nn_only",
        "chords": False,
        "original_strings": strings,
        "frozen_strings": best["frozen_strings"],
        "keep_pairs": best["keep_pairs"],
        "hub_schedule": [list(p) for p in best["schedule"]],
        "signs": signs,
        "doubles": [list(d) for d in CL2_DOUBLES],
        "budget": {**bud.as_dict(), "depth": depth},
        "depth": depth,
        "rzx_pairs": [list(p) for p in rzx_pairs(gates)],
        "overlap_min": ov,
        "gates": gates,
        "cross_model": "spin_split_alpha_beta",
        "rzx_constraint": "no q3/q8 hub; hubs from {0,1,2}x{5,6,7} when feasible",
        "hub_restriction_applied": bool(restricted),
        "rules": RULES_CL2_NN + [
            "q3/q8 forbidden as RZX hub; α∈{0,1,2}, β∈{5,6,7} when feasible",
        ],
    }


def main():
    hf_skip = hf_freeze_fixed(nn_only=False)
    hf_nn = hf_freeze_fixed(nn_only=True)
    cl2 = cl2_nn_baseline()
    chorded_cl2 = json.loads((_HERE / "Cl2_10q_2doubles_disjoint_rzx.json").read_text())

    def _save(stem, data, n):
        (_HERE / f"{stem}.json").write_text(json.dumps(data, indent=2) + "\n")
        qc = gates_to_qc(data["gates"], n)
        fig = qc.draw(output="mpl", fold=-1, style=gen.IQP_STYLE, idle_wires=True)
        fig.savefig(_HERE / f"{stem}_circuit.png", dpi=160, bbox_inches="tight")
        print(f"wrote {stem}.json / {stem}_circuit.png")

    _save("HF_6q_freeze_fixed_rzx", hf_skip, 6)
    _save("HF_6q_freeze_fixed_rzx_nn_only", hf_nn, 6)
    _save("Cl2_10q_2doubles_disjoint_rzx_nn_only", cl2, 10)

    # Remove obsolete identical-to-winner HF nn_only stem.
    for legacy in (
        "HF_6q_disjoint_rzx_nn_only.json",
        "HF_6q_disjoint_rzx_nn_only_circuit.png",
    ):
        p = _HERE / legacy
        if p.exists():
            p.unlink()
            print(f"removed {legacy}")

    def delta(with_skip, nn, *, rzx_key="rzx_pairs"):
        ce, ne = with_skip["budget"]["error"], nn["budget"]["error"]
        return {
            "with_skip_error": ce,
            "nn_only_error": ne,
            "error_saved_by_skip_edges": ne - ce,
            "with_skip_n_onchip": with_skip["budget"]["n_onchip"],
            "nn_only_n_onchip": nn["budget"]["n_onchip"],
            "onchip_cz_saved_by_skip_edges": (
                nn["budget"]["n_onchip"] - with_skip["budget"]["n_onchip"]
            ),
            "with_skip_depth": with_skip["budget"].get("depth") or with_skip.get("depth"),
            "nn_only_depth": nn["budget"].get("depth") or nn.get("depth"),
            "with_skip_rzx": with_skip.get(rzx_key) or with_skip.get("rzx_pairs"),
            "nn_only_rzx": nn.get("rzx_pairs"),
            "with_skip_cz": with_skip.get("cz_pairs"),
            "nn_only_cz": nn.get("cz_pairs"),
        }

    summary = {
        "description": (
            "HF: freeze+fixed RZX(2,5) with skip edges (0,2)/(3,5) vs NN-only. "
            "Cl2: freeze+disjoint RZX with chords (0,3)/(5,8) vs NN-only."
        ),
        "HF_6q": {
            **delta(hf_skip, hf_nn),
            "note": (
                "Skip edges = direct CZ(0,2)/(3,5) in unconstrained row fan-in. "
                "NN-only keeps Z1Z4 so paths walk 0–1–2 / 3–4–5."
            ),
        },
        "Cl2_10q_2doubles": {
            "chorded_error": chorded_cl2["budget"]["error"],
            "nn_only_error": cl2["budget"]["error"],
            "error_saved_by_chords": cl2["budget"]["error"] - chorded_cl2["budget"]["error"],
            "chorded_n_onchip": chorded_cl2["budget"]["n_onchip"],
            "nn_only_n_onchip": cl2["budget"]["n_onchip"],
            "onchip_cz_saved_by_chords": (
                cl2["budget"]["n_onchip"] - chorded_cl2["budget"]["n_onchip"]
            ),
            "chorded_depth": chorded_cl2.get("depth"),
            "nn_only_depth": cl2.get("depth"),
            "chorded_rzx": chorded_cl2["rzx_pairs"],
            "nn_only_rzx": cl2["rzx_pairs"],
        },
    }
    (_HERE / "chord_savings.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("wrote chord_savings.json")
    print("HF:", summary["HF_6q"])
    print("Cl2:", summary["Cl2_10q_2doubles"])


if __name__ == "__main__":
    main()
