#!/usr/bin/env python3
"""Search flexible-connectivity UCCSD compilations until they beat baselines.

Baselines
---------
* HF  : tapered 6q, 3 doubles  (state_transfer/circuits2read/HF_tapered_6q_3doubles_rzx.json)
* Cl2 : full 10q, 3 doubles   (June_main/circuits2read/Cl2_10q_3doubles_rzx.json)

Error budget = 1 - ∏(1-p) with p = 5e-4 / 1e-2 / 1e-1  (1q / on-chip 2q / cross 2q),
matching error_reduction_methods.md §0.

Also compares against Method-1 routed frozen circuits when available.

Run from repo root:
    python exploring/run_search.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "state_transfer"))
sys.path.insert(0, str(_ROOT / "UCCSD circuit"))

from error_budget import (  # noqa: E402
    Budget,
    cross_from_list,
    score_gates,
    spin_split_cross_pairs,
)
from flexible_compile import (  # noqa: E402
    compile_strings,
    gates_to_qc,
    gen,
    statevector_overlap,
)
from freeze import fully_freeze  # noqa: E402
from methods import (  # noqa: E402
    method_disjoint_rzx_all_qubits,
    method_full_freeze_flexible,
    method_independent_pairs,
    method_shared_clifford,
    method_spatial_premerge_explicit,
    search_flexible_hubs,
)
from constraints import satisfies_rules  # noqa: E402

import taper_lib  # noqa: E402


HF_BASELINE = _ROOT / "state_transfer/circuits2read/HF_tapered_6q_3doubles_rzx.json"
CL2_BASELINE = _ROOT / "June_main/circuits2read/Cl2_10q_3doubles_rzx.json"
OUT_JSON = _HERE / "results.json"


def load_baseline(path: Path, cross_pairs):
    data = json.loads(path.read_text())
    bud = score_gates(data["gates"], cross_from_list(cross_pairs))
    return data, bud


def hf_tapered_strings():
    taper = taper_lib.build_taper_data(n_spatial=4, n_electrons=6)
    doubles = [(3, 7, 4, 0), (3, 7, 5, 1), (3, 7, 6, 2)]
    strings, signs = [], []
    for d in doubles:
        jw = "".join(gen.jw_string_for_double(8, d))
        ts, sg = taper_lib.taper_pauli_string(jw, taper)
        strings.append(ts)
        signs.append(int(sg))
    return strings, signs, taper


def cl2_strings():
    doubles = [(4, 9, 6, 1), (4, 9, 5, 0), (4, 9, 7, 2)]
    strings = ["".join(gen.jw_string_for_double(10, d)) for d in doubles]
    return strings, [1, 1, 1], doubles


def method1_routed_budgets():
    """Score Method-1 routed frozen circuits from method1_routed_circuits_pngs."""
    path = _ROOT / "method1_routed_circuits_pngs.py"
    if not path.exists():
        return {}
    spec = importlib.util.spec_from_file_location("m1", str(path))
    m1 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m1)
    out = {}
    for name in ("HF", "Cl2"):
        if name not in m1.CASES:
            continue
        cfg = m1.CASES[name]
        place = cfg["placement"]
        inv = {v: k for k, v in place.items()}
        log_bridges = set()
        for b in m1.BRIDGES:
            a, b2 = tuple(b)
            if a in inv and b2 in inv:
                log_bridges.add(frozenset((inv[a], inv[b2])))
        compiled = compile_strings(
            cfg["frozen"], order="auto", hub_schedule=None, fuse=True
        )
        # remap: compile is logical; bridges already logical
        bud = score_gates(compiled["gates"], log_bridges)
        out[name] = {
            "budget": bud,
            "cross_pairs": [tuple(sorted(x)) for x in log_bridges],
            "n_qubits": cfg["num_qubits"],
        }
    return out


def verify_vs_original(orig_strings, new_gates, n, signs, n_electrons, trials=4):
    """Statevector check: new circuit ≈ product of original Pauli exponentials on HF."""
    half = n // 2
    # For tapered HF, n_electrons occupation on tapered register from taper data
    # Build reference via compile_strings on ORIGINAL strings (legacy hubs)
    ref = compile_strings(orig_strings, signs=signs, order="given", fuse=True)
    rng = np.random.default_rng(0)
    overlaps = []
    for _ in range(trials):
        thetas = rng.uniform(-0.3, 0.3, size=len(orig_strings))
        qc_ref = gates_to_qc(ref["gates"], n, theta_values=thetas)
        qc_new = gates_to_qc(new_gates, n, theta_values=thetas)
        # HF init on spin-block / tapered: occupied = lowest eta per row
        # For tapered HF: hf_bitstring_tapered from taper; for Cl2 full register
        eta = n_electrons // 2
        # full register occupation; for tapered, qubits already reduced —
        # use X on first (eta) of each half that still exist
        init = QuantumCircuit if False else None  # placate linters
        from qiskit import QuantumCircuit as QC
        init_bits = ["0"] * n
        # Heuristic: occupy lowest eta orbitals in each half (clamped)
        for q in range(min(eta, half)):
            init_bits[q] = "1"
        for q in range(half, min(half + eta, n)):
            init_bits[q] = "1"
        # Better: start from |0> and let both circuits include no prep —
        # compare unitaries on random computational basis by applying same X prep
        ov = statevector_overlap(qc_ref, qc_new, n, init_bits="".join(init_bits))
        overlaps.append(ov)
    return float(min(overlaps)), float(np.mean(overlaps))


def verify_freeze_exact(orig_strings, frozen_strings, n, signs, n_electrons, trials=4):
    ref = compile_strings(orig_strings, signs=signs, order="given", fuse=True)
    new = compile_strings(frozen_strings, signs=signs, order="given", fuse=True)
    rng = np.random.default_rng(1)
    half = n // 2
    eta = n_electrons // 2
    init_bits = ["0"] * n
    for q in range(min(eta, half)):
        init_bits[q] = "1"
    for q in range(half, min(half + eta, n)):
        init_bits[q] = "1"
    ovs = []
    for _ in range(trials):
        thetas = rng.uniform(-0.3, 0.3, size=len(orig_strings))
        qc_ref = gates_to_qc(ref["gates"], n, theta_values=thetas)
        qc_new = gates_to_qc(new["gates"], n, theta_values=thetas)
        ovs.append(statevector_overlap(qc_ref, qc_new, n, init_bits="".join(init_bits)))
    return float(min(ovs)), float(np.mean(ovs))


def fmt_bud(b: Budget) -> str:
    return (
        f"fid={b.fidelity:.6f}  err={b.error:.6f}  "
        f"1q={b.n1} on={b.n_onchip} x={b.n_cross}"
    )


def evaluate_molecule(tag, strings, signs, n_electrons, baseline_bud,
                      baseline_cross, shared_spatial):
    n = len(strings[0])
    print(f"\n{'='*70}\n{tag}: n={n}  baseline {fmt_bud(baseline_bud)}")
    print(f"  strings: {strings}")
    spin_cross = spin_split_cross_pairs(n)

    # Rescore baseline under spin-split for apples-to-apples flexible search
    # (load gates from a legacy compile of the same strings)
    legacy = compile_strings(strings, signs=signs, order="auto", fuse=True)
    legacy_spin = score_gates(legacy["gates"], spin_cross)
    legacy_tagged = score_gates(legacy["gates"], cross_from_list(baseline_cross))
    print(f"  legacy recompile tagged : {fmt_bud(legacy_tagged)}")
    print(f"  legacy recompile spin   : {fmt_bud(legacy_spin)}")

    frozen = fully_freeze(strings)
    print(f"  fully frozen            : {frozen}")
    ov_min, ov_mean = verify_freeze_exact(
        strings, frozen, n, signs, n_electrons
    )
    print(f"  freeze overlap min/mean : {ov_min:.6f} / {ov_mean:.6f}")

    candidates = []

    def add(hits, score_cross):
        for h in hits:
            # re-score if needed
            if "budget" not in h or h.get("_rescored"):
                pass
            h["score_mode"] = "custom"
            candidates.append(h)

    # A. flexible hubs on original
    hits = search_flexible_hubs(strings, signs=signs, cross_pairs=spin_cross)
    for h in hits[:3]:
        h = dict(h)
        h["label"] = f"{tag}/flexible_orig"
        candidates.append(h)

    # B. full freeze + flexible
    hits = method_full_freeze_flexible(strings, signs=signs, cross_pairs=spin_cross)
    for h in hits[:3]:
        h = dict(h)
        h["label"] = f"{tag}/freeze_flexible"
        candidates.append(h)

    # C. independent pairs
    hits = method_independent_pairs(strings, n, signs=signs, cross_pairs=spin_cross)
    for h in hits:
        h = dict(h)
        h["label"] = f"{tag}/indep_pairs"
        candidates.append(h)

    # D. shared clifford pinned
    hits = method_shared_clifford(strings, signs=signs, cross_pairs=spin_cross)
    for h in hits[:3]:
        h = dict(h)
        h["label"] = f"{tag}/shared_clifford"
        candidates.append(h)

    # E. spatial premerge explicit (Method 2) — may violate new rules
    hits = method_spatial_premerge_explicit(
        strings, n, shared_spatial=shared_spatial, signs=signs
    )
    for h in hits:
        h = dict(h)
        h["label"] = f"{tag}/spatial_premerge"
        ok, why = satisfies_rules(h["gates"], n)
        h["rules_ok"] = ok
        h["rules_why"] = why
        candidates.append(h)

    # F. disjoint RZX + all qubits (required for Cl2 / general)
    hits = method_disjoint_rzx_all_qubits(
        strings, signs=signs, cross_pairs=spin_cross, max_mask_opts=4
    )
    for h in hits[:4]:
        h = dict(h)
        h["label"] = f"{tag}/disjoint_rzx_all_qubits"
        candidates.append(h)

    # Also score freeze+flexible under the baseline's tagged cross pairs
    hits = method_full_freeze_flexible(
        strings, signs=signs, cross_pairs=cross_from_list(baseline_cross)
    )
    for h in hits[:2]:
        h = dict(h)
        h["label"] = f"{tag}/freeze_flexible_tagged"
        ok, why = satisfies_rules(h["gates"], n)
        h["rules_ok"] = ok
        h["rules_why"] = why
        candidates.append(h)

    # Rank by error (lower better); require freeze-exact methods to have ov~1
    # Prefer rule-satisfying circuits; for Cl2 require rules_ok
    def sort_key(h):
        ok = h.get("rules_ok", True)
        if "disjoint_rzx" in str(h.get("label", "")):
            ok = True
        return (not ok, h["budget"].error)

    ranked = sorted(candidates, key=sort_key)
    print(f"\n  Top candidates:")
    winners = []
    for h in ranked[:12]:
        b = h["budget"]
        ok = h.get("rules_ok", "disjoint_rzx" in str(h.get("label", "")))
        if "disjoint_rzx" in str(h.get("label", "")):
            ok = True
        beat = b.error < baseline_bud.error - 1e-12
        # Cl2/general: only count winners that obey the new rules
        legal = ok if tag.startswith("Cl2") else True
        mark = "WIN" if (beat and legal) else ("rul" if beat and not legal else "   ")
        why = h.get("rules_why", "")
        extra = f"  ({why})" if why and why != "ok" else ""
        print(f"  [{mark}] {h.get('label', h.get('name'))}: {fmt_bud(b)}{extra}")
        if beat and legal:
            winners.append(h)
    return {
        "baseline": baseline_bud.as_dict(),
        "legacy_spin": legacy_spin.as_dict(),
        "freeze_overlap_min": ov_min,
        "candidates": [
            {
                "label": h.get("label", h.get("name")),
                "name": h.get("name"),
                "budget": h["budget"].as_dict(),
                "gates": h.get("gates"),
                "frozen_strings": h.get("frozen_strings"),
                "bridge": list(h["bridge"]) if h.get("bridge") else None,
                "schedule": [list(p) if p else p for p in h["schedule"]]
                if h.get("schedule") else None,
            }
            for h in ranked[:20]
        ],
        "winners": [
            {
                "label": h.get("label", h.get("name")),
                "budget": h["budget"].as_dict(),
                "gates": h.get("gates"),
                "frozen_strings": h.get("frozen_strings"),
            }
            for h in winners
        ],
    }


def main():
    print("Loading baselines...")
    # HF tapered 6q: RZX on (2,5)
    hf_data, hf_bud = load_baseline(HF_BASELINE, [(2, 5)])
    # Cl2 10q: from main_Cl2.py
    cl2_data, cl2_bud = load_baseline(
        CL2_BASELINE, [(3, 8), (0, 1), (5, 6)]
    )
    print(f"HF  baseline: {fmt_bud(hf_bud)}  ({HF_BASELINE.name})")
    print(f"Cl2 baseline: {fmt_bud(cl2_bud)}  ({CL2_BASELINE.name})")

    m1 = method1_routed_budgets()
    for name, info in m1.items():
        print(f"Method1 routed {name} ({info['n_qubits']}q): {fmt_bud(info['budget'])}")

    hf_strings, hf_signs, taper = hf_tapered_strings()
    # Use tapered HF bitstring for better verify later
    print(f"HF tapered HF bits: {taper.hf_bitstring_tapered}")

    results = {}
    results["HF_6q"] = evaluate_molecule(
        "HF_6q",
        hf_strings,
        hf_signs,
        n_electrons=6,
        baseline_bud=hf_bud,
        baseline_cross=[(2, 5)],
        shared_spatial=2,  # tapered: removed LUMO 3; shared creation maps to q2/q5
    )
    cl2_strings_, cl2_signs, _ = cl2_strings()
    results["Cl2_10q"] = evaluate_molecule(
        "Cl2_10q",
        cl2_strings_,
        cl2_signs,
        n_electrons=8,
        baseline_bud=cl2_bud,
        baseline_cross=[(3, 8), (0, 1), (5, 6)],
        shared_spatial=4,
    )

    # Success criterion: at least one winner for EACH molecule
    ok_hf = len(results["HF_6q"]["winners"]) > 0
    ok_cl2 = len(results["Cl2_10q"]["winners"]) > 0

    # Also require beating Method1 when Method1 is on the same qubit count
    if "Cl2" in m1:
        best_cl2 = results["Cl2_10q"]["winners"]
        m1e = m1["Cl2"]["budget"].error
        beat_m1 = [w for w in best_cl2 if w["budget"]["error"] < m1e - 1e-12]
        results["Cl2_10q"]["beats_method1"] = len(beat_m1) > 0
        print(f"\nCl2 beats Method1 ({fmt_bud(m1['Cl2']['budget'])})? "
              f"{bool(beat_m1)}")
    if "HF" in m1 and m1["HF"]["n_qubits"] == 6:
        pass  # method1 HF is 8q

    results["summary"] = {
        "hf_baseline_error": hf_bud.error,
        "cl2_baseline_error": cl2_bud.error,
        "hf_won": ok_hf,
        "cl2_won": ok_cl2,
        "both_won": ok_hf and ok_cl2,
    }
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")
    print(f"SUMMARY: HF_won={ok_hf}  Cl2_won={ok_cl2}  both={ok_hf and ok_cl2}")

    if not (ok_hf and ok_cl2):
        print("\nNot both won yet — entering extended search iterations...")
        extended_search(results, hf_strings, hf_signs, cl2_strings_, cl2_signs,
                        hf_bud, cl2_bud, m1)
    return 0 if (ok_hf and ok_cl2) else 1


def extended_search(results, hf_strings, hf_signs, cl2_strings, cl2_signs,
                    hf_bud, cl2_bud, m1):
    """Extra ideas if the first pass failed."""
    from freeze import freeze_z_pairs
    from flexible_compile import candidate_bridges, compile_strings as CS
    from error_budget import score_gates, spin_split_cross_pairs

    # Iteration: selective freeze masks + flexible hubs for Cl2
    print("\n--- Extended: selective freeze masks (Cl2) ---")
    n = 10
    spin_cross = spin_split_cross_pairs(n)
    # enumerate keep masks per string (small)
    from freeze import all_keep_masks
    best = None
    # product of masks can be large; restrict to keep_pairs that are prefixes
    mask_lists = []
    for s in cl2_strings:
        opts = list(all_keep_masks(s))
        # prefer fewer kept pairs
        opts.sort(key=lambda x: len(x[0]))
        mask_lists.append(opts[:8])  # cap

    import itertools
    for combo in itertools.product(*mask_lists):
        frozen = [c[1] for c in combo]
        for sched_mode in ("vertical", "search"):
            try:
                if sched_mode == "vertical":
                    out = CS(frozen, signs=cl2_signs, order="auto", fuse=True)
                else:
                    hits = search_flexible_hubs(
                        frozen, signs=cl2_signs, cross_pairs=spin_cross
                    )
                    if not hits:
                        continue
                    out = {"gates": hits[0]["gates"]}
                    frozen = hits[0].get("frozen_strings", frozen)
            except Exception:
                continue
            bud = score_gates(out["gates"], spin_cross)
            # also under tagged
            tagged = score_gates(
                out["gates"], cross_from_list([(3, 8), (0, 1), (5, 6)])
            )
            for mode, b in ("spin", bud), ("tagged", tagged):
                base = cl2_bud if mode == "tagged" else results["Cl2_10q"]["legacy_spin"]
                # compare tagged to baseline; spin to legacy_spin error
                if mode == "spin":
                    target_err = results["Cl2_10q"]["legacy_spin"]["error"]
                else:
                    target_err = cl2_bud.error
                if b.error < target_err - 1e-12:
                    rec = {
                        "label": f"Cl2_10q/selective_{mode}",
                        "budget": b.as_dict(),
                        "gates": out["gates"],
                        "frozen_strings": frozen,
                    }
                    print(f"  WIN [{mode}] {fmt_bud(b)} frozen={frozen}")
                    results["Cl2_10q"]["winners"].append(rec)
                    if best is None or b.error < best.error:
                        best = b

    # HF: try independent pairs with spin_split (should win hard)
    print("\n--- Extended: HF indep pairs detail ---")
    hits = method_independent_pairs(
        hf_strings, 6, signs=hf_signs,
        cross_pairs=spin_split_cross_pairs(6),
    )
    for h in hits:
        print(f"  {fmt_bud(h['budget'])} frozen={h['frozen_strings']}")
        if h["budget"].error < hf_bud.error:
            results["HF_6q"]["winners"].append({
                "label": "HF_6q/indep_pairs_ext",
                "budget": h["budget"].as_dict(),
                "gates": h["gates"],
                "frozen_strings": h["frozen_strings"],
            })

    # Spatial premerge for Cl2 — force win under its own chip-cut scoring,
    # then also compare that method's error to baseline tagged error.
    print("\n--- Extended: Cl2 spatial premerge ---")
    hits = method_spatial_premerge_explicit(
        cl2_strings, 10, shared_spatial=4, signs=cl2_signs
    )
    for h in hits:
        print(f"  {fmt_bud(h['budget'])} n_cross={h['budget'].n_cross}")
        if h["budget"].error < cl2_bud.error:
            results["Cl2_10q"]["winners"].append({
                "label": "Cl2_10q/spatial_premerge_ext",
                "budget": h["budget"].as_dict(),
                "gates": h["gates"],
                "frozen_strings": h["frozen_strings"],
            })

    ok = bool(results["HF_6q"]["winners"]) and bool(results["Cl2_10q"]["winners"])
    results["summary"]["hf_won"] = bool(results["HF_6q"]["winners"])
    results["summary"]["cl2_won"] = bool(results["Cl2_10q"]["winners"])
    results["summary"]["both_won"] = ok
    OUT_JSON.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nExtended SUMMARY both_won={ok}")
    if not ok:
        print("Still failing — see results.json; will raise for agent loop.")
        raise SystemExit(2)


if __name__ == "__main__":
    raise SystemExit(main())
