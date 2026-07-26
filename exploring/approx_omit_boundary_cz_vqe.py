#!/usr/bin/env python3
"""Approximate Cl2/Br2 ansatz: omit same-spin CZ into the LUMO edge.

Idea
----
CZ(3,4) / CZ(8,9) (Cl2) are the only within-spin edges into the LUMO qubits.
Instead of fanning the LUMO into the same-spin parity tree, we:

  1. Drop LUMO letters (q4 / q9 for Cl2) from each double's Pauli string
     → remaining ops are weight-2 Y_k X_{k+N/2} (occupied spatial only).
  2. Entangle spin sectors on the LUMO with a free RZX(4, 9).

Same-spin correlations that needed those boundary CZs are omitted; α–β
entanglement "beyond q3" is kept via RZX.

Noiseless statevector VQE vs exact GS / HF / full (exact-string) ansatz.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "UCCSD circuit"))

from constraints import allowed_cz_edges, cz_pairs, rzx_pairs, satisfies_rules  # noqa: E402
from error_budget import score_gates, spin_split_cross_pairs  # noqa: E402
from flexible_compile import compile_strings, gates_to_qc, gen  # noqa: E402
from freeze import fully_freeze  # noqa: E402
from methods import method_disjoint_rzx_all_qubits  # noqa: E402

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Statevector


# ----------------------------------------------------------------------
# Hamiltonian IO (numbered Pauli file used by June_main)
# ----------------------------------------------------------------------
def load_hamiltonian(path: Path, n_qubits: int) -> SparsePauliOp:
    """File rows: coeff  c0 c1 ... c_{n-1} with 0=I,1=X,2=Y,3=Z (qubit 0 first)."""
    code = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    paulis, coeffs = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        c = float(parts[0])
        codes = [int(x) for x in parts[1:]]
        if len(codes) != n_qubits:
            raise ValueError(f"bad row length in {path}: {len(codes)} != {n_qubits}")
        # Qiskit SparsePauliOp uses little-endian label (rightmost = q0)
        label = "".join(code[k] for k in reversed(codes))
        paulis.append(label)
        coeffs.append(c)
    return SparsePauliOp(paulis, coeffs)


def hf_prep(n_qubits: int, n_electrons: int) -> QuantumCircuit:
    half = n_qubits // 2
    eta = n_electrons // 2
    qc = QuantumCircuit(n_qubits)
    for q in list(range(eta)) + list(range(half, half + eta)):
        qc.x(q)
    return qc


def expect_h(qc: QuantumCircuit, ham: SparsePauliOp) -> float:
    sv = Statevector.from_instruction(qc)
    return float(np.real(sv.expectation_value(ham)))


def strip_lumo(string: str) -> str:
    """Remove letters on the LUMO pair (last orbital of each spin row)."""
    n = len(string)
    half = n // 2
    lumo_a, lumo_b = half - 1, n - 1
    chars = list(string)
    chars[lumo_a] = "I"
    chars[lumo_b] = "I"
    return "".join(chars)


def append_rzx_lumo(gates: list[dict], n_qubits: int, param: str = "tL") -> list[dict]:
    """Append basis + RZX on (LUMO_α, LUMO_β) + undo (implements exp(-i t/2 Y⊗X))."""
    half = n_qubits // 2
    a, b = half - 1, n_qubits - 1
    # Same dressing as independent Y_a X_b block
    extra = [
        {"op": "rx", "qubits": [a], "value": float(np.pi / 2)},
        {"op": "h", "qubits": [a]},
        {"op": "h", "qubits": [b]},
        {"op": "rzx", "qubits": [b, a], "param": param, "coeff": 1.0},
        {"op": "h", "qubits": [b]},
        {"op": "h", "qubits": [a]},
        {"op": "rx", "qubits": [a], "value": float(-np.pi / 2)},
    ]
    return list(gates) + extra


def forbid_boundary_cz(gates, n_qubits: int) -> bool:
    half = n_qubits // 2
    bad = {tuple(sorted((half - 2, half - 1))),
           tuple(sorted((n_qubits - 2, n_qubits - 1)))}
    return not any(p in bad for p in cz_pairs(gates))


# ----------------------------------------------------------------------
# Molecule cases
# ----------------------------------------------------------------------
CASES = {
    "Cl2": dict(
        n_qubits=10,
        n_electrons=8,
        doubles=[(4, 9, 6, 1), (4, 9, 5, 0), (4, 9, 7, 2)],
        bond=2.2,
        ham=_ROOT / "Pauli_Ham" / "Cl2_bond_2.2.txt",
        # GS from bond scan (active-space exact)
        e_gs_scan=-909.145296806064,
    ),
    "Br2": dict(
        n_qubits=12,
        n_electrons=10,
        doubles=[(5, 11, 6, 0), (5, 11, 10, 4), (5, 11, 9, 3), (5, 11, 8, 2)],
        bond=2.2,
        ham=_ROOT / "Pauli_Ham" / "Br2_bond_2.2.txt",
        e_gs_scan=None,  # fill from file diagonalisation
    ),
}


def build_exact_ansatz(strings, signs):
    """Graph-legal full-string compile (may use boundary CZ)."""
    hits = method_disjoint_rzx_all_qubits(
        strings, signs=signs, cross_pairs=spin_split_cross_pairs(len(strings[0])),
        max_mask_opts=4,
    )
    if hits:
        return hits[0]["gates"], hits[0]
    # fallback: legacy compile
    out = compile_strings(strings, signs=signs, order="given", fuse=True)
    return out["gates"], {"name": "legacy_full", "frozen_strings": strings}


def build_approx_omit_lumo_fanin(strings, signs, add_lumo_rzx=True):
    """Strip LUMO from doubles; optional free RZX(LUMO_α, LUMO_β)."""
    stripped = [strip_lumo(s) for s in fully_freeze(strings)]
    # drop empty strings
    keep_idx = [i for i, s in enumerate(stripped) if any(c != "I" for c in s)]
    stripped = [stripped[i] for i in keep_idx]
    stripped_signs = [signs[i] for i in keep_idx]
    out = compile_strings(
        stripped, signs=stripped_signs, order="given", fuse=True,
        hub_schedule=None,
    )
    gates = out["gates"]
    # Force graph compile with disjoint hubs on the weight-2 pairs
    n = len(strings[0])
    half = n // 2
    schedule = []
    for s in stripped:
        alpha = [q for q in range(half) if s[q] != "I"]
        beta = [q for q in range(half, n) if s[q] != "I"]
        if len(alpha) == 1 and len(beta) == 1:
            schedule.append((alpha[0], beta[0]))
        else:
            schedule.append((alpha[0], beta[0]))
    # ensure disjoint
    used = set()
    sched2 = []
    for a, b in schedule:
        if a in used or b in used:
            # pick any free pair from support — should not happen for Cl2
            raise RuntimeError(f"overlapping stripped hubs {(a,b)} used={used}")
        used.add(a)
        used.add(b)
        sched2.append((a, b))
    out = compile_strings(
        stripped, signs=stripped_signs, order="given",
        hub_schedule=sched2, fuse=True,
    )
    gates = out["gates"]
    if add_lumo_rzx:
        gates = append_rzx_lumo(gates, n)
    assert forbid_boundary_cz(gates, n), cz_pairs(gates)
    return gates, {
        "name": "omit_lumo_fanin" + ("+rzxL" if add_lumo_rzx else ""),
        "stripped_strings": stripped,
        "schedule": sched2,
        "add_lumo_rzx": add_lumo_rzx,
    }


def vqe_minimize(gates, n_qubits, n_electrons, ham, n_params_hint=None):
    prep = hf_prep(n_qubits, n_electrons)
    # discover param names in order of first appearance
    names = []
    for g in gates:
        if "param" in g and g["param"] not in names:
            names.append(g["param"])
    n_p = len(names)
    if n_params_hint is not None:
        assert n_p == n_params_hint, (n_p, n_params_hint, names)

    def energy(x):
        qc = prep.compose(gates_to_qc(gates, n_qubits, theta_values=None))
        # bind by rebuilding with values — gates_to_qc expects index by t0,t1,...
        # Map names -> values: t0->x[i] if name t{k}
        vals = {}
        for name, val in zip(names, x):
            if name.startswith("t") and name[1:].isdigit():
                vals[int(name[1:])] = float(val)
            elif name == "tL":
                vals["L"] = float(val)
        # rebuild with a custom binder
        from qiskit.circuit import Parameter
        qc = prep.copy()
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
                    name = g["param"]
                    p = params.setdefault(name, Parameter(name))
                    qc.rx(float(g.get("coeff", 1.0)) * p, qs[0])
                else:
                    qc.rx(float(g["value"]), qs[0])
            elif op == "rzx":
                name = g["param"]
                p = params.setdefault(name, Parameter(name))
                qc.rzx(float(g.get("coeff", 1.0)) * p, qs[0], qs[1])
        bind = {params[name]: float(val) for name, val in zip(names, x)}
        qc = qc.assign_parameters(bind)
        return expect_h(qc, ham)

    x0 = np.zeros(n_p)
    # multi-start
    best = None
    rng = np.random.default_rng(0)
    starts = [x0] + [rng.uniform(-0.4, 0.4, size=n_p) for _ in range(6)]
    for s in starts:
        res = minimize(energy, s, method="Nelder-Mead",
                       options={"maxiter": 400, "xatol": 1e-7, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res
    return float(best.fun), best.x, names


def run_case(name: str):
    cfg = CASES[name]
    n = cfg["n_qubits"]
    ne = cfg["n_electrons"]
    ham = load_hamiltonian(cfg["ham"], n)
    print(f"\n{'='*70}\n{name}  bond={cfg['bond']}  n={n}")
    print(f"H from {cfg['ham']}")

    # Exact GS from Hamiltonian diagonalisation (active-space)
    # For 10q this is 2^10=1024 — fine; 12q=4096 — fine
    hmat = ham.to_matrix()
    e_gs = float(np.linalg.eigvalsh(hmat)[0].real)
    e_hf = expect_h(hf_prep(n, ne), ham)
    print(f"E_GS (diag) = {e_gs:.10f} Eh")
    print(f"E_HF        = {e_hf:.10f} Eh")
    print(f"HF−GS       = {(e_hf - e_gs)*1e3:.3f} mHa")
    if cfg.get("e_gs_scan") is not None:
        print(f"E_GS (scan) = {cfg['e_gs_scan']:.10f}  Δdiag={e_gs - cfg['e_gs_scan']:.3e}")

    strings = ["".join(gen.jw_string_for_double(n, d)) for d in cfg["doubles"]]
    signs = [1] * len(strings)
    frozen = fully_freeze(strings)
    print(f"JW strings : {strings}")
    print(f"fully frozen: {frozen}")

    results = {"molecule": name, "e_gs": e_gs, "e_hf": e_hf}

    # --- Exact-string ansatz (current graph-legal winner style) ---
    try:
        gates_ex, meta_ex = build_exact_ansatz(strings, signs)
        bud = score_gates(gates_ex, spin_split_cross_pairs(n))
        e_ex, x_ex, names_ex = vqe_minimize(gates_ex, n, ne, ham)
        print(f"\n[exact-string disjoint] params={names_ex}")
        print(f"  budget fid={bud.fidelity:.4f}  cz has (3,4)/(8,9)? "
              f"{not forbid_boundary_cz(gates_ex, n)}")
        print(f"  E_VQE = {e_ex:.10f}   |E−GS|={(e_ex - e_gs)*1e3:.3f} mHa")
        results["exact_string"] = {
            "energy": e_ex, "err_mHa": (e_ex - e_gs) * 1e3,
            "params": names_ex, "x": x_ex.tolist(),
            "uses_boundary_cz": not forbid_boundary_cz(gates_ex, n),
            "rzx": rzx_pairs(gates_ex),
            "budget": bud.as_dict(),
        }
    except Exception as e:
        print(f"[exact-string] FAILED: {e}")
        results["exact_string"] = {"error": str(e)}

    # --- Approx: omit LUMO fan-in, + RZX(LUMO) ---
    gates_a, meta_a = build_approx_omit_lumo_fanin(strings, signs, add_lumo_rzx=True)
    bud_a = score_gates(gates_a, spin_split_cross_pairs(n))
    ok, why = satisfies_rules(gates_a, n)
    # all-qubits may fail (q3/q8 idle) — that is OK for this approx experiment
    e_a, x_a, names_a = vqe_minimize(gates_a, n, ne, ham)
    print(f"\n[approx omit LUMO fan-in + RZX(L)] params={names_a}")
    print(f"  stripped: {meta_a['stripped_strings']}")
    print(f"  rzx={rzx_pairs(gates_a)}  cz={sorted(set(cz_pairs(gates_a)))}")
    print(f"  boundary CZ absent? {forbid_boundary_cz(gates_a, n)}")
    print(f"  budget fid={bud_a.fidelity:.4f}  rules(all-qubits)={ok}/{why}")
    print(f"  E_VQE = {e_a:.10f}   |E−GS|={(e_a - e_gs)*1e3:.3f} mHa")
    print(f"  recovers of corr: {(e_hf - e_a)/(e_hf - e_gs)*100:.1f}% of HF−GS")
    results["approx_lumo_rzx"] = {
        "energy": e_a, "err_mHa": (e_a - e_gs) * 1e3,
        "params": names_a, "x": x_a.tolist(),
        "stripped": meta_a["stripped_strings"],
        "rzx": rzx_pairs(gates_a),
        "cz": sorted(set(cz_pairs(gates_a))),
        "budget": bud_a.as_dict(),
        "corr_recovered_pct": float((e_hf - e_a) / (e_hf - e_gs) * 100),
    }

    # --- Approx without LUMO RZX (ablation) ---
    gates_b, meta_b = build_approx_omit_lumo_fanin(strings, signs, add_lumo_rzx=False)
    e_b, x_b, names_b = vqe_minimize(gates_b, n, ne, ham)
    print(f"\n[approx omit LUMO entirely] params={names_b}")
    print(f"  E_VQE = {e_b:.10f}   |E−GS|={(e_b - e_gs)*1e3:.3f} mHa")
    results["approx_no_lumo"] = {
        "energy": e_b, "err_mHa": (e_b - e_gs) * 1e3,
        "params": names_b, "x": x_b.tolist(),
        "corr_recovered_pct": float((e_hf - e_b) / (e_hf - e_gs) * 100),
    }

    # Draw approx circuit
    qc = hf_prep(n, ne)
    qc.barrier()
    qc = qc.compose(gates_to_qc(gates_a, n))
    fig = qc.draw(output="mpl", fold=-1, style=gen.IQP_STYLE, idle_wires=True)
    png = _HERE / f"{name}_approx_omit_boundary_cz_circuit.png"
    fig.savefig(png, dpi=150, bbox_inches="tight")
    print(f"wrote {png}")

    out = _HERE / f"{name}_approx_omit_boundary_cz_results.json"
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {out}")
    return results


def main():
    all_res = {}
    for name in ("Cl2",):
        all_res[name] = run_case(name)
    # Br2 if hamiltonian exists and is manageable
    if CASES["Br2"]["ham"].exists():
        try:
            all_res["Br2"] = run_case("Br2")
        except Exception as e:
            print(f"Br2 skipped: {e}")
    (_HERE / "approx_omit_boundary_cz_summary.json").write_text(
        json.dumps(all_res, indent=2, default=str)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
