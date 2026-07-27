"""Hardware-efficient ansatze under constraints.py vs current UCCSD winners."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import SparsePauliOp, Statevector

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "exploring"))
sys.path.insert(0, str(ROOT / "state_transfer"))

from constraints import satisfies_rules, allowed_cz_edges, qubits_used  # noqa: E402


def load_hamiltonian(path, n):
    code = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    paulis, coeffs = [], []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        c = float(parts[0])
        codes = [int(x) for x in parts[1:]]
        assert len(codes) == n
        # Qiskit SparsePauliOp uses little-endian label (qubit 0 = rightmost).
        paulis.append("".join(code[k] for k in reversed(codes)))
        coeffs.append(c)
    return SparsePauliOp(paulis, coeffs)


def hf_prep(n, ne):
    half = n // 2
    eta = ne // 2
    qc = QuantumCircuit(n)
    for q in list(range(eta)) + list(range(half, half + eta)):
        qc.x(q)
    return qc


def hf_prep_bits(n, bits):
    qc = QuantumCircuit(n)
    for q, b in enumerate(bits):
        if b == "1":
            qc.x(q)
    return qc


def count_2q(gates):
    return sum(len(g["qubits"]) == 2 for g in gates)


def param_names(gates):
    names = []
    for g in gates:
        if "param" in g and g["param"] not in names:
            names.append(g["param"])
    return names


def build_param_circuit(gates, n, prep=None):
    qc = prep.copy() if prep is not None else QuantumCircuit(n)
    params = {}
    for g in gates:
        op = g["op"].lower()
        qs = g["qubits"]
        if op == "h":
            qc.h(qs[0])
        elif op == "cz":
            qc.cz(qs[0], qs[1])
        elif op in ("ry", "rx", "rz"):
            if "param" in g:
                p = params.setdefault(g["param"], Parameter(g["param"]))
                getattr(qc, op)(float(g.get("coeff", 1)) * p, qs[0])
            else:
                getattr(qc, op)(float(g["value"]), qs[0])
        elif op == "rzx":
            p = params.setdefault(g["param"], Parameter(g["param"]))
            qc.rzx(float(g.get("coeff", 1)) * p, qs[0], qs[1])
        else:
            raise ValueError(op)
    names = param_names(gates)
    return qc, names, params


def depth_of(gates, n):
    qc, _, _ = build_param_circuit(gates, n)
    return qc.depth()


def pad_unused(gates, n):
    used = qubits_used(gates)
    missing = sorted(set(range(n)) - used)
    out = list(gates)
    for q in missing:
        out.append({"op": "ry", "qubits": [q], "param": f"tpad{q}", "coeff": 1.0})
    return out


def hea_rzx_only(n, pairs, n_layers=1):
    gates = []
    p = 0
    for _ in range(n_layers):
        for q in range(n):
            gates.append({"op": "ry", "qubits": [q], "param": f"t{p}", "coeff": 1.0})
            p += 1
        for a, b in pairs:
            gates.append({"op": "h", "qubits": [b]})
            gates.append({"op": "rzx", "qubits": [a, b], "param": f"t{p}", "coeff": 1.0})
            p += 1
            gates.append({"op": "h", "qubits": [b]})
        for q in range(n):
            gates.append({"op": "ry", "qubits": [q], "param": f"t{p}", "coeff": 1.0})
            p += 1
    return gates


def hea_cz_plus_rzx(n, cz_edges, rzx_pairs, n_layers=1):
    gates = []
    p = 0
    for _ in range(n_layers):
        for q in range(n):
            gates.append({"op": "ry", "qubits": [q], "param": f"t{p}", "coeff": 1.0})
            p += 1
        for a, b in cz_edges:
            gates.append({"op": "cz", "qubits": [a, b]})
        for a, b in rzx_pairs:
            gates.append({"op": "h", "qubits": [b]})
            gates.append({"op": "rzx", "qubits": [a, b], "param": f"t{p}", "coeff": 1.0})
            p += 1
            gates.append({"op": "h", "qubits": [b]})
        for q in range(n):
            gates.append({"op": "ry", "qubits": [q], "param": f"t{p}", "coeff": 1.0})
            p += 1
    return gates


def optimize(gates, n, Hmat, prep_qc, n_starts=3, seed=0, maxiter=120):
    """BFGS with template circuit + dense H matrix for fast <H>."""
    qc_t, names, params = build_param_circuit(gates, n, prep=prep_qc)
    if not names:
        sv = Statevector.from_instruction(qc_t)
        psi = sv.data
        return float(np.real(np.vdot(psi, Hmat @ psi))), 0

    plist = [params[nm] for nm in names]

    def energy(x):
        bound = qc_t.assign_parameters(dict(zip(plist, x)))
        psi = Statevector.from_instruction(bound).data
        return float(np.real(np.vdot(psi, Hmat @ psi)))

    rng = np.random.default_rng(seed)
    starts = [np.zeros(len(names))] + [
        rng.uniform(-0.5, 0.5, len(names)) for _ in range(n_starts - 1)
    ]
    best = None
    for x0 in starts:
        res = minimize(energy, x0, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-6})
        if best is None or res.fun < best.fun:
            best = res
    return float(best.fun), len(names)


def expect_h_mat(qc, Hmat):
    psi = Statevector.from_instruction(qc).data
    return float(np.real(np.vdot(psi, Hmat @ psi)))


def run_molecule(tag, n, ham_path, prep, winner_gates, candidates, n_starts=3, maxiter=120):
    print("=" * 70)
    print(tag)
    ham = load_hamiltonian(ham_path, n)
    Hmat = ham.to_matrix()
    e_gs = float(np.linalg.eigvalsh(Hmat)[0].real)
    e_hf = expect_h_mat(prep, Hmat)
    gap = (e_hf - e_gs) * 1e3
    print(f"E_GS={e_gs:.10f}  E_HF={e_hf:.10f}  gap={gap:.3f} mHa", flush=True)

    t0 = time.time()
    e_cur, np_cur = optimize(winner_gates, n, Hmat, prep, n_starts=n_starts, seed=1, maxiter=maxiter)
    cur_2q, cur_d = count_2q(winner_gates), depth_of(winner_gates, n)
    print(
        f"CURRENT: E={e_cur:.10f} err={(e_cur - e_gs) * 1e3:.3f} mHa "
        f"2q={cur_2q} depth={cur_d} npar={np_cur} wall={time.time() - t0:.1f}s",
        flush=True,
    )

    rows = []
    for name, gates in candidates:
        gates = pad_unused(gates, n)
        ok, why = satisfies_rules(gates, n)
        n2, d = count_2q(gates), depth_of(gates, n)
        print(f"\n-- {name}: rules={ok}/{why} 2q={n2} depth={d}", flush=True)
        if not ok:
            print("  SKIP", flush=True)
            continue
        t0 = time.time()
        # Heavier ansatze: fewer starts
        starts = 2 if n2 >= 20 or "L2" in name else n_starts
        e, npar = optimize(gates, n, Hmat, prep, n_starts=starts, seed=2, maxiter=maxiter)
        err = (e - e_gs) * 1e3
        corr = (e_hf - e) / (e_hf - e_gs) * 100 if e_hf != e_gs else 0.0
        print(
            f"  E={e:.10f} err={err:.3f} mHa corr={corr:.1f}% npar={npar} "
            f"wall={time.time() - t0:.1f}s",
            flush=True,
        )
        rows.append(
            {
                "name": name,
                "E": e,
                "err_mHa": err,
                "corr_pct": corr,
                "n2": n2,
                "depth": d,
                "nparams": npar,
                "beats_E": bool(e < e_cur - 1e-8),
                "beats_2q": bool(n2 < cur_2q),
                "beats_depth": bool(d < cur_d),
                "beats_E_and_leq_resources": bool(
                    e < e_cur - 1e-8 and n2 <= cur_2q and d <= cur_d
                ),
            }
        )

    print(f"\n=== {tag} SUMMARY ===")
    hdr = f'{"name":28} {"err":>8} {"corr%":>7} {"2q":>4} {"dep":>4} {"npar":>4} {"<E":>4} {"≤2q":>4} {"≤d":>4}'
    print(hdr)
    print(
        f'{"CURRENT":28} {(e_cur - e_gs) * 1e3:8.3f} '
        f'{(e_hf - e_cur) / (e_hf - e_gs) * 100:7.1f} {cur_2q:4d} {cur_d:4d} {np_cur:4d}'
    )
    for r in rows:
        print(
            f'{r["name"]:28} {r["err_mHa"]:8.3f} {r["corr_pct"]:7.1f} '
            f'{r["n2"]:4d} {r["depth"]:4d} {r["nparams"]:4d} '
            f'{str(r["beats_E"]):>4} {str(r["n2"] <= cur_2q):>4} {str(r["depth"] <= cur_d):>4}'
        )

    return {
        "tag": tag,
        "E_GS": e_gs,
        "E_HF": e_hf,
        "current": {
            "E": e_cur,
            "err_mHa": (e_cur - e_gs) * 1e3,
            "corr_pct": (e_hf - e_cur) / (e_hf - e_gs) * 100,
            "n2": cur_2q,
            "depth": cur_d,
            "nparams": np_cur,
        },
        "candidates": rows,
    }


def main():
    cl2w = json.loads((ROOT / "exploring/Cl2_10q_disjoint_rzx.json").read_text())
    hfw = json.loads((ROOT / "exploring/HF_6q_disjoint_rzx.json").read_text())

    n = 10
    nn = [(0, 1), (1, 2), (2, 3), (3, 4), (5, 6), (6, 7), (7, 8), (8, 9)]
    allowed = sorted(tuple(sorted(e)) for e in allowed_cz_edges(n))
    cl2_cands = [
        ("HEA_5vertRZX_L1", hea_rzx_only(n, [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)], 1)),
        ("HEA_3RZX_only", hea_rzx_only(n, [(0, 5), (2, 7), (3, 8)], 1)),
        ("HEA_NN+3RZX_L1", hea_cz_plus_rzx(n, nn, [(0, 5), (2, 7), (3, 8)], 1)),
        ("HEA_allCZ+3RZX", hea_cz_plus_rzx(n, allowed, [(0, 5), (2, 7), (3, 8)], 1)),
        ("HEA_NN+chord+5RZX", hea_cz_plus_rzx(n, nn + [(0, 3), (5, 8)], [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)], 1)),
        ("HEA_5vertRZX_L2", hea_rzx_only(n, [(0, 5), (1, 6), (2, 7), (3, 8), (4, 9)], 2)),
        ("HEA_NN+3RZX_L2", hea_cz_plus_rzx(n, nn, [(0, 5), (2, 7), (3, 8)], 2)),
    ]

    cl2 = run_molecule(
        "Cl2 bond 2.2",
        10,
        ROOT / "Pauli_Ham/Cl2_bond_2.2.txt",
        hf_prep(10, 8),
        cl2w["gates"],
        cl2_cands,
        n_starts=3,
        maxiter=120,
    )

    import taper_lib

    taper = taper_lib.build_taper_data(4, 6)
    init = taper.hf_bitstring_tapered
    hf_cands = [
        ("HEA_3vertRZX_L1", hea_rzx_only(6, [(0, 3), (1, 4), (2, 5)], 1)),
        ("HEA_NN+3RZX", hea_cz_plus_rzx(6, [(0, 1), (1, 2), (3, 4), (4, 5)], [(0, 3), (1, 4), (2, 5)], 1)),
        ("HEA_3vertRZX_L2", hea_rzx_only(6, [(0, 3), (1, 4), (2, 5)], 2)),
    ]
    hf = run_molecule(
        "HF tapered bond 1.0",
        6,
        ROOT / "state_transfer/Pauli_Ham/HF_tapered_bond_1.txt",
        hf_prep_bits(6, init),
        hfw["gates"],
        hf_cands,
        n_starts=3,
        maxiter=120,
    )

    summary = {"Cl2": cl2, "HF": hf, "init_hf_tapered": init}
    # Clear win = better energy with fewer/equal 2q and depth
    winners = []
    for mol, block in (("Cl2", cl2), ("HF", hf)):
        for r in block["candidates"]:
            if r["beats_E_and_leq_resources"]:
                winners.append({"molecule": mol, **r})
    summary["clear_wins"] = winners
    summary["verdict"] = (
        "HEA clear win(s) found" if winners else "No HEA beat current winners on energy with ≤ resources"
    )

    out = ROOT / "exploring/hea_vs_winners_summary.json"
    if winners:
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved clear-win summary -> {out}")
    else:
        # Still save a compact comparison (not overwriting winner JSON/PNG)
        out.write_text(json.dumps(summary, indent=2))
        print(f"\nSaved comparison summary (no clear win) -> {out}")
        print("Did NOT overwrite Cl2_10q_disjoint_rzx / HF_6q_disjoint_rzx JSON/PNG.")

    print("\nVERDICT:", summary["verdict"])
    print("Done.")


if __name__ == "__main__":
    main()
