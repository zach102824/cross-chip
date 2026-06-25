"""Hyperparameter sweep for the two-stage CDR+GP mitigator on the REAL HF problem.

Reconstructs the 8-qubit / 3-param HF ansatz + Hamiltonian exactly as the notebook
does, then measures energy-level MAE (mitigated vs ideal) over a fixed set of
thetas around theta_init -- the region a VQE explores -- as a function of the
hyperparameters that most affect accuracy and circuit count.

Run:  ../../../.venv_h4_tencirchem/bin/python _gp_hparam_sweep.py
"""

from __future__ import annotations

import json
import sys
import warnings
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import cirq
import sympy

_HERE = Path(__file__).resolve().parent
_REPO = _HERE
while not (_REPO / "Pauli_Ham").is_dir() and _REPO != _REPO.parent:
    _REPO = _REPO.parent
for _p in (str(_HERE), str(_REPO / "June_main"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gp_mitigator_ML_sep as gpm
from main_cursor_lib import RZXGate, CZ_CROSS_CHIP_TAG

warnings.simplefilter("ignore")

# --- Problem constants (match the notebook) ---
MOLECULE = "HF"
BOND_LENGTH = 1.2
N_SPATIAL_ORBITALS = 4
N_ACTIVE_ELECTRONS = 6
ETA = N_ACTIVE_ELECTRONS // 2
CROSS_CHIP_PAIRS = {frozenset((2, 6))}
CIRCUIT_JSON = _REPO / "June_main" / "circuits2read" / "HF_8q_3doubles_rzx.json"
HAM_TXT = _REPO / "Pauli_Ham" / f"{MOLECULE}_bond_{BOND_LENGTH:.1f}.txt"


def load_circuit_from_json(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    n = int(data["num_qubits"])
    q = cirq.LineQubit.range(n)
    param_names = list(data["param_names"])
    if not param_names:
        for g in data["gates"]:
            name = g.get("param")
            if name is not None and name not in param_names:
                param_names.append(name)
    sym = {name: sympy.Symbol(f"th_{i}") for i, name in enumerate(param_names)}
    syms = [sym[name] for name in param_names]
    c = cirq.Circuit()
    for g in data["gates"]:
        op = g["op"]
        qs = [q[i] for i in g["qubits"]]
        if op == "h":
            new_op = cirq.H(qs[0])
        elif op == "x":
            new_op = cirq.X(qs[0])
        elif op == "ry":
            new_op = cirq.ry(float(g["value"])).on(qs[0])
        elif op == "cx":
            new_op = cirq.CNOT(qs[0], qs[1])
        elif op == "cz":
            new_op = cirq.CZ(qs[0], qs[1])
        elif op == "rzx":
            if "param" in g:
                rzx_angle = float(g.get("coeff", 1.0)) * sym[g["param"]]
            else:
                rzx_angle = float(g.get("angle", g.get("value", 0.0)))
            new_op = RZXGate(rzx_angle).on(qs[0], qs[1])
        elif op in ("rx", "rz"):
            angle = float(g["coeff"]) * sym[g["param"]] if "param" in g else float(g["value"])
            gate = cirq.rx if op == "rx" else cirq.rz
            new_op = gate(angle).on(qs[0])
        else:
            raise ValueError(f"Unhandled op: {op!r}")
        if len(new_op.qubits) == 2 and frozenset(g["qubits"]) in CROSS_CHIP_PAIRS:
            new_op = new_op.with_tags(CZ_CROSS_CHIP_TAG)
        c.append(new_op)
    return c, list(q), syms


def load_pauli_sum(path, qubits):
    idx_to_pauli = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    out = cirq.PauliSum()
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        coeff = float(parts[0])
        codes = [int(x) for x in parts[1:]]
        ps = cirq.PauliString()
        for qb, code in zip(qubits, codes):
            if code:
                ps *= idx_to_pauli[code](qb)
        out += coeff * ps
    return out


def build_problem():
    ansatz, qubits, symbols = load_circuit_from_json(CIRCUIT_JSON)
    pauli_sum = load_pauli_sum(HAM_TXT, qubits)
    half, eta = N_SPATIAL_ORBITALS, ETA
    occupied = list(range(eta)) + list(range(half, half + eta))
    prep = cirq.Circuit([cirq.X(qubits[k]) for k in occupied])
    circuit = prep + ansatz
    return circuit, qubits, symbols, pauli_sum


# --- Noise / shot / readout config (match notebook defaults) ---
from main_cursor_lib import (
    ONE_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPOL_PROB,
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
)

BASE_NOISE = {
    "two_qubit_depol_prob": TWO_QUBIT_GATE_DEPOL_PROB,
    "one_qubit_depol_prob": ONE_QUBIT_GATE_DEPOL_PROB,
    "cross_chip_two_qubit_depol_prob": CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
}
# Kept modest so a full sweep runs in a few minutes; raise for final numbers.
SHOTS = 4000
EVAL_N = 8
SHOT_CFG = {"num_shots": SHOTS, "measurement_scheme": "direct_pauli", "apply_readout_noise": True, "epsilon": 0.1}
READOUT = {"p_0_success": np.full(8, 0.98), "p_1_success": np.full(8, 0.96)}


def make_adapter():
    circuit, qubits, symbols, pauli_sum = build_problem()
    adapter = gpm.CirqBackendAdapter(
        circuit=circuit, qubits=qubits, symbols=symbols, pauli_sum=pauli_sum,
        base_noise_cfg=BASE_NOISE, shot_cfg=SHOT_CFG, readout_cal=READOUT,
        simulator_seed=1234, use_rem_branch=True,
    )
    return adapter


def base_config(adapter):
    return gpm.MitigatorConfig(
        n_params=len(adapter.symbols),
        n_qubits=len(adapter.qubits),
        theta_init=np.zeros(len(adapter.symbols)),
        n_warmstart_circuits=30,
        n_nonclifford_gates=2,
        warmstart_spread=0.3,
        shots=SHOTS,
        rng_seed=0,
        max_harmonic=1,
        include_noisy_feature_in_gp=True,
        kernel_type="matern",
        matern_nu=2.5,
        use_product_kernel=True,
        gp_n_restarts=2,
        n_observables=len(adapter.obs_labels),
    )


def run_one(adapter, cfg, eval_set):
    cfg = deepcopy(cfg)
    mit = gpm.Mitigator(adapter, hamiltonian=adapter.pauli_sum, config=cfg)
    mit.warmstart()
    res = gpm.evaluate_mitigator_on_thetas(mit, eval_set)
    return res


def fmt(res):
    return (
        f"circuits={res['warmstart_circuits']:>3}  "
        f"MAE_full={res['mae_full']*1e3:7.3f}  "
        f"MAE_bb={res['mae_backbone']*1e3:7.3f}  "
        f"MAE_noisy={res['mae_noisy']*1e3:7.3f}  "
        f"(mEh)  med_unc={res['median_weighted_uncertainty']:.4e}"
    )


def main():
    adapter = make_adapter()
    print(f"HF problem: {len(adapter.qubits)} qubits, {len(adapter.symbols)} params, "
          f"{len(adapter.obs_labels)} Pauli observables.")
    base = base_config(adapter)

    # Fixed eval set around theta_init within a VQE-sized ball (radius 0.4 rad).
    eval_set = gpm.make_eval_set(adapter, base.theta_init, radius=0.4, n=EVAL_N, seed=4242, shots=SHOTS)
    e_ideal = np.array(eval_set["ideal_energies"])
    e_noisy = np.array([adapter.energy_from_values(m) for m in eval_set["measured_list"]])
    print(f"Eval set: {len(e_ideal)} thetas in radius 0.4 around theta_init.")
    print(f"  ideal energy range [{e_ideal.min():.4f}, {e_ideal.max():.4f}], "
          f"raw noisy MAE = {np.mean(np.abs(e_noisy-e_ideal))*1e3:.3f} mEh\n")

    print("=== BASELINE (notebook defaults) ===")
    print("  " + fmt(run_one(adapter, base, eval_set)) + "\n")

    print("=== 1) n_warmstart_circuits  x  n_nonclifford_gates (fewer circuits!) ===")
    for nnc in (1, 2, 3):
        for nw in (10, 20, 30):
            cfg = replace(base, n_warmstart_circuits=nw, n_nonclifford_gates=nnc)
            print(f"  nnc={nnc}  " + fmt(run_one(adapter, cfg, eval_set)))
        print()

    print("=== 2) warmstart_spread (coverage of the VQE region) @ nw=12, nnc=2 ===")
    for sp in (0.2, 0.3, 0.5, 0.8):
        cfg = replace(base, n_warmstart_circuits=12, n_nonclifford_gates=2, warmstart_spread=sp)
        print(f"  spread={sp:<4}  " + fmt(run_one(adapter, cfg, eval_set)))
    print()

    print("=== 3) max_harmonic (Fourier feature richness) @ nw=12, nnc=2, spread=0.5 ===")
    for mh in (1, 2, 3):
        cfg = replace(base, n_warmstart_circuits=12, n_nonclifford_gates=2, warmstart_spread=0.5, max_harmonic=mh)
        print(f"  max_harmonic={mh}  " + fmt(run_one(adapter, cfg, eval_set)))
    print()

    print("=== 4) kernel structure @ nw=12, nnc=2, spread=0.5, max_harmonic=2 ===")
    g = dict(n_warmstart_circuits=12, n_nonclifford_gates=2, warmstart_spread=0.5, max_harmonic=2)
    for desc, kw in [
        ("matern2.5 product ", dict(kernel_type="matern", use_product_kernel=True)),
        ("matern2.5 additive", dict(kernel_type="matern", use_product_kernel=False)),
        ("matern1.5 product ", dict(kernel_type="matern", matern_nu=1.5, use_product_kernel=True)),
        ("rbf       product ", dict(kernel_type="rbf", use_product_kernel=True)),
    ]:
        cfg = replace(base, **g, **kw)
        print(f"  {desc}  " + fmt(run_one(adapter, cfg, eval_set)))
    print()

    print("=== 5) include_noisy_feature_in_gp @ nw=12, nnc=2, spread=0.5, max_harmonic=2 ===")
    for inc in (True, False):
        cfg = replace(base, **g, include_noisy_feature_in_gp=inc)
        print(f"  include_noisy={inc!s:<5}  " + fmt(run_one(adapter, cfg, eval_set)))
    print()

    print("=== 6) uncertainty_threshold guidance: top-up fire-rate over eval thetas ===")
    # Build the recommended lean model, then see how often a top-up would trigger
    # at various thresholds across the eval region (lower fire-rate = fewer circuits).
    lean = replace(base, n_warmstart_circuits=12, n_nonclifford_gates=2,
                   warmstart_spread=0.5, max_harmonic=2)
    mit = gpm.Mitigator(adapter, hamiltonian=adapter.pauli_sum, config=lean)
    mit.warmstart()
    uncs = []
    for theta, measured in zip(eval_set["thetas"], eval_set["measured_list"]):
        _, std = mit.predict_with_uncertainty(theta, measured)
        uncs.append(gpm.weighted_uncertainty(std, mit.coeff_by_pauli))
    uncs = np.array(uncs)
    print(f"  weighted-uncertainty over eval set: median={np.median(uncs):.4e}, "
          f"p90={np.percentile(uncs,90):.4e}, max={uncs.max():.4e}")
    for thr in (0.01, 0.02, 0.05, 0.1):
        rate = float(np.mean(uncs > thr))
        print(f"  threshold={thr:<5}  top-up fire-rate={rate*100:5.1f}%")


if __name__ == "__main__":
    main()
