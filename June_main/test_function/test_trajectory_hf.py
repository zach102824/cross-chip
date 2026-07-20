#!/usr/bin/env python3
"""HF accuracy + speed tests for the trajectory noisy-sim backend.

Compares density-matrix vs Monte Carlo trajectory sampling on the real
``HF_8q_3doubles_rzx`` circuit / Hamiltonian assets used by ``main_HF.py``.

Run:

    .venv_py311/bin/python -m pytest -q \\
        June_main/test_function/test_trajectory_hf.py -s
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cirq
import numpy as np
import pytest
import sympy

REPO_ROOT = Path(__file__).resolve().parents[2]
JUNE_MAIN = REPO_ROOT / "June_main"
EXPORT_DIR = JUNE_MAIN / "export2cloud"
EXPORT_LIB = EXPORT_DIR / "June_main"
# Prefer export2cloud copies (trajectory-aware shot_measurement lives there).
_preferred = [str(EXPORT_LIB), str(EXPORT_DIR)]
sys.path[:] = _preferred + [p for p in sys.path if p not in _preferred]

import shot_measurement as sm  # noqa: E402
from trajectory_sampling import (  # noqa: E402
    _compile_noisy_ops,
    evolve_noisy_trajectory,
)


def _load_hf_circuit():
    """Load the same HF circuit JSON used by ``main_HF.py``."""
    # Prefer the loader from main_HF if importable; otherwise mirror its JSON format.
    path = EXPORT_LIB / "circuits2read" / "HF_8q_3doubles_rzx.json"
    if not path.is_file():
        path = JUNE_MAIN / "circuits2read" / "HF_8q_3doubles_rzx.json"
    data = json.loads(path.read_text(encoding="utf-8"))
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
            c.append(cirq.H(qs[0]))
        elif op == "x":
            c.append(cirq.X(qs[0]))
        elif op == "ry":
            c.append(cirq.ry(float(g["value"])).on(qs[0]))
        elif op == "cx":
            c.append(cirq.CNOT(qs[0], qs[1]))
        elif op == "cz":
            c.append(cirq.CZ(qs[0], qs[1]))
        elif op == "rzx":
            from main_cursor_lib import RZXGate

            if "param" in g:
                angle = float(g.get("coeff", 1.0)) * sym[g["param"]]
            else:
                angle = float(g.get("angle", g.get("value", 0.0)))
            c.append(RZXGate(angle).on(qs[0], qs[1]))
        elif op in ("rx", "rz"):
            angle = float(g["coeff"]) * sym[g["param"]] if "param" in g else float(g["value"])
            gate = cirq.rx if op == "rx" else cirq.rz
            c.append(gate(angle).on(qs[0]))
        else:
            raise ValueError(op)
    return c, list(q), syms, path


def _load_hf_hamiltonian(qubits: list[cirq.Qid], bond: float = 1.2):
    ham_path = EXPORT_DIR / "Pauli_Ham" / f"HF_bond_{bond:.1f}.txt"
    if not ham_path.is_file():
        ham_path = JUNE_MAIN / "Pauli_Ham" / f"HF_bond_{bond:.1f}.txt"
    idx_to_pauli = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    out = cirq.PauliSum()
    with ham_path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            coeff = float(parts[0])
            codes = [int(x) for x in parts[1:]]
            if len(codes) != len(qubits):
                raise ValueError(f"{ham_path}:{lineno} bad Pauli width")
            term = cirq.PauliString()
            for q, code in zip(qubits, codes):
                if code == 0:
                    continue
                term *= idx_to_pauli[code](q)
            out += coeff * term
    return out, ham_path


def _hf_problem(bond: float = 1.2):
    circuit, qubits, symbols, _ = _load_hf_circuit()
    # Prepend HF reference prep like main_HF (X on occupied).
    # For HF_8q, n_electrons typically 8 -> X on first 4 spatial? Use circuit meta.
    path = EXPORT_LIB / "circuits2read" / "HF_8q_3doubles_rzx.json"
    if not path.is_file():
        path = JUNE_MAIN / "circuits2read" / "HF_8q_3doubles_rzx.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    n_electrons = int(meta.get("n_electrons", 8))
    prep = cirq.Circuit(cirq.X(qubits[i]) for i in range(n_electrons))
    full = prep + circuit
    observable, ham_path = _load_hf_hamiltonian(qubits, bond=bond)
    ogm = (
        EXPORT_LIB
        / "OGM_measurement_basis"
        / f"OGM_HF_bond_{bond:.1f}.txt"
    )
    if not ogm.is_file():
        ogm = JUNE_MAIN / "OGM_measurement_basis" / f"OGM_HF_bond_{bond:.1f}.txt"
    resolver = {s: 0.15 * (i + 1) for i, s in enumerate(symbols)}
    noise_params = {
        "two_qubit_depol_prob": 0.01,
        "one_qubit_depol_prob": 0.001,
        "cross_chip_two_qubit_depol_prob": 0.02,
    }
    return {
        "circuit": full,
        "qubits": qubits,
        "symbols": symbols,
        "resolver": resolver,
        "observable": observable,
        "ogm_file": ogm if ogm.is_file() else None,
        "noise_params": noise_params,
        "ham_path": ham_path,
    }


def test_trajectory_unitary_matches_cirq_simulator_on_hf():
    prob = _hf_problem()
    compiled = _compile_noisy_ops(
        prob["circuit"],
        prob["resolver"],
        prob["qubits"],
        {
            "two_qubit_depol_prob": 0.0,
            "one_qubit_depol_prob": 0.0,
            "cross_chip_two_qubit_depol_prob": 0.0,
        },
    )
    psi = evolve_noisy_trajectory(compiled, len(prob["qubits"]), np.random.default_rng(0))
    ref = cirq.Simulator().simulate(
        cirq.resolve_parameters(prob["circuit"], prob["resolver"]),
        qubit_order=prob["qubits"],
    ).final_state_vector
    assert abs(abs(np.vdot(ref, psi)) - 1.0) < 1e-6


def test_hf_trajectory_energy_matches_density_matrix_within_shot_noise(monkeypatch):
    """Same settings: CDR-style shot estimate should agree within ~O(1/sqrt(shots))."""
    prob = _hf_problem()
    assert prob["ogm_file"] is not None, "HF OGM file missing"
    num_shots = 2048
    kwargs = dict(
        ansatz_circuit=prob["circuit"],
        resolver=prob["resolver"],
        observable_h=prob["observable"],
        qubits=prob["qubits"],
        noise_params=prob["noise_params"],
        simulator_seed=11,
        num_shots=num_shots,
        measurement_scheme="ogm",
        p_0_success=np.full(len(prob["qubits"]), 0.97),
        p_1_success=np.full(len(prob["qubits"]), 0.90),
        apply_rem=True,
        apply_readout_noise=True,
        sampling_seed=22,
        ogm_file=prob["ogm_file"],
        return_per_term=False,
    )

    monkeypatch.setenv("NOISY_SIM_BACKEND", "density_matrix")
    t0 = time.perf_counter()
    dm = sm.estimate_noisy_shots_for_resolver(**kwargs)
    t_dm = time.perf_counter() - t0

    # Average several trajectory seeds — channel MC + shot noise.
    traj_energies = []
    t_traj = 0.0
    monkeypatch.setenv("NOISY_SIM_BACKEND", "trajectory")
    for seed in (11, 1011, 2011, 3011, 4011):
        kw = dict(kwargs)
        kw["simulator_seed"] = seed
        t0 = time.perf_counter()
        out = sm.estimate_noisy_shots_for_resolver(**kw)
        t_traj += time.perf_counter() - t0
        traj_energies.append(float(out["energy_rem"]))
    t_traj /= len(traj_energies)
    traj_mean = float(np.mean(traj_energies))
    traj_std = float(np.std(traj_energies, ddof=1))
    dm_e = float(dm["energy_rem"])

    # Statistical tolerance: a few combined standard errors.
    tol = max(0.05, 4.0 * traj_std / np.sqrt(len(traj_energies)), 3.0 / np.sqrt(num_shots))
    print(
        f"\n[HF energy] DM={dm_e:.6f}  traj_mean={traj_mean:.6f}±{traj_std:.6f}  "
        f"|Δ|={abs(dm_e - traj_mean):.6f}  tol={tol:.6f}"
    )
    print(f"[HF timing] DM={t_dm:.3f}s  traj_avg={t_traj:.3f}s  speedup={t_dm / t_traj:.2f}x")
    assert abs(dm_e - traj_mean) <= tol


def test_hf_trajectory_cdr_call_speed_and_accuracy(monkeypatch):
    """One full ``run_mitigation('cdr')`` on HF: report speedup, check CDR energy."""
    prob = _hf_problem()
    assert prob["ogm_file"] is not None
    num_shots = 512
    num_train = 6
    common = dict(
        ansatz_circuit=prob["circuit"],
        observable_h=prob["observable"],
        qubits=prob["qubits"],
        target_resolver=prob["resolver"],
        target_params=prob["resolver"],
        symbols=prob["symbols"],
        base_noise_cfg=prob["noise_params"],
        shot_cfg={
            "num_shots": num_shots,
            "measurement_scheme": "ogm",
            "apply_readout_noise": True,
            "sampling_seed": 7,
            "ogm_file": prob["ogm_file"],
        },
        readout_cal={
            "p_0_success": np.full(len(prob["qubits"]), 0.97),
            "p_1_success": np.full(len(prob["qubits"]), 0.90),
        },
        cdr_cfg={
            "num_circuits": num_train,
            "t_max": 2,
            "seed": 42,
            "cdr_fit_scope": "per_pauli",
        },
        simulator_seed=99,
    )

    monkeypatch.setenv("NOISY_SIM_BACKEND", "density_matrix")
    monkeypatch.setenv("PARALLEL_CDR_WORKERS", "1")
    t0 = time.perf_counter()
    dm = sm.run_mitigation("cdr", **common)
    t_dm = time.perf_counter() - t0

    monkeypatch.setenv("NOISY_SIM_BACKEND", "trajectory")
    t0 = time.perf_counter()
    traj = sm.run_mitigation("cdr", **common)
    t_traj = time.perf_counter() - t0

    dm_e = float(dm["cdr_rem_corrected"])
    traj_e = float(traj["cdr_rem_corrected"])
    # Exact noiseless reference for scale.
    psi = cirq.Simulator().simulate(
        cirq.resolve_parameters(prob["circuit"], prob["resolver"]),
        qubit_order=prob["qubits"],
    ).final_state_vector
    e_exact = float(
        np.real(
            prob["observable"].expectation_from_state_vector(
                psi, qubit_map={q: i for i, q in enumerate(prob["qubits"])}
            )
        )
    )
    # Both backends should land in the same neighborhood of the noiseless energy.
    # Allow generous absolute tolerance because this is a short CDR smoke test.
    print(
        f"\n[HF CDR] exact={e_exact:.6f}  DM={dm_e:.6f}  traj={traj_e:.6f}  "
        f"|DM-traj|={abs(dm_e - traj_e):.6f}"
    )
    print(f"[HF CDR timing] DM={t_dm:.3f}s  traj={t_traj:.3f}s  speedup={t_dm / t_traj:.2f}x")
    assert abs(dm_e - traj_e) < 0.35
    # Speed: trajectory may be slower on 8q/high-shot; still record ratio.
    # Require it finishes and is not catastrophically slower (>20x) on this HF smoke.
    assert t_traj < 20.0 * t_dm + 5.0


def test_install_trajectory_backend_sets_env():
    from trajectory_backend import install_trajectory_backend

    os.environ.pop("NOISY_SIM_BACKEND", None)
    install_trajectory_backend()
    assert sm.noisy_sim_backend() == "trajectory"


def test_br2_scale_trajectory_faster_than_density_matrix(monkeypatch):
    """At 12 qubits the DM ``4^n`` cost dominates; trajectories should win.

    Uses a shallow synthetic circuit (Br2-sized width) and direct Pauli shots so
    the comparison isolates noisy-state simulation cost.
    """
    n = 12
    qubits = list(cirq.LineQubit.range(n))
    symbols = list(sympy.symbols(f"th_0:{4}"))
    circuit = cirq.Circuit()
    circuit.append(cirq.X(qubits[i]) for i in range(8))
    for i, sym in enumerate(symbols):
        circuit.append(cirq.ry(sym).on(qubits[i]))
        circuit.append(cirq.CZ(qubits[i], qubits[i + 4]))
        circuit.append(cirq.rx(0.3 * sym).on(qubits[i + 6]))
    observable = (
        -5000.0 * cirq.PauliString()
        + 0.5 * cirq.Z(qubits[0])
        - 0.3 * cirq.Z(qubits[3]) * cirq.Z(qubits[7])
        + 0.2 * cirq.X(qubits[1]) * cirq.X(qubits[5])
        + 0.1 * cirq.Y(qubits[2]) * cirq.Y(qubits[6])
    )
    resolver = {s: 0.2 * (i + 1) for i, s in enumerate(symbols)}
    noise_params = {
        "two_qubit_depol_prob": 0.01,
        "one_qubit_depol_prob": 0.001,
        "cross_chip_two_qubit_depol_prob": 0.02,
    }
    kwargs = dict(
        ansatz_circuit=circuit,
        resolver=resolver,
        observable_h=observable,
        qubits=qubits,
        noise_params=noise_params,
        simulator_seed=3,
        num_shots=256,
        measurement_scheme="direct_pauli",
        p_0_success=np.full(n, 0.97),
        p_1_success=np.full(n, 0.90),
        apply_rem=True,
        apply_readout_noise=True,
        sampling_seed=9,
    )

    monkeypatch.setenv("NOISY_SIM_BACKEND", "density_matrix")
    t0 = time.perf_counter()
    dm = sm.estimate_noisy_shots_for_resolver(**kwargs)
    t_dm = time.perf_counter() - t0

    monkeypatch.setenv("NOISY_SIM_BACKEND", "trajectory")
    t0 = time.perf_counter()
    traj = sm.estimate_noisy_shots_for_resolver(**kwargs)
    t_traj = time.perf_counter() - t0

    print(
        f"\n[12q scale] DM={float(dm['energy_rem']):.6f} ({t_dm:.3f}s)  "
        f"traj={float(traj['energy_rem']):.6f} ({t_traj:.3f}s)  "
        f"speedup={t_dm / t_traj:.2f}x"
    )
    # Energies need only be in the same ballpark (few Pauli terms, 256 shots).
    assert abs(float(dm["energy_rem"]) - float(traj["energy_rem"])) < 1.5
    assert t_traj < t_dm, f"expected trajectory faster at 12q, got DM={t_dm:.3f}s traj={t_traj:.3f}s"
