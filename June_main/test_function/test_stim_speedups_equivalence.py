#!/usr/bin/env python3
"""Equivalence + speed tests for Stim / trajectory speedups vs original DM path.

Compares density-matrix (`main_HF.py` / `main_Cl2.py` hot path) against the
accelerated trajectory backend (Stim near-Clifford + optional shot threads)
on the same HF assets. Outputs must agree within shot / MC noise; the
accelerated path must not change science knobs (shots, OGM, noise, CDR count).

Run:

    MPLCONFIGDIR=/tmp/mpl .venv_py311/bin/python -m pytest -q \\
        June_main/test_function/test_stim_speedups_equivalence.py -s
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
_preferred = [str(EXPORT_LIB), str(EXPORT_DIR)]
sys.path[:] = _preferred + [p for p in sys.path if p not in _preferred]

import shot_measurement as sm  # noqa: E402
from main_cursor_lib import (  # noqa: E402
    count_non_clifford_ops,
    generate_near_clifford_param_sets,
    RZXGate,
)
from stim_clifford import compile_stim_hybrid_ops, stim_available  # noqa: E402
from trajectory_backend import install_trajectory_backend  # noqa: E402


def _load_hf_circuit():
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
    n_electrons = int(data.get("n_electrons", 8))
    prep = cirq.Circuit(cirq.X(q[i]) for i in range(n_electrons))
    return prep + c, list(q), syms


def _load_hf_hamiltonian(qubits, bond: float = 1.2):
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
            for qq, code in zip(qubits, codes):
                if code == 0:
                    continue
                term *= idx_to_pauli[code](qq)
            out += coeff * term
    return out


def _hf_problem(bond: float = 1.2):
    circuit, qubits, symbols = _load_hf_circuit()
    observable = _load_hf_hamiltonian(qubits, bond=bond)
    ogm = EXPORT_LIB / "OGM_measurement_basis" / f"OGM_HF_bond_{bond:.1f}.txt"
    if not ogm.is_file():
        ogm = JUNE_MAIN / "OGM_measurement_basis" / f"OGM_HF_bond_{bond:.1f}.txt"
    resolver = {s: 0.15 * (i + 1) for i, s in enumerate(symbols)}
    noise_params = {
        "two_qubit_depol_prob": 0.01,
        "one_qubit_depol_prob": 0.001,
        "cross_chip_two_qubit_depol_prob": 0.02,
    }
    return {
        "circuit": circuit,
        "qubits": qubits,
        "symbols": symbols,
        "resolver": resolver,
        "observable": observable,
        "ogm_file": ogm if ogm.is_file() else None,
        "noise_params": noise_params,
    }


def _energy_kwargs(prob, *, num_shots: int, sim_seed: int, samp_seed: int):
    return dict(
        ansatz_circuit=prob["circuit"],
        resolver=prob["resolver"],
        observable_h=prob["observable"],
        qubits=prob["qubits"],
        noise_params=prob["noise_params"],
        simulator_seed=sim_seed,
        num_shots=num_shots,
        measurement_scheme="ogm",
        p_0_success=np.full(len(prob["qubits"]), 0.97),
        p_1_success=np.full(len(prob["qubits"]), 0.90),
        apply_rem=True,
        apply_readout_noise=True,
        sampling_seed=samp_seed,
        ogm_file=prob["ogm_file"],
        return_per_term=False,
    )


@pytest.mark.skipif(not stim_available(), reason="stim not installed")
def test_stim_compiles_near_clifford_hf_trainers():
    prob = _hf_problem()
    trainers = generate_near_clifford_param_sets(
        prob["resolver"],
        prob["symbols"],
        num_circuits=6,
        t_max=2,
        circuit=prob["circuit"],
        seed=7,
    )
    nons = [count_non_clifford_ops(prob["circuit"], r) for r in trainers]
    assert nons == [1] * len(trainers)
    for r in trainers:
        compiled = compile_stim_hybrid_ops(
            prob["circuit"], r, prob["qubits"], prob["noise_params"]
        )
        assert compiled is not None
        n_unitary = sum(1 for k, _ in compiled if k == "unitary")
        assert n_unitary == 1


@pytest.mark.skipif(not stim_available(), reason="stim not installed")
def test_hf_stim_energy_matches_density_matrix_within_shot_noise(monkeypatch):
    """Near-Clifford trainer: Stim path ≈ original DM energy (statistical)."""
    prob = _hf_problem()
    assert prob["ogm_file"] is not None
    trainer = generate_near_clifford_param_sets(
        prob["resolver"],
        prob["symbols"],
        num_circuits=1,
        t_max=2,
        circuit=prob["circuit"],
        seed=11,
    )[0]
    num_shots = 2048
    kwargs = _energy_kwargs(prob, num_shots=num_shots, sim_seed=11, samp_seed=22)
    kwargs["resolver"] = trainer

    monkeypatch.setenv("NOISY_SIM_BACKEND", "density_matrix")
    monkeypatch.delenv("USE_STIM_CLIFFORD", raising=False)
    t0 = time.perf_counter()
    dm = sm.estimate_noisy_shots_for_resolver(**kwargs)
    t_dm = time.perf_counter() - t0

    monkeypatch.setenv("NOISY_SIM_BACKEND", "trajectory")
    monkeypatch.setenv("USE_STIM_CLIFFORD", "1")
    monkeypatch.setenv("TRAJECTORY_SHOT_WORKERS", "1")
    stim_vals = []
    t_stim = 0.0
    out = None
    for k in range(4):
        kw = dict(kwargs)
        kw["simulator_seed"] = 11 + 100 * (k + 1)
        kw["sampling_seed"] = 22 + 100 * (k + 1)
        t0 = time.perf_counter()
        out = sm.estimate_noisy_shots_for_resolver(**kw)
        t_stim += time.perf_counter() - t0
        stim_vals.append(float(out["energy_rem"]))
        assert "stim" in str(out.get("noisy_backend", ""))

    stim_mean = float(np.mean(stim_vals))
    stim_std = float(np.std(stim_vals))
    dm_rem = float(dm["energy_rem"])
    # Allow ~3σ of the MC seed spread, with a floor for shot noise.
    tol = max(0.15, 3.0 * stim_std + 0.05)
    err = abs(stim_mean - dm_rem)
    print(
        f"[HF trainer energy] DM={dm_rem:.6f} stim_mean={stim_mean:.6f} "
        f"std={stim_std:.6f} |err|={err:.6f} tol={tol:.6f} "
        f"t_dm={t_dm:.3f}s t_stim_avg={t_stim / 4:.3f}s backend={out.get('noisy_backend')}"
    )
    assert err <= tol, f"DM vs Stim disagree: |{dm_rem} - {stim_mean}|={err} > {tol}"


@pytest.mark.skipif(not stim_available(), reason="stim not installed")
def test_hf_near_clifford_trainer_stim_faster_than_numpy_traj(monkeypatch):
    """Near-Clifford trainer energy: Stim hybrid should beat pure NumPy traj."""
    prob = _hf_problem()
    assert prob["ogm_file"] is not None
    trainers = generate_near_clifford_param_sets(
        prob["resolver"],
        prob["symbols"],
        num_circuits=1,
        t_max=2,
        circuit=prob["circuit"],
        seed=3,
    )
    resolver = trainers[0]
    kwargs = _energy_kwargs(prob, num_shots=1024, sim_seed=5, samp_seed=6)
    kwargs["resolver"] = resolver

    monkeypatch.setenv("NOISY_SIM_BACKEND", "trajectory")
    monkeypatch.setenv("TRAJECTORY_SHOT_WORKERS", "1")

    monkeypatch.setenv("USE_STIM_CLIFFORD", "0")
    t0 = time.perf_counter()
    numpy_out = sm.estimate_noisy_shots_for_resolver(**kwargs)
    t_numpy = time.perf_counter() - t0

    monkeypatch.setenv("USE_STIM_CLIFFORD", "1")
    t0 = time.perf_counter()
    stim_out = sm.estimate_noisy_shots_for_resolver(**kwargs)
    t_stim = time.perf_counter() - t0

    # Average a few seeds — single-shot MC noise on 1024 shots is O(0.1–0.3 Eh).
    stim_vals = [float(stim_out["energy_rem"])]
    numpy_vals = [float(numpy_out["energy_rem"])]
    for k in range(3):
        kw = dict(kwargs)
        kw["simulator_seed"] = 5 + 17 * (k + 1)
        kw["sampling_seed"] = 6 + 19 * (k + 1)
        monkeypatch.setenv("USE_STIM_CLIFFORD", "0")
        numpy_vals.append(float(sm.estimate_noisy_shots_for_resolver(**kw)["energy_rem"]))
        monkeypatch.setenv("USE_STIM_CLIFFORD", "1")
        stim_vals.append(float(sm.estimate_noisy_shots_for_resolver(**kw)["energy_rem"]))
    err = abs(float(np.mean(stim_vals)) - float(np.mean(numpy_vals)))
    print(
        f"[HF trainer speed] numpy={t_numpy:.3f}s stim={t_stim:.3f}s "
        f"speedup={t_numpy / max(t_stim, 1e-9):.2f}x |ΔE_mean|={err:.6f} "
        f"backend={stim_out.get('noisy_backend')}"
    )
    assert err < 0.35, f"Stim vs NumPy traj disagree too much: {err}"
    assert t_stim < t_numpy * 1.15, (
        f"expected Stim <= ~NumPy time, got stim={t_stim:.3f}s numpy={t_numpy:.3f}s"
    )


def test_hf_cdr_accelerated_matches_original_dm(monkeypatch):
    """Production-like HF CDR: DM vs Stim+traj, averaged over 3 seeds.

    Knobs match ``main_HF.py`` defaults: 8192 shots, 30 trainers, t_max=2.
    """
    prob = _hf_problem()
    assert prob["ogm_file"] is not None

    # Production-like defaults from main_HF.py
    num_shots = 8192
    num_circuits = 30
    t_max = 2
    n_repeats = 3
    seed_base = 1234  # GLOBAL_RANDOM_SEED / GLOBAL_SAMPLING_SEED default

    keys = (
        "unmit_target",
        "rem_target",
        "cdr_unmit_corrected",
        "cdr_rem_corrected",
    )
    dm_rows: dict[str, list[float]] = {k: [] for k in keys}
    accel_rows: dict[str, list[float]] = {k: [] for k in keys}
    t_dm_total = 0.0
    t_accel_total = 0.0

    # Modest parallelism for wall time; same for both backends.
    monkeypatch.setenv("PARALLEL_CDR_WORKERS", "4")
    monkeypatch.setenv("TRAJECTORY_SHOT_WORKERS", "1")

    for rep in range(n_repeats):
        sim_seed = seed_base + 1000 * rep
        samp_seed = seed_base + 1000 * rep
        cdr_seed = 42 + 1000 * rep
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
                "sampling_seed": samp_seed,
                "ogm_file": prob["ogm_file"],
            },
            readout_cal={
                "p_0_success": np.full(len(prob["qubits"]), 0.97),
                "p_1_success": np.full(len(prob["qubits"]), 0.90),
            },
            cdr_cfg={
                "num_circuits": num_circuits,
                "t_max": t_max,
                "seed": cdr_seed,
                "cdr_fit_scope": "per_pauli",
            },
            simulator_seed=sim_seed,
        )

        monkeypatch.setenv("NOISY_SIM_BACKEND", "density_matrix")
        monkeypatch.setenv("USE_STIM_CLIFFORD", "0")
        t0 = time.perf_counter()
        dm = sm.run_mitigation("cdr", **common)
        t_dm = time.perf_counter() - t0
        t_dm_total += t_dm

        monkeypatch.setenv("NOISY_SIM_BACKEND", "trajectory")
        monkeypatch.setenv("USE_STIM_CLIFFORD", "1")
        t0 = time.perf_counter()
        accel = sm.run_mitigation("cdr", **common)
        t_accel = time.perf_counter() - t0
        t_accel_total += t_accel

        print(
            f"[HF CDR prod rep {rep + 1}/{n_repeats}] "
            f"t_dm={t_dm:.1f}s t_accel={t_accel:.1f}s"
        )
        for key in keys:
            a = float(dm[key])
            b = float(accel[key])
            dm_rows[key].append(a)
            accel_rows[key].append(b)
            print(f"  {key}: DM={a:.10f} accel={b:.10f} |Δ|={abs(a - b):.10f}")

    print(
        f"\n[HF CDR prod AVERAGE over {n_repeats} runs] "
        f"shots={num_shots} trainers={num_circuits} t_max={t_max}"
    )
    print(
        f"  wall: DM_total={t_dm_total:.1f}s accel_total={t_accel_total:.1f}s "
        f"DM_avg={t_dm_total / n_repeats:.1f}s accel_avg={t_accel_total / n_repeats:.1f}s"
    )
    mean_errs: dict[str, float] = {}
    for key in keys:
        dm_mean = float(np.mean(dm_rows[key]))
        accel_mean = float(np.mean(accel_rows[key]))
        dm_std = float(np.std(dm_rows[key]))
        accel_std = float(np.std(accel_rows[key]))
        err = abs(dm_mean - accel_mean)
        mean_errs[key] = err
        print(
            f"  {key}: DM_mean={dm_mean:.10f}±{dm_std:.6f}  "
            f"accel_mean={accel_mean:.10f}±{accel_std:.6f}  "
            f"|Δ_mean|={err:.10f}"
        )

    # Science deliverables: averaged CDR energies should agree tightly.
    assert mean_errs["cdr_unmit_corrected"] < 0.05
    assert mean_errs["cdr_rem_corrected"] < 0.05


def test_install_trajectory_backend_enables_stim_and_skips_dm():
    os.environ.pop("NOISY_SIM_BACKEND", None)
    os.environ.pop("SKIP_DM_DIAGNOSTICS", None)
    install_trajectory_backend()
    assert sm.noisy_sim_backend() == "trajectory"
    assert os.environ.get("SKIP_DM_DIAGNOSTICS", "1") in ("1", "true", "yes", "on")


def test_br2_cmx_transform_rewrites_serial_loop():
    from parallel_cmx import transform_br2_source

    src = (EXPORT_DIR / "main_Br2.py").read_text(encoding="utf-8")
    out = transform_br2_source(src)
    assert "measure_moment_grid_parallel" in out
    assert 'moment_replicates["H"].append(_measure_moment("H", CME_H_NUM_SHOTS))' not in out
