#!/usr/bin/env python3
"""Compare exact vs REM energies along the VQE optimization trajectory.

Reverse-engineering tool for the error-mitigation pipeline in
``ML_CDR/ML_CDR_together/main_HF_ML_tog.ipynb``: runs the same 15-iteration
noiseless VQE path as the notebook (parameter-shift GD from theta=0), then
samples ``N_SAMPLES`` nearby points by adding small Gaussian jitter to
trajectory thetas. Evaluates noiseless exact energy vs noisy+REM shot energy
and plots patterns (linearity, residuals, theta dependence).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cirq
import matplotlib.pyplot as plt
import numpy as np
import sympy
from cirq.ops import GlobalPhaseGate

setattr(cirq, "GlobalPhaseGate", GlobalPhaseGate)

# ---------------------------------------------------------------------------
# Config (edit these)
# ---------------------------------------------------------------------------
N_SAMPLES = 100
# VQE trajectory (matches main_HF_ML_tog.ipynb: GP_VQE_ITERS, GP_LEARNING_RATE, etc.)
VQE_ITERS = 15
VQE_LR = 0.5
VQE_STEP_MAX = 0.1  # GP_OPTIMIZER_STEP_MAX trust-region clip
THETA_INIT = None  # None -> zeros(n_params)
# Gaussian jitter added to a randomly chosen trajectory point per sample.
THETA_JITTER_STD = 0.05  # rad; keep small so energies stay in the VQE basin (~ -98 Eh)
PLOT_EXACT_X_LIM = (-98.4, -98.3)

RNG_SEED = 1234
SIMULATOR_SEED = 1234
SAMPLING_SEED_BASE = 1234

MOLECULE = "HF"
BOND_LENGTH = 2.2
N_ACTIVE_ELECTRONS = 6
N_SPATIAL_ORBITALS = 4
N_QUBITS = 2 * N_SPATIAL_ORBITALS
ETA = N_ACTIVE_ELECTRONS // 2

GLOBAL_NUM_SHOTS = 8192
GLOBAL_MEASUREMENT_SCHEME = "ogm"
GLOBAL_APPLY_READOUT_NOISE = True
GLOBAL_READOUT_P0_SUCCESS = np.array([0.97, 0.96, 0.93, 0.96, 0.92, 0.93, 0.94, 0.92])
GLOBAL_READOUT_P1_SUCCESS = np.array([0.85, 0.90, 0.88, 0.90, 0.86, 0.89, 0.87, 0.85])

CIRCUIT_NAME = "HF_8q_3doubles_rzx"
CZ_CROSS_CHIP_TAG = "cz_cross_chip"
CROSS_CHIP_QUBIT_PAIRS = {(2, 6)}
SHADOWGROUPING_ROOT = Path("/Users/zacharyhe/shadowgrouping")
EPSILON = 0.1

# ---------------------------------------------------------------------------
# Repo / import paths
# ---------------------------------------------------------------------------
_REPO = Path(__file__).resolve().parent.parent
_JUNE_MAIN = _REPO / "June_main"
_ML_CDR = _JUNE_MAIN / "ML_CDR" / "ML_CDR_together"
for _p in (str(_ML_CDR), str(_JUNE_MAIN), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from main_cursor_lib_ML_tog import (  # noqa: E402
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    GateArityDepolarizingNoise,
    ONE_QUBIT_GATE_DEPOL_PROB,
    RZXGate,
    TWO_QUBIT_GATE_DEPOL_PROB,
)
from shot_measurement_ML_tog import estimate_energy_from_noisy_rho_shots  # noqa: E402

_CROSS_CHIP_PAIR_SET = {frozenset(pair) for pair in CROSS_CHIP_QUBIT_PAIRS}
CIRCUITS_DIR = _JUNE_MAIN / "circuits2read"
CIRCUIT_JSON = CIRCUITS_DIR / f"{CIRCUIT_NAME}.json"
OGM_FILE = _JUNE_MAIN / "OGM_measurement_basis" / f"OGM_{MOLECULE}_bond_{BOND_LENGTH:.1f}.txt"
HAM_PATH = _REPO / "Pauli_Ham" / f"{MOLECULE}_bond_{BOND_LENGTH:.1f}.txt"
OUTPUT_DIR = _JUNE_MAIN


def _is_cross_chip_pair(qubit_indices: list[int]) -> bool:
    return frozenset(qubit_indices) in _CROSS_CHIP_PAIR_SET


def load_circuit_from_json(path: Path):
    """Build a cirq.Circuit from a saved UCCSD circuit JSON (matches notebook)."""
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
            raise ValueError(f"Unhandled op in circuit JSON: {op!r}")

        if len(new_op.qubits) == 2 and _is_cross_chip_pair(g["qubits"]):
            new_op = new_op.with_tags(CZ_CROSS_CHIP_TAG)
        c.append(new_op)
    return c, list(q), syms, data


def load_pauli_sum_from_numbered_file(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
    idx_to_pauli = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    out = cirq.PauliSum()
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            coeff = float(parts[0])
            pauli_codes = [int(x) for x in parts[1:]]
            if len(pauli_codes) != len(qubits):
                raise ValueError(
                    f"{path}:{lineno} has {len(pauli_codes)} Pauli indices, expected {len(qubits)}."
                )
            pauli_string = cirq.PauliString()
            for q, code in zip(qubits, pauli_codes):
                if code == 0:
                    continue
                if code not in idx_to_pauli:
                    raise ValueError(f"{path}:{lineno} has invalid Pauli code {code}.")
                pauli_string *= idx_to_pauli[code](q)
            out += coeff * pauli_string
    return out


def resolver_from_params(params_vec: np.ndarray, symbols: list, n_params: int) -> cirq.ParamResolver:
    p = np.asarray(params_vec, dtype=float).reshape(n_params)
    return cirq.ParamResolver({symbols[i]: float(p[i]) for i in range(n_params)})


def hf_prep_circuit(qubits: list[cirq.Qid]) -> cirq.Circuit:
    half = N_SPATIAL_ORBITALS
    eta = ETA
    occupied = list(range(eta)) + list(range(half, half + eta))
    return cirq.Circuit([cirq.X(qubits[k]) for k in occupied])


def build_full_circuit():
    ansatz_circuit, qubits, symbols, _meta = load_circuit_from_json(CIRCUIT_JSON)
    prep_circuit = hf_prep_circuit(qubits)
    circuit = prep_circuit + ansatz_circuit
    n_params = len(symbols)
    assert len(qubits) == N_QUBITS
    return circuit, qubits, symbols, n_params


def exact_energy(
    circuit: cirq.Circuit,
    resolver: cirq.ParamResolver,
    qubits: list[cirq.Qid],
    pauli_sum: cirq.PauliSum,
    qubit_map: dict,
) -> float:
    resolved = cirq.resolve_parameters(circuit, resolver)
    psi = np.asarray(
        cirq.Simulator(dtype=np.complex128).simulate(resolved, qubit_order=qubits).final_state_vector,
        dtype=np.complex128,
    )
    return float(np.real(pauli_sum.expectation_from_state_vector(psi, qubit_map=qubit_map)))


def noisy_rho(
    circuit: cirq.Circuit,
    resolver: cirq.ParamResolver,
    qubits: list[cirq.Qid],
    gate_noise: GateArityDepolarizingNoise,
    *,
    simulator_seed: int,
) -> np.ndarray:
    noisy_circuit = circuit.with_noise(gate_noise)
    resolved_noisy = cirq.resolve_parameters(noisy_circuit, resolver)
    return np.asarray(
        cirq.DensityMatrixSimulator(seed=simulator_seed)
        .simulate(resolved_noisy, qubit_order=qubits)
        .final_density_matrix,
        dtype=np.complex128,
    )


def exact_energy_at_theta(
    circuit: cirq.Circuit,
    theta: np.ndarray,
    symbols: list,
    n_params: int,
    qubits: list[cirq.Qid],
    pauli_sum: cirq.PauliSum,
    qubit_map: dict,
) -> float:
    return exact_energy(
        circuit,
        resolver_from_params(theta, symbols, n_params),
        qubits,
        pauli_sum,
        qubit_map,
    )


def run_noiseless_vqe_trajectory(
    circuit: cirq.Circuit,
    symbols: list,
    n_params: int,
    qubits: list[cirq.Qid],
    pauli_sum: cirq.PauliSum,
    qubit_map: dict,
    *,
    n_iters: int,
    learning_rate: float,
    step_max: float,
    theta_init: np.ndarray,
) -> tuple[list[np.ndarray], list[float]]:
    """Parameter-shift GD on exact energy (matches notebook noiseless reference + step clip)."""
    shift = np.pi / 2.0
    grad_scale = 0.5
    theta = np.asarray(theta_init, dtype=float).ravel()[:n_params].copy()
    thetas = [theta.copy()]
    energies = [exact_energy_at_theta(circuit, theta, symbols, n_params, qubits, pauli_sum, qubit_map)]

    for _ in range(int(n_iters)):
        grad = np.zeros(n_params, dtype=float)
        for j in range(n_params):
            tp = theta.copy()
            tm = theta.copy()
            tp[j] += shift
            tm[j] -= shift
            grad[j] = (
                exact_energy_at_theta(circuit, tp, symbols, n_params, qubits, pauli_sum, qubit_map)
                - exact_energy_at_theta(circuit, tm, symbols, n_params, qubits, pauli_sum, qubit_map)
            ) * grad_scale
        step = -float(learning_rate) * grad
        step_norm = float(np.max(np.abs(step)))
        if step_norm > float(step_max):
            step *= float(step_max) / step_norm
        theta = theta + step
        thetas.append(theta.copy())
        energies.append(exact_energy_at_theta(circuit, theta, symbols, n_params, qubits, pauli_sum, qubit_map))
    return thetas, energies


def sample_jittered_trajectory_thetas(
    rng: np.random.Generator,
    trajectory: list[np.ndarray],
    n_samples: int,
    *,
    jitter_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pick a trajectory point per sample and add i.i.d. Gaussian jitter."""
    n_traj = len(trajectory)
    traj_idx = rng.integers(0, n_traj, size=n_samples)
    base = np.stack([trajectory[i] for i in traj_idx], axis=0)
    jitter = rng.normal(0.0, jitter_std, size=(n_samples, base.shape[1]))
    return base + jitter, traj_idx


def run_sweep(
    circuit: cirq.Circuit,
    qubits: list[cirq.Qid],
    symbols: list,
    n_params: int,
    pauli_sum: cirq.PauliSum,
    qubit_map: dict,
    gate_noise: GateArityDepolarizingNoise,
) -> list[dict]:
    if not OGM_FILE.is_file():
        raise FileNotFoundError(f"OGM file missing: {OGM_FILE}")
    if not SHADOWGROUPING_ROOT.is_dir():
        raise FileNotFoundError(f"SHADOWGROUPING_ROOT missing: {SHADOWGROUPING_ROOT}")

    trajectory, traj_energies = run_noiseless_vqe_trajectory(
        circuit,
        symbols,
        n_params,
        qubits,
        pauli_sum,
        qubit_map,
        n_iters=VQE_ITERS,
        learning_rate=VQE_LR,
        step_max=VQE_STEP_MAX,
        theta_init=np.zeros(n_params, dtype=float),
    )
    print(f"VQE trajectory ({len(trajectory)} points, {VQE_ITERS} iters, LR={VQE_LR}, step_max={VQE_STEP_MAX}):")
    for it, (th, e) in enumerate(zip(trajectory, traj_energies)):
        print(f"  iter {it:2d}: theta={np.round(th, 5).tolist()}  E_exact={e:.6f} Eh")

    rng = np.random.default_rng(RNG_SEED)
    thetas, traj_idx = sample_jittered_trajectory_thetas(
        rng, trajectory, N_SAMPLES, jitter_std=THETA_JITTER_STD
    )
    rows: list[dict] = []

    for i, theta in enumerate(thetas):
        resolver = resolver_from_params(theta, symbols, n_params)
        e_exact = exact_energy(circuit, resolver, qubits, pauli_sum, qubit_map)
        rho = noisy_rho(circuit, resolver, qubits, gate_noise, simulator_seed=SIMULATOR_SEED)
        shot_est = estimate_energy_from_noisy_rho_shots(
            rho,
            pauli_sum,
            qubits,
            num_shots=GLOBAL_NUM_SHOTS,
            measurement_scheme=GLOBAL_MEASUREMENT_SCHEME,
            p_0_success=GLOBAL_READOUT_P0_SUCCESS,
            p_1_success=GLOBAL_READOUT_P1_SUCCESS,
            apply_rem=True,
            apply_readout_noise=GLOBAL_APPLY_READOUT_NOISE,
            sampling_seed=SAMPLING_SEED_BASE + i,
            epsilon=EPSILON,
            ogm_file=OGM_FILE,
            shadowgrouping_root=SHADOWGROUPING_ROOT,
        )
        e_rem = float(shot_est["energy_rem"])
        e_unmit = float(shot_est["energy_unmitigated"])
        rows.append(
            {
                "traj_iter": int(traj_idx[i]),
                "theta1": float(theta[0]),
                "theta2": float(theta[1]),
                "theta3": float(theta[2]),
                "E_exact": e_exact,
                "E_rem": e_rem,
                "E_unmit": e_unmit,
                "residual_rem": e_rem - e_exact,
                "residual_unmit": e_unmit - e_exact,
            }
        )
        if (i + 1) % 10 == 0 or i == 0:
            print(
                f"  [{i + 1:3d}/{N_SAMPLES}] traj_iter={int(traj_idx[i]):2d} "
                f"theta={theta.round(3).tolist()}  "
                f"E_exact={e_exact:.6f}  E_rem={e_rem:.6f}  delta={e_rem - e_exact:+.6f}"
            )
    return rows


def fit_affine(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Least-squares y ≈ a*x + b; returns (coeffs [a, b], R^2)."""
    if len(x) < 2 or float(np.std(x)) == 0.0:
        bias = float(np.mean(y - x))
        return np.array([1.0, bias]), float("nan")
    coeffs = np.polyfit(x, y, deg=1)
    y_pred = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coeffs, r2


def padded_limits(values: np.ndarray, *, pad_fraction: float = 0.08, min_pad: float = 1e-4) -> tuple[float, float]:
    """Return tight plot limits with a small visual margin."""
    lo = float(np.min(values))
    hi = float(np.max(values))
    span = hi - lo
    pad = max(pad_fraction * span, min_pad)
    return lo - pad, hi + pad


def save_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "traj_iter",
        "theta1",
        "theta2",
        "theta3",
        "E_exact",
        "E_rem",
        "E_unmit",
        "residual_rem",
        "residual_unmit",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV: {path}")


def make_plots(rows: list[dict]) -> None:
    e_exact = np.array([r["E_exact"] for r in rows], dtype=float)
    e_rem = np.array([r["E_rem"] for r in rows], dtype=float)
    rem_vs_exact_coeffs, rem_vs_exact_r2 = fit_affine(e_exact, e_rem)

    # One diagnostic only: noisy/REM energy as a function of exact noiseless energy.
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(e_exact, e_rem, alpha=0.65, s=32, label="Noisy REM energy", c="C0")
    x_lim = PLOT_EXACT_X_LIM
    y_lim = padded_limits(e_rem)
    ax.plot(x_lim, x_lim, "k--", lw=1, label="y = x (perfect)")
    x_fit = np.linspace(x_lim[0], x_lim[1], 100)
    ax.plot(
        x_fit,
        np.polyval(rem_vs_exact_coeffs, x_fit),
        "C0",
        lw=2,
        label=(
            f"REM measured fit: y = {rem_vs_exact_coeffs[0]:.4f}x "
            f"+ {rem_vs_exact_coeffs[1]:.4f}  (R²={rem_vs_exact_r2:.4f})"
        ),
    )
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("Exact energy (noiseless statevector) [Eh]")
    ax.set_ylabel("Noisy REM energy [Eh]")
    ax.set_title(
        f"Noisy REM vs exact ({N_SAMPLES} jittered VQE-trajectory points, "
        f"jitter σ={THETA_JITTER_STD}, {MOLECULE} bond {BOND_LENGTH} Å)"
    )
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    scatter_path = OUTPUT_DIR / "exact_vs_rem_scatter.png"
    fig.savefig(scatter_path, dpi=150)
    print(f"Saved plot: {scatter_path}")

    plt.show()


def print_summary(rows: list[dict], coeffs_rem: np.ndarray, r2_rem: float) -> None:
    e_exact = np.array([r["E_exact"] for r in rows], dtype=float)
    e_rem = np.array([r["E_rem"] for r in rows], dtype=float)
    e_unmit = np.array([r["E_unmit"] for r in rows], dtype=float)
    residual_rem = e_rem - e_exact
    e_rem_corrected = np.polyval(coeffs_rem, e_rem)
    residual_corrected = e_rem_corrected - e_exact

    coeffs_unmit, r2_unmit = fit_affine(e_unmit, e_exact)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Samples: {N_SAMPLES}  |  VQE iters: {VQE_ITERS}  |  jitter σ: {THETA_JITTER_STD} rad"
    )
    print(f"Shots: {GLOBAL_NUM_SHOTS}  |  Scheme: {GLOBAL_MEASUREMENT_SCHEME}")
    print(f"E_exact range: [{e_exact.min():.6f}, {e_exact.max():.6f}] Eh")
    print(f"REM affine fit (E_exact ~ a*E_rem + b):")
    print(f"  slope a     = {coeffs_rem[0]:.6f}")
    print(f"  intercept b = {coeffs_rem[1]:.6f}")
    print(f"  R²          = {r2_rem:.6f}")
    print(f"Unmit affine fit (E_exact ~ a*E_unmit + b):")
    print(f"  slope a     = {coeffs_unmit[0]:.6f}")
    print(f"  intercept b = {coeffs_unmit[1]:.6f}")
    print(f"  R²          = {r2_unmit:.6f}")
    print()
    print(f"REM vs exact:")
    print(f"  mean |error| = {np.mean(np.abs(residual_rem)):.6f} Eh")
    print(f"  max  |error| = {np.max(np.abs(residual_rem)):.6f} Eh")
    print(f"  std  error   = {np.std(residual_rem):.6f} Eh")
    print(f"Affine-corrected REM vs exact:")
    print(f"  mean |error| = {np.mean(np.abs(residual_corrected)):.6f} Eh")
    print(f"  max  |error| = {np.max(np.abs(residual_corrected)):.6f} Eh")
    print(f"Unmit vs exact:")
    print(f"  mean |error| = {np.mean(np.abs(e_unmit - e_exact)):.6f} Eh")
    print("=" * 60)


def main() -> None:
    print(f"Loading circuit: {CIRCUIT_JSON.name}")
    print(f"Hamiltonian:     {HAM_PATH}")
    print(f"OGM file:        {OGM_FILE}  (exists={OGM_FILE.is_file()})")
    print(f"Shadowgrouping:  {SHADOWGROUPING_ROOT}  (exists={SHADOWGROUPING_ROOT.is_dir()})")

    circuit, qubits, symbols, n_params = build_full_circuit()
    pauli_sum = load_pauli_sum_from_numbered_file(HAM_PATH, qubits)
    qubit_map = {q: i for i, q in enumerate(qubits)}

    gate_noise = GateArityDepolarizingNoise(
        two_qubit_depol_prob=TWO_QUBIT_GATE_DEPOL_PROB,
        one_qubit_depol_prob=ONE_QUBIT_GATE_DEPOL_PROB,
        cross_chip_two_qubit_depol_prob=CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    )
    print(
        f"Noise: two_qubit_depol={gate_noise.two_qubit_depol_prob} "
        f"one_qubit_depol={gate_noise.one_qubit_depol_prob} "
        f"cross_chip_two_qubit_depol={gate_noise.cross_chip_two_qubit_depol_prob}"
    )
    print(
        f"\nBuilding VQE trajectory then sweeping {N_SAMPLES} jittered points "
        f"(σ={THETA_JITTER_STD} rad) ..."
    )

    rows = run_sweep(circuit, qubits, symbols, n_params, pauli_sum, qubit_map, gate_noise)

    csv_path = OUTPUT_DIR / "exact_vs_rem_results.csv"
    save_csv(rows, csv_path)

    e_exact = np.array([r["E_exact"] for r in rows], dtype=float)
    e_rem = np.array([r["E_rem"] for r in rows], dtype=float)
    coeffs_rem, r2_rem = fit_affine(e_rem, e_exact)

    print_summary(rows, coeffs_rem, r2_rem)
    make_plots(rows)


if __name__ == "__main__":
    main()
