#!/usr/bin/env python3
"""Compare gradient mitigation methods on HF (standalone; does not modify library code).

Implements the ablation from ``~/OTOC/gradient_cdr_plan.md`` against a noiseless
parameter-shift reference, using the same HF assets / optimized 3-parameter probe
as ``June_main/export2cloud/main_HF.py``.

Methods (all built from the *same* noisy ±π/2 shot measurements):
  1. rem              — REM energies, then parameter-shift
  2. cdr_per_pauli    — absolute per-Pauli CDR (``exact ≈ a x + b``), then PS
  3. nhat             — shared scale-strip ``x / N̂`` on Pauli expectations, then PS
  4. nhat_residual    — (3) then multiply gradient by residual slope ``a_res``
                        fit from training ``E_exact ≈ a_res E_#`` (through origin
                        on demeaned energies; additive offset cancels in PS)

Run from repo root, for example:

    .venv_py311/bin/python gradient_improvement_test.py
    .venv_py311/bin/python gradient_improvement_test.py --shots 2048 --num-train 8
    .venv_py311/bin/python gradient_improvement_test.py --grad-residual-train 2

Environment (optional):
  HF_BOND_LENGTH, GLOBAL_NUM_SHOTS, CDR_NUM_TRAINING_CIRCUITS, NOISY_SIM_BACKEND
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cirq
import numpy as np
import sympy

REPO_ROOT = Path(__file__).resolve().parent
JUNE_MAIN = REPO_ROOT / "June_main"
EXPORT_DIR = JUNE_MAIN / "export2cloud"
EXPORT_LIB = EXPORT_DIR / "June_main"

# Prefer export2cloud copies of shot_measurement / main_cursor_lib.
_preferred = [str(EXPORT_LIB), str(EXPORT_DIR), str(JUNE_MAIN)]
sys.path[:] = _preferred + [p for p in sys.path if p not in _preferred]

import shot_measurement as sm  # noqa: E402
from main_cursor_lib import (  # noqa: E402
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    CZ_CROSS_CHIP_TAG,
    ONE_QUBIT_GATE_DEPOL_PROB,
    RZXGate,
    TWO_QUBIT_GATE_DEPOL_PROB,
    generate_near_clifford_param_sets,
)

# Same probe used by main_HF.py before the VQE loop.
HF_OPTIMIZED_PARAMS = np.array([0.0020292770, 0.0020329417, -4.9689013693], dtype=float)
CROSS_CHIP_QUBIT_PAIRS = {(2, 6)}
_CROSS_CHIP_PAIR_SET = {frozenset(pair) for pair in CROSS_CHIP_QUBIT_PAIRS}
PARAM_SHIFT = 0.5 * np.pi

# Active space matching main_HF.py
N_ACTIVE_ELECTRONS = 6
N_SPATIAL_ORBITALS = 4
ETA = N_ACTIVE_ELECTRONS // 2


def _is_cross_chip_pair(qubit_indices) -> bool:
    return frozenset(qubit_indices) in _CROSS_CHIP_PAIR_SET


def load_hf_ansatz() -> tuple[cirq.Circuit, list[cirq.Qid], list, dict]:
    path = JUNE_MAIN / "circuits2read" / "HF_8q_3doubles_rzx.json"
    if not path.is_file():
        path = EXPORT_LIB / "circuits2read" / "HF_8q_3doubles_rzx.json"
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
                angle = float(g.get("coeff", 1.0)) * sym[g["param"]]
            else:
                angle = float(g.get("angle", g.get("value", 0.0)))
            new_op = RZXGate(angle).on(qs[0], qs[1])
        elif op in ("rx", "rz"):
            angle = float(g["coeff"]) * sym[g["param"]] if "param" in g else float(g["value"])
            gate = cirq.rx if op == "rx" else cirq.rz
            new_op = gate(angle).on(qs[0])
        else:
            raise ValueError(f"Unhandled op {op!r}")
        if len(new_op.qubits) == 2 and _is_cross_chip_pair(g["qubits"]):
            new_op = new_op.with_tags(CZ_CROSS_CHIP_TAG)
        c.append(new_op)
    return c, list(q), syms, data


def load_hf_hamiltonian(qubits: list[cirq.Qid], bond: float) -> cirq.PauliSum:
    ham_path = EXPORT_DIR / "Pauli_Ham" / f"HF_bond_{bond:.1f}.txt"
    if not ham_path.is_file():
        ham_path = REPO_ROOT / "Pauli_Ham" / f"HF_bond_{bond:.1f}.txt"
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
    return out


def build_hf_problem(bond: float) -> dict:
    ansatz, qubits, symbols, meta = load_hf_ansatz()
    half = N_SPATIAL_ORBITALS
    occupied = list(range(ETA)) + list(range(half, half + ETA))
    prep = cirq.Circuit(cirq.X(qubits[k]) for k in occupied)
    circuit = prep + ansatz
    observable = load_hf_hamiltonian(qubits, bond=bond)
    ogm = EXPORT_LIB / "OGM_measurement_basis" / f"OGM_HF_bond_{bond:.1f}.txt"
    if not ogm.is_file():
        ogm = JUNE_MAIN / "OGM_measurement_basis" / f"OGM_HF_bond_{bond:.1f}.txt"
    if not ogm.is_file():
        raise FileNotFoundError(f"Missing OGM basis for HF bond {bond:.1f}")
    return {
        "circuit": circuit,
        "qubits": qubits,
        "symbols": symbols,
        "observable": observable,
        "ogm_file": ogm,
        "meta": meta,
        "occupied": occupied,
        "bond": float(bond),
    }


def resolver_from_params(params: np.ndarray, symbols: list) -> dict:
    p = np.asarray(params, dtype=float).ravel()
    if p.size != len(symbols):
        raise ValueError(f"params length {p.size} != n_params {len(symbols)}")
    return {symbols[i]: float(p[i]) for i in range(len(symbols))}


def noiseless_energy(
    circuit: cirq.Circuit,
    observable: cirq.PauliSum,
    qubits: list[cirq.Qid],
    params: np.ndarray,
    symbols: list,
    sim: cirq.Simulator,
) -> float:
    resolver = resolver_from_params(params, symbols)
    psi = np.asarray(
        sim.simulate(
            cirq.resolve_parameters(circuit, resolver),
            qubit_order=qubits,
        ).final_state_vector,
        dtype=np.complex128,
    )
    qmap = {q: i for i, q in enumerate(qubits)}
    return float(np.real(observable.expectation_from_state_vector(psi, qubit_map=qmap)))


def parameter_shift_from_energy_fn(params: np.ndarray, energy_fn) -> np.ndarray:
    p = np.asarray(params, dtype=float).ravel()
    g = np.zeros_like(p)
    for i in range(p.size):
        plus = p.copy()
        minus = p.copy()
        plus[i] += PARAM_SHIFT
        minus[i] -= PARAM_SHIFT
        g[i] = 0.5 * (float(energy_fn(plus)) - float(energy_fn(minus)))
    return g


def estimate_nhat(
    training_exact_per_term: np.ndarray,
    training_noisy_per_term: np.ndarray,
    *,
    exact_cutoff: float = 0.2,
) -> tuple[float, int]:
    """Median ratio noisy/exact over informative Pauli-training pairs."""
    exact = np.asarray(training_exact_per_term, dtype=float)
    noisy = np.asarray(training_noisy_per_term, dtype=float)
    mask = np.abs(exact) > float(exact_cutoff)
    if not np.any(mask):
        # Fallback: loosen cutoff to top-quartile |exact|.
        flat = np.abs(exact).ravel()
        thr = float(np.quantile(flat, 0.75)) if flat.size else 0.0
        mask = np.abs(exact) > max(thr, 1e-3)
    if not np.any(mask):
        return 1.0, 0
    ratios = (noisy[mask] / exact[mask]).astype(float)
    ratios = ratios[np.isfinite(ratios)]
    ratios = ratios[ratios > 0.0]
    if ratios.size == 0:
        return 1.0, 0
    return float(np.median(ratios)), int(ratios.size)


def energy_from_per_term(
    per_term: np.ndarray,
    weights: np.ndarray,
    offset: float,
    *,
    scale: float = 1.0,
    residual_a: float = 1.0,
) -> float:
    """E = offset + a_res * sum_k w_k * (x_k / scale). Identity offset is not scaled."""
    x = np.asarray(per_term, dtype=float).ravel()
    w = np.asarray(weights, dtype=float).ravel()
    if scale == 0.0:
        raise ValueError("scale (N̂) must be nonzero")
    body = float(np.dot(w, x / float(scale)))
    return float(offset) + float(residual_a) * body


def fit_residual_a_from_energies(
    e_exact: np.ndarray,
    e_nhat: np.ndarray,
    offset: float,
) -> float:
    """Through-origin slope on demeaned (E - offset) after scale strip."""
    xs = np.asarray(e_nhat, dtype=float) - float(offset)
    ys = np.asarray(e_exact, dtype=float) - float(offset)
    xs = xs - float(np.mean(xs))
    ys = ys - float(np.mean(ys))
    denom = float(np.dot(xs, xs))
    if denom <= 0.0:
        return 1.0
    a = float(np.dot(xs, ys) / denom)
    if not np.isfinite(a) or a <= 0.0:
        return 1.0
    return a


def fit_residual_a_from_gradients(
    g_id: np.ndarray,
    g_hash: np.ndarray,
) -> float:
    """Shared through-origin residual: g_id ≈ a * g_# (flattened over train × params)."""
    x = np.asarray(g_hash, dtype=float).ravel()
    y = np.asarray(g_id, dtype=float).ravel()
    denom = float(np.dot(x, x))
    if denom <= 0.0:
        return 1.0
    a = float(np.dot(x, y) / denom)
    if not np.isfinite(a) or a <= 0.0:
        return 1.0
    return a


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-15 or nb < 1e-15:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def grad_metrics(g: np.ndarray, g_true: np.ndarray) -> dict[str, float]:
    err = g - g_true
    n_true = float(np.linalg.norm(g_true))
    return {
        "l2_error": float(np.linalg.norm(err)),
        "rel_l2_error": float(np.linalg.norm(err) / n_true) if n_true > 0 else float("nan"),
        "cosine": cosine_similarity(g, g_true),
        "grad_norm": float(np.linalg.norm(g)),
    }


def measure_point(
    *,
    circuit: cirq.Circuit,
    observable: cirq.PauliSum,
    qubits: list[cirq.Qid],
    resolver: dict,
    noise_params: dict,
    shot_cfg: dict,
    readout_cal: dict,
    simulator_seed: int,
) -> dict:
    est = sm.estimate_noisy_shots_for_resolver(
        circuit,
        resolver,
        observable,
        qubits,
        noise_params,
        simulator_seed=simulator_seed,
        num_shots=int(shot_cfg["num_shots"]),
        measurement_scheme=str(shot_cfg["measurement_scheme"]),
        p_0_success=readout_cal["p_0_success"],
        p_1_success=readout_cal["p_1_success"],
        apply_rem=True,
        apply_readout_noise=bool(shot_cfg["apply_readout_noise"]),
        sampling_seed=int(shot_cfg["sampling_seed"]),
        epsilon=float(shot_cfg.get("epsilon", 0.1)),
        ogm_file=shot_cfg["ogm_file"],
        return_per_term=True,
    )
    return {
        "energy_unmitigated": float(est["energy_unmitigated"]),
        "energy_rem": float(est["energy_rem"]),
        "per_term_unmitigated": np.asarray(est["per_term_unmitigated"], dtype=float),
        "per_term_rem": np.asarray(est["per_term_rem"], dtype=float),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bond", type=float, default=float(os.environ.get("HF_BOND_LENGTH", "1.2")))
    p.add_argument(
        "--shots",
        type=int,
        default=int(os.environ.get("GLOBAL_NUM_SHOTS", "2048")),
        help="Shots per energy evaluation (default 2048 for a manageable local run).",
    )
    p.add_argument(
        "--num-train",
        type=int,
        default=int(os.environ.get("CDR_NUM_TRAINING_CIRCUITS", "10")),
        help="Near-Clifford training circuits for CDR / N̂.",
    )
    p.add_argument("--t-max", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--simulator-seed", type=int, default=1234)
    p.add_argument("--exact-cutoff", type=float, default=0.2, help="|exact| cutoff for N̂ ratios.")
    p.add_argument(
        "--grad-residual-train",
        type=int,
        default=0,
        help="If >0, also fit residual a from noiseless vs N̂ gradients on this many "
        "training resolvers (extra shot cost: N * 2 * n_params measurements).",
    )
    p.add_argument(
        "--backend",
        choices=("density_matrix", "trajectory"),
        default=os.environ.get("NOISY_SIM_BACKEND", "density_matrix"),
    )
    p.add_argument(
        "--params",
        type=str,
        default="",
        help="Comma-separated params override (default: main_HF optimized 3-param probe).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    os.environ["NOISY_SIM_BACKEND"] = str(args.backend)

    if args.params.strip():
        params = np.array([float(x) for x in args.params.split(",")], dtype=float)
    else:
        params = HF_OPTIMIZED_PARAMS.copy()

    prob = build_hf_problem(args.bond)
    circuit = prob["circuit"]
    qubits = prob["qubits"]
    symbols = prob["symbols"]
    observable = prob["observable"]
    n_params = len(symbols)
    if params.size != n_params:
        raise SystemExit(f"Need {n_params} params, got {params.size}")

    noise_params = {
        "two_qubit_depol_prob": float(TWO_QUBIT_GATE_DEPOL_PROB),
        "one_qubit_depol_prob": float(ONE_QUBIT_GATE_DEPOL_PROB),
        "cross_chip_two_qubit_depol_prob": float(CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB),
    }
    shot_cfg = {
        "num_shots": int(args.shots),
        "measurement_scheme": "ogm",
        "apply_readout_noise": True,
        "sampling_seed": int(args.seed) + 17,
        "epsilon": 0.1,
        "ogm_file": prob["ogm_file"],
    }
    readout_cal = {
        "p_0_success": np.resize(
            np.array([0.97, 0.96, 0.93, 0.96, 0.92, 0.93, 0.94, 0.92], dtype=float),
            len(qubits),
        ),
        "p_1_success": np.resize(
            np.array([0.85, 0.90, 0.88, 0.90, 0.86, 0.89, 0.87, 0.85], dtype=float),
            len(qubits),
        ),
    }

    sim = cirq.Simulator(seed=int(args.simulator_seed), dtype=np.complex128)
    target_resolver = resolver_from_params(params, symbols)

    print("=" * 72)
    print("HF gradient improvement test (standalone)")
    print("=" * 72)
    print(f"bond           : {args.bond:.1f} Å")
    print(f"params         : {params.tolist()}")
    print(f"occupied prep  : {prob['occupied']}")
    print(f"shots          : {args.shots}")
    print(f"num_train      : {args.num_train}")
    print(f"backend        : {args.backend}")
    print(f"noise          : {noise_params}")
    print(f"ogm            : {prob['ogm_file']}")

    t0 = time.perf_counter()
    e_exact = noiseless_energy(circuit, observable, qubits, params, symbols, sim)
    g_true = parameter_shift_from_energy_fn(
        params,
        lambda p: noiseless_energy(circuit, observable, qubits, p, symbols, sim),
    )
    print(f"\n[exact] E={e_exact:.10f} Eh")
    print(f"[exact] g={np.array2string(g_true, precision=6, separator=', ')}")
    print(f"[exact] ||g||={float(np.linalg.norm(g_true)):.6e}")

    # --- Train CDR / estimate N̂ once at the test point (shared across methods) ---
    print("\n[train] generating near-Clifford resolvers + per-Pauli CDR models ...")
    try:
        resolvers = generate_near_clifford_param_sets(
            target_resolver,
            list(symbols),
            num_circuits=int(args.num_train),
            t_max=int(args.t_max),
            circuit=circuit,
            min_snap_fraction=0.0,
            seed=int(args.seed),
        )
    except ValueError as err:
        if "Unrecognized symbol naming convention" not in str(err):
            raise
        resolvers = sm._generate_near_clifford_resolvers_fallback(
            target_resolver,
            list(symbols),
            num_circuits=int(args.num_train),
            t_max=int(args.t_max),
            circuit=circuit,
            min_snap_fraction=0.0,
            seed=int(args.seed),
        )

    models = sm.train_cf_models_per_pauli(
        circuit,
        observable,
        qubits,
        resolvers,
        noise_params=noise_params,
        simulator_seed=int(args.simulator_seed),
        num_shots=int(args.shots),
        measurement_scheme="ogm",
        p_0_success=readout_cal["p_0_success"],
        p_1_success=readout_cal["p_1_success"],
        apply_readout_noise=True,
        sampling_seed=int(shot_cfg["sampling_seed"]),
        ogm_file=prob["ogm_file"],
    )
    weights = np.asarray(models["weights"], dtype=float)
    offset = float(models["hamiltonian_offset"])
    tex = np.asarray(models["training_exact_per_term"], dtype=float)
    trem = np.asarray(models["training_rem_per_term"], dtype=float)

    nhat, n_ratio = estimate_nhat(tex, trem, exact_cutoff=float(args.exact_cutoff))
    e_exact_train = offset + tex @ weights
    e_nhat_train = np.array(
        [energy_from_per_term(trem[k], weights, offset, scale=nhat) for k in range(trem.shape[0])],
        dtype=float,
    )
    a_res_energy = fit_residual_a_from_energies(e_exact_train, e_nhat_train, offset)

    print(f"[train] N̂={nhat:.6f}  (from {n_ratio} ratios, cutoff={args.exact_cutoff})")
    print(f"[train] residual a_res (energy fit) = {a_res_energy:.6f}")
    print(f"[train] implied CDR-like stretch 1/N̂ = {1.0 / nhat:.6f}")

    # --- Measure each ±π/2 point once; reuse for all methods ---
    print("\n[measure] parameter-shift circuit points (shared shot data) ...")
    shift_meas: dict[tuple[int, int], dict] = {}
    sampling_rng = np.random.default_rng(int(args.seed) + 99)
    for i in range(n_params):
        for sign in (+1, -1):
            p = params.copy()
            p[i] += sign * PARAM_SHIFT
            local_shot = dict(shot_cfg)
            local_shot["sampling_seed"] = int(sampling_rng.integers(1, 2**31 - 1))
            shift_meas[(i, sign)] = measure_point(
                circuit=circuit,
                observable=observable,
                qubits=qubits,
                resolver=resolver_from_params(p, symbols),
                noise_params=noise_params,
                shot_cfg=local_shot,
                readout_cal=readout_cal,
                simulator_seed=int(args.simulator_seed),
            )
            print(f"  measured param[{i}] {'+' if sign > 0 else '-'}π/2")

    def energy_at_shift(i: int, sign: int, method: str, a_res: float = 1.0) -> float:
        m = shift_meas[(i, sign)]
        if method == "rem":
            return float(m["energy_rem"])
        if method == "unmit":
            return float(m["energy_unmitigated"])
        if method == "cdr_per_pauli":
            out = sm.apply_cf_models_per_pauli(
                m["per_term_unmitigated"],
                m["per_term_rem"],
                models,
            )
            return float(out["cdr_rem_corrected"])
        if method == "nhat":
            return energy_from_per_term(m["per_term_rem"], weights, offset, scale=nhat, residual_a=1.0)
        if method == "nhat_residual":
            return energy_from_per_term(
                m["per_term_rem"], weights, offset, scale=nhat, residual_a=a_res
            )
        raise ValueError(method)

    def grad_from_method(method: str, a_res: float = 1.0) -> np.ndarray:
        g = np.zeros(n_params, dtype=float)
        for i in range(n_params):
            e_plus = energy_at_shift(i, +1, method, a_res=a_res)
            e_minus = energy_at_shift(i, -1, method, a_res=a_res)
            g[i] = 0.5 * (e_plus - e_minus)
        return g

    methods: list[tuple[str, np.ndarray, dict]] = []
    for name, a_res in (
        ("unmit", 1.0),
        ("rem", 1.0),
        ("cdr_per_pauli", 1.0),
        ("nhat", 1.0),
        ("nhat_residual", a_res_energy),
    ):
        g = grad_from_method(name if name != "nhat_residual" else "nhat_residual", a_res=a_res)
        methods.append((name, g, grad_metrics(g, g_true)))

    # Optional: residual a from training-circuit gradients (plan §3 step 3).
    a_res_grad = None
    if int(args.grad_residual_train) > 0:
        n_gt = min(int(args.grad_residual_train), len(resolvers))
        print(f"\n[train] fitting residual a from gradients on {n_gt} training resolvers ...")
        g_id_rows = []
        g_hash_rows = []
        for ridx in range(n_gt):
            resolver = resolvers[ridx]
            p_train = np.array([float(resolver[s]) for s in symbols], dtype=float)

            def e_id(p, _res=resolver):
                return noiseless_energy(circuit, observable, qubits, p, symbols, sim)

            g_id = parameter_shift_from_energy_fn(p_train, e_id)

            # Noisy N̂-cleaned energies at ± shifts for this training point.
            g_hash = np.zeros(n_params, dtype=float)
            for i in range(n_params):
                es = []
                for sign in (+1, -1):
                    p = p_train.copy()
                    p[i] += sign * PARAM_SHIFT
                    local_shot = dict(shot_cfg)
                    local_shot["sampling_seed"] = int(sampling_rng.integers(1, 2**31 - 1))
                    meas = measure_point(
                        circuit=circuit,
                        observable=observable,
                        qubits=qubits,
                        resolver=resolver_from_params(p, symbols),
                        noise_params=noise_params,
                        shot_cfg=local_shot,
                        readout_cal=readout_cal,
                        simulator_seed=int(args.simulator_seed),
                    )
                    es.append(
                        energy_from_per_term(
                            meas["per_term_rem"], weights, offset, scale=nhat, residual_a=1.0
                        )
                    )
                g_hash[i] = 0.5 * (es[0] - es[1])
            g_id_rows.append(g_id)
            g_hash_rows.append(g_hash)
            print(f"  train[{ridx}] ||g_id||={np.linalg.norm(g_id):.3e}  ||g_#||={np.linalg.norm(g_hash):.3e}")

        a_res_grad = fit_residual_a_from_gradients(np.asarray(g_id_rows), np.asarray(g_hash_rows))
        g_nhat = next(g for n, g, _ in methods if n == "nhat")
        g_mit = a_res_grad * g_nhat
        methods.append(("nhat_residual_grad", g_mit, grad_metrics(g_mit, g_true)))
        print(f"[train] residual a_res (gradient fit) = {a_res_grad:.6f}")

    elapsed = time.perf_counter() - t0

    # --- Report ---
    print("\n" + "=" * 72)
    print("Gradient accuracy vs noiseless parameter-shift reference")
    print("=" * 72)
    header = f"{'method':22s} {'||g-g*||':>12s} {'rel_l2':>10s} {'cosine':>10s} {'||g||':>12s}"
    print(header)
    print("-" * len(header))
    baseline_l2 = None
    rows_out = []
    for name, g, met in methods:
        if name == "rem":
            baseline_l2 = met["l2_error"]
        mark = ""
        if baseline_l2 is not None and name not in ("unmit", "rem") and baseline_l2 > 0:
            improve = 100.0 * (1.0 - met["l2_error"] / baseline_l2)
            mark = f"  ({improve:+.1f}% vs rem)"
        print(
            f"{name:22s} {met['l2_error']:12.6e} {met['rel_l2_error']:10.4f} "
            f"{met['cosine']:10.4f} {met['grad_norm']:12.6e}{mark}"
        )
        print(f"{'':22s} g={np.array2string(g, precision=6, separator=', ')}")
        rows_out.append({"method": name, "grad": g.tolist(), **met})

    # Rank by L2 error (lower better); NaN cosine last.
    ranked = sorted(rows_out, key=lambda r: r["l2_error"])
    best = ranked[0]
    print("\n" + "-" * 72)
    print(
        f"Best L2 method: {best['method']}  "
        f"||g-g*||={best['l2_error']:.6e}  cosine={best['cosine']:.4f}"
    )
    if baseline_l2 is not None and baseline_l2 > 0:
        print(
            f"Improvement vs REM: {100.0 * (1.0 - best['l2_error'] / baseline_l2):+.1f}%"
        )
    print(f"N̂={nhat:.6f}  a_res_energy={a_res_energy:.6f}", end="")
    if a_res_grad is not None:
        print(f"  a_res_grad={a_res_grad:.6f}", end="")
    print()
    print(f"Wall time: {elapsed:.1f}s")
    print("=" * 72)

    # Nonzero exit if everything failed catastrophically (optional CI hook).
    if not np.isfinite(best["l2_error"]):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
