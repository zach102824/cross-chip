"""Generate the concrete worked-example numbers used in ``gp_mitigated_energy.tex``.

This standalone script reconstructs the EXACT pipeline of ``main_HF_ML_tog.ipynb``
(HF, bond 1.2 A) without Jupyter/qiskit, warm-starts the combined CDR+GP mitigator,
and then replicates the inner loop of ``gp_mitigator_ML_tog.run_vqe_with_mitigator``
(analytic-gradient mode) for the first TWO VQE iterations -- capturing every number
needed to show, with real values:

    E_mit(theta) = offset + sum_i w_i * O_mit_i           (mitigated energy)
    dE_mit/dtheta_j = sum_i w_i [ dm_i/dangle . dangle/dtheta_j
                                  + dm_i/dO_noisy * dO_noisy_i/dtheta_j ]   (gradient)

It prints a human-readable breakdown plus a machine-readable JSON block that is
transcribed verbatim into the LaTeX doc. Re-run after changing the GP_* settings
to regenerate the example.

Usage:
    python docs/make_example_numbers.py
(run with the ``cross_chips_sim`` environment that has cirq + sklearn + shadowgrouping).
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

# --- make the local *_ML_tog modules + repo importable -----------------------
_THIS_DIR = Path(__file__).resolve().parent          # .../ML_CDR_together/docs
_TOG_DIR = _THIS_DIR.parent                           # .../ML_CDR_together
_REPO = _TOG_DIR
while not (_REPO / "June_main").is_dir() and _REPO != _REPO.parent:
    _REPO = _REPO.parent
for _p in (str(_TOG_DIR), str(_REPO / "June_main"), str(_REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cirq
import sympy

from main_cursor_lib_ML_tog import (
    RZXGate,
    TWO_QUBIT_GATE_DEPOL_PROB,
    ONE_QUBIT_GATE_DEPOL_PROB,
    CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPHASING_PROB,
    ONE_QUBIT_GATE_DEPHASING_PROB,
    CROSS_CHIP_TWO_QUBIT_GATE_DEPHASING_PROB,
    TWO_QUBIT_GATE_OVER_ROTATION,
    ONE_QUBIT_GATE_OVER_ROTATION,
    CROSS_CHIP_TWO_QUBIT_GATE_OVER_ROTATION,
)
import gp_mitigator_ML_tog as gpm

# =====================================================================
# 1. Constants -- mirror the notebook's first cell + hyperparameter cell
# =====================================================================
GLOBAL_NUM_SHOTS = 8192
GLOBAL_RANDOM_SEED = 1234
GLOBAL_SAMPLING_SEED = 1234
GLOBAL_MEASUREMENT_SCHEME = "ogm"
GLOBAL_APPLY_READOUT_NOISE = True
GLOBAL_READOUT_P0_SUCCESS = np.array([0.97, 0.96, 0.93, 0.96, 0.92, 0.93, 0.94, 0.92])
GLOBAL_READOUT_P1_SUCCESS = np.array([0.85, 0.90, 0.88, 0.90, 0.86, 0.89, 0.87, 0.85])

MOLECULE = "HF"
BOND_LENGTH = 1.2
bond_length = BOND_LENGTH
N_ACTIVE_ELECTRONS = 6
N_SPATIAL_ORBITALS = 4
N_QUBITS = 2 * N_SPATIAL_ORBITALS
ETA = N_ACTIVE_ELECTRONS // 2

CIRCUITS_DIR = _REPO / "June_main" / "circuits2read"
CIRCUIT_NAME = "HF_8q_3doubles_rzx"
CIRCUIT_JSON = CIRCUITS_DIR / f"{CIRCUIT_NAME}.json"
CZ_CROSS_CHIP_TAG = "cz_cross_chip"
CROSS_CHIP_QUBIT_PAIRS = {(2, 6)}
_CROSS_CHIP_PAIR_SET = {frozenset(p) for p in CROSS_CHIP_QUBIT_PAIRS}

# --- current (faithful) GP_* settings, copied from the notebook cell ---------
GP_N_WARMSTART_CIRCUITS = 60
GP_N_NONCLIFFORD_GATES = 2
GP_WARMSTART_SPREAD = 0.8
GP_SHOTS = 1 * int(GLOBAL_NUM_SHOTS)
GP_MAX_HARMONIC = 2
GP_INCLUDE_NOISY_FEATURE = True
GP_PAULI_ONEHOT = True
GP_PAULI_SUMMARIES = True
GP_AFFINE_REGULARIZATION = 0.0
GP_REFIT_AFFINE_ON_TOPUP = True
GP_KERNEL_TYPE = "matern"
GP_MATERN_NU = 2.5
GP_USE_ARD = True
GP_USE_PRODUCT_KERNEL = True
GP_NOISE_VARIANCE_INIT = 1e-3
GP_NORMALIZE_TARGETS = True
GP_N_RESTARTS = 2
GP_MAX_TRAIN_POINTS = 1000
GP_UNCERTAINTY_THRESHOLD = 0.02
GP_TOPUP_BATCH_SIZE = 12
GP_TOPUP_RADIUS = 0.5
GP_DROP_FARAWAY_POINTS = True
GP_MAX_GP_POINTS = 1500
GP_VQE_ITERS = 10
GP_LEARNING_RATE = 0.5
GP_OPTIMIZER_STEP_MAX = 0.2
GP_FD_EPS = 0.05
GP_HOLDOUT_FRACTION = 0.2
GP_RNG_SEED = int(GLOBAL_SAMPLING_SEED)


def _is_cross_chip_pair(qubit_indices):
    return frozenset(qubit_indices) in _CROSS_CHIP_PAIR_SET


def load_circuit_from_json(path):
    """Same loader as the notebook's first cell (visualization parts removed)."""
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
            raise ValueError(f"Unhandled op in circuit JSON: {op!r}")
        if len(new_op.qubits) == 2 and _is_cross_chip_pair(g["qubits"]):
            new_op = new_op.with_tags(CZ_CROSS_CHIP_TAG)
        c.append(new_op)
    return c, list(q), syms, data


def load_pauli_sum_from_numbered_file(path: Path, qubits: list) -> cirq.PauliSum:
    idx_to_pauli = {1: cirq.X, 2: cirq.Y, 3: cirq.Z}
    out = cirq.PauliSum()
    with Path(path).open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            coeff = float(parts[0])
            pauli_codes = [int(x) for x in parts[1:]]
            if len(pauli_codes) != len(qubits):
                raise ValueError(f"{path}:{lineno} has {len(pauli_codes)} Pauli indices, expected {len(qubits)}.")
            pauli_string = cirq.PauliString()
            for q, code in zip(qubits, pauli_codes):
                if code == 0:
                    continue
                pauli_string *= idx_to_pauli[code](q)
            out += coeff * pauli_string
    return out


# =====================================================================
# 2. Build the inputs (circuit + Hamiltonian + HF prep)
# =====================================================================
circuit_bare, qubits, symbols, circuit_meta = load_circuit_from_json(CIRCUIT_JSON)
ansatz_circuit = circuit_bare
n_params = len(symbols)
n_qubits = len(qubits)
qubit_map = {q: i for i, q in enumerate(qubits)}

ham_path = _REPO / "Pauli_Ham" / f"{MOLECULE}_bond_{bond_length:.1f}.txt"
pauli_sum = load_pauli_sum_from_numbered_file(ham_path, list(qubits))
e_gs = float(np.linalg.eigvalsh(pauli_sum.matrix(qubits=qubits))[0].real)

# HF-determinant prep: X on occupied spin-orbitals.
_half = N_SPATIAL_ORBITALS
_eta = ETA
_occupied = list(range(_eta)) + list(range(_half, _half + _eta))
prep_circuit = cirq.Circuit([cirq.X(qubits[k]) for k in _occupied])
circuit = prep_circuit + ansatz_circuit

# =====================================================================
# 3. Noise / shot / readout configuration
# =====================================================================
base_noise_cfg = {
    "two_qubit_depol_prob": TWO_QUBIT_GATE_DEPOL_PROB,
    "one_qubit_depol_prob": ONE_QUBIT_GATE_DEPOL_PROB,
    "cross_chip_two_qubit_depol_prob": CROSS_CHIP_TWO_QUBIT_GATE_DEPOL_PROB,
    "two_qubit_dephasing_prob": TWO_QUBIT_GATE_DEPHASING_PROB,
    "one_qubit_dephasing_prob": ONE_QUBIT_GATE_DEPHASING_PROB,
    "cross_chip_two_qubit_dephasing_prob": CROSS_CHIP_TWO_QUBIT_GATE_DEPHASING_PROB,
    "two_qubit_over_rotation": TWO_QUBIT_GATE_OVER_ROTATION,
    "one_qubit_over_rotation": ONE_QUBIT_GATE_OVER_ROTATION,
    "cross_chip_two_qubit_over_rotation": CROSS_CHIP_TWO_QUBIT_GATE_OVER_ROTATION,
}
ogm_file = _REPO / "June_main" / "OGM_measurement_basis" / f"OGM_{MOLECULE}_bond_{bond_length:.1f}.txt"
shot_cfg = {
    "num_shots": int(GP_SHOTS),
    "measurement_scheme": str(GLOBAL_MEASUREMENT_SCHEME),
    "apply_readout_noise": bool(GLOBAL_APPLY_READOUT_NOISE),
    "sampling_seed": int(GLOBAL_SAMPLING_SEED),
    "epsilon": 0.1,
    "ogm_file": ogm_file,
    "shadowgrouping_root": "/Users/zacharyhe/shadowgrouping",
}
readout_cal = {
    "p_0_success": np.array(GLOBAL_READOUT_P0_SUCCESS, dtype=float),
    "p_1_success": np.array(GLOBAL_READOUT_P1_SUCCESS, dtype=float),
}

theta_init_gp = np.zeros(n_params, dtype=float)

# =====================================================================
# 4. Config + adapter + mitigator + warm start
# =====================================================================
gp_config = gpm.MitigatorConfig(
    n_params=n_params,
    n_qubits=n_qubits,
    theta_init=theta_init_gp,
    n_warmstart_circuits=int(GP_N_WARMSTART_CIRCUITS),
    n_nonclifford_gates=int(GP_N_NONCLIFFORD_GATES),
    warmstart_spread=float(GP_WARMSTART_SPREAD),
    shots=int(GP_SHOTS),
    rng_seed=int(GP_RNG_SEED),
    max_harmonic=int(GP_MAX_HARMONIC),
    include_noisy_feature_in_gp=bool(GP_INCLUDE_NOISY_FEATURE),
    pauli_onehot=bool(GP_PAULI_ONEHOT),
    pauli_summaries=bool(GP_PAULI_SUMMARIES),
    affine_regularization=float(GP_AFFINE_REGULARIZATION),
    refit_affine_on_topup=bool(GP_REFIT_AFFINE_ON_TOPUP),
    kernel_type=str(GP_KERNEL_TYPE),
    matern_nu=float(GP_MATERN_NU),
    use_ard=bool(GP_USE_ARD),
    use_product_kernel=bool(GP_USE_PRODUCT_KERNEL),
    noise_variance_init=float(GP_NOISE_VARIANCE_INIT),
    normalize_targets=bool(GP_NORMALIZE_TARGETS),
    gp_n_restarts=int(GP_N_RESTARTS),
    max_gp_train_points=int(GP_MAX_TRAIN_POINTS),
    uncertainty_threshold=float(GP_UNCERTAINTY_THRESHOLD),
    topup_batch_size=int(GP_TOPUP_BATCH_SIZE),
    topup_radius=GP_TOPUP_RADIUS,
    drop_faraway_points=bool(GP_DROP_FARAWAY_POINTS),
    max_gp_points=int(GP_MAX_GP_POINTS),
    optimizer_step_max=float(GP_OPTIMIZER_STEP_MAX),
    max_vqe_iterations=int(GP_VQE_ITERS),
    holdout_fraction=float(GP_HOLDOUT_FRACTION),
    use_rem_branch=True,
)

gp_adapter = gpm.CirqBackendAdapter(
    circuit=circuit,
    qubits=qubits,
    symbols=symbols,
    pauli_sum=pauli_sum,
    base_noise_cfg=base_noise_cfg,
    shot_cfg=shot_cfg,
    readout_cal=readout_cal,
    simulator_seed=int(GLOBAL_RANDOM_SEED),
    use_rem_branch=gp_config.use_rem_branch,
)
gp_config.n_observables = len(gp_adapter.obs_labels)

print(f"[GP] {n_qubits} qubits, {n_params} params, {gp_config.n_observables} Pauli observables.")
print(f"[GP] offset (identity coeff) = {gp_adapter.offset:.10f} Eh")
print(f"[GP] e_gs (exact GS) = {e_gs:.10f} Eh")

mitigator = gpm.Mitigator(gp_adapter, hamiltonian=pauli_sum, config=gp_config)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    mitigator.warmstart()
print(f"[GP] warm start done: {len(mitigator.rows)} rows "
      f"({gp_config.n_warmstart_circuits} circuits x {gp_config.n_observables} observables).")


# =====================================================================
# 5. Replicate run_vqe_with_mitigator (analytic mode) for 2 iterations
# =====================================================================
labels = list(gp_adapter.obs_labels)
weights = np.asarray(gp_adapter.weights, dtype=float)
offset = float(gp_adapter.offset)
cfg = gp_config

rng = np.random.default_rng(int(cfg.rng_seed) + 777)   # SAME stream as run_vqe_with_mitigator


def measure(th):
    return gp_adapter.run_noisy(
        gp_adapter.resolver_from_theta(th),
        shots=int(cfg.shots),
        sampling_seed=int(rng.integers(1, 2**31 - 1)),
    )


def gradient_breakdown(theta, measured, do_noisy, focus_param=0, focus_label=None):
    """Mirror Mitigator.energy_gradient, but also return per-piece numbers for display."""
    theta = np.asarray(theta, dtype=float).ravel()
    rows = [{"theta": theta, "pauli": lab, "o_noisy": float(measured[lab])} for lab in labels]
    X = mitigator._feature_matrix(rows)
    dmean = mitigator.gp.predict_input_gradient(X)         # (P, D)
    idx = gpm.feature_index_map(cfg)
    angle_idx = idx["angle"]
    noisy_idx = idx["noisy"]
    n_harm = int(cfg.max_harmonic)
    n_angle = len(angle_idx)
    dangle = np.zeros((n_angle, n_params), dtype=float)
    for i in range(n_params):
        ti = float(theta[i])
        base = i * 2 * n_harm
        for kk in range(1, n_harm + 1):
            c = base + (kk - 1) * 2
            dangle[c, i] = -kk * np.sin(kk * ti)
            dangle[c + 1, i] = kk * np.cos(kk * ti)
    d_omit_dtheta = dmean[:, angle_idx] @ dangle
    do = np.stack([np.asarray(do_noisy[lab], dtype=float).ravel() for lab in labels], axis=0)
    if len(noisy_idx) > 0:
        d_omit_dtheta = d_omit_dtheta + dmean[:, noisy_idx[0]][:, None] * do
    grad = weights @ d_omit_dtheta

    # focus chain for one (label, param) to print explicitly
    if focus_label is None:
        focus_label = max(labels, key=lambda L: abs(weights[labels.index(L)]))
    li = labels.index(focus_label)
    # angle dims of focus_param
    base = focus_param * 2 * n_harm
    fa_dims = [angle_idx[base + d] for d in range(2 * n_harm)]
    chain = {
        "focus_label": focus_label,
        "focus_param": int(focus_param),
        "weight": float(weights[li]),
        "dangle_block": dangle[base:base + 2 * n_harm, focus_param].tolist(),
        "dmean_angle_block": dmean[li, fa_dims].tolist(),
        "angle_contrib": float(dmean[li, angle_idx] @ dangle[:, focus_param]),
        "dmean_dnoisy": float(dmean[li, noisy_idx[0]]) if noisy_idx else 0.0,
        "do_noisy": float(do[li, focus_param]),
        "noisy_contrib": float((dmean[li, noisy_idx[0]] if noisy_idx else 0.0) * do[li, focus_param]),
    }
    chain["d_omit_dtheta"] = chain["angle_contrib"] + chain["noisy_contrib"]
    chain["term_w_times"] = chain["weight"] * chain["d_omit_dtheta"]
    return np.asarray(grad).ravel(), chain


def top_terms(o_mit, o_ideal, o_noisy, k=6):
    order = np.argsort(-np.abs(weights))
    out = []
    for j in order[:k]:
        lab = labels[j]
        out.append({
            "label": lab,
            "w": float(weights[j]),
            "o_mit": float(o_mit[lab]),
            "o_ideal": float(o_ideal[lab]),
            "o_noisy": float(o_noisy[lab]),
        })
    return out


theta = np.asarray(theta_init_gp, dtype=float).ravel()

# focus (label, query) used for the worked kernel reconstruction in the doc
FOCUS_LABEL = "IIIIIZII"
x_star_focus = None
o_mit_focus = None
o_noisy_focus = None
theta_iter2_focus = None

dump = {
    "settings": {
        "n_warmstart_circuits": GP_N_WARMSTART_CIRCUITS,
        "warmstart_spread": GP_WARMSTART_SPREAD,
        "shots": GP_SHOTS,
        "max_harmonic": GP_MAX_HARMONIC,
        "learning_rate": GP_LEARNING_RATE,
        "rng_seed": GP_RNG_SEED,
        "n_warmstart_rows": len(mitigator.rows),
    },
    "offset": offset,
    "e_gs": e_gs,
    "n_observables": len(labels),
    "iterations": [],
}

for it in range(2):
    bundle = measure(theta)
    measured = bundle["primary"]
    o_mit, std = mitigator.predict_with_uncertainty(theta, measured)
    topped = 0
    if gpm.needs_topup(std, mitigator.coeff_by_pauli, cfg):
        new_rows = gpm.sample_local_rows(gp_adapter, theta, cfg, seed=int(cfg.rng_seed) + 10_000 + it)
        mitigator.update_with_rows(new_rows, current_theta=theta)
        o_mit, std = mitigator.predict_with_uncertainty(theta, measured)
        topped = 1
    e_mit = gp_adapter.energy_from_values(o_mit)
    e_ideal = gp_adapter.ideal_energy(theta)
    o_ideal = gp_adapter.simulate_ideal(gp_adapter.resolver_from_theta(theta))

    if it == 1:
        _fl = FOCUS_LABEL if FOCUS_LABEL in measured else labels[0]
        _rows_focus = [{"theta": theta, "pauli": _fl, "o_noisy": float(measured[_fl])}]
        x_star_focus = mitigator._feature_matrix(_rows_focus)[0]
        o_mit_focus = float(o_mit[_fl])
        o_noisy_focus = float(measured[_fl])
        theta_iter2_focus = theta.copy()
        FOCUS_LABEL = _fl

    # analytic gradient: device parameter-shift on the REAL noisy measurement
    half = float(np.pi) / 2.0
    do_noisy = {lab: np.zeros(len(theta)) for lab in measured}
    for j in range(len(theta)):
        tp = theta.copy(); tp[j] += half
        tm = theta.copy(); tm[j] -= half
        bp = measure(tp)["primary"]
        bm = measure(tm)["primary"]
        for lab in measured:
            do_noisy[lab][j] = 0.5 * (bp[lab] - bm[lab])
    grad, chain = gradient_breakdown(theta, measured, do_noisy, focus_param=0)
    step = -float(GP_LEARNING_RATE) * grad
    theta_next = theta + step

    rec = {
        "iter": it,
        "theta": theta.tolist(),
        "topped_up": int(topped),
        "E_mit": float(e_mit),
        "E_ideal": float(e_ideal),
        "abs_err_vs_gs": float(abs(e_mit - e_gs)),
        "top_terms": top_terms(o_mit, o_ideal, measured),
        "grad": grad.tolist(),
        "step": step.tolist(),
        "theta_next": theta_next.tolist(),
        "grad_chain_param0": chain,
        # partial sum E using offset + the top terms (for the worked expansion)
        "E_from_top_terms_partial": float(offset + sum(t["w"] * t["o_mit"] for t in top_terms(o_mit, o_ideal, measured))),
    }
    dump["iterations"].append(rec)

    print("\n" + "=" * 70)
    print(f"ITERATION {it + 1}   theta = {np.round(theta, 6).tolist()}   topped_up={topped}")
    print("=" * 70)
    print(f"  offset (b_off)         = {offset:.6f} Eh")
    print(f"  E_mit  (blue)          = {e_mit:.6f} Eh")
    print(f"  E_ideal(green, exact)  = {e_ideal:.6f} Eh")
    print(f"  |E_mit - e_gs|         = {abs(e_mit - e_gs)*1e3:.3f} mEh   (e_gs={e_gs:.6f})")
    print("  Top terms by |w_i|:  w_i        O_mit       O_ideal     w_i*O_mit")
    for t in rec["top_terms"]:
        print(f"    {t['label']}  {t['w']:+.5f}  {t['o_mit']:+.5f}  {t['o_ideal']:+.5f}  {t['w']*t['o_mit']:+.5f}")
    print(f"  grad = {np.round(grad, 6).tolist()}")
    print(f"  step (-lr*grad) = {np.round(step, 6).tolist()}")
    print(f"  theta_next = {np.round(theta_next, 6).tolist()}")
    print("  gradient chain for theta_0, focus Pauli =", chain["focus_label"])
    print(f"    angle Jacobian block (d[cos,sin,..]/dtheta0) = {np.round(chain['dangle_block'],4).tolist()}")
    print(f"    dmean/dangle block                           = {np.round(chain['dmean_angle_block'],6).tolist()}")
    print(f"    angle_contrib = {chain['angle_contrib']:+.6e}")
    print(f"    dmean/dO_noisy = {chain['dmean_dnoisy']:+.6f}   dO_noisy/dtheta0 = {chain['do_noisy']:+.6f}")
    print(f"    noisy_contrib = {chain['noisy_contrib']:+.6e}")
    print(f"    dO_mit/dtheta0 = {chain['d_omit_dtheta']:+.6e}   -> w*dO_mit = {chain['term_w_times']:+.6e}")

    theta = theta_next

# =====================================================================
# 6. Kernel inspection -- fitted hyperparameters + a fully worked GP-mean
#    reconstruction for the focus query (so the doc can show O_mit by hand).
# =====================================================================
from sklearn.gaussian_process.kernels import (
    Sum, Product, ConstantKernel, WhiteKernel, DotProduct, Matern, RBF,
)

gp_sk = mitigator.gp.gp                         # fitted sklearn GaussianProcessRegressor
kernel = gp_sk.kernel_                          # kernel with FITTED hyperparameters
Xtr = np.asarray(gp_sk.X_train_, dtype=float)   # (N, 48) training features actually used
alpha = np.asarray(gp_sk.alpha_, dtype=float).ravel()   # (N,) dual weights
ybar = float(np.ravel(gp_sk._y_train_mean)[0])
ysig = float(np.ravel(getattr(gp_sk, "_y_train_std", np.array([1.0])))[0])
Ntr = int(Xtr.shape[0])
y_train_norm = np.asarray(getattr(gp_sk, "y_train_", np.zeros(Ntr)), dtype=float).ravel()


def describe(k):
    if isinstance(k, Sum):
        return {"op": "sum", "terms": [describe(k.k1), describe(k.k2)]}
    if isinstance(k, Product):
        return {"op": "prod", "terms": [describe(k.k1), describe(k.k2)]}
    if isinstance(k, gpm._SubsetKernel):
        d = describe(k.kernel)
        d["on_indices"] = [int(i) for i in k.indices]
        return d
    if isinstance(k, ConstantKernel):
        return {"type": "Constant", "constant_value": float(k.constant_value)}
    if isinstance(k, WhiteKernel):
        return {"type": "White", "noise_level": float(k.noise_level)}
    if isinstance(k, DotProduct):
        return {"type": "DotProduct", "sigma_0": float(k.sigma_0)}
    if isinstance(k, Matern):
        return {"type": "Matern", "nu": float(k.nu),
                "length_scale": np.atleast_1d(k.length_scale).astype(float).tolist()}
    if isinstance(k, RBF):
        return {"type": "RBF",
                "length_scale": np.atleast_1d(k.length_scale).astype(float).tolist()}
    return {"type": type(k).__name__}


kdesc = describe(kernel)

# worked reconstruction of O_mit at the focus query
kvec = kernel(x_star_focus[None, :], Xtr).ravel()        # (N,) = k(x*, x_n)
contrib = ysig * alpha * kvec                            # per-train contribution to mean
recon = ybar + float(np.dot(ysig * alpha, kvec))
order = np.argsort(-np.abs(contrib))[:8]
top_contribs = []
for n in order:
    top_contribs.append({
        "n": int(n),
        "k_xstar_xn": float(kvec[n]),
        "alpha_n": float(alpha[n]),
        "contrib_ysig_alpha_k": float(contrib[n]),
        "y_target_O_ideal": float(ybar + ysig * y_train_norm[n]),
        "xn_o_noisy_feature": float(Xtr[n, gpm.feature_index_map(cfg)["noisy"][0]]),
    })

# component breakdown of k(x*, x_n) for the single top training point
n0 = int(order[0]); xn0 = Xtr[n0]


def _safe(fn):
    try:
        return float(fn())
    except Exception:
        return None


k_components = {
    "n": n0,
    "k_total": _safe(lambda: kernel(x_star_focus[None, :], xn0[None, :])[0, 0]),
    "k_lin_times_obs": _safe(lambda: kernel.k1.k1(x_star_focus[None, :], xn0[None, :])[0, 0]),
    "k_base": _safe(lambda: kernel.k1.k2(x_star_focus[None, :], xn0[None, :])[0, 0]),
    "k_white": _safe(lambda: kernel.k2(x_star_focus[None, :], xn0[None, :])[0, 0]),
}

idxmap = gpm.feature_index_map(cfg)
dump["kernel"] = {
    "N_train_used": Ntr,
    "y_train_mean": ybar,
    "y_train_std": ysig,
    "kernel_str": str(kernel),
    "kernel_struct": kdesc,
    "feature_index_map": {k: list(map(int, v)) for k, v in idxmap.items()},
    "focus_reconstruction": {
        "label": FOCUS_LABEL,
        "theta": theta_iter2_focus.tolist(),
        "o_noisy": float(o_noisy_focus),
        "o_mit_predicted_by_gp": float(o_mit_focus),
        "o_mit_reconstructed": recon,
        "x_star": x_star_focus.tolist(),
        "top_contributors": top_contribs,
        "top1_component_breakdown": k_components,
        "top1_x_n": xn0.tolist(),
    },
}

# Save full arrays so the doc's claims can be checked WITHOUT re-running the fit.
np.savez(
    _THIS_DIR / "fitted_gp_arrays.npz",
    X_train=Xtr, alpha=alpha, y_train_norm=y_train_norm,
    y_train_mean=np.array([ybar]), y_train_std=np.array([ysig]),
    x_star=x_star_focus, kvec=kvec,
    angle_idx=np.array(idxmap["angle"]), noisy_idx=np.array(idxmap["noisy"]),
    pauli_idx=np.array(idxmap["pauli"]),
)

print("\n" + "=" * 70)
print("KERNEL INSPECTION")
print("=" * 70)
print(f"  N_train used = {Ntr}   y_mean = {ybar:.6f}   y_std = {ysig:.6f}")
print(f"  fitted kernel:\n    {str(kernel)}")
print(f"  focus query  label={FOCUS_LABEL}  theta={np.round(theta_iter2_focus,6).tolist()}  o_noisy={o_noisy_focus:.6f}")
print(f"  O_mit (GP.predict)      = {o_mit_focus:.6f}")
print(f"  O_mit (reconstructed)   = {recon:.6f}   (ybar + ysig * sum_n alpha_n k(x*,x_n))")
print("  top training-point contributions  ysig*alpha_n*k(x*,x_n):")
for c in top_contribs:
    print(f"    n={c['n']:4d}  k={c['k_xstar_xn']:+.5f}  alpha={c['alpha_n']:+.5f}  "
          f"contrib={c['contrib_ysig_alpha_k']:+.6f}  y(O_ideal)={c['y_target_O_ideal']:+.4f}")
print(f"  component breakdown of k(x*, x_n0) for n0={n0}:")
print(f"    k_lin*obs = {k_components['k_lin_times_obs']}   k_base = {k_components['k_base']}   "
      f"white = {k_components['k_white']}   total = {k_components['k_total']}")
print(f"\n[written] {_THIS_DIR / 'fitted_gp_arrays.npz'}")

print("\n\n===== MACHINE-READABLE DUMP (JSON) =====")
print(json.dumps(dump, indent=2))

# also write it next to the doc for reproducibility
out_path = _THIS_DIR / "example_numbers.json"
out_path.write_text(json.dumps(dump, indent=2), encoding="utf-8")
print(f"\n[written] {out_path}")
