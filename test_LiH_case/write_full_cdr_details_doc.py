"""Generate full per-Pauli CDR details markdown (LiH fig13 notebook settings)."""

from __future__ import annotations

import sys
from pathlib import Path

import cirq
import numpy as np
import sympy

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
if str(_REPO / "test_LiH_case") not in sys.path:
    sys.path.insert(0, str(_REPO / "test_LiH_case"))

from main_cursor_lib_test_LiH import (  # noqa: E402
    ONE_QUBIT_GATE_DEPOL_PROB,
    TWO_QUBIT_GATE_DEPOL_PROB,
    generate_near_clifford_param_sets,
)
from shot_measurement_test_LiH import (  # noqa: E402
    _generate_near_clifford_resolvers_fallback,
    _is_setting_compatible,
    _load_shadowgrouping_scheme,
    _simulate_noiseless_state_for_resolver,
    _simulate_noisy_rho_for_resolver,
    _unique_settings_with_counts,
    ensure_shadowgrouping_importable,
    estimate_energy_from_noisy_rho_shots,
    exact_pauli_expectation_from_int_row,
    pauli_sum_to_int_observables,
    run_mitigation,
)


def _load_pauli_sum(path: Path, qubits: list[cirq.Qid]) -> cirq.PauliSum:
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
                raise ValueError(f"{path}:{lineno}: expected {len(qubits)} Pauli codes.")
            pstr = cirq.PauliString()
            for q, code in zip(qubits, pauli_codes):
                if code == 0:
                    continue
                pstr *= idx_to_pauli[code](q)
            out += coeff * pstr
    return out


def _lih_fig13_circuit(
    theta1: sympy.Symbol, theta2: sympy.Symbol, theta3: sympy.Symbol
) -> tuple[cirq.Circuit, list[cirq.LineQubit]]:
    q = cirq.LineQubit.range(6)
    q0, q1, q2, q3, q4, q5 = q
    c = cirq.Circuit()
    c.append(cirq.ry(-0.1).on(q0))
    c.append(cirq.H(q1))
    c.append(cirq.CZ(q0, q1))
    c.append(cirq.H(q1))
    c.append(cirq.H(q4))
    c.append(cirq.CZ(q1, q4))
    c.append(cirq.H(q4))
    c.append(cirq.H(q3))
    c.append(cirq.CZ(q4, q3))
    c.append(cirq.H(q3))
    c.append(cirq.X(q0))
    c.append(cirq.X(q3))
    c.append(cirq.identity_each(*q))
    c.append(cirq.rx(np.pi / 2).on(q0))
    c.append(cirq.H(q2))
    c.append(cirq.H(q3))
    c.append(cirq.H(q4))
    c.append(cirq.H(q5))
    c.append([cirq.CZ(q0, q1), cirq.CZ(q3, q4)])
    c.append([cirq.CZ(q4, q5), cirq.H(q4)])
    c.append(cirq.CZ(q1, q4))
    c.append(cirq.rx(theta1).on(q1))
    c.append(cirq.CZ(q1, q4))
    c.append(cirq.CZ(q0, q1))
    c.append(cirq.H(q1))
    c.append(cirq.CZ(q0, q1))
    c.append(cirq.CZ(q1, q2))
    c.append(cirq.CZ(q1, q4))
    c.append(cirq.rx(theta2).on(q1))
    c.append(cirq.CZ(q1, q4))
    c.append(cirq.H(q4))
    c.append(cirq.CZ(q4, q5))
    c.append(cirq.H(q5))
    c.append(cirq.CZ(q3, q4))
    c.append(cirq.H(q4))
    c.append(cirq.CZ(q3, q4))
    c.append(cirq.H(q4))
    c.append(cirq.CZ(q1, q4))
    c.append(cirq.rx(theta3).on(q1))
    c.append(cirq.CZ(q1, q4))
    c.append(cirq.H(q4))
    c.append(cirq.CZ(q1, q2))
    c.append(cirq.H(q2))
    c.append(cirq.CZ(q0, q1))
    c.append(cirq.CZ(q3, q4))
    c.append(cirq.rx(-np.pi / 2).on(q0))
    c.append([cirq.H(q1), cirq.H(q3)])
    return c, list(q)


def _int_row_to_cirq_term(obs_row: np.ndarray, weight: float, qubits: list[cirq.Qid]) -> str:
    int_to_char = {1: "X", 2: "Y", 3: "Z"}
    factors: list[str] = []
    for i, v in enumerate(obs_row):
        vi = int(v)
        if vi == 0:
            continue
        factors.append(f"{int_to_char[vi]}(q({qubits[i].x}))")
    if not factors:
        return f"({weight})*I"
    return f"({weight})*{'*'.join(factors)}"


def _fmt(x: float) -> str:
    return f"{float(x):.16e}"


def _ogm_shots_per_term(
    observables_int: np.ndarray,
    weights: np.ndarray,
    *,
    num_shots: int,
    epsilon: float,
    ogm_file: Path,
    shadowgrouping_root: Path,
) -> list[int]:
    ensure_shadowgrouping_importable(shadowgrouping_root)
    method = _load_shadowgrouping_scheme("ogm", observables_int, weights, epsilon, ogm_file)
    settings, _ = method.find_setting(N_samples=num_shots)
    sampled = np.asarray(settings, dtype=int)
    if sampled.ndim == 1:
        sampled = sampled.reshape(1, -1)
    unique_settings, unique_counts = _unique_settings_with_counts(sampled)
    out: list[int] = []
    for obs_row in observables_int:
        total = sum(
            cnt
            for srow, cnt in zip(unique_settings, unique_counts)
            if _is_setting_compatible(obs_row, srow)
        )
        out.append(int(total if total > 0 else 1))
    return out


def _collect_training_arrays(
    ansatz_circuit: cirq.Circuit,
    observable_h: cirq.PauliSum,
    qubits: list[cirq.Qid],
    resolvers: list[dict],
    *,
    noise_params: dict,
    simulator_seed: int,
    num_shots: int,
    measurement_scheme: str,
    p_0_success: np.ndarray,
    p_1_success: np.ndarray,
    apply_readout_noise: bool,
    sampling_seed: int,
    epsilon: float,
    ogm_file: Path,
    shadowgrouping_root: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int]]:
    observables_int, weights, _offset = pauli_sum_to_int_observables(observable_h, qubits)
    n_terms = len(weights)
    n = len(resolvers)
    tex_exact = np.zeros((n, n_terms), dtype=float)
    tunmit = np.zeros((n, n_terms), dtype=float)
    trem = np.zeros((n, n_terms), dtype=float)
    t_rem_list: list[int] = []

    from main_cursor_lib_test_LiH import count_non_clifford_ops

    for i, resolver in enumerate(resolvers):
        state = _simulate_noiseless_state_for_resolver(
            ansatz_circuit, resolver, qubits, simulator_seed=simulator_seed
        )
        for k in range(n_terms):
            tex_exact[i, k] = exact_pauli_expectation_from_int_row(state, observables_int[k], qubits)
        rho = _simulate_noisy_rho_for_resolver(
            ansatz_circuit, resolver, qubits, noise_params, simulator_seed=simulator_seed
        )
        est = estimate_energy_from_noisy_rho_shots(
            rho,
            observable_h,
            qubits,
            num_shots=num_shots,
            measurement_scheme=measurement_scheme,
            p_0_success=p_0_success,
            p_1_success=p_1_success,
            apply_rem=True,
            apply_readout_noise=apply_readout_noise,
            sampling_seed=sampling_seed,
            epsilon=epsilon,
            ogm_file=ogm_file,
            shadowgrouping_root=shadowgrouping_root,
            return_per_term=True,
        )
        tunmit[i, :] = est["per_term_unmitigated"]
        trem[i, :] = est["per_term_rem"]
        t_rem_list.append(int(count_non_clifford_ops(ansatz_circuit, resolver)))
    return tex_exact, tunmit, trem, t_rem_list


def write_doc(out_path: Path) -> None:
    np.random.seed(1234)

    bond_length = 1.5
    params = np.array([-0.128705, 0.233859, 0.114671], dtype=float)
    num_shots = 8192
    sampling_seed = 1234
    simulator_seed = 1234
    epsilon = 0.1
    cdr_num_circuits = 10
    cdr_t_max = 2
    cdr_seed = 42
    max_terms_to_record = 20
    shadowgrouping_root = Path("/Users/zacharyhe/shadowgrouping")
    ogm_file = (
        shadowgrouping_root
        / "haozhaowu/LiH/hamil_class/ogm_outputs"
        / f"OGM_ogm_LiH_{bond_length:.1f}.txt"
    )
    p0 = np.array([0.9756, 0.9748, 0.9738, 0.9656, 0.9585, 0.9514], dtype=float)
    p1 = p0.copy()
    base_noise = {
        "two_qubit_depol_prob": TWO_QUBIT_GATE_DEPOL_PROB,
        "one_qubit_depol_prob": ONE_QUBIT_GATE_DEPOL_PROB,
    }

    theta1, theta2, theta3 = sympy.symbols("theta1 theta2 theta3")
    circuit, qubits = _lih_fig13_circuit(theta1, theta2, theta3)
    ham_path = _REPO / "Pauli_Ham" / f"LiH_bond_{bond_length:.1f}.txt"
    pauli_sum = _load_pauli_sum(ham_path, qubits)
    target_resolver = {
        theta1: float(params[0]),
        theta2: float(params[1]),
        theta3: float(params[2]),
    }
    symbols = [theta1, theta2, theta3]

    mit = run_mitigation(
        "cdr",
        ansatz_circuit=circuit,
        observable_h=pauli_sum,
        qubits=qubits,
        target_resolver=target_resolver,
        target_params=target_resolver,
        symbols=symbols,
        base_noise_cfg=base_noise,
        shot_cfg={
            "num_shots": num_shots,
            "measurement_scheme": "ogm",
            "apply_readout_noise": True,
            "sampling_seed": sampling_seed,
            "ogm_file": ogm_file,
            "shadowgrouping_root": shadowgrouping_root,
        },
        readout_cal={"p_0_success": p0, "p_1_success": p1},
        cdr_cfg={
            "num_circuits": cdr_num_circuits,
            "t_max": cdr_t_max,
            "seed": cdr_seed,
            "cdr_fit_scope": "per_pauli",
        },
        simulator_seed=simulator_seed,
    )

    models = mit["cdr_models"]
    observables_int, weights, offset = pauli_sum_to_int_observables(pauli_sum, qubits)
    coeffs_u = np.asarray(models["coeffs_unmit_to_exact_per_term"], dtype=float)
    coeffs_r = np.asarray(models["coeffs_rem_to_exact_per_term"], dtype=float)
    r2_r = np.asarray(models["r2_rem_to_exact_per_term"], dtype=float)

    try:
        resolvers = generate_near_clifford_param_sets(
            target_resolver,
            symbols,
            num_circuits=cdr_num_circuits,
            t_max=cdr_t_max,
            circuit=circuit,
            min_snap_fraction=0.0,
            seed=cdr_seed,
        )
    except ValueError as err:
        if "Unrecognized symbol naming convention" not in str(err):
            raise
        resolvers = _generate_near_clifford_resolvers_fallback(
            target_resolver,
            symbols,
            num_circuits=cdr_num_circuits,
            t_max=cdr_t_max,
            circuit=circuit,
            min_snap_fraction=0.0,
            seed=cdr_seed,
        )
    tex_exact, tunmit, trem, t_rem_list = _collect_training_arrays(
        circuit,
        pauli_sum,
        qubits,
        resolvers,
        noise_params=base_noise,
        simulator_seed=simulator_seed,
        num_shots=num_shots,
        measurement_scheme="ogm",
        p_0_success=p0,
        p_1_success=p1,
        apply_readout_noise=True,
        sampling_seed=sampling_seed,
        epsilon=epsilon,
        ogm_file=ogm_file,
        shadowgrouping_root=shadowgrouping_root,
    )

    rho_target = _simulate_noisy_rho_for_resolver(
        circuit, target_resolver, qubits, base_noise, simulator_seed=simulator_seed
    )
    target_est = estimate_energy_from_noisy_rho_shots(
        rho_target,
        pauli_sum,
        qubits,
        num_shots=num_shots,
        measurement_scheme="ogm",
        p_0_success=p0,
        p_1_success=p1,
        apply_rem=True,
        apply_readout_noise=True,
        sampling_seed=sampling_seed,
        epsilon=epsilon,
        ogm_file=ogm_file,
        shadowgrouping_root=shadowgrouping_root,
        return_per_term=True,
    )
    x_u_target = np.asarray(target_est["per_term_unmitigated"], dtype=float)
    x_r_target = np.asarray(target_est["per_term_rem"], dtype=float)
    if coeffs_r.shape[0] != x_r_target.shape[0] or r2_r.shape[0] != coeffs_r.shape[0]:
        raise RuntimeError("CDR arrays are inconsistent in size; cannot build diagnostics section.")

    ogm_shots = _ogm_shots_per_term(
        observables_int,
        weights,
        num_shots=num_shots,
        epsilon=epsilon,
        ogm_file=ogm_file,
        shadowgrouping_root=shadowgrouping_root,
    )

    state_exact = _simulate_noiseless_state_for_resolver(
        circuit, target_resolver, qubits, simulator_seed=simulator_seed
    )
    eref = float(
        np.vdot(
            state_exact,
            pauli_sum.matrix(qubits=qubits) @ state_exact,
        ).real
    )

    contrib_u: list[float] = []
    contrib_r: list[float] = []
    for k in range(len(weights)):
        au, bu = coeffs_u[k]
        ar, br = coeffs_r[k]
        contrib_u.append(float(weights[k]) * float(np.polyval([au, bu], x_u_target[k])))
        contrib_r.append(float(weights[k]) * float(np.polyval([ar, br], x_r_target[k])))

    a_vals = np.asarray(coeffs_r[:, 0], dtype=float)
    b_vals = np.asarray(coeffs_r[:, 1], dtype=float)
    zero_pair_mask = np.isclose(a_vals, 0.0, atol=5e-3) & np.isclose(b_vals, 0.0, atol=5e-3)
    panel_a_mask = ~zero_pair_mask
    if not np.any(panel_a_mask):
        raise RuntimeError("Panel a mask removed every term; diagnostics would be meaningless.")
    a_vals_panel = a_vals[panel_a_mask]
    slope_lt1_all = int(np.sum(a_vals < 1.0))
    slope_lt1_panel = int(np.sum(a_vals_panel < 1.0))
    x_std = np.std(trem, axis=0)
    y_std = np.std(tex_exact, axis=0)
    intercept_dominated = np.abs(b_vals) > (np.abs(a_vals) * (x_std + 1e-12))
    low_r2_mask = r2_r < 0.9
    weak_signal_mask = y_std < 1e-3
    weak_x_mask = x_std < 1e-2
    bad_case_mask = low_r2_mask & (zero_pair_mask | weak_signal_mask | weak_x_mask | intercept_dominated)
    bad_case_ids = np.where(bad_case_mask)[0]
    worst_ids = np.argsort(r2_r)[: min(8, len(r2_r))]
    abs_weight = np.abs(weights)
    r2_manual = np.zeros(len(weights), dtype=float)
    ss_res_arr = np.zeros(len(weights), dtype=float)
    ss_tot_arr = np.zeros(len(weights), dtype=float)
    for k in range(len(weights)):
        y_true = tex_exact[:, k]
        y_pred = a_vals[k] * trem[:, k] + b_vals[k]
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        ss_res_arr[k] = ss_res
        ss_tot_arr[k] = ss_tot
        if ss_tot <= 1e-15:
            r2_manual[k] = np.nan
        else:
            r2_manual[k] = 1.0 - ss_res / ss_tot
    terms_to_record = min(max_terms_to_record, len(weights))

    lines: list[str] = []
    w = lines.append
    w("# full CDR details for LiH testing case (2nd version)")
    w("")
    w("Source: `test_LiH_case/lih_fig13_compiled_ansatz.ipynb` per-Pauli CDR cell.")
    w("")
    w("## Configuration used")
    w("")
    w(f"- bond_length: `{bond_length}`")
    w(f"- target params [theta1, theta2, theta3]: `{params.tolist()}`")
    w("- measurement_scheme: `ogm`")
    w(f"- num_shots: `{num_shots}`")
    w(f"- sampling_seed: `{sampling_seed}`")
    w("- apply_readout_noise: `True`")
    w("- apply_rem in CDR training/target estimation: `True`")
    w(f"- readout p_0_success: `{p0.tolist()}`")
    w(f"- readout p_1_success: `{p1.tolist()}`")
    w(f"- base_noise_cfg: `{base_noise}`")
    w(
        '- cdr_cfg: '
        f'`{{"num_circuits": {cdr_num_circuits}, "t_max": {cdr_t_max}, '
        f'"seed": {cdr_seed}, "cdr_fit_scope": "per_pauli"}}`'
    )
    w(f"- per-term records included in this document: first `{terms_to_record}` terms")
    w(f"- training resolvers count: `{len(resolvers)}`")
    w(f"- training non-Clifford counts (t_remaining): `{t_rem_list}`")
    w(f"- OGM file: `{ogm_file}`")
    w("")
    w("## Important note on epsilon")
    w("")
    w(
        "- For OGM `SettingSampler`, shot allocation is sampled from file probabilities "
        "`p` and total `N_samples` only; epsilon is not used in the sampler math."
    )
    w("")
    w("## Model definition used")
    w("")
    w("- Per-pauli CDR applies, for each term `k`:")
    w("  - unmit branch: `y_k ~= a_u[k] * x_u[k] + b_u[k]`")
    w("  - rem branch:   `y_k ~= a_r[k] * x_r[k] + b_r[k]`")
    w("- Total corrected energies:")
    w("  - `E_cdr_unmit = offset + sum_k w_k * (a_u[k] * x_u_target[k] + b_u[k])`")
    w("  - `E_cdr_rem   = offset + sum_k w_k * (a_r[k] * x_r_target[k] + b_r[k])`")
    w("")
    w("## Fit quality diagnostics (REM per-term fits)")
    w("")
    w(f"- LiH terms total: `{len(weights)}`")
    w(
        f"- Panel a terms (excl. a≈b≈0): `{int(panel_a_mask.sum())}` "
        f"(dropped `{int(zero_pair_mask.sum())}`)"
    )
    w(f"- Panel a slope (a): mean=`{np.mean(a_vals_panel):.6f}`, std=`{np.std(a_vals_panel):.6f}`")
    w(f"- All terms slope (a): mean=`{np.mean(a_vals):.6f}`, std=`{np.std(a_vals):.6f}`")
    w(f"- Intercept (b): mean=`{np.mean(b_vals):.6f}`, std=`{np.std(b_vals):.6f}`")
    w(f"- R^2: mean=`{np.mean(r2_r):.6f}`, std=`{np.std(r2_r):.6f}`")
    w(f"- Terms with slope < 1: panel-a `{slope_lt1_panel}/{int(panel_a_mask.sum())}`, all `{slope_lt1_all}/{len(weights)}`")
    w(f"- Terms with R^2 < 0.9: `{int(np.sum(low_r2_mask))}/{len(weights)}`")
    w(
        "- Terms flagged as likely bad fits "
        "(R^2<0.9 and at least one of near-zero (a,b), weak signal, weak x spread, intercept-dominated): "
        f"`{len(bad_case_ids)}/{len(weights)}`"
    )
    w("")
    w("### Representative bad-fitting terms (lowest R^2)")
    w("")
    for rank, k in enumerate(worst_ids, start=1):
        reasons: list[str] = []
        if zero_pair_mask[k]:
            reasons.append("near-zero-(a,b)")
        if weak_signal_mask[k]:
            reasons.append("weak-y-signal")
        if weak_x_mask[k]:
            reasons.append("weak-x-variation")
        if intercept_dominated[k]:
            reasons.append("intercept-dominated")
        reason_text = ", ".join(reasons) if reasons else "mostly stochastic residual pattern"
        w(
            f"- rank {rank}, term {int(k)}: R^2=`{r2_r[k]:.6f}`, "
            f"a=`{a_vals[k]:.6f}`, b=`{b_vals[k]:.6f}`, "
            f"|w_k|=`{abs_weight[k]:.6e}`, x_std=`{x_std[k]:.6f}`, y_std=`{y_std[k]:.6e}`, "
            f"reasons: `{reason_text}`"
        )
    w("")
    w("### Why slope can be far below 1")
    w("")
    w(
        "- In this per-term CDR setup, each Pauli term is fit independently, so slope is not a global "
        "noise attenuation factor and can vary widely across terms."
    )
    w(
        "- Many terms have near-zero exact expectation over training circuits; for those weak-signal terms, "
        "linear regression can return tiny or unstable slopes (often much smaller than 1)."
    )
    w(
        "- Including all terms therefore pulls down the all-term slope mean; panel-a removes near-zero "
        "coefficient pairs and better reflects active terms."
    )
    w("")
    w("### What cases give bad fitting (R^2 not close to 1)")
    w("")
    w(
        "- Weak target signal (`std(y_exact)` tiny): shot noise and finite-sampling fluctuations dominate "
        "the regression residuals."
    )
    w(
        "- Weak feature spread (`std(x_rem)` tiny): the fitted slope is poorly constrained by training data."
    )
    w(
        "- Intercept-dominated fits (`|b| > |a|*std(x_rem)`): correction behaves like a constant offset "
        "rather than a strong linear relation."
    )
    w(
        "- Near-zero `(a,b)` pairs: these terms carry little calibratable linear signal and commonly "
        "show low/unstable R^2."
    )
    w("")
    w("## Constant term (offset)")
    w("")
    w(f"- Hamiltonian identity term: `({_fmt(offset)}) * I`")
    w(f"- offset: `{_fmt(offset)}`")
    w(f"- E_cdr_unmit: `{_fmt(mit['cdr_unmit_corrected'])}`")
    w(f"- E_cdr_rem: `{_fmt(mit['cdr_rem_corrected'])}`")
    w(f"- delta (unmit - rem): `{_fmt(mit['cdr_unmit_corrected'] - mit['cdr_rem_corrected'])}`")
    w("")
    w("## Int encoding map")
    w("")
    w("- `I=0, X=1, Y=2, Z=3`")
    w("")
    w("## Per-term full details (first 20 terms only)")
    w("")

    for k in range(terms_to_record):
        au, bu = coeffs_u[k]
        ar, br = coeffs_r[k]
        w(f"### term {k}")
        w("")
        w(f"- pauli term from int row: `{_int_row_to_cirq_term(observables_int[k], weights[k], qubits)}`")
        w(f"- int observable row: `{observables_int[k].tolist()}`")
        w(f"- Hamiltonian weight w_{k}: `{_fmt(weights[k])}`")
        w(f"- OGM effective shots used for this term: `{ogm_shots[k]}`")
        w(f"- fitted unmit coeffs: `a_u={_fmt(au)}`, `b_u={_fmt(bu)}`")
        w(f"- fitted rem coeffs: `a_r={_fmt(ar)}`, `b_r={_fmt(br)}`")
        w("- training data pairs (`x_unmit`, `x_rem`, `y_exact`) by training circuit:")
        for i in range(len(resolvers)):
            w(
                f"  - train[{i}] t_remaining={t_rem_list[i]}: "
                f"`x_unmit={_fmt(tunmit[i, k])}`, `x_rem={_fmt(trem[i, k])}`, "
                f"`y_exact={_fmt(tex_exact[i, k])}`"
            )
        w("- explicit R^2 calculation for this term (REM branch):")
        w(
            "  - formula: "
            "`R^2 = 1 - SS_res/SS_tot`, "
            "`SS_res = sum_i(y_exact_i - (a_r*x_rem_i + b_r))^2`, "
            "`SS_tot = sum_i(y_exact_i - mean(y_exact))^2`"
        )
        w(
            f"  - values: `SS_res={_fmt(ss_res_arr[k])}`, `SS_tot={_fmt(ss_tot_arr[k])}`, "
            f"`R^2_manual={r2_manual[k]:.6f}`, `R^2_model={r2_r[k]:.6f}`"
        )
        w(f"- target x values: `x_u_target={_fmt(x_u_target[k])}`, `x_r_target={_fmt(x_r_target[k])}`")
        w(f"- target contribution to E_cdr_unmit: `{_fmt(contrib_u[k])}`")
        w(f"- target contribution to E_cdr_rem: `{_fmt(contrib_r[k])}`")
        w("")

    w("## Complete expanded energy expressions (first 20 terms only)")
    w("")
    w("### E_cdr_unmit")
    w("")
    parts_u = [_fmt(offset)] + [_fmt(c) for c in contrib_u[:terms_to_record]]
    w(f"`E_cdr_unmit = {' + '.join(parts_u)}`")
    w("")
    w("### E_cdr_rem")
    w("")
    parts_r = [_fmt(offset)] + [_fmt(c) for c in contrib_r[:terms_to_record]]
    w(f"`E_cdr_rem = {' + '.join(parts_r)}`")
    w("")
    w("## Headline values (target circuit)")
    w("")
    w(
        f"- raw finite-shot (unmit / REM): "
        f"`{mit['unmit_target']:.12f} / {mit['rem_target']:.12f} Eh`"
    )
    w(
        f"- cdr corrected (unmit / REM): "
        f"`{mit['cdr_unmit_corrected']:.12f} / {mit['cdr_rem_corrected']:.12f} Eh`"
    )
    w(f"- reference exact noiseless: `{eref:.12f} Eh`")
    w(f"- Energy error (exact - cdr_rem): `{eref - mit['cdr_rem_corrected']:.12f} Eh`")
    w("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path} ({len(lines)} lines)")


if __name__ == "__main__":
    write_doc(_REPO / "docs" / "full_CDR_details_for_LiH_testing_case_2nd version.md")
